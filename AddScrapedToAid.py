#!/usr/bin/env python3
"""
Production ShipSpotting → AID Gallery Augmentor
Creates a new timestamped gallery and augments ALL groups with ShipSpotting frames.
"""

import argparse
import json
import logging
import os
import sys
from datetime import datetime, timezone
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass, field

from google.cloud import storage
from google.oauth2 import service_account


@dataclass
class AugmentationResult:
    """Results from augmentation operation"""
    group_id: str
    imo: Optional[str] = None
    frames_copied: int = 0
    frames_skipped: int = 0
    appearances_added: int = 0
    total_images: int = 0
    metadata_updated: bool = False
    error: Optional[str] = None


@dataclass 
class GalleryReport:
    """Overall gallery processing report"""
    gallery_name: str
    total_groups: int = 0
    groups_with_imo: int = 0
    groups_augmented: int = 0
    total_frames_added: int = 0
    total_frames_skipped: int = 0
    groups_failed: List[str] = field(default_factory=list)
    group_results: List[AugmentationResult] = field(default_factory=list)


class GCSManager:
    """Manages Google Cloud Storage operations"""
    
    def __init__(self, credentials_path: Optional[str] = None):
        """Initialize GCS client with credentials"""
        if credentials_path and os.path.exists(credentials_path):
            credentials = service_account.Credentials.from_service_account_file(
                credentials_path
            )
            self.client = storage.Client(credentials=credentials)
        else:
            # Use default credentials
            self.client = storage.Client()
    
    def parse_gcs_path(self, gcs_path: str) -> Tuple[str, str]:
        """Parse GCS path into bucket and blob name"""
        if not gcs_path.startswith('gs://'):
            raise ValueError(f"Invalid GCS path: {gcs_path}")
        
        path = gcs_path[5:]  # Remove 'gs://'
        parts = path.split('/', 1)
        bucket_name = parts[0]
        blob_name = parts[1] if len(parts) > 1 else ''
        
        return bucket_name, blob_name
    
    def blob_exists(self, gcs_path: str) -> bool:
        """Check if a blob exists"""
        bucket_name, blob_name = self.parse_gcs_path(gcs_path)
        bucket = self.client.bucket(bucket_name)
        blob = bucket.blob(blob_name)
        return blob.exists()
    
    def list_blobs(self, gcs_path: str) -> List[str]:
        """List all blobs under a prefix"""
        bucket_name, prefix = self.parse_gcs_path(gcs_path)
        bucket = self.client.bucket(bucket_name)
        
        blobs = []
        for blob in bucket.list_blobs(prefix=prefix):
            if not blob.name.endswith('/'):  # Skip directories
                blobs.append(f"gs://{bucket_name}/{blob.name}")
        
        return blobs
    
    def list_directories(self, gcs_path: str) -> List[str]:
        """List all directories under a prefix"""
        bucket_name, prefix = self.parse_gcs_path(gcs_path)
        if not prefix.endswith('/'):
            prefix += '/'
        
        bucket = self.client.bucket(bucket_name)
        
        # Use delimiter to get directory-like listing
        directories = set()
        for blob in bucket.list_blobs(prefix=prefix, delimiter='/'):
            pass  # We just need to iterate to populate prefixes
        
        # Get prefixes (directories)
        for prefix_item in bucket.list_blobs(prefix=prefix, delimiter='/').prefixes:
            # Extract directory name
            dir_name = prefix_item[len(prefix):].rstrip('/')
            if dir_name:  # Skip empty
                directories.add(dir_name)
        
        return sorted(list(directories))
    
    def copy_blob(self, source_path: str, dest_path: str, skip_existing: bool = True) -> bool:
        """Copy a blob from source to destination. Returns True if copied, False if skipped."""
        if skip_existing and self.blob_exists(dest_path):
            logging.debug(f"Skipping existing blob: {dest_path}")
            return False
        
        source_bucket_name, source_blob_name = self.parse_gcs_path(source_path)
        dest_bucket_name, dest_blob_name = self.parse_gcs_path(dest_path)
        
        source_bucket = self.client.bucket(source_bucket_name)
        source_blob = source_bucket.blob(source_blob_name)
        dest_bucket = self.client.bucket(dest_bucket_name)
        
        source_bucket.copy_blob(source_blob, dest_bucket, dest_blob_name)
        logging.debug(f"Copied: {source_path} → {dest_path}")
        return True
    
    def read_json(self, gcs_path: str) -> Dict:
        """Read JSON from GCS"""
        bucket_name, blob_name = self.parse_gcs_path(gcs_path)
        bucket = self.client.bucket(bucket_name)
        blob = bucket.blob(blob_name)
        
        content = blob.download_as_text()
        return json.loads(content)
    
    def write_json(self, gcs_path: str, data: Dict, atomic: bool = True):
        """Write JSON to GCS (optionally atomic via temp file)"""
        bucket_name, blob_name = self.parse_gcs_path(gcs_path)
        bucket = self.client.bucket(bucket_name)
        
        json_content = json.dumps(data, indent=2, ensure_ascii=False)
        
        if atomic:
            # Write to temp blob first, then rename
            temp_blob_name = f"{blob_name}.tmp.{datetime.now().timestamp()}"
            temp_blob = bucket.blob(temp_blob_name)
            temp_blob.upload_from_string(json_content, content_type='application/json')
            
            # Rename temp to final
            bucket.rename_blob(temp_blob, blob_name)
        else:
            blob = bucket.blob(blob_name)
            blob.upload_from_string(json_content, content_type='application/json')
        
        logging.debug(f"Wrote JSON to: {gcs_path}")


class ProductionGalleryAugmentor:
    """Production augmentor for all gallery groups"""
    
    def __init__(self, gcs_manager: GCSManager, dry_run: bool = False):
        self.gcs = gcs_manager
        self.dry_run = dry_run
        
        # Source gallery
        self.source_gallery = "gs://azimut_data/reidentification/silver/galleries/verified/AID_Gallery_20250914_170413"
        
        # Generate new gallery name with timestamp
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.dest_gallery = f"gs://azimut_data/reidentification/silver/galleries/verified/AID_Gallery_{timestamp}"
        
        # ShipSpotting paths
        self.shipspotting_base = "gs://outsource_data/reidentification/bronze/raw_crops/ship_spotting"
        self.shipspotting_json_base = "gs://outsource_data/reidentification/bronze/json_lables/ship_spotting"
        
        logging.info(f"New gallery will be created at: {self.dest_gallery}")
    
    def find_matching_json(self, image_filename: str, json_files: List[str]) -> Optional[str]:
        """Find matching JSON file for an image"""
        image_stem = os.path.splitext(image_filename)[0]
        
        for json_path in json_files:
            json_filename = json_path.split('/')[-1]
            json_stem = os.path.splitext(json_filename)[0]
            if json_stem == image_stem:
                return json_path
        
        return None
    
    def copy_gallery_group(self, group_id: str) -> int:
        """Copy a single gallery group from source to destination"""
        source_group = f"{self.source_gallery}/{group_id}"
        dest_group = f"{self.dest_gallery}/{group_id}"
        
        files_copied = 0
        
        # Copy metadata.json
        source_metadata = f"{source_group}/metadata.json"
        dest_metadata = f"{dest_group}/metadata.json"
        
        if not self.dry_run:
            if self.gcs.copy_blob(source_metadata, dest_metadata):
                files_copied += 1
        else:
            logging.debug(f"[DRY-RUN] Would copy: {source_metadata} → {dest_metadata}")
        
        # Copy images directory
        source_images = f"{source_group}/images"
        dest_images = f"{dest_group}/images"
        
        image_files = self.gcs.list_blobs(source_images)
        for source_image in image_files:
            filename = source_image.split('/')[-1]
            dest_image = f"{dest_images}/{filename}"
            
            if not self.dry_run:
                if self.gcs.copy_blob(source_image, dest_image):
                    files_copied += 1
            else:
                logging.debug(f"[DRY-RUN] Would copy image: {source_image} → {dest_image}")
        
        logging.info(f"Group {group_id}: Copied {files_copied} files")
        return files_copied
    
    def augment_group_with_shipspotting(self, group_id: str) -> AugmentationResult:
        """Augment a single group with ShipSpotting frames"""
        result = AugmentationResult(group_id=group_id)
        
        # Read metadata
        metadata_path = f"{self.dest_gallery}/{group_id}/metadata.json"
        
        try:
            metadata = self.gcs.read_json(metadata_path)
        except Exception as e:
            result.error = f"Failed to read metadata: {e}"
            logging.error(f"Group {group_id}: {result.error}")
            return result
        
        # Extract IMO
        imo = metadata.get('imo')
        if not imo:
            logging.info(f"Group {group_id}: No IMO found - skipping augmentation")
            return result
        
        result.imo = str(imo)
        logging.info(f"Group {group_id}: Found IMO {imo}")
        
        # Build ShipSpotting paths
        imo_dir = f"IMO_{imo}"
        ss_frames_path = f"{self.shipspotting_base}/{imo_dir}"
        ss_jsons_path = f"{self.shipspotting_json_base}/{imo_dir}"
        
        # List ShipSpotting frames and JSONs
        try:
            ss_frames = self.gcs.list_blobs(ss_frames_path)
            logging.info(f"Group {group_id}: Found {len(ss_frames)} ShipSpotting frames for IMO {imo}")
        except Exception as e:
            logging.warning(f"Group {group_id}: No ShipSpotting frames found for IMO {imo}: {e}")
            ss_frames = []
        
        try:
            ss_jsons = self.gcs.list_blobs(ss_jsons_path)
            logging.info(f"Group {group_id}: Found {len(ss_jsons)} ShipSpotting JSONs")
        except Exception as e:
            logging.warning(f"Group {group_id}: No ShipSpotting JSONs found: {e}")
            ss_jsons = []
        
        if not ss_frames:
            logging.info(f"Group {group_id}: No ShipSpotting frames to process")
            return result
        
        # Get existing appearances
        appearances = metadata.get('appearances', [])
        existing_images = {app['gallery_image_name'] for app in appearances}
        
        # Build set of already imported ShipSpotting files
        already_imported_files = set()
        for app in appearances:
            orig_path = app.get('original_img_path', '')
            if 'ship_spotting' in orig_path:
                filename = orig_path.split('/')[-1]
                already_imported_files.add(filename)
        
        if already_imported_files:
            logging.info(f"Group {group_id}: {len(already_imported_files)} ShipSpotting images already imported")
        
        # Get the next image index
        next_index = len(appearances)
        
        # Copy frames and update appearances
        dest_images_path = f"{self.dest_gallery}/{group_id}/images"
        new_appearances = []
        
        for frame_path in ss_frames:
            filename = frame_path.split('/')[-1]
            
            # Check if already imported
            if filename in already_imported_files:
                result.frames_skipped += 1
                logging.debug(f"Group {group_id}: Skipping already imported {filename}")
                continue
            
            # Create gallery image name with index prefix
            gallery_image_name = f"{next_index}_{filename}"
            
            # Skip if already exists in gallery
            if gallery_image_name in existing_images:
                result.frames_skipped += 1
                continue
            
            # Copy frame to destination
            dest_frame_path = f"{dest_images_path}/{gallery_image_name}"
            
            if not self.dry_run:
                if self.gcs.copy_blob(frame_path, dest_frame_path):
                    result.frames_copied += 1
                else:
                    result.frames_skipped += 1
                    continue
            else:
                logging.debug(f"[DRY-RUN] Would copy: {frame_path} → {dest_frame_path}")
                result.frames_copied += 1
            
            # Find matching JSON
            matching_json = self.find_matching_json(filename, ss_jsons)
            
            # Create appearance entry
            original_img_path = frame_path.replace('gs://', '')
            original_json_path = matching_json.replace('gs://', '') if matching_json else None
            
            appearance = {
                "gallery_image_name": gallery_image_name,
                "image_index": next_index,
                "bbox_padding": [50, 50, 50, 50],
                "original_img_path": original_img_path,
                "original_json_path": original_json_path or original_img_path.replace('.jpg', '.json'),
                "source": "web_imos"
            }
            
            new_appearances.append(appearance)
            next_index += 1
        
        # Update metadata if we have new appearances
        if not self.dry_run and new_appearances:
            metadata['appearances'].extend(new_appearances)
            result.appearances_added = len(new_appearances)
            
            metadata['total_images'] = len(metadata['appearances'])
            result.total_images = metadata['total_images']
            
            if 'audit' not in metadata:
                metadata['audit'] = {}
            
            metadata['audit']['last_augmented_from'] = 'ship_spotting'
            metadata['audit']['last_updated_utc'] = datetime.now(timezone.utc).isoformat()
            
            self.gcs.write_json(metadata_path, metadata, atomic=True)
            result.metadata_updated = True
            
            logging.info(f"Group {group_id}: Added {result.appearances_added} new appearances")
        elif self.dry_run:
            result.appearances_added = len(new_appearances)
            result.total_images = len(appearances) + len(new_appearances)
        
        return result
    
    def process_all_groups(self) -> GalleryReport:
        """Process all groups in the source gallery"""
        report = GalleryReport(gallery_name=self.dest_gallery)
        
        # List all groups in source gallery
        logging.info(f"Listing groups in {self.source_gallery}")
        
        # Get all subdirectories (groups) in the source gallery
        groups = self.gcs.list_directories(self.source_gallery)
        
        if not groups:
            # Alternative method: list all blobs and extract group IDs
            all_blobs = self.gcs.list_blobs(self.source_gallery)
            groups = set()
            for blob in all_blobs:
                # Extract group ID from path
                path_parts = blob.replace(f"{self.source_gallery}/", "").split('/')
                if len(path_parts) > 0:
                    groups.add(path_parts[0])
            groups = sorted(list(groups))
        
        report.total_groups = len(groups)
        logging.info(f"Found {report.total_groups} groups to process: {groups}")
        
        # Process each group
        for i, group_id in enumerate(groups, 1):
            logging.info(f"\n{'='*60}")
            logging.info(f"Processing group {i}/{report.total_groups}: {group_id}")
            logging.info(f"{'='*60}")
            
            try:
                # Copy the group
                self.copy_gallery_group(group_id)
                
                # Augment with ShipSpotting
                result = self.augment_group_with_shipspotting(group_id)
                report.group_results.append(result)
                
                if result.imo:
                    report.groups_with_imo += 1
                    if result.appearances_added > 0:
                        report.groups_augmented += 1
                        report.total_frames_added += result.frames_copied
                    report.total_frames_skipped += result.frames_skipped
                
                if result.error:
                    report.groups_failed.append(group_id)
                    
            except Exception as e:
                logging.error(f"Failed to process group {group_id}: {e}")
                report.groups_failed.append(group_id)
                result = AugmentationResult(group_id=group_id, error=str(e))
                report.group_results.append(result)
        
        return report


def print_gallery_report(report: GalleryReport):
    """Print comprehensive gallery processing report"""
    print("\n" + "="*80)
    print("PRODUCTION GALLERY AUGMENTATION REPORT")
    print("="*80)
    print(f"New Gallery: {report.gallery_name}")
    print(f"Total Groups Processed: {report.total_groups}")
    print(f"Groups with IMO: {report.groups_with_imo}")
    print(f"Groups Successfully Augmented: {report.groups_augmented}")
    print(f"Total Frames Added: {report.total_frames_added}")
    print(f"Total Frames Skipped (duplicates): {report.total_frames_skipped}")
    
    if report.groups_failed:
        print(f"\nFailed Groups ({len(report.groups_failed)}):")
        for group_id in report.groups_failed:
            print(f"  - {group_id}")
    
    print("\n" + "-"*80)
    print("DETAILED GROUP RESULTS:")
    print("-"*80)
    
    for result in report.group_results:
        if result.imo:  # Only show groups with IMO
            print(f"\nGroup {result.group_id}:")
            print(f"  IMO: {result.imo}")
            print(f"  Frames Added: {result.frames_copied}")
            print(f"  Frames Skipped: {result.frames_skipped}")
            print(f"  Total Images: {result.total_images}")
            if result.error:
                print(f"  ERROR: {result.error}")
    
    print("\n" + "="*80)
    print(f"Gallery creation completed: {report.gallery_name}")
    print("="*80)


def main():
    """Main entry point"""
    parser = argparse.ArgumentParser(
        description="Production: Create new AID Gallery and augment ALL groups with ShipSpotting"
    )
    parser.add_argument(
        '--credentials',
        default=r"C:\Users\OrGil.AzureAD\OneDrive - AMPC\Desktop\Azimut.ai\recognition_labeling\resources\credentials.json",
        help='Path to GCS credentials JSON file'
    )
    parser.add_argument(
        '--dry-run',
        action='store_true',
        help='Run without making changes (preview mode)'
    )
    parser.add_argument(
        '--verbose',
        action='store_true',
        help='Enable verbose logging'
    )
    parser.add_argument(
        '--no-report',
        action='store_true',
        help='Skip printing the final report'
    )
    
    args = parser.parse_args()
    
    # Setup logging
    log_level = logging.DEBUG if args.verbose else logging.INFO
    logging.basicConfig(
        level=log_level,
        format='%(asctime)s - %(levelname)s - %(message)s'
    )
    
    if args.dry_run:
        logging.info("DRY-RUN MODE: No changes will be made")
    
    try:
        # Initialize GCS manager
        gcs_manager = GCSManager(credentials_path=args.credentials)
        
        # Initialize production augmentor
        augmentor = ProductionGalleryAugmentor(gcs_manager, dry_run=args.dry_run)
        
        # Confirm before proceeding (unless dry-run)
        if not args.dry_run:
            print(f"\n{'='*80}")
            print(f"PRODUCTION RUN - This will create a new gallery:")
            print(f"Source: {augmentor.source_gallery}")
            print(f"Destination: {augmentor.dest_gallery}")
            print(f"{'='*80}")
            response = input("\nProceed with production run? (yes/no): ")
            if response.lower() != 'yes':
                print("Operation cancelled.")
                sys.exit(0)
        
        # Process all groups
        logging.info("\nStarting gallery processing...")
        report = augmentor.process_all_groups()
        
        # Print report
        if not args.no_report:
            print_gallery_report(report)
        
        # Exit with appropriate code
        if report.groups_failed:
            logging.warning(f"Completed with {len(report.groups_failed)} failures")
            sys.exit(1)
        else:
            logging.info("All groups processed successfully!")
        
    except Exception as e:
        logging.error(f"Fatal error: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()