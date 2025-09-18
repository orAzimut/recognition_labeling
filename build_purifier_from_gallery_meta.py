#!/usr/bin/env python3
"""
Gallery to Label Studio Batch Converter

Purpose
-------
Process ALL gallery metadata JSON files in a specific gallery batch and convert 
them to Label Studio format for annotation/verification tasks.

What it does
------------
- Scans all subdirectories in the hardcoded gallery path
- For each group/subdirectory found:
  - Reads the metadata_transformed.json file
  - Fetches all associated JSON metadata to get bounding box information
  - Converts bounding boxes to Label Studio percentage format
  - Outputs a Label Studio JSON named after the group

Hardcoded Input Path
--------------------
gs://azimut_data/reidentification/silver/galleries/verified/AID_Gallery_20250915_135425/

Outputs
-------
gs://azimut_data/reidentification/bronze/labeling/AID_Gallery_Purifier/group_{group_name}.json

Usage
-----
python gallery_label_converter.py

Author: Gallery Processing Team
"""

import os
import json
import logging
import argparse
from datetime import datetime
from pathlib import Path
from typing import List, Dict, Optional, Any

from google.cloud import storage


# ============================= HARDCODED PATH ================================
GALLERY_BASE_PATH = "gs://azimut_data/reidentification/silver/galleries/verified/AID_Gallery_20250915_135425"
OUTPUT_BASE_PATH = "gs://azimut_data/reidentification/bronze/labeling/AID_Gallery_Purifier"
# =============================================================================


# ----------------------------- GCS Client ------------------------------------
class GCSClient:
    """Google Cloud Storage client for gallery processing."""

    def __init__(self, credentials_path: Optional[str] = None):
        self.logger = logging.getLogger(f"{__name__}.{self.__class__.__name__}")
        if credentials_path and os.path.exists(credentials_path):
            os.environ['GOOGLE_APPLICATION_CREDENTIALS'] = credentials_path
        try:
            self.client = storage.Client()
            self.logger.info("GCS client initialized successfully")
        except Exception as e:
            self.logger.error(f"Failed to initialize GCS client: {e}")
            raise

    def list_gallery_groups(self, base_path: str) -> List[str]:
        """List all group directories in the gallery base path."""
        if not base_path.startswith('gs://'):
            self.logger.error(f"Invalid GCS path format: {base_path}")
            return []
        
        try:
            # Parse GCS path
            path_parts = base_path[5:].split('/', 1)
            bucket_name = path_parts[0]
            prefix = path_parts[1] if len(path_parts) > 1 else ""
            
            if not prefix.endswith('/'):
                prefix += '/'
            
            bucket = self.client.bucket(bucket_name)
            
            # Method 1: Try using delimiter to get prefixes
            blobs = bucket.list_blobs(prefix=prefix, delimiter='/')
            
            # Consume the iterator to populate prefixes
            list(blobs)  # This is needed to populate the prefixes
            
            groups = []
            if blobs.prefixes:
                for prefix_obj in blobs.prefixes:
                    # Extract group name from path
                    group_name = prefix_obj.rstrip('/').split('/')[-1]
                    groups.append(group_name)
                self.logger.info(f"Found {len(groups)} groups using delimiter method")
            
            # Method 2: If no prefixes found, scan for metadata_transformed.json files
            if not groups:
                self.logger.info("No prefixes found, scanning for metadata_transformed.json files...")
                all_blobs = bucket.list_blobs(prefix=prefix)
                metadata_files = set()
                
                for blob in all_blobs:
                    if blob.name.endswith('/metadata_transformed.json'):
                        # Extract the group name from the path
                        # Path format: .../verified/AID_Gallery.../GROUP_NAME/metadata_transformed.json
                        path_parts = blob.name[len(prefix):].split('/')
                        if len(path_parts) >= 2:  # Should be GROUP_NAME/metadata_transformed.json
                            group_name = path_parts[0]
                            metadata_files.add(group_name)
                
                groups = sorted(list(metadata_files))
                self.logger.info(f"Found {len(groups)} groups by scanning metadata_transformed.json files")
            
            if groups:
                self.logger.info(f"Groups found: {groups[:10]}..." if len(groups) > 10 else f"Groups found: {groups}")
            else:
                self.logger.warning(f"No groups found in {base_path}")
                self.logger.warning("Please verify the path exists and contains subdirectories with metadata_transformed.json files")
            
            return sorted(groups)
            
        except Exception as e:
            self.logger.error(f"Failed to list groups: {e}")
            import traceback
            traceback.print_exc()
            return []

    def download_json(self, gcs_path: str) -> Optional[Dict[str, Any]]:
        """Download and parse JSON from GCS path."""
        if not gcs_path.startswith('gs://'):
            self.logger.error(f"Invalid GCS path format: {gcs_path}")
            return None
        
        try:
            # Parse GCS path
            path_parts = gcs_path[5:].split('/', 1)
            if len(path_parts) != 2:
                self.logger.error(f"Invalid GCS path structure: {gcs_path}")
                return None
            
            bucket_name, blob_name = path_parts
            bucket = self.client.bucket(bucket_name)
            blob = bucket.blob(blob_name)
            
            if not blob.exists():
                self.logger.warning(f"File not found: {gcs_path}")
                return None
            
            content = blob.download_as_text()
            return json.loads(content)
        except Exception as e:
            self.logger.error(f"Failed to download/parse JSON from {gcs_path}: {e}")
            return None

    def save_label_studio_json(self, output_path: str, data: List[Dict[str, Any]]) -> bool:
        """Save Label Studio format JSON to GCS."""
        try:
            if not output_path.startswith('gs://'):
                self.logger.error(f"Invalid GCS path format: {output_path}")
                return False
            
            # Parse GCS path
            path_parts = output_path[5:].split('/', 1)
            if len(path_parts) != 2:
                self.logger.error(f"Invalid GCS path structure: {output_path}")
                return False
            
            bucket_name, blob_name = path_parts
            bucket = self.client.bucket(bucket_name)
            blob = bucket.blob(blob_name)
            
            # Upload JSON data
            blob.upload_from_string(json.dumps(data, indent=2))
            self.logger.info(f"Saved Label Studio JSON to {output_path}")
            return True
        except Exception as e:
            self.logger.error(f"Failed to save Label Studio JSON: {e}")
            return False


# ----------------------------- Gallery Processor -----------------------------
class GalleryProcessor:
    """Process gallery metadata and create Label Studio format output."""

    def __init__(self, gcs_client: GCSClient):
        self.gcs_client = gcs_client
        self.logger = logging.getLogger(f"{__name__}.{self.__class__.__name__}")

    def process_gallery(self, gallery_metadata_path: str, group_name: str) -> Optional[Dict[str, Any]]:
        """Process a gallery metadata JSON and create Label Studio format output."""
        
        # Download gallery metadata
        self.logger.info(f"Processing group '{group_name}' from {gallery_metadata_path}")
        metadata = self.gcs_client.download_json(gallery_metadata_path)
        
        if not metadata:
            self.logger.error(f"Failed to load gallery metadata for group {group_name}")
            return None
        
        azimut_id = metadata.get('azimut_id', 'unknown')
        imo = metadata.get('imo', 'unknown')
        appearances = metadata.get('appearances', [])
        
        self.logger.info(f"Group {group_name}: azimut_id={azimut_id}, IMO={imo}, appearances={len(appearances)}")
        
        # Process all appearances
        images = []
        jsons = []
        predictions_result = []
        
        for idx, appearance in enumerate(appearances):
            img_path = appearance.get('original_img_path', '')
            json_path = appearance.get('original_json_path', '')
            bbox_padding = appearance.get('bbox_padding', [0, 0, 0, 0])
            
            if not img_path or not json_path:
                self.logger.warning(f"Skipping appearance {idx}: missing paths")
                continue
            
            # Convert to full GCS paths if needed
            img_gcs = self._to_gcs_path(img_path)
            json_gcs = self._to_gcs_path(json_path)
            
            images.append(img_gcs)
            jsons.append(json_gcs)
            
            # Get bounding box from JSON metadata
            bbox_data = self._extract_bbox_data(json_gcs, bbox_padding)
            
            # Create prediction entry
            pred = self._create_prediction_entry(idx, bbox_data)
            predictions_result.append(pred)
        
        if not images:
            self.logger.warning(f"No valid images found in group {group_name}")
            return None
        
        # Create Label Studio format output
        group = {
            'data': {
                'r_id': group_name,  # Use group name as ID
                'group_name': group_name,
                'azimut_id': azimut_id,
                'images': images,
                'jsons': jsons,
                'imo': imo,
                'created_at': datetime.utcnow().isoformat() + 'Z',
            },
            'predictions': [{
                'model_version': 'gallery-converter',
                'score': 1.0,
                'result': predictions_result,
            }],
        }
        
        return group

    def _to_gcs_path(self, path: str) -> str:
        """Convert path to full GCS format."""
        if path.startswith('gs://'):
            return path
        # If path doesn't start with gs://, prepend it
        if path.startswith('/'):
            path = path[1:]
        return f"gs://{path}"

    def _extract_bbox_data(self, json_path: str, bbox_padding: List[int]) -> Dict[str, Any]:
        """Extract bounding box data from JSON metadata."""
        metadata = self.gcs_client.download_json(json_path)
        
        if not metadata:
            self.logger.warning(f"Could not load metadata from {json_path}")
            return {'bbox': None, 'padded_bbox': None, 'width': 0, 'height': 0}
        
        target = metadata.get('target', {})
        bbox_info = target.get('bounding_box', {})
        
        bbox = bbox_info.get('bounding_box')  # [x1, y1, x2, y2]
        padded_bbox = bbox_info.get('padded_bounding_box')
        
        # Calculate dimensions
        width = 0
        height = 0
        
        if padded_bbox and len(padded_bbox) == 4:
            width = padded_bbox[2] - padded_bbox[0]
            height = padded_bbox[3] - padded_bbox[1]
        elif bbox and len(bbox) == 4:
            # If no padded bbox, calculate from original bbox + padding
            pad_top, pad_right, pad_bottom, pad_left = bbox_padding
            width = (bbox[2] - bbox[0]) + pad_left + pad_right
            height = (bbox[3] - bbox[1]) + pad_top + pad_bottom
        
        return {
            'bbox': bbox,
            'padded_bbox': padded_bbox,
            'padding': bbox_padding,
            'width': width,
            'height': height
        }

    def _create_prediction_entry(self, idx: int, bbox_data: Dict[str, Any]) -> Dict[str, Any]:
        """Create a Label Studio prediction entry."""
        pred = {
            'from_name': 'bbox_labels',
            'to_name': 'images',
            'type': 'rectanglelabels',
            'original_width': bbox_data['width'],
            'original_height': bbox_data['height'],
            'value': {
                'item_index': idx,
                'x': 0.0,
                'y': 0.0,
                'width': 0.0,
                'height': 0.0,
                'rotation': 0,
                'rectanglelabels': ['Ship'],
                'image_rotation': 0,
                'image_index': idx,
            },
            'meta': {'item_index': idx, 'image_index': idx},
            'item_index': idx,
            'image_index': idx,
            'image_rotation': 0,
        }
        
        # Calculate percentage coordinates if we have bbox data
        if bbox_data['bbox'] and bbox_data['width'] > 0 and bbox_data['height'] > 0:
            bbox = bbox_data['bbox']  # [x1, y1, x2, y2]
            
            if bbox_data['padded_bbox']:
                # Use padded bbox as reference
                pb = bbox_data['padded_bbox']
                x_offset = pb[0]
                y_offset = pb[1]
            else:
                # Calculate offset from padding
                pad_top, _, _, pad_left = bbox_data['padding']
                x_offset = bbox[0] - pad_left
                y_offset = bbox[1] - pad_top
            
            # Calculate percentages
            x_perc = ((bbox[0] - x_offset) / bbox_data['width']) * 100.0
            y_perc = ((bbox[1] - y_offset) / bbox_data['height']) * 100.0
            w_perc = ((bbox[2] - bbox[0]) / bbox_data['width']) * 100.0
            h_perc = ((bbox[3] - bbox[1]) / bbox_data['height']) * 100.0
            
            # Ensure values are within valid range
            x_perc = max(0, min(100, x_perc))
            y_perc = max(0, min(100, y_perc))
            w_perc = max(0, min(100, w_perc))
            h_perc = max(0, min(100, h_perc))
            
            pred['value'].update({
                'x': x_perc,
                'y': y_perc,
                'width': w_perc,
                'height': h_perc,
            })
        
        return pred


# ----------------------------- Batch Processor -------------------------------
class BatchProcessor:
    """Process all galleries in a batch."""
    
    def __init__(self, gcs_client: GCSClient, gallery_processor: GalleryProcessor):
        self.gcs_client = gcs_client
        self.gallery_processor = gallery_processor
        self.logger = logging.getLogger(f"{__name__}.{self.__class__.__name__}")
    
    def process_all_galleries(self, dry_run: bool = False):
        """Process all gallery groups in the hardcoded path."""
        self.logger.info(f"Starting batch processing for: {GALLERY_BASE_PATH}")
        
        # List all groups
        groups = self.gcs_client.list_gallery_groups(GALLERY_BASE_PATH)
        
        if not groups:
            self.logger.warning("No groups found to process")
            return
        
        self.logger.info(f"Found {len(groups)} groups to process: {groups}")
        
        success_count = 0
        failed_groups = []
        
        for group_name in groups:
            try:
                # Build metadata path
                metadata_path = f"{GALLERY_BASE_PATH}/{group_name}/metadata_transformed.json"
                
                # Process the gallery
                group_data = self.gallery_processor.process_gallery(metadata_path, group_name)
                
                if not group_data:
                    self.logger.warning(f"Skipping group {group_name} - no data generated")
                    failed_groups.append(group_name)
                    continue
                
                # Build output path
                output_path = f"{OUTPUT_BASE_PATH}/group_{group_name}.json"
                
                if dry_run:
                    self.logger.info(f"[DRY RUN] Would save group {group_name} to: {output_path}")
                    self.logger.debug(f"Would process {len(group_data['data']['images'])} images")
                else:
                    # Save the output
                    success = self.gcs_client.save_label_studio_json(output_path, [group_data])
                    
                    if success:
                        success_count += 1
                        self.logger.info(f"✓ Saved group {group_name} ({len(group_data['data']['images'])} images)")
                    else:
                        failed_groups.append(group_name)
                        self.logger.error(f"✗ Failed to save group {group_name}")
                
            except Exception as e:
                self.logger.error(f"Error processing group {group_name}: {e}")
                failed_groups.append(group_name)
                continue
        
        # Summary
        self.logger.info("=" * 60)
        self.logger.info(f"Batch processing complete!")
        self.logger.info(f"Successfully processed: {success_count}/{len(groups)} groups")
        if failed_groups:
            self.logger.warning(f"Failed groups: {failed_groups}")


# ----------------------------- CLI / Main -------------------------------------

def setup_logging(log_level: str = 'INFO') -> logging.Logger:
    """Setup logging configuration."""
    logging.basicConfig(
        level=getattr(logging, log_level.upper()),
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.StreamHandler(),
            logging.FileHandler('gallery_batch_converter.log') if os.access('.', os.W_OK) else logging.NullHandler(),
        ],
    )
    return logging.getLogger(__name__)


def parse_arguments() -> argparse.Namespace:
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description='Batch convert all gallery metadata files to Label Studio format',
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        '--credentials',
        default=r"C:\Users\OrGil.AzureAD\OneDrive - AMPC\Desktop\Azimut.ai\recognition_labeling\resources\credentials.json",
        help='Path to GCS service account JSON'
    )
    parser.add_argument(
        '--log-level',
        default='INFO',
        choices=['DEBUG', 'INFO', 'WARNING', 'ERROR'],
        help='Logging level'
    )
    parser.add_argument(
        '--dry-run',
        action='store_true',
        help='Process but do not save output'
    )
    parser.add_argument(
        '--debug-list',
        action='store_true',
        help='Debug mode: list all files in the gallery path'
    )
    return parser.parse_args()


def main() -> int:
    """Main function."""
    args = parse_arguments()
    logger = setup_logging(args.log_level)
    
    try:
        logger.info("=" * 60)
        logger.info("Gallery Batch Converter Starting")
        logger.info(f"Processing gallery: {GALLERY_BASE_PATH}")
        logger.info(f"Output directory: {OUTPUT_BASE_PATH}")
        logger.info("=" * 60)
        
        logger.info("Initializing GCS client...")
        gcs_client = GCSClient(args.credentials)
        
        # Debug mode: list files in the path
        if getattr(args, 'debug_list', False):
            logger.info("DEBUG MODE: Listing files in gallery path...")
            path_parts = GALLERY_BASE_PATH[5:].split('/', 1)
            bucket_name = path_parts[0]
            prefix = path_parts[1] if len(path_parts) > 1 else ""
            
            bucket = gcs_client.client.bucket(bucket_name)
            blobs = bucket.list_blobs(prefix=prefix, max_results=50)
            
            logger.info(f"First 50 files in {GALLERY_BASE_PATH}:")
            for i, blob in enumerate(blobs, 1):
                logger.info(f"  {i}. {blob.name}")
            
            return 0
        
        logger.info("Creating processors...")
        gallery_processor = GalleryProcessor(gcs_client)
        batch_processor = BatchProcessor(gcs_client, gallery_processor)
        
        logger.info("Starting batch processing...")
        batch_processor.process_all_galleries(dry_run=args.dry_run)
        
        logger.info("All processing complete!")
        return 0
        
    except Exception as e:
        logger.error(f"Fatal error: {e}")
        import traceback
        traceback.print_exc()
        return 1


if __name__ == '__main__':
    import sys
    sys.exit(main())