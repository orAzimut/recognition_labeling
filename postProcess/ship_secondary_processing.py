#!/usr/bin/env python3
"""
Ship Secondary Processing Tool

A standalone script that processes Label Studio tagged data from Stage 1 and performs:
1. Group extraction: Extract multiple clean groups from labeled tasks (MainGroup, OutGroup1, etc.)
2. Secondary matching: Find matches between all clean groups using Ship-Recognition-Service

IMPORTANT CHANGES (Updated Labeling Format):
- Human labelers now create separate clean groups instead of marking outsiders
- Each labeled task can produce multiple clean groups (MainGroup + OutGroups)
- All groups proceed directly to Stage 2 secondary matching
- No feedback loop back to Stage 1
- Vessels are added to 'system_gallery' via clean REST API
- Requires the Ship-Recognition-Service to be running and accessible

Usage:
    python ship_secondary_processing.py \
        --bucket azimut_data \
        --labeled-json "/reidentification/silver/Initial_groups_phase_cleaned/label_studio_exports/2025-09-08_13-22_36Tasks.json" \
        --service-url "http://localhost:8080"

Author: Ship Recognition Team
"""

import os
import sys
import json
import logging
import argparse
import base64
import uuid
from datetime import datetime
from pathlib import Path
from typing import List, Dict, Set, Optional, Tuple, Any
from collections import defaultdict
import requests
from google.cloud import storage



class GCSClient:
    """Google Cloud Storage client for managing ship recognition data."""
    
    def __init__(self, credentials_path: Optional[str] = None):
        """Initialize GCS client."""
        self.logger = logging.getLogger(f"{__name__}.{self.__class__.__name__}")
        
        if credentials_path and os.path.exists(credentials_path):
            os.environ['GOOGLE_APPLICATION_CREDENTIALS'] = credentials_path
        
        try:
            self.client = storage.Client()
            self.logger.info("GCS client initialized successfully")
        except Exception as e:
            self.logger.error(f"Failed to initialize GCS client: {e}")
            raise
    
    def load_labeled_json(self, bucket_name: str, gcs_path: str) -> List[Dict[str, Any]]:
        """Load Label Studio export JSON from GCS with auto-retry for common path variations."""
        bucket = self.client.bucket(bucket_name)
        
        # Try the provided path first
        paths_to_try = [gcs_path.lstrip('/')]
        
        # Add common spelling variations if not already tried
        original_path = gcs_path.lstrip('/')
        if 'label_studio_exports' in original_path:
            # Try with 'lable' spelling
            variant_path = original_path.replace('label_studio_exports', 'lable_studio_exports')
            if variant_path != original_path:
                paths_to_try.append(variant_path)
        elif 'lable_studio_exports' in original_path:
            # Try with 'label' spelling
            variant_path = original_path.replace('lable_studio_exports', 'label_studio_exports')
            if variant_path != original_path:
                paths_to_try.append(variant_path)
        
        # Try each path variation
        for attempt, blob_path in enumerate(paths_to_try, 1):
            try:
                blob = bucket.blob(blob_path)
                
                if blob.exists():
                    if attempt > 1:
                        self.logger.info(f"Found file at alternate path: gs://{bucket_name}/{blob_path}")
                    
                    json_content = blob.download_as_text()
                    labeled_data = json.loads(json_content)
                    
                    if not isinstance(labeled_data, list):
                        raise ValueError(f"Expected JSON array, got {type(labeled_data)}")
                    
                    if len(labeled_data) == 0:
                        raise ValueError("Label Studio JSON is empty")
                    
                    # Validate first entry has required fields (support both old and new formats)
                    first_group = labeled_data[0]
                    
                    # Check for r_id (required in both formats)
                    if 'r_id' not in first_group:
                        raise ValueError("Label Studio JSON missing required field: r_id")
                    
                    # Check for either old format (images/jsons) or new format (images_data)
                    has_old_format = 'images' in first_group and 'jsons' in first_group
                    has_new_format = 'images_data' in first_group
                    
                    if not has_old_format and not has_new_format:
                        raise ValueError("Label Studio JSON missing required fields. Expected either ['images', 'jsons'] (old format) or ['images_data'] (new format)")
                    
                    format_type = "new format (images_data)" if has_new_format else "old format (images/jsons)"
                    self.logger.info(f"Detected Label Studio export format: {format_type}")
                    
                    self.logger.info(f"Loaded {len(labeled_data)} labeled groups from gs://{bucket_name}/{blob_path}")
                    self.logger.info(f"Sample group keys: {list(first_group.keys())}")
                    return labeled_data
                    
            except Exception as e:
                if attempt == len(paths_to_try):  # Last attempt failed
                    self.logger.error(f"Failed to load labeled JSON from {blob_path}: {e}")
                else:
                    self.logger.debug(f"Path attempt {attempt} failed: {blob_path}")
        
        # All paths failed
        self.logger.error("Label Studio JSON not found at any of the attempted paths:")
        for path in paths_to_try:
            self.logger.error(f"  - gs://{bucket_name}/{path}")
        
        self.logger.info("")
        self.logger.info("Common Label Studio export locations:")
        self.logger.info("  - /reidentification/silver/Initial_groups_phase_cleaned/lable_studio_exports/")
        self.logger.info("  - /reidentification/silver/Initial_groups_phase_cleaned/label_studio_exports/")
        self.logger.info("  - /reidentification/bronze/labeling/label_studio_exports/")
        self.logger.info("")
        self.logger.info("To find the correct file, check your GCS bucket:")
        self.logger.info(f"  gsutil ls gs://{bucket_name}/reidentification/*/Initial_groups_phase_cleaned/*/")
        
        raise FileNotFoundError(f"Label Studio JSON not found at any attempted path in gs://{bucket_name}")
    
    def download_image_as_base64(self, gcs_path: str) -> str:
        """Download an image from GCS and return as base64 string."""
        if not gcs_path.startswith('gs://'):
            raise ValueError(f"Invalid GCS path format: {gcs_path}")
        
        path_parts = gcs_path[5:].split('/', 1)
        bucket_name = path_parts[0]
        blob_name = path_parts[1]
        
        bucket = self.client.bucket(bucket_name)
        blob = bucket.blob(blob_name)
        
        try:
            image_data = blob.download_as_bytes()
            return base64.b64encode(image_data).decode('utf-8')
        except Exception as e:
            self.logger.error(f"Failed to download {gcs_path}: {e}")
            raise
    
    def download_json_metadata(self, gcs_path: str) -> Optional[Dict[str, Any]]:
        """
        Download and parse JSON metadata from GCS.
        
        Args:
            gcs_path: Full GCS path (gs://bucket/path/to/file.json)
            
        Returns:
            Parsed JSON data as dictionary, or None if failed
        """
        # Parse GCS path
        if not gcs_path.startswith('gs://'):
            self.logger.error(f"Invalid GCS path format: {gcs_path}")
            return None
        
        path_parts = gcs_path[5:].split('/', 1)  # Remove 'gs://' prefix
        bucket_name = path_parts[0]
        blob_name = path_parts[1]
        
        bucket = self.client.bucket(bucket_name)
        blob = bucket.blob(blob_name)
        
        try:
            if not blob.exists():
                self.logger.warning(f"JSON metadata file does not exist: {gcs_path}")
                return None
                
            json_data = blob.download_as_text()
            return json.loads(json_data)
        except Exception as e:
            self.logger.error(f"Failed to download or parse JSON metadata {gcs_path}: {e}")
            return None
    
    def save_match_pair_json(self, bucket_name: str, match_data: Dict[str, Any]) -> bool:
        """Save a match pair JSON file to GCS."""
        try:
            bucket = self.client.bucket(bucket_name)
            r_id_1 = match_data['r_id_1']
            r_id_2 = match_data['r_id_2']
            blob_path = f"reidentification/bronze/labeling/secondary_matching_phase/match_{r_id_1}_{r_id_2}.json"
            blob = bucket.blob(blob_path)
            
            # Ensure directory exists (GCS creates it automatically)
            blob.upload_from_string(json.dumps(match_data, indent=2))
            self.logger.info(f"Saved match pair {r_id_1}-{r_id_2} to GCS")
            return True
            
        except Exception as e:
            self.logger.error(f"Failed to save match pair {match_data.get('r_id_1')}-{match_data.get('r_id_2')}: {e}")
            return False
    
    
    def delete_match_pair_json(self, bucket_name: str, r_id_1: str, r_id_2: str) -> bool:
        """Delete a match pair JSON file from GCS."""
        try:
            bucket = self.client.bucket(bucket_name)
            blob_path = f"reidentification/bronze/labeling/secondary_matching_phase/match_{r_id_1}_{r_id_2}.json"
            blob = bucket.blob(blob_path)
            
            if blob.exists():
                blob.delete()
                self.logger.info(f"Deleted duplicate match pair {r_id_1}-{r_id_2}")
                return True
            else:
                self.logger.warning(f"Match pair file {r_id_1}-{r_id_2} not found for deletion")
                return False
                
        except Exception as e:
            self.logger.error(f"Failed to delete match pair {r_id_1}-{r_id_2}: {e}")
            return False
    
    def load_processed_files(self, bucket_name: str, list_filename: str) -> Set[str]:
        """
        Load the list of already processed files from GCS.
        
        Args:
            bucket_name: GCS bucket name
            list_filename: Name of the tracking file (e.g., 'secondary_processing_files_list.json')
            
        Returns:
            Set of processed file paths
        """
        try:
            bucket = self.client.bucket(bucket_name)
            blob_path = f"reidentification/bronze/labeling/{list_filename}"
            blob = bucket.blob(blob_path)
            
            if blob.exists():
                data = json.loads(blob.download_as_text())
                processed_files = set(data.get('processed_files', []))
                self.logger.info(f"Loaded {len(processed_files)} processed files from GCS")
                return processed_files
            else:
                self.logger.info("No existing processed files list found in GCS")
                return set()
                
        except Exception as e:
            self.logger.error(f"Failed to load processed files list: {e}")
            return set()
    
    def save_processed_files(self, bucket_name: str, list_filename: str, processed_files: Set[str]) -> bool:
        """
        Save the list of processed files to GCS.
        
        Args:
            bucket_name: GCS bucket name
            list_filename: Name of the tracking file
            processed_files: Set of processed file paths
            
        Returns:
            True if successful, False otherwise
        """
        try:
            bucket = self.client.bucket(bucket_name)
            blob_path = f"reidentification/bronze/labeling/{list_filename}"
            blob = bucket.blob(blob_path)
            
            data = {
                'processed_files': sorted(list(processed_files)),
                'last_updated': datetime.utcnow().isoformat() + 'Z',
                'total_count': len(processed_files)
            }
            
            blob.upload_from_string(json.dumps(data, indent=2))
            self.logger.info(f"Saved {len(processed_files)} processed files to GCS")
            return True
            
        except Exception as e:
            self.logger.error(f"Failed to save processed files list: {e}")
            return False


class RecognitionClient:
    """Enhanced client for communicating with the Ship-Recognition-Service."""
    
    def __init__(self, service_url: str = "http://localhost:8080"):
        """Initialize recognition client."""
        self.service_url = service_url.rstrip('/')
        self.session = requests.Session()
        self.logger = logging.getLogger(f"{__name__}.{self.__class__.__name__}")
    
    def health_check(self) -> bool:
        """Check if the recognition service is healthy."""
        try:
            response = self.session.get(f"{self.service_url}/health", timeout=10)
            is_healthy = response.status_code == 200
            self.logger.info(f"Recognition service health check: {'OK' if is_healthy else 'FAILED'}")
            return is_healthy
        except Exception as e:
            self.logger.error(f"Health check failed: {e}")
            return False
    
    def reset_gallery(self) -> bool:
        """Reset (clear) only the system_gallery, preserving other galleries like azimut_gallery."""
        self.logger.info("Resetting system_gallery (preserving other galleries for reference data)...")
        
        try:
            # Use the new API endpoint to clear system_gallery
            response = self.session.delete(f"{self.service_url}/galleries/system_gallery/reset", timeout=60)
            response.raise_for_status()
            
            result = response.json()
            status = result.get('status', 'unknown')
            message = result.get('message', 'No message provided')
            vessels_removed = result.get('vessels_removed', 0)
            
            if status == 'success':
                self.logger.info(f"Gallery reset successful: {message}")
                if vessels_removed > 0:
                    self.logger.info(f"Removed {vessels_removed} vessels from system_gallery")
                return True
            elif status == 'partial_success':
                failed_removals = result.get('failed_removals', 0)
                self.logger.warning(f"Gallery reset partially successful: {message}")
                self.logger.warning(f"Removed {vessels_removed} vessels, {failed_removals} failures")
                return True  # Still consider partial success as acceptable
            else:
                self.logger.error(f"Gallery reset failed: {message}")
                return False
                
        except requests.exceptions.RequestException as e:
            self.logger.error(f"Failed to reset gallery via API: {e}")
            return False
        except Exception as e:
            self.logger.error(f"Failed to reset gallery: {e}")
            return False
    
    def _get_gallery_vessels(self, gallery_name: str) -> List[str]:
        """Get list of vessels in a specific gallery."""
        try:
            if gallery_name == 'legacy':
                # For legacy gallery, try the general vessels endpoint
                response = self.session.get(f"{self.service_url}/gallery/vessels", timeout=30)
            else:
                # For named galleries, use the specific endpoint
                response = self.session.get(f"{self.service_url}/galleries/{gallery_name}/vessels", timeout=30)
            
            response.raise_for_status()
            vessels_data = response.json()
            
            # Extract vessel IDs based on response format
            vessel_ids = []
            if isinstance(vessels_data, list):
                # Handle list of vessel objects or simple strings
                for item in vessels_data:
                    if isinstance(item, dict):
                        # Extract vessel_id from vessel object
                        vessel_id = item.get('vessel_id')
                        if vessel_id:
                            vessel_ids.append(vessel_id)
                    elif isinstance(item, str):
                        vessel_ids.append(item)
            elif isinstance(vessels_data, dict):
                # Handle dict with vessels key
                vessels_list = vessels_data.get('vessels', [])
                for item in vessels_list:
                    if isinstance(item, dict):
                        vessel_id = item.get('vessel_id')
                        if vessel_id:
                            vessel_ids.append(vessel_id)
                    elif isinstance(item, str):
                        vessel_ids.append(item)
            
            self.logger.debug(f"Gallery '{gallery_name}': Found {len(vessel_ids)} vessel IDs")
            return vessel_ids
                
        except requests.exceptions.HTTPError as e:
            if e.response.status_code == 404:
                self.logger.debug(f"Gallery '{gallery_name}' not found or has no vessels endpoint")
                return []
            else:
                self.logger.error(f"HTTP error getting vessels for gallery {gallery_name}: {e}")
                return []
        except Exception as e:
            self.logger.error(f"Error getting vessels for gallery {gallery_name}: {e}")
            return []
    
    def _remove_vessel(self, vessel_id: str, gallery_name: str = 'system_gallery') -> bool:
        """Remove a single vessel from the gallery using REST API."""
        try:
            response = self.session.delete(
                f"{self.service_url}/gallery/vessels/{vessel_id}",
                params={'gallery_name': gallery_name},
                timeout=30
            )
            
            if response.status_code in [200, 204, 404]:  # 404 means already removed
                self.logger.debug(f"Removed vessel {vessel_id} from {gallery_name}")
                return True
            else:
                self.logger.debug(f"Failed to remove vessel {vessel_id}: HTTP {response.status_code}")
                return False
                
        except Exception as e:
            self.logger.debug(f"Error removing vessel {vessel_id}: {e}")
            return False
    
    
    def add_vessel_to_gallery(self, vessel_id: str, base64_images: List[str], metadata: Optional[Dict] = None, gallery_name: str = 'system_gallery') -> bool:
        """Add a vessel to the gallery using REST API with gallery_name parameter."""
        payload = {
            'vessel_id': vessel_id,
            'images': base64_images,
            'metadata': metadata or {},
            'gallery_name': gallery_name
        }
        
        try:
            response = self.session.post(
                f"{self.service_url}/gallery/vessels",
                json=payload,
                headers={'Content-Type': 'application/json'},
                timeout=120  # Longer timeout for batch operations
            )
            response.raise_for_status()
            
            result = response.json()
            if result.get('status') == 'success':
                self.logger.info(f"Successfully added vessel {vessel_id} to {gallery_name} gallery")
                return True
            else:
                self.logger.error(f"Failed to add vessel {vessel_id}: {result}")
                return False
                
        except requests.exceptions.RequestException as e:
            self.logger.error(f"Failed to add vessel {vessel_id} to {gallery_name} gallery: {e}")
            return False
        except Exception as e:
            self.logger.error(f"Unexpected error adding vessel {vessel_id}: {e}")
            return False
    
    def recognize_without_adding(self, base64_images: List[str], metadata: Optional[Dict] = None) -> Dict[str, Any]:
        """
        Recognize images against gallery without adding them.
        Uses the standard recognition endpoint which doesn't add to gallery.
        """
        payload = {
            'images': base64_images,
            'metadata': metadata or {'source': 'ship_secondary_processing'},
            'config': {
                'silhouette_threshold': 0.1,
                'mean_a_threshold': 0.7
            }
        }
        
        try:
            response = self.session.post(
                f"{self.service_url}/recognize/multi-gallery",
                json=payload,
                headers={'Content-Type': 'application/json'},
                timeout=60
            )
            response.raise_for_status()
            return response.json()
            
        except requests.exceptions.Timeout:
            self.logger.error("Recognition request timed out")
            return {'status': 'error', 'error_message': 'Request timeout'}
        except requests.exceptions.RequestException as e:
            self.logger.error(f"Recognition request failed: {e}")
            return {'status': 'error', 'error_message': str(e)}
        except Exception as e:
            self.logger.error(f"Unexpected error during recognition: {e}")
            return {'status': 'error', 'error_message': str(e)}


class LabelStudioProcessor:
    """Processor for Label Studio tagged data."""
    
    def __init__(self):
        """Initialize the processor."""
        self.logger = logging.getLogger(f"{__name__}.{self.__class__.__name__}")
    
    def extract_clean_groups(self, labeled_data: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Extract clean groups from labeled data, creating separate groups for each class (MainGroup, OutGroup1, etc.)."""
        clean_groups = []
        
        for group in labeled_data:
            try:
                original_r_id = group.get('r_id')
                images_data = group.get('images_data', [])
                
                if not images_data:
                    self.logger.warning(f"Group {original_r_id} has no images_data, skipping")
                    continue
                
                # Group images by their class (MainGroup, OutGroup1, OutGroup2, etc.)
                groups_by_class = {}
                
                self.logger.debug(f"Processing group {original_r_id} with {len(images_data)} images")
                
                for img_data in images_data:
                    is_trash = img_data.get('is_trash', False)
                    
                    if is_trash:
                        # Skip trash images completely
                        continue
                    
                    # Get the bounding boxes to determine class membership
                    bboxes = img_data.get('bboxes', [])
                    
                    if not bboxes:
                        # Image has no bounding boxes - skip it
                        continue
                    
                    # Get the class from the first bounding box (should be consistent)
                    bbox_class = bboxes[0].get('class', 'MainGroup')  # Default to MainGroup
                    
                    if bbox_class not in groups_by_class:
                        groups_by_class[bbox_class] = {
                            'images': [],
                            'jsons': []
                        }
                    
                    groups_by_class[bbox_class]['images'].append(img_data.get('url'))
                    groups_by_class[bbox_class]['jsons'].append(img_data.get('json_url'))
                
                # Create clean groups for each class
                for class_name, class_data in groups_by_class.items():
                    if class_data['images']:  # Only create groups with images
                        # Generate new r_id for each group
                        if class_name == 'MainGroup':
                            new_r_id = original_r_id  # Keep original r_id for main group
                        else:
                            # Create new r_id for out groups (e.g. "abc_out1" for "OutGroup1")
                            class_suffix = class_name.lower().replace('outgroup', 'out')
                            new_r_id = f"{original_r_id}_{class_suffix}"
                        
                        clean_group = {
                            'r_id': new_r_id,
                            'images': class_data['images'],
                            'jsons': class_data['jsons'],
                            'original_group': group,  # Keep reference to original for metadata
                            'class_name': class_name,  # Track which class this group represents
                            'original_r_id': original_r_id  # Track original r_id for reference
                        }
                        clean_groups.append(clean_group)
                        
                        self.logger.info(f"Extracted clean group {new_r_id} ({class_name}): "
                                       f"{len(class_data['images'])} images, {len(class_data['jsons'])} JSONs")
                
                if not groups_by_class:
                    self.logger.warning(f"Group {original_r_id} has no clean images after processing")
                    
            except Exception as e:
                self.logger.error(f"Error processing group {group.get('r_id', 'unknown')}: {e}")
                continue
        
        self.logger.info(f"Extracted {len(clean_groups)} clean groups from {len(labeled_data)} labeled tasks")
        return clean_groups
    
    
    def _corresponding_image_path(self, json_path: str) -> str:
        """Convert JSON metadata path to corresponding image path."""
        if '/json_metadata/' in json_path:
            # First convert the path structure
            image_path = json_path.replace('/json_metadata/', '/raw_crops/')
            
            # Handle GCS URL format to ensure proper gs:// protocol
            if image_path.startswith('gs:/') and not image_path.startswith('gs://'):
                image_path = image_path.replace('gs:/', 'gs://', 1)
            
            # Change extension from .json to image extensions (try common ones)
            for ext in ['.jpg', '.jpeg', '.png']:
                if image_path.endswith('.json'):
                    potential_image = image_path.replace('.json', ext)
                    return potential_image
        return json_path  # Fallback


class SecondaryMatcher:
    """Engine for secondary matching between clean groups."""
    
    def __init__(self, recognition_client: RecognitionClient, gcs_client: GCSClient):
        """Initialize the secondary matcher."""
        self.recognition_client = recognition_client
        self.gcs_client = gcs_client
        self.logger = logging.getLogger(f"{__name__}.{self.__class__.__name__}")
        self.match_pairs = []  # List of (r_id_1, r_id_2, match_data) tuples
    
    def load_groups_to_gallery(self, clean_groups: List[Dict[str, Any]]) -> bool:
        """Load all clean groups as vessels in the gallery."""
        self.logger.info(f"Loading {len(clean_groups)} clean groups to gallery...")
        
        successful_loads = 0
        for i, group in enumerate(clean_groups, 1):
            try:
                r_id = group['r_id']
                images = group['images']
                
                self.logger.info(f"Loading group {i}/{len(clean_groups)}: {r_id} ({len(images)} images)")
                
                # Download images as base64
                base64_images = []
                for image_path in images:
                    try:
                        b64_image = self.gcs_client.download_image_as_base64(image_path)
                        base64_images.append(b64_image)
                    except Exception as e:
                        self.logger.warning(f"Failed to download {image_path}: {e}")
                
                if not base64_images:
                    self.logger.warning(f"No images could be downloaded for group {r_id}")
                    continue
                
                # Add to gallery (specify system_gallery explicitly)
                success = self.recognition_client.add_vessel_to_gallery(
                    vessel_id=r_id,
                    base64_images=base64_images,
                    metadata={
                        'source': 'ship_secondary_processing',
                        'num_images': len(images),
                        'created_at': datetime.utcnow().isoformat() + 'Z'
                    },
                    gallery_name='system_gallery'
                )
                
                if success:
                    successful_loads += 1
                else:
                    self.logger.error(f"Failed to load group {r_id} to gallery")
                    
            except Exception as e:
                self.logger.error(f"Error loading group {group.get('r_id', 'unknown')}: {e}")
                continue
        
        self.logger.info(f"Successfully loaded {successful_loads}/{len(clean_groups)} groups to gallery")
        return successful_loads > 0
    
    def find_secondary_matches(self, clean_groups: List[Dict[str, Any]]) -> List[Tuple[str, str, Dict]]:
        """Find secondary matches for all groups."""
        self.logger.info(f"Finding secondary matches for {len(clean_groups)} groups...")
        
        matches_found = 0
        for i, group in enumerate(clean_groups, 1):
            try:
                r_id = group['r_id']
                images = group['images']
                
                self.logger.info(f"Processing group {i}/{len(clean_groups)}: {r_id}")
                
                # Download images as base64
                base64_images = []
                for image_path in images:
                    try:
                        b64_image = self.gcs_client.download_image_as_base64(image_path)
                        base64_images.append(b64_image)
                    except Exception as e:
                        self.logger.warning(f"Failed to download {image_path}: {e}")
                
                if not base64_images:
                    self.logger.warning(f"No images available for recognition of group {r_id}")
                    continue
                
                # Recognize against gallery (without adding)
                result = self.recognition_client.recognize_without_adding(
                    base64_images,
                    metadata={'source': 'secondary_matching', 'r_id': r_id}
                )
                
                if result.get('status') == 'error':
                    self.logger.error(f"Recognition failed for group {r_id}: {result.get('error_message')}")
                    continue
                
                # Extract second best match
                second_match = self._extract_second_match(result, r_id, clean_groups)
                if second_match:
                    match_data = self._create_match_data(group, second_match, clean_groups)
                    if match_data:
                        self.match_pairs.append((r_id, second_match['vessel_id'], match_data))
                        matches_found += 1
                        self.logger.info(f"Found match: {r_id} -> {second_match['vessel_id']}")
                else:
                    self.logger.warning(f"No second match found for group {r_id}")
                    
            except Exception as e:
                self.logger.error(f"Error processing group {group.get('r_id', 'unknown')}: {e}")
                continue
        
        self.logger.info(f"Found {matches_found} secondary matches")
        return self.match_pairs
    
    def _extract_second_match(self, result: Dict[str, Any], r_id: str, clean_groups: List[Dict[str, Any]]) -> Optional[Dict]:
        """Extract the second best match from recognition results, only considering matches within current batch."""
        # Handle multi-gallery results
        if 'results_per_gallery' in result:
            for gallery_name, gallery_result in result['results_per_gallery'].items():
                top_matches = gallery_result.get('top_matches', [])
                self.logger.debug(f"Gallery '{gallery_name}' has {len(top_matches)} matches for {r_id}")
                
                if len(top_matches) >= 2:
                    # Skip first match (should be self-match)
                    first_match = top_matches[0]
                    if first_match.get('vessel_id') == r_id:
                        # Check remaining matches for ones in current batch
                        for match in top_matches[1:]:
                            vessel_id = match.get('vessel_id')
                            if self._vessel_exists_in_current_batch(vessel_id, clean_groups):
                                self.logger.info(f"Found valid second match for {r_id}: {vessel_id} (from {gallery_name})")
                                return match
                            else:
                                self.logger.debug(f"Skipping match {vessel_id} for {r_id} - not in current batch (from {gallery_name})")
                    else:
                        # If first match is not self, check if it's in current batch
                        vessel_id = first_match.get('vessel_id')
                        if self._vessel_exists_in_current_batch(vessel_id, clean_groups):
                            self.logger.info(f"Found valid first match for {r_id}: {vessel_id} (from {gallery_name})")
                            return first_match
                        else:
                            self.logger.debug(f"Skipping first match {vessel_id} for {r_id} - not in current batch (from {gallery_name})")
                            # Check remaining matches
                            for match in top_matches[1:]:
                                vessel_id = match.get('vessel_id')
                                if self._vessel_exists_in_current_batch(vessel_id, clean_groups):
                                    self.logger.info(f"Found valid match for {r_id}: {vessel_id} (from {gallery_name})")
                                    return match
                                else:
                                    self.logger.debug(f"Skipping match {vessel_id} for {r_id} - not in current batch (from {gallery_name})")
        
        # Handle single gallery results
        else:
            top_matches = result.get('top_matches', [])
            self.logger.debug(f"Single gallery has {len(top_matches)} matches for {r_id}")
            
            if len(top_matches) >= 2:
                first_match = top_matches[0]
                if first_match.get('vessel_id') == r_id:
                    # Check remaining matches for ones in current batch
                    for match in top_matches[1:]:
                        vessel_id = match.get('vessel_id')
                        if self._vessel_exists_in_current_batch(vessel_id, clean_groups):
                            self.logger.info(f"Found valid second match for {r_id}: {vessel_id}")
                            return match
                        else:
                            self.logger.debug(f"Skipping match {vessel_id} for {r_id} - not in current batch")
                else:
                    # Check all matches for ones in current batch
                    for match in top_matches:
                        vessel_id = match.get('vessel_id')
                        if self._vessel_exists_in_current_batch(vessel_id, clean_groups):
                            self.logger.info(f"Found valid match for {r_id}: {vessel_id}")
                            return match
                        else:
                            self.logger.debug(f"Skipping match {vessel_id} for {r_id} - not in current batch")
        
        self.logger.warning(f"No valid matches found for {r_id} within current batch")
        return None
    
    def _vessel_exists_in_current_batch(self, vessel_id: str, clean_groups: List[Dict[str, Any]]) -> bool:
        """Check if a vessel_id exists in the current batch of clean groups."""
        if not vessel_id:
            return False
        
        for group in clean_groups:
            if group.get('r_id') == vessel_id:
                return True
        return False
    
    def _create_match_data(self, group1: Dict, second_match: Dict, all_groups: List[Dict]) -> Optional[Dict]:
        """Create match pair data structure."""
        try:
            r_id_1 = group1['r_id']
            r_id_2 = second_match['vessel_id']
            
            # Find group2 data
            group2 = None
            for g in all_groups:
                if g['r_id'] == r_id_2:
                    group2 = g
                    break
            
            if not group2:
                self.logger.error(f"Could not find group data for {r_id_2}")
                return None
            
            # Extract bounding boxes for both groups
            r_id_1_bboxes = self._extract_bboxes_from_jsons(group1['jsons'])
            r_id_2_bboxes = self._extract_bboxes_from_jsons(group2['jsons'])
            
            match_data = {
                'r_id_1': r_id_1,
                'r_id_1_images': group1['images'],
                'r_id_1_jsons': group1['jsons'],
                'r_id_1_bboxes': r_id_1_bboxes,
                'r_id_2': r_id_2,
                'r_id_2_images': group2['images'],
                'r_id_2_jsons': group2['jsons'],
                'r_id_2_bboxes': r_id_2_bboxes,
                'similarity_score': second_match.get('similarity_score', 0.0),
                'created_at': datetime.utcnow().isoformat() + 'Z'
            }
            
            return match_data
            
        except Exception as e:
            self.logger.error(f"Error creating match data: {e}")
            return None
    
    def _extract_bboxes_from_jsons(self, json_paths: List[str]) -> List[List[int]]:
        """
        Extract bounding boxes from JSON metadata files.
        
        Args:
            json_paths: List of GCS paths to JSON metadata files
            
        Returns:
            List of bounding boxes in [x1, y1, x2, y2] format
        """
        bboxes = []
        
        for json_path in json_paths:
            try:
                # Download and parse JSON metadata
                json_metadata = self.gcs_client.download_json_metadata(json_path)
                
                if (json_metadata and 
                    'target' in json_metadata and 
                    'bounding_box' in json_metadata['target'] and
                    'bounding_box' in json_metadata['target']['bounding_box']):
                    
                    bbox = json_metadata['target']['bounding_box']['bounding_box']
                    
                    # Validate bbox format - should be [x1, y1, x2, y2]
                    if isinstance(bbox, list) and len(bbox) == 4:
                        # Ensure all values are integers
                        bbox_int = [int(coord) for coord in bbox]
                        bboxes.append(bbox_int)
                    else:
                        self.logger.warning(f"Invalid bounding box format in {json_path}: {bbox}")
                        bboxes.append([0, 0, 0, 0])  # Default bbox
                else:
                    self.logger.warning(f"No bounding box data found in {json_path}")
                    bboxes.append([0, 0, 0, 0])  # Default bbox
                    
            except Exception as e:
                self.logger.warning(f"Failed to extract bounding box from {json_path}: {e}")
                bboxes.append([0, 0, 0, 0])  # Default bbox
        
        return bboxes



def setup_logging(log_level: str = "INFO") -> logging.Logger:
    """Setup logging configuration."""
    logging.basicConfig(
        level=getattr(logging, log_level.upper()),
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.StreamHandler(),
            logging.FileHandler('ship_secondary_processing.log') if os.access('.', os.W_OK) else logging.NullHandler()
        ]
    )
    return logging.getLogger(__name__)


def parse_arguments() -> argparse.Namespace:
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description='Ship Secondary Processing Tool - Stage 2',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python ship_secondary_processing.py \\
    --bucket azimut_data \\
    --labeled-json "/reidentification/silver/Initial_groups_phase_cleaned/label_studio_exports/2025-09-04_15-36_7Tasks.json"

  python ship_secondary_processing.py \\
    --bucket azimut_data \\
    --labeled-json "/path/to/labeled_export.json" \\
    --service-url "http://192.168.1.100:8080" \\
    --credentials "/path/to/gcs-key.json"
        """
    )
    
    parser.add_argument(
        '--bucket',
        default="azimut_data",
        help='GCS bucket name (e.g., "azimut_data")'
    )
    
    parser.add_argument(
        '--labeled-json',
        default="/reidentification/silver/Initial_groups_phase_cleaned/lable_studio_exports/2025-09-18_09-19_677Tasks.json",
        help='GCS path to Label Studio export JSON file'
    )

    parser.add_argument(
        '--service-url',
        default='http://localhost:8080',
        help='URL of the Ship-Recognition-Service (default: http://localhost:8080)'
    )
    
    parser.add_argument(
        '--credentials',
        default=r"C:\Users\OrGil.AzureAD\OneDrive - AMPC\Desktop\Azimut.ai\recognition_labeling\resources\credentials.json",
        help='Path to GCS service account credentials JSON file'
    )
    
    parser.add_argument(
        '--log-level',
        default='INFO',
        choices=['DEBUG', 'INFO', 'WARNING', 'ERROR'],
        help='Logging level (default: INFO)'
    )
    
    parser.add_argument(
        '--dry-run',
        action='store_true',
        default=False,
        help='Perform a dry run without saving results to GCS'
    )
    
    parser.add_argument(
        '--skip-duplicates',
        action='store_true',
        default=False,
        help='Skip duplicate elimination step (keep all match pairs)'
    )
    
    return parser.parse_args()


def main():
    """Main entry point for the ship secondary processing tool."""
    # Parse arguments and setup logging
    args = parse_arguments()
    logger = setup_logging(args.log_level)
    
    logger.info("Starting Ship Secondary Processing Tool (Stage 2)")
    logger.info(f"Arguments: {vars(args)}")
    
    try:
        # Initialize clients
        logger.info("Initializing clients...")
        gcs_client = GCSClient(args.credentials)
        recognition_client = RecognitionClient(args.service_url)
        
        # Health check
        logger.info("Checking recognition service health...")
        if not recognition_client.health_check():
            logger.error("Recognition service is not healthy. Exiting.")
            return 1
        
        # Automatic gallery reset
        logger.info("Resetting system_gallery (preserving other galleries for reference)...")
        if not args.dry_run:
            if not recognition_client.reset_gallery():
                logger.error("Failed to reset gallery. Exiting.")
                return 1
        else:
            logger.info("Dry run mode - gallery reset skipped")
        
        # Load processed files tracking list
        logger.info("Loading processed files tracking list...")
        processed_files = gcs_client.load_processed_files(args.bucket, "secondary_processing_files_list.json")
        logger.info(f"Found {len(processed_files)} previously processed files")
        
        # Check if current input file was already processed
        if args.labeled_json in processed_files:
            logger.info(f"File {args.labeled_json} was already processed. Skipping.")
            logger.info("To reprocess this file, remove it from the processed files list or use a different input file.")
            logger.info("Ship Secondary Processing Tool completed (skipped - already processed)")
            return 0
        
        # Load Label Studio data
        logger.info(f"Loading Label Studio data from {args.labeled_json}...")
        labeled_data = gcs_client.load_labeled_json(args.bucket, args.labeled_json)
        
        # Process labeled data
        logger.info("Processing labeled data...")
        processor = LabelStudioProcessor()
        
        # Extract clean groups (which may include multiple groups per original task)
        clean_groups = processor.extract_clean_groups(labeled_data)
        
        if not clean_groups:
            logger.error("No clean groups found in labeled data. Exiting.")
            return 1
        
        # Initialize secondary matcher
        logger.info("Initializing secondary matcher...")
        matcher = SecondaryMatcher(recognition_client, gcs_client)
        
        # Load groups to gallery
        logger.info("Loading clean groups to gallery...")
        if not matcher.load_groups_to_gallery(clean_groups):
            logger.error("Failed to load groups to gallery. Exiting.")
            return 1
        
        # Find secondary matches
        logger.info("Finding secondary matches...")
        match_pairs = matcher.find_secondary_matches(clean_groups)
        
        if not match_pairs:
            logger.warning("No secondary matches found.")
        else:
            # Save match pairs
            if not args.dry_run:
                logger.info("Saving match pairs to GCS...")
                saved_pairs = 0
                for r_id_1, r_id_2, match_data in match_pairs:
                    if gcs_client.save_match_pair_json(args.bucket, match_data):
                        saved_pairs += 1
                
                logger.info(f"Saved {saved_pairs}/{len(match_pairs)} match pairs")
                
                # Duplicate elimination
                eliminated_count = 0
                if not args.skip_duplicates:
                    logger.info("Eliminating duplicate pairs...")
                    eliminated_count = eliminate_duplicate_pairs(match_pairs, gcs_client, args.bucket)
                    logger.info(f"Eliminated {eliminated_count} duplicate pairs")
            
            else:
                logger.info("Dry run - match pairs not saved:")
                for r_id_1, r_id_2, match_data in match_pairs:
                    similarity = match_data.get('similarity_score', 0)
                    logger.info(f"  Match: {r_id_1} -> {r_id_2} (similarity: {similarity:.3f})")
                eliminated_count = 0
        
        
        # Update processed files tracking list
        if not args.dry_run:
            logger.info("Updating processed files tracking list...")
            processed_files.add(args.labeled_json)
            if gcs_client.save_processed_files(args.bucket, "secondary_processing_files_list.json", processed_files):
                logger.info(f"Added {args.labeled_json} to processed files list")
            else:
                logger.warning("Failed to update processed files tracking list")
        
        logger.info("Ship Secondary Processing Tool completed successfully")
        logger.info("")
        logger.info("Results saved to GCS:")
        if not args.dry_run:
            final_match_pairs = len(match_pairs) - eliminated_count if match_pairs else 0
            if final_match_pairs > 0:
                logger.info(f"   • Match pairs ({final_match_pairs}): gs://{args.bucket}/reidentification/bronze/labeling/secondary_matching_phase/")
            if final_match_pairs == 0:
                logger.info("   • No match pairs created")
        else:
            logger.info("   • Dry run - no files saved")
        return 0
        
    except Exception as e:
        logger.error(f"Fatal error: {e}")
        import traceback
        traceback.print_exc()
        return 1


def eliminate_duplicate_pairs(match_pairs: List[Tuple[str, str, Dict]], gcs_client: GCSClient, bucket_name: str) -> int:
    """Eliminate duplicate bidirectional pairs."""
    logger = logging.getLogger(__name__)
    
    # Create set of all pairs for fast lookup
    pairs_set = set()
    pairs_to_delete = []
    
    for r_id_1, r_id_2, match_data in match_pairs:
        pair = (r_id_1, r_id_2)
        reverse_pair = (r_id_2, r_id_1)
        
        if reverse_pair in pairs_set:
            # This pair is a reverse of an existing pair - mark for deletion
            pairs_to_delete.append(pair)
            logger.info(f"Marked pair {r_id_1}-{r_id_2} for deletion (reverse of {r_id_2}-{r_id_1})")
        else:
            pairs_set.add(pair)
    
    # Delete marked pairs
    deleted_count = 0
    for r_id_1, r_id_2 in pairs_to_delete:
        if gcs_client.delete_match_pair_json(bucket_name, r_id_1, r_id_2):
            deleted_count += 1
    
    return deleted_count


if __name__ == "__main__":
    sys.exit(main())
