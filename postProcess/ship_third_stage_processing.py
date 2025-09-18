#!/usr/bin/env python3
"""
Ship Third Stage Processing Tool

A standalone script that processes Label Studio labeled outputs from Stage 2 and performs:
1. Group merging: Merges groups based on labeler decisions (same_vessel field)
2. Secondary matching: Finds matches between merged/separate groups using Ship-Recognition-Service
3. Output generation: Creates match pairs in the same format as Stage 2 for consistency

IMPORTANT CHANGES (Clean REST API Integration):
- Vessels are now added directly to 'system_gallery' instead of legacy gallery
- Uses clean REST API communication with Ship-Recognition-Service
- Requires the Ship-Recognition-Service to be running and accessible

Usage:
    python ship_third_stage_processing.py \
        --bucket azimut_data \
        --labeled-exports-path "/reidentification/silver/Groups_Association_Phase_cleaned/lable_studio_exports/" \
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
    
    def load_labeled_json(self, bucket_name: str, json_path: str) -> List[Dict[str, Any]]:
        """Load a single labeled JSON file containing match pairs with bboxes."""
        if json_path.startswith('gs://'):
            # Handle full GCS URL
            path_parts = json_path[5:].split('/', 1)
            bucket_name = path_parts[0]
            blob_name = path_parts[1]
        else:
            # Handle relative path
            blob_name = json_path.lstrip('/')
        
        bucket = self.client.bucket(bucket_name)
        blob = bucket.blob(blob_name)
        
        try:
            if not blob.exists():
                raise FileNotFoundError(f"JSON file not found: gs://{bucket_name}/{blob_name}")
            
            self.logger.info(f"Loading labeled JSON: gs://{bucket_name}/{blob_name}")
            json_content = blob.download_as_text()
            labeled_data = json.loads(json_content)
            
            if not isinstance(labeled_data, list):
                raise ValueError(f"Expected JSON array, got {type(labeled_data)}")
            
            self.logger.info(f"Loaded {len(labeled_data)} labeled entries from JSON file")
            return labeled_data
            
        except Exception as e:
            self.logger.error(f"Failed to load labeled JSON from gs://{bucket_name}/{blob_name}: {e}")
            raise
    
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
    
    def save_third_stage_match_json(self, bucket_name: str, match_data: Dict[str, Any]) -> bool:
        """Save a third stage match pair JSON file to GCS."""
        try:
            bucket = self.client.bucket(bucket_name)
            r_id_1 = match_data['r_id_1']
            r_id_2 = match_data['r_id_2']
            blob_path = f"reidentification/bronze/labeling/third_matching_phase/match_{r_id_1}_{r_id_2}.json"
            blob = bucket.blob(blob_path)
            
            # Ensure directory exists (GCS creates it automatically)
            blob.upload_from_string(json.dumps(match_data, indent=2))
            self.logger.info(f"Saved third stage match pair {r_id_1}-{r_id_2} to GCS")
            return True
            
        except Exception as e:
            self.logger.error(f"Failed to save third stage match pair {match_data.get('r_id_1')}-{match_data.get('r_id_2')}: {e}")
            return False
    
    def delete_third_stage_match_json(self, bucket_name: str, r_id_1: str, r_id_2: str) -> bool:
        """Delete a third stage match pair JSON file from GCS."""
        try:
            bucket = self.client.bucket(bucket_name)
            blob_path = f"reidentification/bronze/labeling/third_matching_phase/match_{r_id_1}_{r_id_2}.json"
            blob = bucket.blob(blob_path)
            
            if blob.exists():
                blob.delete()
                self.logger.info(f"Deleted duplicate third stage match pair {r_id_1}-{r_id_2}")
                return True
            else:
                self.logger.warning(f"Third stage match pair file {r_id_1}-{r_id_2} not found for deletion")
                return False
                
        except Exception as e:
            self.logger.error(f"Failed to delete third stage match pair {r_id_1}-{r_id_2}: {e}")
            return False
    
# Processed files tracking methods removed - not needed for single file input


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
            'metadata': metadata or {'source': 'ship_third_stage_processing'},
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


class LabelStudioStage3Processor:
    """Processor for Label Studio Stage 3 (third stage) labeled data."""
    
    def __init__(self):
        """Initialize the processor."""
        self.logger = logging.getLogger(f"{__name__}.{self.__class__.__name__}")
    
    def extract_labeling_decisions(self, labeled_data: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Extract labeling decisions from labeled JSON with bboxes."""
        decisions = []
        
        for entry in labeled_data:
            try:
                # Extract key fields from labeled JSON
                r_id_1 = entry.get('r_id_1')
                r_id_2 = entry.get('r_id_2')
                same_vessel = entry.get('same_vessel', 'No')  # Default to 'No' if missing
                
                if not r_id_1 or not r_id_2:
                    self.logger.warning(f"Missing r_id fields in entry: {entry.keys()}")
                    continue
                
                # Extract image and JSON paths
                r_id_1_images = entry.get('r_id_1_images', [])
                r_id_1_jsons = entry.get('r_id_1_jsons', [])
                r_id_2_images = entry.get('r_id_2_images', [])
                r_id_2_jsons = entry.get('r_id_2_jsons', [])
                
                # Extract bounding boxes (new format)
                r_id_1_bboxes = entry.get('r_id_1_bboxes', [])
                r_id_2_bboxes = entry.get('r_id_2_bboxes', [])
                
                decision = {
                    'r_id_1': r_id_1,
                    'r_id_2': r_id_2,
                    'same_vessel': same_vessel,
                    'r_id_1_images': r_id_1_images,
                    'r_id_1_jsons': r_id_1_jsons,
                    'r_id_1_bboxes': r_id_1_bboxes,
                    'r_id_2_images': r_id_2_images,
                    'r_id_2_jsons': r_id_2_jsons,
                    'r_id_2_bboxes': r_id_2_bboxes,
                    'original_entry': entry
                }
                
                decisions.append(decision)
                self.logger.debug(f"Extracted decision: {r_id_1} + {r_id_2} = {same_vessel} (with {len(r_id_1_bboxes)} + {len(r_id_2_bboxes)} bboxes)")
                
            except Exception as e:
                self.logger.error(f"Error processing labeled entry: {e}")
                continue
        
        self.logger.info(f"Extracted {len(decisions)} labeling decisions from {len(labeled_data)} entries")
        return decisions


class GroupMerger:
    """Handles merging of groups based on Label Studio decisions."""
    
    def __init__(self):
        """Initialize the group merger."""
        self.logger = logging.getLogger(f"{__name__}.{self.__class__.__name__}")
    
    def merge_groups_by_decisions(self, decisions: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        Merge groups based on same_vessel decisions.
        Creates consolidated groups for processing.
        """
        # Track which groups should be merged
        merge_map = {}  # r_id -> merged_group_id
        merged_groups = {}  # merged_group_id -> group_data
        individual_groups = {}  # r_id -> group_data for non-merged groups
        processed_pairs = set()
        
        # Process merge decisions
        for decision in decisions:
            r_id_1 = decision['r_id_1']
            r_id_2 = decision['r_id_2']
            same_vessel = decision['same_vessel']
            
            # Create a canonical pair key to avoid duplicate processing
            pair_key = tuple(sorted([r_id_1, r_id_2]))
            if pair_key in processed_pairs:
                continue
            processed_pairs.add(pair_key)
            
            if same_vessel == 'Yes':
                # Merge these groups
                merged_id = self._get_or_create_merged_group(r_id_1, r_id_2, merge_map, merged_groups)
                self._add_to_merged_group(merged_id, decision, merged_groups)
                self.logger.info(f"Merging groups {r_id_1} and {r_id_2} into merged group {merged_id}")
            else:
                # Keep groups separate
                self._add_individual_groups(r_id_1, r_id_2, decision, individual_groups, merge_map)
        
        # Combine merged groups and individual groups
        final_groups = []
        
        # Add merged groups
        for merged_id, group_data in merged_groups.items():
            final_groups.append(group_data)
        
        # Add individual groups (those not involved in merging)
        for r_id, group_data in individual_groups.items():
            if r_id not in merge_map:  # Only add if not already merged
                final_groups.append(group_data)
        
        self.logger.info(f"Created {len(final_groups)} consolidated groups ({len(merged_groups)} merged, {len([g for g in individual_groups.values() if g['r_id'] not in merge_map])} individual)")
        return final_groups
    
    def _get_or_create_merged_group(self, r_id_1: str, r_id_2: str, merge_map: dict, merged_groups: dict) -> str:
        """Get existing merged group ID or create a new one."""
        # Check if either r_id is already in a merged group
        existing_merged_id = merge_map.get(r_id_1) or merge_map.get(r_id_2)
        
        if existing_merged_id:
            # Use existing merged group
            merge_map[r_id_1] = existing_merged_id
            merge_map[r_id_2] = existing_merged_id
            return existing_merged_id
        else:
            # Create new merged group
            merged_id = self._generate_merged_id()
            merge_map[r_id_1] = merged_id
            merge_map[r_id_2] = merged_id
            
            merged_groups[merged_id] = {
                'r_id': merged_id,
                'images': [],
                'jsons': [],
                'bboxes': [],
                'source_r_ids': set(),
                'is_merged': True
            }
            return merged_id
    
    def _add_to_merged_group(self, merged_id: str, decision: Dict[str, Any], merged_groups: dict):
        """Add decision data to a merged group."""
        group = merged_groups[merged_id]
        
        # Add images, jsons, and bboxes from both r_ids, avoiding duplicates
        all_images = decision['r_id_1_images'] + decision['r_id_2_images']
        all_jsons = decision['r_id_1_jsons'] + decision['r_id_2_jsons']
        all_bboxes = decision['r_id_1_bboxes'] + decision['r_id_2_bboxes']
        
        # Use sets to avoid duplicates, then convert back to lists
        existing_images = set(group['images'])
        existing_jsons = set(group['jsons'])
        
        for i, img in enumerate(all_images):
            if img not in existing_images:
                group['images'].append(img)
                existing_images.add(img)
                # Add corresponding bbox if available
                if i < len(all_bboxes):
                    group['bboxes'].append(all_bboxes[i])
                else:
                    # Default bbox if missing
                    group['bboxes'].append([0.0, 0.0, 0.0, 0.0])
        
        for json_path in all_jsons:
            if json_path not in existing_jsons:
                group['jsons'].append(json_path)
                existing_jsons.add(json_path)
        
        # Track source r_ids
        group['source_r_ids'].add(decision['r_id_1'])
        group['source_r_ids'].add(decision['r_id_2'])
    
    def _add_individual_groups(self, r_id_1: str, r_id_2: str, decision: Dict[str, Any], individual_groups: dict, merge_map: dict):
        """Add individual groups for r_ids that should remain separate."""
        # Only add if not already in a merged group
        if r_id_1 not in merge_map and r_id_1 not in individual_groups:
            individual_groups[r_id_1] = {
                'r_id': r_id_1,
                'images': decision['r_id_1_images'],
                'jsons': decision['r_id_1_jsons'],
                'bboxes': decision['r_id_1_bboxes'],
                'is_merged': False
            }
        
        if r_id_2 not in merge_map and r_id_2 not in individual_groups:
            individual_groups[r_id_2] = {
                'r_id': r_id_2,
                'images': decision['r_id_2_images'],
                'jsons': decision['r_id_2_jsons'],
                'bboxes': decision['r_id_2_bboxes'],
                'is_merged': False
            }
    
    def _generate_merged_id(self) -> str:
        """Generate a unique ID for a merged group."""
        import random
        import string
        return 'merged_' + ''.join(random.choices(string.ascii_lowercase + string.digits, k=3))


class ThirdStageSecondaryMatcher:
    """Engine for secondary matching between consolidated groups from Stage 3."""
    
    def __init__(self, recognition_client: RecognitionClient, gcs_client: GCSClient):
        """Initialize the third stage secondary matcher."""
        self.recognition_client = recognition_client
        self.gcs_client = gcs_client
        self.logger = logging.getLogger(f"{__name__}.{self.__class__.__name__}")
        self.match_pairs = []  # List of (r_id_1, r_id_2, match_data) tuples
    
    def load_groups_to_gallery(self, consolidated_groups: List[Dict[str, Any]]) -> bool:
        """Load all consolidated groups as vessels in the gallery."""
        self.logger.info(f"Loading {len(consolidated_groups)} consolidated groups to gallery...")
        
        successful_loads = 0
        for i, group in enumerate(consolidated_groups, 1):
            try:
                r_id = group['r_id']
                images = group['images']
                
                self.logger.info(f"Loading group {i}/{len(consolidated_groups)}: {r_id} ({len(images)} images)")
                
                # Download images as base64
                base64_images = []
                for image_path in images:
                    try:
                        # Convert Label Studio URL format to GCS path if needed
                        gcs_path = self._convert_to_gcs_path(image_path)
                        b64_image = self.gcs_client.download_image_as_base64(gcs_path)
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
                        'source': 'ship_third_stage_processing',
                        'num_images': len(images),
                        'is_merged': group.get('is_merged', False),
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
        
        self.logger.info(f"Successfully loaded {successful_loads}/{len(consolidated_groups)} groups to gallery")
        return successful_loads > 0
    
    def find_secondary_matches(self, consolidated_groups: List[Dict[str, Any]]) -> List[Tuple[str, str, Dict]]:
        """Find secondary matches for all consolidated groups."""
        self.logger.info(f"Finding secondary matches for {len(consolidated_groups)} groups...")
        
        matches_found = 0
        for i, group in enumerate(consolidated_groups, 1):
            try:
                r_id = group['r_id']
                images = group['images']
                
                self.logger.info(f"Processing group {i}/{len(consolidated_groups)}: {r_id}")
                
                # Download images as base64
                base64_images = []
                for image_path in images:
                    try:
                        gcs_path = self._convert_to_gcs_path(image_path)
                        b64_image = self.gcs_client.download_image_as_base64(gcs_path)
                        base64_images.append(b64_image)
                    except Exception as e:
                        self.logger.warning(f"Failed to download {image_path}: {e}")
                
                if not base64_images:
                    self.logger.warning(f"No images available for recognition of group {r_id}")
                    continue
                
                # Recognize against gallery (without adding)
                result = self.recognition_client.recognize_without_adding(
                    base64_images,
                    metadata={'source': 'third_stage_matching', 'r_id': r_id}
                )
                
                if result.get('status') == 'error':
                    self.logger.error(f"Recognition failed for group {r_id}: {result.get('error_message')}")
                    continue
                
                # Extract second best match
                second_match = self._extract_second_match(result, r_id, consolidated_groups)
                if second_match:
                    match_data = self._create_match_data(group, second_match, consolidated_groups)
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
    
    def _convert_to_gcs_path(self, image_path: str) -> str:
        """Convert Label Studio URL format to GCS path if needed."""
        if image_path.startswith('gs://'):
            return image_path
        elif image_path.startswith('/tasks/') and 'fileuri=' in image_path:
            # Extract base64-encoded GCS path from Label Studio URL
            import base64
            import urllib.parse
            
            try:
                # Extract the fileuri parameter
                parsed = urllib.parse.urlparse(image_path)
                query_params = urllib.parse.parse_qs(parsed.query)
                fileuri = query_params.get('fileuri', [None])[0]
                
                if fileuri:
                    # Decode base64 to get GCS path
                    decoded_path = base64.b64decode(fileuri).decode('utf-8')
                    if decoded_path.startswith('gs://'):
                        return decoded_path
            except Exception as e:
                self.logger.warning(f"Failed to decode Label Studio URL {image_path}: {e}")
        
        # Fallback - assume it's already a valid GCS path
        return image_path
    
    def _extract_second_match(self, result: Dict[str, Any], r_id: str, consolidated_groups: List[Dict[str, Any]]) -> Optional[Dict]:
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
                            if self._vessel_exists_in_current_batch(vessel_id, consolidated_groups):
                                self.logger.info(f"Found valid second match for {r_id}: {vessel_id} (from {gallery_name})")
                                return match
                            else:
                                self.logger.debug(f"Skipping match {vessel_id} for {r_id} - not in current batch (from {gallery_name})")
                    else:
                        # If first match is not self, check if it's in current batch
                        vessel_id = first_match.get('vessel_id')
                        if self._vessel_exists_in_current_batch(vessel_id, consolidated_groups):
                            self.logger.info(f"Found valid first match for {r_id}: {vessel_id} (from {gallery_name})")
                            return first_match
                        else:
                            self.logger.debug(f"Skipping first match {vessel_id} for {r_id} - not in current batch (from {gallery_name})")
                            # Check remaining matches
                            for match in top_matches[1:]:
                                vessel_id = match.get('vessel_id')
                                if self._vessel_exists_in_current_batch(vessel_id, consolidated_groups):
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
                        if self._vessel_exists_in_current_batch(vessel_id, consolidated_groups):
                            self.logger.info(f"Found valid second match for {r_id}: {vessel_id}")
                            return match
                        else:
                            self.logger.debug(f"Skipping match {vessel_id} for {r_id} - not in current batch")
                else:
                    # Check all matches for ones in current batch
                    for match in top_matches:
                        vessel_id = match.get('vessel_id')
                        if self._vessel_exists_in_current_batch(vessel_id, consolidated_groups):
                            self.logger.info(f"Found valid match for {r_id}: {vessel_id}")
                            return match
                        else:
                            self.logger.debug(f"Skipping match {vessel_id} for {r_id} - not in current batch")
        
        self.logger.warning(f"No valid matches found for {r_id} within current batch")
        return None
    
    def _vessel_exists_in_current_batch(self, vessel_id: str, consolidated_groups: List[Dict[str, Any]]) -> bool:
        """Check if a vessel_id exists in the current batch of consolidated groups."""
        if not vessel_id:
            return False
        
        for group in consolidated_groups:
            if group.get('r_id') == vessel_id:
                return True
        return False
    
    def _create_match_data(self, group1: Dict, second_match: Dict, all_groups: List[Dict]) -> Optional[Dict]:
        """Create match pair data structure in Stage 2 format with bboxes."""
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
            
            match_data = {
                'r_id_1': r_id_1,
                'r_id_1_images': [self._convert_to_gcs_path(img) for img in group1['images']],
                'r_id_1_jsons': [self._convert_to_gcs_path(json_path) for json_path in group1['jsons']],
                'r_id_1_bboxes': group1.get('bboxes', []),
                'r_id_2': r_id_2,
                'r_id_2_images': [self._convert_to_gcs_path(img) for img in group2['images']],
                'r_id_2_jsons': [self._convert_to_gcs_path(json_path) for json_path in group2['jsons']],
                'r_id_2_bboxes': group2.get('bboxes', []),
                'similarity_score': second_match.get('similarity_score', 0.0),
                'created_at': datetime.utcnow().isoformat() + 'Z'
            }
            
            return match_data
            
        except Exception as e:
            self.logger.error(f"Error creating match data: {e}")
            return None


def setup_logging(log_level: str = "INFO") -> logging.Logger:
    """Setup logging configuration."""
    logging.basicConfig(
        level=getattr(logging, log_level.upper()),
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.StreamHandler(),
            logging.FileHandler('ship_third_stage_processing.log') if os.access('.', os.W_OK) else logging.NullHandler()
        ]
    )
    return logging.getLogger(__name__)


def parse_arguments() -> argparse.Namespace:
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description='Ship Third Stage Processing Tool - Stage 3',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python ship_third_stage_processing.py \\
    --bucket azimut_data \\
    --labeled-json "/reidentification/silver/labeled_match_pairs.json"

  python ship_third_stage_processing.py \\
    --bucket azimut_data \\
    --labeled-json "/path/to/labeled_pairs.json" \\
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
        default="/reidentification/silver/Groups_Association_Phase_cleaned/lable_studio_exports/LS_185882_ACCEPTED_2025-09-18_14-11_9Tasks.json",
        help='GCS path to the labeled JSON file containing match pairs with bboxes'
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
    
    return parser.parse_args()


def main():
    """Main entry point for the ship third stage processing tool."""
    # Parse arguments and setup logging
    args = parse_arguments()
    logger = setup_logging(args.log_level)
    
    logger.info("Starting Ship Third Stage Processing Tool (Stage 3)")
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
        
        # Load labeled JSON file
        logger.info(f"Loading labeled JSON from {args.labeled_json}...")
        labeled_data = gcs_client.load_labeled_json(args.bucket, args.labeled_json)
        
        if not labeled_data:
            logger.error("No labeled data found in JSON file. Exiting.")
            return 1
        
        # Process labeled data
        logger.info("Processing labeled data...")
        processor = LabelStudioStage3Processor()
        decisions = processor.extract_labeling_decisions(labeled_data)
        
        if not decisions:
            logger.error("No labeling decisions found in exported data. Exiting.")
            return 1
        
        # Merge groups based on decisions
        logger.info("Merging groups based on labeling decisions...")
        merger = GroupMerger()
        consolidated_groups = merger.merge_groups_by_decisions(decisions)
        
        if not consolidated_groups:
            logger.error("No consolidated groups created. Exiting.")
            return 1
        
        # Initialize third stage secondary matcher
        logger.info("Initializing third stage secondary matcher...")
        matcher = ThirdStageSecondaryMatcher(recognition_client, gcs_client)
        
        # Load groups to gallery
        logger.info("Loading consolidated groups to gallery...")
        if not matcher.load_groups_to_gallery(consolidated_groups):
            logger.error("Failed to load groups to gallery. Exiting.")
            return 1
        
        # Find secondary matches
        logger.info("Finding secondary matches...")
        match_pairs = matcher.find_secondary_matches(consolidated_groups)
        
        if not match_pairs:
            logger.warning("No secondary matches found.")
        else:
            # Save match pairs
            if not args.dry_run:
                logger.info("Saving match pairs to GCS...")
                saved_pairs = 0
                for r_id_1, r_id_2, match_data in match_pairs:
                    if gcs_client.save_third_stage_match_json(args.bucket, match_data):
                        saved_pairs += 1
                
                logger.info(f"Saved {saved_pairs}/{len(match_pairs)} match pairs")
                
                # Duplicate elimination
                eliminated_count = 0
                logger.info("Eliminating duplicate pairs...")
                eliminated_count = eliminate_duplicate_pairs(match_pairs, gcs_client, args.bucket)
                logger.info(f"Eliminated {eliminated_count} duplicate pairs")
            
            else:
                logger.info("Dry run - match pairs not saved:")
                for r_id_1, r_id_2, match_data in match_pairs:
                    similarity = match_data.get('similarity_score', 0)
                    logger.info(f"  Match: {r_id_1} -> {r_id_2} (similarity: {similarity:.3f})")
                eliminated_count = 0
        
# No processed files tracking needed for single file input
        
        logger.info("Ship Third Stage Processing Tool completed successfully")
        logger.info("")
        logger.info("Results saved to GCS:")
        if not args.dry_run:
            final_match_pairs = len(match_pairs) - eliminated_count if match_pairs else 0
            if final_match_pairs > 0:
                logger.info(f"   • Match pairs ({final_match_pairs}): gs://{args.bucket}/reidentification/bronze/labeling/third_matching_phase/")
            else:
                logger.info("   • No files created (no matches found)")
        else:
            logger.info("   • Dry run - no files saved")
        return 0
        
    except Exception as e:
        logger.error(f"Fatal error: {e}")
        import traceback
        traceback.print_exc()
        return 1


def eliminate_duplicate_pairs(match_pairs: List[Tuple[str, str, Dict]], gcs_client: GCSClient, bucket_name: str) -> int:
    """Eliminate duplicate bidirectional pairs for third stage processing."""
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
        if gcs_client.delete_third_stage_match_json(bucket_name, r_id_1, r_id_2):
            deleted_count += 1
    
    return deleted_count


if __name__ == "__main__":
    sys.exit(main())
