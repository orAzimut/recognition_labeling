#!/usr/bin/env python3
"""
Ship Third Stage Processing Tool

A standalone script that processes Label Studio labeled outputs from Stage 2 and performs:
1. Group merging: Merges groups based on labeler decisions (same_vessel field)
2. Secondary matching: Finds matches between merged/separate groups using Ship-Recognition-Service
3. Output generation: Creates match pairs in the same format as Stage 2 for consistency

Uses ADC (no JSON keys). Ensure the runtime has:
    gcloud auth application-default login
and bucket-level IAM on the target bucket.

Usage:
    python ship_third_stage_processing.py \
        --bucket azimut_data \
    --labeled-json "/recognition/silver/Groups_Association_Phase_cleaned/lable_studio_exports/LS_185882_ACCEPTED_2025-10-05_11-36_166Tasks.json" \
        --service-url "http://localhost:8080"
"""

import os
import sys
import json
import logging
import argparse
import base64
from datetime import datetime
from pathlib import Path
from typing import List, Dict, Set, Optional, Tuple, Any
import requests

from google.cloud import storage
import google.auth


# ----------------------------- ADC GCS Client --------------------------------
class GCSClient:
    """Google Cloud Storage client for managing ship recognition data via ADC."""
    def __init__(self, project: Optional[str] = None):
        self.logger = logging.getLogger(f"{__name__}.{self.__class__.__name__}")
        try:
            resolved = project or os.getenv("GOOGLE_CLOUD_PROJECT") or os.getenv("GCLOUD_PROJECT")
            if not resolved:
                _, detected = google.auth.default(scopes=["https://www.googleapis.com/auth/cloud-platform"])
                resolved = detected
            self.client = storage.Client(project=resolved)
            self.project = resolved
            self.logger.info(f"GCS client initialized with ADC (project={self.project!r})")
        except Exception as e:
            self.logger.error(f"Failed to initialize GCS client: {e}")
            raise

    def bucket_exists_or_die(self, bucket_name: str):
        bucket = self.client.bucket(bucket_name)
        if not bucket.exists():
            raise RuntimeError(
                f"Bucket gs://{bucket_name} does not exist or is not visible. "
                "If 403 -> grant bucket-level IAM (Storage Object Viewer/Creator/Admin)."
            )

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

        image_data = blob.download_as_bytes()
        import base64 as _b64
        return _b64.b64encode(image_data).decode('utf-8')

    def save_third_stage_match_json(self, bucket_name: str, match_data: Dict[str, Any]) -> bool:
        """Save a third stage match pair JSON file to GCS."""
        try:
            bucket = self.client.bucket(bucket_name)
            r_id_1 = match_data['r_id_1']
            r_id_2 = match_data['r_id_2']
            blob_path = f"recognition/bronze/labeling/third_matching_phase/match_{r_id_1}_{r_id_2}.json"
            bucket.blob(blob_path).upload_from_string(json.dumps(match_data, indent=2))
            self.logger.info(f"Saved third stage match pair {r_id_1}-{r_id_2} to GCS")
            return True
        except Exception as e:
            self.logger.error(f"Failed to save third stage match pair {match_data.get('r_id_1')}-{match_data.get('r_id_2')}: {e}")
            return False

    def delete_third_stage_match_json(self, bucket_name: str, r_id_1: str, r_id_2: str) -> bool:
        """Delete a third stage match pair JSON file from GCS."""
        try:
            blob_path = f"recognition/bronze/labeling/third_matching_phase/match_{r_id_1}_{r_id_2}.json"
            blob = self.client.bucket(bucket_name).blob(blob_path)
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


# ----------------------------- Recognition Client ----------------------------
class RecognitionClient:
    """Enhanced client for communicating with the Ship-Recognition-Service."""

    def __init__(self, service_url: str = "http://localhost:8080"):
        self.service_url = service_url.rstrip('/')
        self.session = requests.Session()
        self.logger = logging.getLogger(f"{__name__}.{self.__class__.__name__}")

    def health_check(self) -> bool:
        try:
            response = self.session.get(f"{self.service_url}/health", timeout=10)
            is_healthy = response.status_code == 200
            self.logger.info(f"Recognition service health check: {'OK' if is_healthy else 'FAILED'}")
            return is_healthy
        except Exception as e:
            self.logger.error(f"Health check failed: {e}")
            return False

    def reset_gallery(self) -> bool:
        self.logger.info("Resetting system_gallery (preserving other galleries for reference data)...")
        try:
            response = self.session.delete(f"{self.service_url}/galleries/system_gallery/reset", timeout=60)
            response.raise_for_status()
            result = response.json()
            status = result.get('status', 'unknown')
            if status in ('success', 'partial_success'):
                self.logger.info(f"Gallery reset response: {result}")
                return True
            self.logger.error(f"Gallery reset failed: {result}")
            return False
        except requests.exceptions.RequestException as e:
            self.logger.error(f"Failed to reset gallery via API: {e}")
            return False

    def add_vessel_to_gallery(self, vessel_id: str, base64_images: List[str], metadata: Optional[Dict] = None, gallery_name: str = 'system_gallery') -> bool:
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
                timeout=120
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
            self.logger.error(f"Failed to add vessel {vessel_id}: {e}")
            return False

    def recognize_without_adding(self, base64_images: List[str], metadata: Optional[Dict] = None) -> Dict[str, Any]:
        payload = {
            'images': base64_images,
            'metadata': metadata or {'source': 'ship_third_stage_processing'},
            'config': {'silhouette_threshold': 0.1, 'mean_a_threshold': 0.7}
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


# ----------------------------- Stage 3 Processor -----------------------------
class LabelStudioStage3Processor:
    """Processor for Label Studio Stage 3 (third stage) labeled data."""

    def __init__(self):
        self.logger = logging.getLogger(f"{__name__}.{self.__class__.__name__}")

    def extract_labeling_decisions(self, labeled_data: List[Dict[str, Any]]) -> List[Dict[str, Any]]:

        decisions = []

        for entry in labeled_data:
            try:
                r_id_1 = entry.get('r_id_1')
                r_id_2 = entry.get('r_id_2')
                same_vessel = entry.get('same_vessel', 'No')

                r_id_1_timestamp = entry.get('r_id_1_timestamp')
                r_id_2_timestamp = entry.get('r_id_2_timestamp')
                r_id_1_uuid = entry.get('r_id_1_uuid', '')
                r_id_2_uuid = entry.get('r_id_2_uuid', '')

                if not r_id_1 or not r_id_2:
                    self.logger.warning(f"Missing r_id fields in entry: {entry.keys()}")
                    continue

                decision = {
                    'r_id_1': r_id_1,
                    'r_id_2': r_id_2,
                    'same_vessel': same_vessel,
                    'r_id_1_images': entry.get('r_id_1_images', []),
                    'r_id_1_jsons': entry.get('r_id_1_jsons', []),
                    'r_id_1_bboxes': entry.get('r_id_1_bboxes', []),
                    'r_id_1_timestamp': r_id_1_timestamp,
                    'r_id_1_uuid': r_id_1_uuid,
                    'r_id_2_images': entry.get('r_id_2_images', []),
                    'r_id_2_jsons': entry.get('r_id_2_jsons', []),
                    'r_id_2_bboxes': entry.get('r_id_2_bboxes', []),
                    'r_id_2_timestamp': r_id_2_timestamp,
                    'r_id_2_uuid': r_id_2_uuid,
                    'original_entry': entry
                }

                decisions.append(decision)
                self.logger.debug(f"Extracted decision: {r_id_1} + {r_id_2} = {same_vessel} "
                                  f"(timestamps: {r_id_1_timestamp}, {r_id_2_timestamp})")

            except Exception as e:
                self.logger.error(f"Error processing labeled entry: {e}")
                continue

        self.logger.info(f"Extracted {len(decisions)} labeling decisions from {len(labeled_data)} entries")
        return decisions


class GroupMerger:
    """Handles merging of groups based on Label Studio decisions."""

    def __init__(self):
        self.logger = logging.getLogger(f"{__name__}.{self.__class__.__name__}")

    def merge_groups_by_decisions(self, decisions: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        Merge groups based on same_vessel decisions.
        Creates consolidated groups for processing.
        """
        merge_map = {}        # r_id -> merged_group_id
        merged_groups = {}    # merged_group_id -> group_data
        individual_groups = {}  # r_id -> group_data for non-merged groups
        processed_pairs = set()

        for decision in decisions:
            r_id_1 = decision['r_id_1']
            r_id_2 = decision['r_id_2']
            same_vessel = decision['same_vessel']

            pair_key = tuple(sorted([r_id_1, r_id_2]))
            if pair_key in processed_pairs:
                continue
            processed_pairs.add(pair_key)

            if same_vessel == 'Yes':
                merged_id = self._get_or_create_merged_group(r_id_1, r_id_2, merge_map, merged_groups)
                self._add_to_merged_group(merged_id, decision, merged_groups)
                self.logger.info(f"Merging groups {r_id_1} and {r_id_2} into merged group {merged_id}")
            else:
                self._add_individual_groups(r_id_1, r_id_2, decision, individual_groups, merge_map)

        final_groups = []
        for merged_id, group_data in merged_groups.items():
            final_groups.append(group_data)
        for r_id, group_data in individual_groups.items():
            if r_id not in merge_map:
                final_groups.append(group_data)

        self.logger.info(f"Created {len(final_groups)} consolidated groups "
                         f"({len(merged_groups)} merged, "
                         f"{len([g for g in individual_groups.values() if g['r_id'] not in merge_map])} individual)")
        return final_groups

    def _get_or_create_merged_group(self, r_id_1: str, r_id_2: str, merge_map: dict, merged_groups: dict) -> str:
        existing_merged_id = merge_map.get(r_id_1) or merge_map.get(r_id_2)
        if existing_merged_id:
            merge_map[r_id_1] = existing_merged_id
            merge_map[r_id_2] = existing_merged_id
            return existing_merged_id
        else:
            import random, string
            merged_id = 'merged_' + ''.join(random.choices(string.ascii_lowercase + string.digits, k=3))
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
        group = merged_groups[merged_id]

        all_images = decision['r_id_1_images'] + decision['r_id_2_images']
        all_jsons = decision['r_id_1_jsons'] + decision['r_id_2_jsons']
        all_bboxes = decision['r_id_1_bboxes'] + decision['r_id_2_bboxes']

        # timestamps (earliest)
        timestamps = []
        if decision.get('r_id_1_timestamp'): timestamps.append(decision['r_id_1_timestamp'])
        if decision.get('r_id_2_timestamp'): timestamps.append(decision['r_id_2_timestamp'])
        if timestamps and not group.get('group_timestamp'):
            group['group_timestamp'] = min(timestamps)

        # uuids
        uuids = []
        if decision.get('r_id_1_uuid'): uuids.append(decision['r_id_1_uuid'])
        if decision.get('r_id_2_uuid'): uuids.append(decision['r_id_2_uuid'])
        if 'uuids' not in group: group['uuids'] = set()
        group['uuids'].update(uuids)

        existing_images = set(group['images'])
        existing_jsons = set(group['jsons'])

        for i, img in enumerate(all_images):
            if img not in existing_images:
                group['images'].append(img)
                existing_images.add(img)
                group['bboxes'].append(all_bboxes[i] if i < len(all_bboxes) else [0.0, 0.0, 0.0, 0.0])

        for j in all_jsons:
            if j not in existing_jsons:
                group['jsons'].append(j)
                existing_jsons.add(j)

        group['source_r_ids'].add(decision['r_id_1'])
        group['source_r_ids'].add(decision['r_id_2'])

    def _add_individual_groups(self, r_id_1: str, r_id_2: str, decision: Dict[str, Any], individual_groups: dict, merge_map: dict):
        if r_id_1 not in merge_map and r_id_1 not in individual_groups:
            individual_groups[r_id_1] = {
                'r_id': r_id_1,
                'images': decision['r_id_1_images'],
                'jsons': decision['r_id_1_jsons'],
                'bboxes': decision['r_id_1_bboxes'],
                'group_timestamp': decision.get('r_id_1_timestamp'),
                'uuid': decision.get('r_id_1_uuid', ''),
                'is_merged': False
            }
        if r_id_2 not in merge_map and r_id_2 not in individual_groups:
            individual_groups[r_id_2] = {
                'r_id': r_id_2,
                'images': decision['r_id_2_images'],
                'jsons': decision['r_id_2_jsons'],
                'bboxes': decision['r_id_2_bboxes'],
                'group_timestamp': decision.get('r_id_2_timestamp'),
                'uuid': decision.get('r_id_2_uuid', ''),
                'is_merged': False
            }


# ----------------------------- Matching Engine -------------------------------
class ThirdStageSecondaryMatcher:
    """Engine for secondary matching between consolidated groups from Stage 3."""

    def __init__(self, recognition_client: RecognitionClient, gcs_client: GCSClient):
        self.recognition_client = recognition_client
        self.gcs_client = gcs_client
        self.logger = logging.getLogger(f"{__name__}.{self.__class__.__name__}")
        self.match_pairs: List[Tuple[str, str, Dict]] = []

    def load_groups_to_gallery(self, consolidated_groups: List[Dict[str, Any]]) -> bool:
        self.logger.info(f"Loading {len(consolidated_groups)} consolidated groups to gallery...")
        ok = 0
        for i, group in enumerate(consolidated_groups, 1):
            try:
                r_id = group['r_id']
                images = group['images']
                self.logger.info(f"Loading group {i}/{len(consolidated_groups)}: {r_id} ({len(images)} images)")
                base64_images = []
                for image_path in images:
                    try:
                        gcs_path = self._convert_to_gcs_path(image_path)
                        base64_images.append(self.gcs_client.download_image_as_base64(gcs_path))
                    except Exception as e:
                        self.logger.warning(f"Failed to download {image_path}: {e}")
                if not base64_images:
                    self.logger.warning(f"No images could be downloaded for group {r_id}")
                    continue
                if self.recognition_client.add_vessel_to_gallery(
                    vessel_id=r_id,
                    base64_images=base64_images,
                    metadata={'source': 'ship_third_stage_processing', 'num_images': len(images),
                              'is_merged': group.get('is_merged', False),
                              'created_at': datetime.utcnow().isoformat() + 'Z'},
                    gallery_name='system_gallery'
                ):
                    ok += 1
                else:
                    self.logger.error(f"Failed to load group {r_id} to gallery")
            except Exception as e:
                self.logger.error(f"Error loading group {group.get('r_id', 'unknown')}: {e}")
                continue
        self.logger.info(f"Successfully loaded {ok}/{len(consolidated_groups)} groups to gallery")
        return ok > 0

    def find_secondary_matches(self, consolidated_groups: List[Dict[str, Any]]) -> List[Tuple[str, str, Dict]]:
        self.logger.info(f"Finding secondary matches for {len(consolidated_groups)} groups...")
        for i, group in enumerate(consolidated_groups, 1):
            try:
                r_id = group['r_id']
                images = group['images']
                self.logger.info(f"Processing group {i}/{len(consolidated_groups)}: {r_id}")
                base64_images = []
                for image_path in images:
                    try:
                        gcs_path = self._convert_to_gcs_path(image_path)
                        base64_images.append(self.gcs_client.download_image_as_base64(gcs_path))
                    except Exception as e:
                        self.logger.warning(f"Failed to download {image_path}: {e}")
                if not base64_images:
                    self.logger.warning(f"No images available for recognition of group {r_id}")
                    continue
                result = self.recognition_client.recognize_without_adding(
                    base64_images, metadata={'source': 'third_stage_matching', 'r_id': r_id}
                )
                if result.get('status') == 'error':
                    self.logger.error(f"Recognition failed for group {r_id}: {result.get('error_message')}")
                    continue
                second_match = self._extract_second_match(result, r_id, consolidated_groups)
                if second_match:
                    match_data = self._create_match_data(group, second_match, consolidated_groups)
                    if match_data:
                        self.match_pairs.append((r_id, second_match['vessel_id'], match_data))
                        self.logger.info(f"Found match: {r_id} -> {second_match['vessel_id']}")
                else:
                    self.logger.warning(f"No valid matches found for {r_id} within current batch")
            except Exception as e:
                self.logger.error(f"Error processing group {group.get('r_id', 'unknown')}: {e}")
                continue
        self.logger.info(f"Found {len(self.match_pairs)} secondary matches")
        return self.match_pairs

    def _convert_to_gcs_path(self, image_path: str) -> str:
        if image_path.startswith('gs://'):
            return image_path
        elif image_path.startswith('/tasks/') and 'fileuri=' in image_path:
            import base64, urllib.parse
            try:
                parsed = urllib.parse.urlparse(image_path)
                query_params = urllib.parse.parse_qs(parsed.query)
                fileuri = query_params.get('fileuri', [None])[0]
                if fileuri:
                    decoded_path = base64.b64decode(fileuri).decode('utf-8')
                    if decoded_path.startswith('gs://'):
                        return decoded_path
            except Exception:
                pass
        return image_path

    def _extract_second_match(self, result: Dict[str, Any], r_id: str, consolidated_groups: List[Dict[str, Any]]) -> Optional[Dict]:
        def in_batch(vessel_id: Optional[str]) -> bool:
            return vessel_id and any(g.get('r_id') == vessel_id for g in consolidated_groups)

        if 'results_per_gallery' in result:
            for _, gallery_result in result['results_per_gallery'].items():
                top_matches = gallery_result.get('top_matches', [])
                if not top_matches:
                    continue
                for m in top_matches:
                    vid = m.get('vessel_id')
                    if vid == r_id:
                        continue
                    if in_batch(vid):
                        return m
        else:
            for m in result.get('top_matches', []):
                vid = m.get('vessel_id')
                if vid == r_id:
                    continue
                if in_batch(vid):
                    return m
        return None

    def _create_match_data(self, group1: Dict, second_match: Dict, all_groups: List[Dict]) -> Optional[Dict]:
        try:
            r_id_1 = group1['r_id']
            r_id_2 = second_match['vessel_id']
            group2 = next((g for g in all_groups if g['r_id'] == r_id_2), None)
            if not group2:
                self.logger.error(f"Could not find group data for {r_id_2}")
                return None

            return {
                'r_id_1': r_id_1,
                'r_id_1_images': [self._convert_to_gcs_path(img) for img in group1['images']],
                'r_id_1_jsons': [self._convert_to_gcs_path(p) for p in group1['jsons']],
                'r_id_1_bboxes': group1.get('bboxes', []),
                'r_id_1_timestamp': group1.get('group_timestamp'),
                'r_id_1_uuid': group1.get('uuid', '') or ','.join(group1.get('uuids', [])) if isinstance(group1.get('uuids', []), (set, list)) else group1.get('uuid', ''),
                'r_id_2': r_id_2,
                'r_id_2_images': [self._convert_to_gcs_path(img) for img in group2['images']],
                'r_id_2_jsons': [self._convert_to_gcs_path(p) for p in group2['jsons']],
                'r_id_2_bboxes': group2.get('bboxes', []),
                'r_id_2_timestamp': group2.get('group_timestamp'),
                'r_id_2_uuid': group2.get('uuid', '') or ','.join(group2.get('uuids', [])) if isinstance(group2.get('uuids', []), (set, list)) else group2.get('uuid', ''),
                'similarity_score': second_match.get('similarity_score', 0.0),
                'created_at': datetime.utcnow().isoformat() + 'Z'
            }
        except Exception as e:
            self.logger.error(f"Error creating match data: {e}")
            return None


# ----------------------------- CLI / Runner ----------------------------------
def setup_logging(log_level: str = "INFO") -> logging.Logger:
    logging.basicConfig(
        level=getattr(logging, log_level.upper()),
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[logging.StreamHandler(),
                  logging.FileHandler('ship_third_stage_processing.log') if os.access('.', os.W_OK) else logging.NullHandler()]
    )
    return logging.getLogger(__name__)

def parse_arguments() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description='Ship Third Stage Processing Tool - Stage 3',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python ship_third_stage_processing.py \
    --bucket azimut_data \
    --labeled-json "/recognition/silver/Groups_Association_Phase_cleaned/lable_studio_exports/LS_185882_ACCEPTED_2025-10-05_11-36_166Tasks.json"
  python ship_third_stage_processing.py \
    --bucket azimut_data \
    --labeled-json "/path/to/labeled_pairs.json" \
    --service-url "http://192.168.1.100:8080"
        """
    )
    p.add_argument('--bucket', default="azimut_data", help='GCS bucket name')
    p.add_argument('--labeled-json', required=True, help='GCS path to the labeled JSON file (leading / ok)')
    p.add_argument('--service-url', default='http://localhost:8080', help='Ship-Recognition-Service URL')
    p.add_argument('--project', default=None, help='GCP project (optional; auto-detected from ADC otherwise)')
    p.add_argument('--log-level', default='INFO', choices=['DEBUG', 'INFO', 'WARNING', 'ERROR'])
    p.add_argument('--dry-run', action='store_true', default=False, help='Perform a dry run without saving results to GCS')
    return p.parse_args()


def eliminate_duplicate_pairs(match_pairs: List[Tuple[str, str, Dict]], gcs_client: GCSClient, bucket_name: str) -> int:
    """Eliminate duplicate bidirectional pairs for third stage processing."""
    logger = logging.getLogger(__name__)
    pairs_set = set()
    pairs_to_delete = []
    for r_id_1, r_id_2, _ in match_pairs:
        pair = (r_id_1, r_id_2)
        reverse_pair = (r_id_2, r_id_1)
        if reverse_pair in pairs_set:
            pairs_to_delete.append(pair)
            logger.info(f"Marked pair {r_id_1}-{r_id_2} for deletion (reverse of {r_id_2}-{r_id_1})")
        else:
            pairs_set.add(pair)
    deleted_count = 0
    for r_id_1, r_id_2 in pairs_to_delete:
        if gcs_client.delete_third_stage_match_json(bucket_name, r_id_1, r_id_2):
            deleted_count += 1
    return deleted_count


def main():
    args = parse_arguments()
    logger = setup_logging(args.log_level)

    logger.info("Starting Ship Third Stage Processing Tool (Stage 3)")
    logger.info(f"Arguments: {{'bucket': '{args.bucket}', 'labeled_json': '{args.labeled_json}', "
                f"'service_url': '{args.service_url}', 'project': '{args.project}', "
                f"'log_level': '{args.log_level}', 'dry_run': {args.dry_run}}}")

    try:
        # Initialize ADC GCS + bucket reachability
        gcs_client = GCSClient(project=args.project)
        gcs_client.bucket_exists_or_die(args.bucket)

        # Health check
        recognition_client = RecognitionClient(args.service_url)
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

        # Load groups to gallery
        logger.info("Loading consolidated groups to gallery...")
        matcher = ThirdStageSecondaryMatcher(recognition_client, gcs_client)
        if not args.dry_run:
            if not matcher.load_groups_to_gallery(consolidated_groups):
                logger.error("Failed to load groups to gallery. Exiting.")
                return 1
        else:
            logger.info("Dry run - skipping gallery load")

        # Find matches
        logger.info("Finding secondary matches...")
        match_pairs = matcher.find_secondary_matches(consolidated_groups)

        eliminated_count = 0
        if not args.dry_run:
            if match_pairs:
                logger.info("Saving match pairs to GCS...")
                saved_pairs = 0
                for r_id_1, r_id_2, match_data in match_pairs:
                    if gcs_client.save_third_stage_match_json(args.bucket, match_data):
                        saved_pairs += 1
                logger.info(f"Saved {saved_pairs}/{len(match_pairs)} match pairs")

                logger.info("Eliminating duplicate pairs...")
                eliminated_count = eliminate_duplicate_pairs(match_pairs, gcs_client, args.bucket)
                logger.info(f"Eliminated {eliminated_count} duplicate pairs")
            else:
                logger.warning("No secondary matches found.")
        else:
            logger.info("Dry run - match pairs not saved")
            for r_id_1, r_id_2, md in match_pairs:
                sim = md.get('similarity_score', 0.0)
                logger.info(f"  Match (dry-run): {r_id_1} -> {r_id_2} (similarity: {sim:.3f})")

        logger.info("Ship Third Stage Processing Tool completed successfully")
        if not args.dry_run:
            final_count = len(match_pairs) - eliminated_count if match_pairs else 0
            if final_count > 0:
                logger.info(f"Results: gs://{args.bucket}/recognition/bronze/labeling/third_matching_phase/")
            else:
                logger.info("No files created (no matches found)")
        else:
            logger.info("Dry run - no files saved")
        return 0

    except Exception as e:
        logger.error(f"Fatal error: {e}")
        import traceback; traceback.print_exc()
        return 1


if __name__ == "__main__":
    sys.exit(main())
