#!/usr/bin/env python3
"""
Ship Secondary Processing Tool (Stage 2, ADC)

Processes Label Studio export from Stage 1 and performs:
1) Group extraction (MainGroup, OutGroupX) → clean groups
2) Secondary matching against gallery via REST

Changes:
- Uses Application Default Credentials (ADC); no service-account file.
- Startup bucket reachability check to catch IAM/ADC misconfig early.
"""

import os
import sys
import json
import logging
import argparse
import base64
from datetime import datetime
from typing import List, Dict, Set, Optional, Tuple, Any
from collections import defaultdict

import requests
from google.cloud import storage
import google.auth


# ----------------------------- ADC Helpers -----------------------------
def _resolve_project() -> Optional[str]:
    """Resolve GCP project for storage.Client(project=...)."""
    env_proj = os.getenv("GOOGLE_CLOUD_PROJECT") or os.getenv("GCLOUD_PROJECT")
    if env_proj:
        return env_proj
    try:
        _, detected = google.auth.default(scopes=["https://www.googleapis.com/auth/cloud-platform"])
        return detected
    except Exception:
        return None

def _adc_storage_client() -> storage.Client:
    project = _resolve_project()
    return storage.Client(project=project)

def _startup_bucket_check(bucket_name: str) -> None:
    """Fail fast if bucket is not reachable with ADC/IAM."""
    client = _adc_storage_client()
    bucket = client.bucket(bucket_name)
    if not bucket.exists():
        raise RuntimeError(
            f"Bucket gs://{bucket_name} does not exist or is not visible. "
            "If 403 → grant Storage Object Viewer/Creator/Admin on the bucket."
        )


# =========================
# GCS Client (ADC)
# =========================
class GCSClient:
    """Google Cloud Storage client for managing ship recognition data."""

    def __init__(self):
        self.logger = logging.getLogger(f"{__name__}.{self.__class__.__name__}")
        try:
            project = _resolve_project()
            self.client = storage.Client(project=project)
            self.logger.info(f"GCS client initialized with ADC (project={project!r})")
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
            variant_path = original_path.replace('label_studio_exports', 'lable_studio_exports')
            if variant_path != original_path:
                paths_to_try.append(variant_path)
        elif 'lable_studio_exports' in original_path:
            variant_path = original_path.replace('lable_studio_exports', 'label_studio_exports')
            if variant_path != original_path:
                paths_to_try.append(variant_path)

        # Try each path variation
        for attempt, blob_path in enumerate(paths_to_try, 1):
            try:
                blob = bucket.blob(blob_path)
                if not blob.exists():
                    continue

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

                if 'r_id' not in first_group:
                    raise ValueError("Label Studio JSON missing required field: r_id")

                has_old_format = 'images' in first_group and 'jsons' in first_group
                has_new_format = 'images_data' in first_group
                if not has_old_format and not has_new_format:
                    raise ValueError(
                        "Label Studio JSON missing required fields. Expected either "
                        "['images', 'jsons'] (old) or ['images_data'] (new)"
                    )

                fmt = "new format (images_data)" if has_new_format else "old format (images/jsons)"
                self.logger.info(f"Detected Label Studio export format: {fmt}")
                self.logger.info(f"Loaded {len(labeled_data)} labeled groups from gs://{bucket_name}/{blob_path}")
                self.logger.info(f"Sample group keys: {list(first_group.keys())}")
                return labeled_data

            except Exception as e:
                # Log at the end if all attempts fail
                last_error = e
                continue

        # All paths failed
        self.logger.error("Label Studio JSON not found at any of the attempted paths:")
        for path in paths_to_try:
            self.logger.error(f"  - gs://{bucket_name}/{path}")

        self.logger.info("")
        self.logger.info("Common Label Studio export locations:")
        self.logger.info("  - /recognition/silver/Initial_groups_phase_cleaned/lable_studio_exports/")
        self.logger.info("  - /recognition/silver/Initial_groups_phase_cleaned/label_studio_exports/")
        self.logger.info("  - /recognition/bronze/labeling/label_studio_exports/")
        self.logger.info("")
        self.logger.info("To find the correct file, check your GCS bucket:")
        self.logger.info(f"  gsutil ls gs://{bucket_name}/recognition/*/Initial_groups_phase_cleaned/*/")

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
        """Download and parse JSON metadata from GCS."""
        if not gcs_path.startswith('gs://'):
            self.logger.error(f"Invalid GCS path format: {gcs_path}")
            return None

        path_parts = gcs_path[5:].split('/', 1)
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
            blob_path = f"recognition/bronze/labeling/secondary_matching_phase/match_{r_id_1}_{r_id_2}.json"
            blob = bucket.blob(blob_path)
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
            blob_path = f"recognition/bronze/labeling/secondary_matching_phase/match_{r_id_1}_{r_id_2}.json"
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

    # Processed-files tracking (kept as-is, now ADC)
    def load_processed_files(self, bucket_name: str, list_filename: str) -> Set[str]:
        try:
            bucket = self.client.bucket(bucket_name)
            blob_path = f"recognition/bronze/labeling/{list_filename}"
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
        try:
            bucket = self.client.bucket(bucket_name)
            blob_path = f"recognition/bronze/labeling/{list_filename}"
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


# =========================
# Recognition Client
# =========================
class RecognitionClient:
    def __init__(self, service_url: str = "http://localhost:8080"):
        self.service_url = service_url.rstrip('/')
        self.session = requests.Session()
        self.logger = logging.getLogger(f"{__name__}.{self.__class__.__name__}")

    def health_check(self) -> bool:
        try:
            r = self.session.get(f"{self.service_url}/health", timeout=10)
            ok = r.status_code == 200
            self.logger.info(f"Recognition service health check: {'OK' if ok else 'FAILED'}")
            return ok
        except Exception as e:
            self.logger.error(f"Health check failed: {e}")
            return False

    def reset_gallery(self) -> bool:
        self.logger.info("Resetting system_gallery (preserving other galleries for reference data)...")
        try:
            r = self.session.delete(f"{self.service_url}/galleries/system_gallery/reset", timeout=60)
            r.raise_for_status()
            result = r.json()
            status = result.get('status', 'unknown')
            msg = result.get('message', '')
            removed = result.get('vessels_removed', 0)
            if status in ('success', 'partial_success'):
                self.logger.info(f"Gallery reset successful: {msg}")
                if removed:
                    self.logger.info(f"Removed {removed} vessels from system_gallery")
                return True
            self.logger.error(f"Gallery reset failed: {result}")
            return False
        except requests.exceptions.RequestException as e:
            self.logger.error(f"Failed to reset gallery via API: {e}")
            return False

    def add_vessel_to_gallery(self, vessel_id: str, base64_images: List[str], metadata: Optional[Dict] = None, gallery_name: str = 'system_gallery') -> bool:
        payload = {'vessel_id': vessel_id, 'images': base64_images, 'metadata': metadata or {}, 'gallery_name': gallery_name}
        try:
            r = self.session.post(f"{self.service_url}/gallery/vessels", json=payload,
                                  headers={'Content-Type': 'application/json'}, timeout=120)
            r.raise_for_status()
            result = r.json()
            if result.get('status') == 'success':
                self.logger.info(f"Successfully added vessel {vessel_id} to {gallery_name} gallery")
                return True
            self.logger.error(f"Failed to add vessel {vessel_id}: {result}")
            return False
        except requests.exceptions.RequestException as e:
            self.logger.error(f"Failed to add vessel {vessel_id} to {gallery_name} gallery: {e}")
            return False

    def recognize_without_adding(self, base64_images: List[str], metadata: Optional[Dict] = None) -> Dict[str, Any]:
        payload = {
            'images': base64_images,
            'metadata': metadata or {'source': 'ship_secondary_processing'},
            'config': {'silhouette_threshold': 0.1, 'mean_a_threshold': 0.7}
        }
        try:
            r = self.session.post(f"{self.service_url}/recognize/multi-gallery", json=payload,
                                  headers={'Content-Type': 'application/json'}, timeout=60)
            r.raise_for_status()
            return r.json()
        except requests.exceptions.Timeout:
            self.logger.error("Recognition request timed out")
            return {'status': 'error', 'error_message': 'Request timeout'}
        except requests.exceptions.RequestException as e:
            self.logger.error(f"Recognition request failed: {e}")
            return {'status': 'error', 'error_message': str(e)}


# =========================
# Label Studio Processor
# =========================
class LabelStudioProcessor:
    """Processor for Label Studio tagged data."""

    def __init__(self):
        self.logger = logging.getLogger(f"{__name__}.{self.__class__.__name__}")

    def extract_clean_groups(self, labeled_data: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        clean_groups = []
        for group in labeled_data:
            try:
                original_r_id = group.get('r_id')
                images_data = group.get('images_data', [])

                group_timestamp = group.get('group_timestamp')
                uuid = group.get('uuid', '')

                if not images_data:
                    self.logger.warning(f"Group {original_r_id} has no images_data, skipping")
                    continue

                # Group images by bbox class (MainGroup / OutGroupX)
                groups_by_class: Dict[str, Dict[str, List[str]]] = {}

                for img_data in images_data:
                    if img_data.get('is_trash', False):
                        continue
                    bboxes = img_data.get('bboxes', [])
                    if not bboxes:
                        continue
                    bbox_class = bboxes[0].get('class', 'MainGroup')
                    groups_by_class.setdefault(bbox_class, {'images': [], 'jsons': []})
                    groups_by_class[bbox_class]['images'].append(img_data.get('url'))
                    groups_by_class[bbox_class]['jsons'].append(img_data.get('json_url'))

                for class_name, class_data in groups_by_class.items():
                    if class_data['images']:
                        if class_name == 'MainGroup':
                            new_r_id = original_r_id
                        else:
                            class_suffix = class_name.lower().replace('outgroup', 'out')
                            new_r_id = f"{original_r_id}_{class_suffix}"

                        clean_group = {
                            'r_id': new_r_id,
                            'images': class_data['images'],
                            'jsons': class_data['jsons'],
                            'original_group': group,
                            'class_name': class_name,
                            'original_r_id': original_r_id,
                            'group_timestamp': group_timestamp,
                            'uuid': uuid,
                        }
                        clean_groups.append(clean_group)
                        self.logger.info(
                            f"Extracted clean group {new_r_id} ({class_name}): "
                            f"{len(class_data['images'])} images, {len(class_data['jsons'])} JSONs, "
                            f"timestamp: {group_timestamp}"
                        )
                if not groups_by_class:
                    self.logger.warning(f"Group {original_r_id} has no clean images after processing")

            except Exception as e:
                self.logger.error(f"Error processing group {group.get('r_id', 'unknown')}: {e}")
                continue

        self.logger.info(f"Extracted {len(clean_groups)} clean groups from {len(labeled_data)} labeled tasks")
        return clean_groups

    def _corresponding_image_path(self, json_path: str) -> str:
        if '/json_metadata/' in json_path:
            image_path = json_path.replace('/json_metadata/', '/raw_crops/')
            if image_path.startswith('gs:/') and not image_path.startswith('gs://'):
                image_path = image_path.replace('gs:/', 'gs://', 1)
            for ext in ['.jpg', '.jpeg', '.png']:
                if image_path.endswith('.json'):
                    return image_path.replace('.json', ext)
        return json_path


# =========================
# Secondary Matching Engine
# =========================
class SecondaryMatcher:
    def __init__(self, recognition_client: RecognitionClient, gcs_client: GCSClient):
        self.recognition_client = recognition_client
        self.gcs_client = gcs_client
        self.logger = logging.getLogger(f"{__name__}.{self.__class__.__name__}")
        self.match_pairs: List[Tuple[str, str, Dict]] = []

    def load_groups_to_gallery(self, clean_groups: List[Dict[str, Any]]) -> bool:
        self.logger.info(f"Loading {len(clean_groups)} clean groups to gallery...")
        ok = 0
        for i, group in enumerate(clean_groups, 1):
            try:
                r_id = group['r_id']
                images = group['images']
                self.logger.info(f"Loading group {i}/{len(clean_groups)}: {r_id} ({len(images)} images)")
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
                if self.recognition_client.add_vessel_to_gallery(
                    vessel_id=r_id,
                    base64_images=base64_images,
                    metadata={'source': 'ship_secondary_processing',
                              'num_images': len(images),
                              'created_at': datetime.utcnow().isoformat() + 'Z'},
                    gallery_name='system_gallery'
                ):
                    ok += 1
                else:
                    self.logger.error(f"Failed to load group {r_id} to gallery")
            except Exception as e:
                self.logger.error(f"Error loading group {group.get('r_id', 'unknown')}: {e}")
                continue
        self.logger.info(f"Successfully loaded {ok}/{len(clean_groups)} groups to gallery")
        return ok > 0

    def find_secondary_matches(self, clean_groups: List[Dict[str, Any]]) -> List[Tuple[str, str, Dict]]:
        self.logger.info(f"Finding secondary matches for {len(clean_groups)} groups...")
        for i, group in enumerate(clean_groups, 1):
            try:
                r_id = group['r_id']
                images = group['images']
                self.logger.info(f"Processing group {i}/{len(clean_groups)}: {r_id}")
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

                result = self.recognition_client.recognize_without_adding(
                    base64_images, metadata={'source': 'secondary_matching', 'r_id': r_id}
                )
                if result.get('status') == 'error':
                    self.logger.error(f"Recognition failed for group {r_id}: {result.get('error_message')}")
                    continue

                second_match = self._extract_second_match(result, r_id, clean_groups)
                if second_match:
                    match_data = self._create_match_data(group, second_match, clean_groups)
                    if match_data:
                        self.match_pairs.append((r_id, second_match['vessel_id'], match_data))
                        self.logger.info(f"Found match: {r_id} -> {second_match['vessel_id']}")
                else:
                    self.logger.warning(f"No second match found for group {r_id}")
            except Exception as e:
                self.logger.error(f"Error processing group {group.get('r_id', 'unknown')}: {e}")
                continue

        self.logger.info(f"Found {len(self.match_pairs)} secondary matches")
        return self.match_pairs

    def _extract_second_match(self, result: Dict[str, Any], r_id: str, clean_groups: List[Dict[str, Any]]) -> Optional[Dict]:
        def in_batch(vessel_id: Optional[str]) -> bool:
            return vessel_id and any(g.get('r_id') == vessel_id for g in clean_groups)

        if 'results_per_gallery' in result:
            for gallery_name, gallery_result in result['results_per_gallery'].items():
                top_matches = gallery_result.get('top_matches', [])
                self.logger.debug(f"Gallery '{gallery_name}' has {len(top_matches)} matches for {r_id}")
                for match in top_matches:
                    vid = match.get('vessel_id')
                    if vid == r_id:
                        continue
                    if in_batch(vid):
                        self.logger.info(f"Found valid match for {r_id}: {vid} (from {gallery_name})")
                        return match
        else:
            for match in result.get('top_matches', []):
                vid = match.get('vessel_id')
                if vid == r_id:
                    continue
                if in_batch(vid):
                    self.logger.info(f"Found valid match for {r_id}: {vid}")
                    return match
        return None

    def _create_match_data(self, group1: Dict, second_match: Dict, all_groups: List[Dict]) -> Optional[Dict]:
        try:
            r_id_1 = group1['r_id']
            r_id_2 = second_match['vessel_id']
            group2 = next((g for g in all_groups if g['r_id'] == r_id_2), None)
            if not group2:
                self.logger.error(f"Could not find group data for {r_id_2}")
                return None

            # Extract bounding boxes for both groups from their JSONs (metadata)
            r_id_1_bboxes = self._extract_bboxes_from_jsons(group1['jsons'])
            r_id_2_bboxes = self._extract_bboxes_from_jsons(group2['jsons'])

            match_data = {
                'r_id_1': r_id_1,
                'r_id_1_images': group1['images'],
                'r_id_1_jsons': group1['jsons'],
                'r_id_1_bboxes': r_id_1_bboxes,
                'r_id_1_timestamp': group1.get('group_timestamp'),
                'r_id_1_uuid': group1.get('uuid', ''),
                'r_id_2': r_id_2,
                'r_id_2_images': group2['images'],
                'r_id_2_jsons': group2['jsons'],
                'r_id_2_bboxes': r_id_2_bboxes,
                'r_id_2_timestamp': group2.get('group_timestamp'),
                'r_id_2_uuid': group2.get('uuid', ''),
                'similarity_score': second_match.get('similarity_score', 0.0),
                'created_at': datetime.utcnow().isoformat() + 'Z'
            }
            return match_data
        except Exception as e:
            self.logger.error(f"Error creating match data: {e}")
            return None

    def _extract_bboxes_from_jsons(self, json_paths: List[str]) -> List[List[int]]:
        bboxes: List[List[int]] = []
        for jp in json_paths:
            try:
                meta = self.gcs_client.download_json_metadata(jp)
                if (meta and 'target' in meta and 'bounding_box' in meta['target']
                        and 'bounding_box' in meta['target']['bounding_box']):
                    bb = meta['target']['bounding_box']['bounding_box']
                    if isinstance(bb, list) and len(bb) == 4:
                        bboxes.append([int(v) for v in bb])
                    else:
                        self.logger.warning(f"Invalid bbox format in {jp}: {bb}")
                        bboxes.append([0, 0, 0, 0])
                else:
                    self.logger.warning(f"No bbox data in {jp}")
                    bboxes.append([0, 0, 0, 0])
            except Exception as e:
                self.logger.warning(f"Failed to extract bbox from {jp}: {e}")
                bboxes.append([0, 0, 0, 0])
        return bboxes


# =========================
# CLI / Runner
# =========================
def setup_logging(level: str = "INFO") -> logging.Logger:
    logging.basicConfig(
        level=getattr(logging, level.upper()),
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.StreamHandler(),
            logging.FileHandler('ship_secondary_processing.log') if os.access('.', os.W_OK) else logging.NullHandler()
        ]
    )
    return logging.getLogger(__name__)

def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description='Ship Secondary Processing Tool - Stage 2 (ADC)',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python ship_secondary_processing.py \
    --bucket azimut_data \
    --labeled-json "/recognition/silver/Initial_groups_phase_cleaned/label_studio_exports/2025-10-15_13-06_791Tasks.json" \
    --service-url "http://localhost:8080"
        """
    )
    parser.add_argument('--bucket', default="azimut_data", help='GCS bucket name')
    parser.add_argument('--labeled-json',default='recognition/silver/Initial_groups_phase_cleaned/label_studio_exports/2025-10-20_07-53_201Tasks.json', help='GCS path to LS export JSON (leading "/" ok)')
    parser.add_argument('--service-url', default='http://localhost:8080', help='Ship-Recognition-Service URL')
    parser.add_argument('--log-level', default='INFO', choices=['DEBUG', 'INFO', 'WARNING', 'ERROR'])
    parser.add_argument('--dry-run', action='store_true', default=False, help='Do not save results to GCS')
    parser.add_argument('--skip-duplicates', action='store_true', default=False, help='Skip duplicate elimination')
    return parser.parse_args()


def eliminate_duplicate_pairs(match_pairs: List[Tuple[str, str, Dict]], gcs_client: GCSClient, bucket_name: str) -> int:
    logger = logging.getLogger(__name__)
    pairs_set = set()
    pairs_to_delete = []
    for r_id_1, r_id_2, _ in match_pairs:
        pair = (r_id_1, r_id_2)
        rev = (r_id_2, r_id_1)
        if rev in pairs_set:
            pairs_to_delete.append(pair)
            logger.info(f"Marked pair {r_id_1}-{r_id_2} for deletion (reverse of {r_id_2}-{r_id_1})")
        else:
            pairs_set.add(pair)

    deleted = 0
    for r_id_1, r_id_2 in pairs_to_delete:
        if gcs_client.delete_match_pair_json(bucket_name, r_id_1, r_id_2):
            deleted += 1
    return deleted


def main():
    args = parse_arguments()
    logger = setup_logging(args.log_level)

    logger.info("Starting Ship Secondary Processing Tool (Stage 2)")
    logger.info(f"Arguments: {vars(args)}")

    try:
        # Fast ADC/bucket reachability check
        _startup_bucket_check(args.bucket)
        logger.info(f"GCS reachable and bucket '{args.bucket}' is accessible.")

        # Initialize clients
        logger.info("Initializing clients...")
        gcs_client = GCSClient()
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
            logger.info("Ship Secondary Processing Tool completed (skipped - already processed)")
            return 0

        # Load Label Studio data
        logger.info(f"Loading Label Studio data from {args.labeled_json}...")
        labeled_data = gcs_client.load_labeled_json(args.bucket, args.labeled_json)

        # Process labeled data
        logger.info("Processing labeled data...")
        processor = LabelStudioProcessor()
        clean_groups = processor.extract_clean_groups(labeled_data)

        if not clean_groups:
            logger.error("No clean groups found in labeled data. Exiting.")
            return 1

        # Initialize matcher
        logger.info("Initializing secondary matcher...")
        matcher = SecondaryMatcher(recognition_client, gcs_client)

        # Load groups to gallery
        logger.info("Loading clean groups to gallery...")
        if not matcher.load_groups_to_gallery(clean_groups):
            logger.error("Failed to load groups to gallery. Exiting.")
            return 1

        # Find matches
        logger.info("Finding secondary matches...")
        match_pairs = matcher.find_secondary_matches(clean_groups)

        if not args.dry_run:
            if match_pairs:
                logger.info("Saving match pairs to GCS...")
                saved = 0
                for r_id_1, r_id_2, md in match_pairs:
                    if gcs_client.save_match_pair_json(args.bucket, md):
                        saved += 1
                logger.info(f"Saved {saved}/{len(match_pairs)} match pairs")

                # Duplicate elimination (unless skipped)
                eliminated_count = 0
                if not args.skip_duplicates:
                    logger.info("Eliminating duplicate pairs...")
                    eliminated_count = eliminate_duplicate_pairs(match_pairs, gcs_client, args.bucket)
                    logger.info(f"Eliminated {eliminated_count} duplicate pairs")
            else:
                logger.warning("No secondary matches found.")
        else:
            logger.info("Dry run - match pairs not saved")
            for r_id_1, r_id_2, md in match_pairs:
                sim = md.get('similarity_score', 0.0)
                logger.info(f"  Match: {r_id_1} -> {r_id_2} (similarity: {sim:.3f})")

        # Update processed files tracking list
        if not args.dry_run:
            logger.info("Updating processed files tracking list...")
            processed_files.add(args.labeled_json)
            if gcs_client.save_processed_files(args.bucket, "secondary_processing_files_list.json", processed_files):
                logger.info(f"Added {args.labeled_json} to processed files list")
            else:
                logger.warning("Failed to update processed files tracking list")

        logger.info("Ship Secondary Processing Tool completed successfully")
        if not args.dry_run:
            final_pairs = len(match_pairs) if match_pairs else 0
            if final_pairs > 0:
                logger.info(f"Results: gs://{args.bucket}/recognition/bronze/labeling/secondary_matching_phase/")
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
