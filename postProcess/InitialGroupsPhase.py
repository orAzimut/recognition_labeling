#!/usr/bin/env python3
"""
Ship UUID Uploader (No-Recognition, No-Grouping)

Purpose
-------
Process every UUID found under a given GCS path and emit **one group JSON per UUID**
(using the same output format/paths as the previous tool), without calling the
Ship-Recognition-Service and without cross-UUID grouping.

What it does
------------
- Scans GCS under --gcs-path to discover UUIDs (legacy/mission/custom layouts).
- Skips UUIDs already listed in the processed-UUIDs ledger.
- For each selected UUID (up to --num-uuids):
  - Lists all image files of that UUID.
  - Infers paired JSON metadata paths per image.
  - Extracts bounding boxes (if available) into Label Studio percentage format.
  - Writes a single group JSON (array-wrapped) to the Initial_groups_phase path,
    with a short random r_id and one rectangle per image.
- Updates the processed UUIDs ledger in GCS.

Outputs
-------
- Groups: gs://{output_bucket}/reidentification/bronze/labeling/Initial_groups_phase/group_{r_id}.json
- Ledger: gs://{bucket}/reidentification/bronze/labeling/Initial_groups_uuid_list.json

Usage
-----
python ship_uuid_uploader.py \
  --bucket azimut_data \
  --gcs-path "/reidentification/bronze/raw_crops/haifa/azimut-haifa" \
  --num-uuids 200 \
  --credentials "C:/path/to/gcs-key.json"

Author: Ship Recognition Team (refactored for UUID-only upload)
"""

import os
import json
import logging
import argparse
from datetime import datetime
from pathlib import Path
from typing import List, Dict, Set, Optional, Tuple, Any
from collections import defaultdict

from google.cloud import storage


# ----------------------------- GCS Client ------------------------------------
class GCSClient:
    """Google Cloud Storage client for managing ship UUID data and outputs."""

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

    # ------------------------- Discovery ------------------------------------
    def list_uuids_in_path(self, bucket_name: str, gcs_path: str) -> List[str]:
        """
        Find UUID folders anywhere under gcs_path.
        Supports legacy, mission, or custom nesting by scanning path segments
        that parse as UUIDs.
        """
        bucket = self.client.bucket(bucket_name)
        prefix = gcs_path.strip('/') + '/'
        uuids: Set[str] = set()
        for blob in bucket.list_blobs(prefix=prefix):
            rel = blob.name[len(prefix):]
            for part in rel.split('/'):
                if part and self._is_valid_uuid(part):
                    uuids.add(part)
                    break
        out = sorted(uuids)
        self.logger.info(f"Found {len(out)} UUIDs in {bucket_name}/{gcs_path}")
        return out

    def list_images_for_uuid(self, bucket_name: str, gcs_path: str, target_uuid: str) -> List[str]:
        bucket = self.client.bucket(bucket_name)
        prefix = gcs_path.strip('/') + '/'
        images: List[str] = []
        for blob in bucket.list_blobs(prefix=prefix):
            if target_uuid in blob.name and self._is_image_file(blob.name):
                images.append(f"gs://{bucket_name}/{blob.name}")
        return sorted(images)

    # ------------------------- IO Helpers -----------------------------------
    def download_json_metadata(self, gcs_path: str) -> Optional[Dict[str, Any]]:
        if not gcs_path.startswith('gs://'):
            self.logger.error(f"Invalid GCS path format: {gcs_path}")
            return None
        bucket_name, blob_name = gcs_path[5:].split('/', 1)
        bucket = self.client.bucket(bucket_name)
        blob = bucket.blob(blob_name)
        try:
            if not blob.exists():
                self.logger.debug(f"JSON metadata not found: {gcs_path}")
                return None
            return json.loads(blob.download_as_text())
        except Exception as e:
            self.logger.warning(f"Failed to load JSON metadata {gcs_path}: {e}")
            return None

    def load_processed_uuids(self, bucket_name: str) -> Set[str]:
        try:
            bucket = self.client.bucket(bucket_name)
            blob = bucket.blob("reidentification/bronze/labeling/Initial_groups_uuid_list.json")
            if not blob.exists():
                self.logger.info("No existing processed UUID list found in GCS")
                return set()
            data = json.loads(blob.download_as_text())
            processed = set(data.get('processed_uuids', []))
            self.logger.info(f"Loaded {len(processed)} processed UUIDs")
            return processed
        except Exception as e:
            self.logger.error(f"Failed to load processed UUIDs: {e}")
            return set()

    def save_processed_uuids(self, bucket_name: str, processed_uuids: Set[str]) -> bool:
        try:
            bucket = self.client.bucket(bucket_name)
            blob = bucket.blob("reidentification/bronze/labeling/Initial_groups_uuid_list.json")
            payload = {
                'processed_uuids': sorted(processed_uuids),
                'last_updated': datetime.utcnow().isoformat() + 'Z',
                'total_count': len(processed_uuids)
            }
            blob.upload_from_string(json.dumps(payload, indent=2))
            self.logger.info(f"Saved processed UUIDs list ({len(processed_uuids)} total)")
            return True
        except Exception as e:
            self.logger.error(f"Failed to save processed UUIDs: {e}")
            return False

    def save_group_json(self, bucket_name: str, group_data: Dict[str, Any]) -> bool:
        try:
            bucket = self.client.bucket(bucket_name)
            r_id = group_data['data']['r_id']
            blob = bucket.blob(f"reidentification/bronze/labeling/Initial_groups_phase/group_{r_id}.json")
            blob.upload_from_string(json.dumps([group_data], indent=2))
            self.logger.info(
                f"Saved group {r_id} with {len(group_data['data']['images'])} images to GCS"
            )
            return True
        except Exception as e:
            self.logger.error(f"Failed to save group {group_data.get('data',{}).get('r_id','?')}: {e}")
            return False

    # ------------------------- Internal utils -------------------------------
    def _is_valid_uuid(self, s: str) -> bool:
        import uuid
        try:
            uuid.UUID(s)
            return True
        except Exception:
            return False

    def _is_image_file(self, filename: str) -> bool:
        return Path(filename).suffix.lower() in {'.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.tif'}


# ----------------------------- Group Builder ---------------------------------
class UUIDGroupBuilder:
    """Build a single LS-style group for one UUID (no recognition)."""

    def __init__(self):
        self.logger = logging.getLogger(f"{__name__}.{self.__class__.__name__}")



    def _extract_timestamp_from_first_json(self, json_path: str, gcs_client: GCSClient) -> Optional[str]:
    
        try:
            meta = gcs_client.download_json_metadata(json_path)
            if meta:
                # Try common timestamp field names
                timestamp_fields = [
                    'timestamp',
                    'created_at', 
                    'sync_timestamp',
                    'capture_time',
                    'frame_time'
                ]
                
                for field in timestamp_fields:
                    if field in meta:
                        timestamp = meta[field]
                        # Handle different timestamp formats
                        if isinstance(timestamp, (int, float)):
                            # Unix timestamp
                            from datetime import datetime
                            return datetime.fromtimestamp(timestamp).isoformat() + 'Z'
                        elif isinstance(timestamp, str):
                            # ISO string or other format
                            return timestamp
                
                # Try nested fields
                if 'metadata' in meta and isinstance(meta['metadata'], dict):
                    for field in timestamp_fields:
                        if field in meta['metadata']:
                            timestamp = meta['metadata'][field]
                            if isinstance(timestamp, (int, float)):
                                from datetime import datetime
                                return datetime.fromtimestamp(timestamp).isoformat() + 'Z'
                            elif isinstance(timestamp, str):
                                return timestamp
            
            self.logger.warning(f"No timestamp found in {json_path}")
            return None
            
        except Exception as e:
            self.logger.warning(f"Failed to extract timestamp from {json_path}: {e}")
            return None

    def build_group_for_uuid(
        self,
        target_uuid: str,
        images: List[str],
        bucket_name: str,
        gcs_client: GCSClient,
    ) -> Optional[Dict[str, Any]]:
        if not images:
            self.logger.warning(f"UUID {target_uuid} has no images, skipping")
            return None

        r_id = self._generate_r_id()
        predictions_result = []
        jsons = []
        item_index = 0
        
        # Extract timestamp from first JSON
        first_json_path = None
        group_timestamp = None

        for idx, img in enumerate(images):
            json_path = self._convert_image_path_to_json(img)
            jsons.append(json_path)
            
            # Extract timestamp from first JSON only
            if idx == 0:
                first_json_path = json_path
                group_timestamp = self._extract_timestamp_from_first_json(json_path, gcs_client)

            pred = self._empty_pred(item_index)
            meta = gcs_client.download_json_metadata(json_path)

            if meta and 'target' in meta and 'bounding_box' in meta['target']:
                bbox_info = meta['target']['bounding_box']
                if 'bounding_box' in bbox_info and 'padded_bounding_box' in bbox_info:
                    bbox = bbox_info['bounding_box']          # [x1, y1, x2, y2]
                    pb = bbox_info['padded_bounding_box']     # [px1, py1, px2, py2]
                    try:
                        ow = pb[2] - pb[0]
                        oh = pb[3] - pb[1]
                        x_perc = ((bbox[0] - pb[0]) / ow) * 100.0
                        y_perc = ((bbox[1] - pb[1]) / oh) * 100.0
                        w_perc = ((bbox[2] - bbox[0]) / ow) * 100.0
                        h_perc = ((bbox[3] - bbox[1]) / oh) * 100.0
                        pred['original_width'] = ow
                        pred['original_height'] = oh
                        pred['value'].update({
                            'x': x_perc,
                            'y': y_perc,
                            'width': w_perc,
                            'height': h_perc,
                        })
                    except Exception as e:
                        self.logger.debug(f"Bad bbox in {json_path}: {e}; leaving zeros")
                else:
                    self.logger.debug(f"Missing bbox keys in {json_path}; leaving zeros")
            else:
                self.logger.debug(f"No bbox in {json_path}; leaving zeros")

            predictions_result.append(pred)
            item_index += 1

        group = {
            'data': {
                'r_id': r_id,
                'images': images,
                'jsons': jsons,
                'created_at': datetime.utcnow().isoformat() + 'Z',
                'group_timestamp': group_timestamp,  # ADD THIS LINE
                'uuid': target_uuid,  # ADD THIS LINE for tracking
            },
            'predictions': [{
                'model_version': 'preloaded-bboxes',
                'score': 1.0,
                'result': predictions_result,
            }],
        }
        return group

    # ------------------------ Helpers ---------------------------------------
    def _generate_r_id(self) -> str:
        import random, string
        return ''.join(random.choices(string.ascii_lowercase + string.digits, k=3))

    def _convert_image_path_to_json(self, image_path: str) -> str:
        if '/raw_crops/' in image_path:
            return self._change_extension_to_json(image_path.replace('/raw_crops/', '/json_metadata/'))
        return self._change_extension_to_json(image_path)

    def _change_extension_to_json(self, path: str) -> str:
        for ext in ('.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.tif'):
            if path.lower().endswith(ext):
                return path[: -len(ext)] + '.json'
        dot = path.rfind('.')
        return (path[:dot] if dot != -1 else path) + '.json'

    def _empty_pred(self, idx: int) -> Dict[str, Any]:
        return {
            'from_name': 'bbox_labels',
            'to_name': 'images',
            'type': 'rectanglelabels',
            'original_width': 0,
            'original_height': 0,
            'value': {
                'item_index': idx,
                'x': 0.0,
                'y': 0.0,
                'width': 0.0,
                'height': 0.0,
                'rotation': 0,
                'rectanglelabels': ['Box'],
                'image_rotation': 0,
                'image_index': idx,
            },
            'meta': {'item_index': idx, 'image_index': idx},
            'item_index': idx,
            'image_index': idx,
            'image_rotation': 0,
        }


# ----------------------------- CLI / Runner ----------------------------------

def setup_logging(log_level: str = 'INFO') -> logging.Logger:
    logging.basicConfig(
        level=getattr(logging, log_level.upper()),
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.StreamHandler(),
            logging.FileHandler('ship_uuid_uploader.log') if os.access('.', os.W_OK) else logging.NullHandler(),
        ],
    )
    return logging.getLogger(__name__)


def parse_arguments() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description='Ship UUID Uploader (no recognition, no grouping)',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog='''Examples:\n  python ship_uuid_uploader.py \\\n    --bucket azimut_data \\\n    --gcs-path "/reidentification/bronze/raw_crops/haifa/azimut-haifa" \\\n    --num-uuids 200\n'''
    )
    p.add_argument('--bucket', default='azimut_data', help='Input GCS bucket')
    p.add_argument('--gcs-path', default='/reidentification/bronze/raw_crops/haifa/azimut-haifa/2025/09/27',
                   help='Base GCS path to scan for UUIDs (legacy/mission/custom supported)')
    p.add_argument('--num-uuids', type=int, default=400, help='Max UUIDs to process this run')
    p.add_argument('--output-bucket', default=None, help='Output GCS bucket (default: same as --bucket)')
    p.add_argument('--credentials', default=r"C:\Users\OrGil.AzureAD\OneDrive - AMPC\Desktop\Azimut.ai\recognition_labeling\resources\credentials.json", help='Path to GCS service account JSON')
    p.add_argument('--log-level', default='INFO', choices=['DEBUG','INFO','WARNING','ERROR'])
    p.add_argument('--dry-run', default=False, action='store_true', help='Run without writing to GCS')
    return p.parse_args()


def main() -> int:
    args = parse_arguments()
    logger = setup_logging(args.log_level)

    output_bucket = args.output_bucket or args.bucket

    try:
        logger.info("Initializing GCS client…")
        gcs = GCSClient(args.credentials)
        builder = UUIDGroupBuilder()

        logger.info("Loading processed UUID ledger…")
        processed = gcs.load_processed_uuids(args.bucket)

        logger.info(f"Discovering UUIDs in gs://{args.bucket}{args.gcs_path}…")
        all_uuids = gcs.list_uuids_in_path(args.bucket, args.gcs_path)
        todo = [u for u in all_uuids if u not in processed]
        logger.info(f"Found {len(todo)} unprocessed out of {len(all_uuids)} total")
        if not todo:
            logger.info("Nothing to do. Exiting.")
            return 0

        to_process = todo[: args.num_uuids]
        logger.info(f"Processing {len(to_process)} UUIDs this run")

        success: List[str] = []
        for i, uid in enumerate(to_process, 1):
            logger.info(f"[{i}/{len(to_process)}] UUID {uid}")
            try:
                images = gcs.list_images_for_uuid(args.bucket, args.gcs_path, uid)
                if not images:
                    logger.warning(f"No images for UUID {uid}; skipping")
                    continue

                group = builder.build_group_for_uuid(uid, images, args.bucket, gcs)
                if group is None:
                    continue

                if not args.dry_run:
                    if not gcs.save_group_json(output_bucket, group):
                        logger.error(f"Failed to save group for UUID {uid}")
                        continue

                success.append(uid)
            except Exception as e:
                logger.error(f"Error on UUID {uid}: {e}")
                continue

        logger.info(f"Successfully emitted {len(success)} group files")

        if not args.dry_run and success:
            updated = processed.union(success)
            if gcs.save_processed_uuids(args.bucket, updated):
                logger.info(f"Updated ledger with {len(updated)} total processed UUIDs")
            else:
                logger.error("Failed to update processed UUID ledger")

        logger.info("Done.")
        return 0

    except Exception as e:
        logger.error(f"Fatal error: {e}")
        import traceback
        traceback.print_exc()
        return 1


if __name__ == '__main__':
    import sys
    sys.exit(main())
