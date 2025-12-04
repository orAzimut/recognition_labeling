#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Ship UUID Uploader (No-Recognition, No-Grouping) — ADC Edition

Purpose
-------
Process every UUID found under a given GCS path and emit **one group JSON per UUID**
(using the same output format/paths as the previous tool), without calling the
Ship-Recognition-Service and without cross-UUID grouping.

Auth Model (THIS VERSION)
-------------------------
- Uses Google Application Default Credentials (ADC).
- Employees authenticate with *their own Google identities*:
    gcloud auth application-default login
- No service-account JSON is required or supported here.
- Ensure each employee's Workspace email has IAM on the bucket:
    Upload only  -> roles/storage.objectCreator
    Read+Upload  -> roles/storage.objectAdmin (avoid if possible)

Outputs
-------
- Groups: gs://{output_bucket}/recognition/bronze/labeling/Initial_groups_phase/group_{r_id}.json
- Ledger: gs://{output_bucket}/recognition/bronze/labeling/Initial_groups_uuid_list.json

Usage
-----
# One-time per user on their laptop (no JSON keys!):
gcloud auth application-default login

# Run
python ship_uuid_uploader.py \
  --bucket ratag \
  --gcs-path "recognition/bronze/raw_jsons/eilat/azimut-eilat/2025/12/01" \
  --num-uuids 200
# optionally:
#   --project your-gcp-project-id
#   --log-level DEBUG
#   --dry-run
"""

import os
import json
import logging
import argparse
from datetime import datetime
from pathlib import Path
from typing import List, Dict, Set, Optional, Any, Tuple

from google.cloud import storage
import google.auth


# ----------------------------- GCS Client ------------------------------------
class GCSClient:
    """Google Cloud Storage client for managing ship UUID data and outputs."""

    def __init__(self, project: Optional[str] = None):
        self.logger = logging.getLogger(f"{__name__}.{self.__class__.__name__}")
        try:
            # Try to resolve project if not provided
            resolved_project = project or os.getenv("GOOGLE_CLOUD_PROJECT") or os.getenv("GCLOUD_PROJECT")
            if not resolved_project:
                # Pull from ADC default if available
                _, detected = google.auth.default(scopes=["https://www.googleapis.com/auth/cloud-platform"])
                resolved_project = detected

            self.client = storage.Client(project=resolved_project)
            self.project = resolved_project
            self.logger.info(
                f"GCS client initialized with ADC (project={self.project!r}). "
                "If you hit 401/403, check IAM on the bucket and that you've run "
                "`gcloud auth application-default login`."
            )
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
        try:
            # Walk every blob under the user path, including subfolders.
            for blob in self._iter_blobs(bucket, prefix):
                rel = blob.name[len(prefix):]
                for part in rel.split('/'):
                    if part and self._is_valid_uuid(part):
                        uuids.add(part)
                        break
        except Exception as e:
            self._hint_auth_iam(e, f"listing blobs under gs://{bucket_name}/{prefix}")
            raise
        out = sorted(uuids)
        self.logger.info(f"Found {len(out)} UUIDs in {bucket_name}/{gcs_path}")
        return out

    def list_images_for_uuid(self, bucket_name: str, gcs_path: str, target_uuid: str) -> List[Tuple[str, str]]:
        bucket = self.client.bucket(bucket_name)
        prefix = gcs_path.strip('/') + '/'
        images_jsons: List[Tuple[str, str]] = []
        try:
            for blob in self._iter_blobs(bucket, prefix):
                if target_uuid not in blob.name:
                    continue
                if blob.name.lower().endswith('.json'):
                    json_path = f"gs://{bucket_name}/{blob.name}"
                    meta = self.download_json_metadata(json_path)
                    image_path = self._extract_image_path(meta, json_path, bucket_name)
                    if image_path:
                        images_jsons.append((image_path, json_path))
                elif self._is_image_file(blob.name):
                    images_jsons.append((f"gs://{bucket_name}/{blob.name}", f"gs://{bucket_name}/{blob.name}"))
        except Exception as e:
            self._hint_auth_iam(e, f"listing images for UUID {target_uuid} under gs://{bucket_name}/{prefix}")
            raise
        return sorted(images_jsons)

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
            blob = bucket.blob("recognition/bronze/labeling/Initial_groups_uuid_list.json")
            if not blob.exists():
                self.logger.info("No existing processed UUID list found in GCS")
                return set()
            data = json.loads(blob.download_as_text())
            processed = set(data.get('processed_uuids', []))
            self.logger.info(f"Loaded {len(processed)} processed UUIDs")
            return processed
        except Exception as e:
            self._hint_auth_iam(e, "reading processed UUIDs ledger")
            return set()

    def save_processed_uuids(self, bucket_name: str, processed_uuids: Set[str]) -> bool:
        try:
            bucket = self.client.bucket(bucket_name)
            blob = bucket.blob("recognition/bronze/labeling/Initial_groups_uuid_list.json")
            payload = {
                'processed_uuids': sorted(processed_uuids),
                'last_updated': datetime.utcnow().isoformat() + 'Z',
                'total_count': len(processed_uuids)
            }
            blob.upload_from_string(json.dumps(payload, indent=2))
            self.logger.info(f"Saved processed UUIDs list ({len(processed_uuids)} total)")
            return True
        except Exception as e:
            self._hint_auth_iam(e, "writing processed UUIDs ledger")
            self.logger.error(f"Failed to save processed UUIDs: {e}")
            return False

    def save_group_json(self, bucket_name: str, group_data: Dict[str, Any], prefix: Optional[str] = None) -> bool:
        try:
            bucket = self.client.bucket(bucket_name)
            r_id = group_data['data']['r_id']
            prefix_path = prefix or "recognition/bronze/labeling/Initial_groups_phase"
            blob = bucket.blob(f"{prefix_path.rstrip('/')}/group_{r_id}.json")
            blob.upload_from_string(json.dumps([group_data], indent=2))
            self.logger.info(
                f"Saved group {r_id} with {len(group_data['data']['images'])} images to GCS at {blob.name}"
            )
            return True
        except Exception as e:
            self._hint_auth_iam(e, f"writing group JSON for r_id={group_data.get('data', {}).get('r_id', '?')}")
            self.logger.error(f"Failed to save group {group_data.get('data',{}).get('r_id','?')}: {e}")
            return False

    # ------------------------- Internal utils -------------------------------
    def _is_valid_uuid(self, s: str) -> bool:
        import re
        import uuid
        # Accept plain UUIDs or names containing one (e.g., "tgt-<uuid>").
        match = re.search(
            r'[0-9a-fA-F]{8}-[0-9a-fA-F]{4}-[1-5][0-9a-fA-F]{3}-[89abAB][0-9a-fA-F]{3}-[0-9a-fA-F]{12}',
            s,
        )
        candidate = match.group(0) if match else s
        try:
            uuid.UUID(candidate)
            return True
        except Exception:
            return False

    def _is_image_file(self, filename: str) -> bool:
        return Path(filename).suffix.lower() in {'.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.tif'}

    def _extract_image_path(self, meta: Optional[Dict[str, Any]], json_path: str, bucket_name: str) -> Optional[str]:
        if not meta or not isinstance(meta, dict):
            return None
        candidate = None
        if 'image' in meta:
            candidate = meta.get('image')
        elif isinstance(meta.get('data'), dict) and 'image' in meta['data']:
            candidate = meta['data'].get('image')
        if not candidate or not isinstance(candidate, str):
            self.logger.debug(f"No usable image path in metadata {json_path}")
            return None
        path = candidate.strip()
        if not path:
            return None
        if path.startswith('gs://'):
            return path
        return f"gs://{bucket_name}/{path.lstrip('/')}"

    def _hint_auth_iam(self, exc: Exception, when: str) -> None:
        msg = str(exc)
        if any(token in msg for token in ("403", "401", "permission", "Permission", "unauthorized", "denied")):
            self.logger.error(
                f"Auth/IAM issue while {when}. Ensure:\n"
                "  1) You ran: gcloud auth application-default login\n"
                "  2) Your Workspace user has the right IAM on the bucket "
                "(e.g., Storage Object Creator / Viewer / Admin).\n"
                f"Original error: {exc}"
            )

    def _iter_blobs(self, bucket: storage.Bucket, prefix: str):
        """
        Yield every blob under the prefix, recursing into subfolders.
        Using delimiter=None (default) already recurses, but we keep this helper
        so intent is explicit and shared between UUID and image discovery.
        """
        return bucket.list_blobs(prefix=prefix, page_size=2000)


# ----------------------------- Group Builder ---------------------------------
class UUIDGroupBuilder:
    """Build a single LS-style group for one UUID (no recognition)."""

    def __init__(self):
        self.logger = logging.getLogger(f"{__name__}.{self.__class__.__name__}")

    def _extract_timestamp_from_first_json(self, json_path: str, gcs_client: GCSClient) -> Optional[str]:
        def _as_iso(val: Any) -> Optional[str]:
            if val is None:
                return None
            if isinstance(val, (int, float)):
                from datetime import datetime as _dt
                try:
                    return _dt.fromtimestamp(val).isoformat() + 'Z'
                except Exception:
                    return None
            if isinstance(val, str):
                return val
            return None

        def _pick_timestamp(meta: Dict[str, Any]) -> Optional[str]:
            timestamp_fields = [
                'timestamp', 'created_at', 'sync_timestamp', 'capture_time', 'frame_time',
                'FIRST_DETECTION', 'last_update_time',
            ]
            for field in timestamp_fields:
                if field in meta:
                    iso = _as_iso(meta[field])
                    if iso:
                        return iso
            return None

        try:
            meta = gcs_client.download_json_metadata(json_path)
            if meta:
                for candidate in (
                    _pick_timestamp(meta),
                    _pick_timestamp(meta.get('data', {})) if isinstance(meta.get('data'), dict) else None,
                    _pick_timestamp(meta.get('metadata', {})) if isinstance(meta.get('metadata'), dict) else None,
                    _pick_timestamp(meta.get('target', {})) if isinstance(meta.get('target'), dict) else None,
                ):
                    if candidate:
                        return candidate
            self.logger.warning(f"No timestamp found in {json_path}")
            return None
        except Exception as e:
            self.logger.warning(f"Failed to extract timestamp from {json_path}: {e}")
            return None

    def build_group_for_uuid(
        self,
        target_uuid: str,
        images_with_jsons: List[Tuple[str, str]],
        bucket_name: str,
        gcs_client: GCSClient,
    ) -> Optional[Dict[str, Any]]:
        if not images_with_jsons:
            self.logger.warning(f"UUID {target_uuid} has no images, skipping")
            return None

        r_id = self._generate_r_id()
        predictions_result = []
        jsons = []
        group_timestamp = None

        images: List[str] = []

        for idx, (img, json_path) in enumerate(images_with_jsons):
            images.append(img)
            jsons.append(json_path)

            if idx == 0:
                group_timestamp = self._extract_timestamp_from_first_json(json_path, gcs_client)

            pred = self._empty_pred(idx)
            meta = gcs_client.download_json_metadata(json_path)

            if meta:
                self._apply_bbox_to_prediction(pred, meta, json_path)
            else:
                self.logger.debug(f"No metadata found for {json_path}; leaving zeros")

            predictions_result.append(pred)

        group = {
            'data': {
                'r_id': r_id,
                'images': images,
                'jsons': jsons,
                'created_at': datetime.utcnow().isoformat() + 'Z',
                'group_timestamp': group_timestamp,
                'uuid': target_uuid,
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
                'x': 0.0, 'y': 0.0, 'width': 0.0, 'height': 0.0,
                'rotation': 0, 'rectanglelabels': ['Box'],
                'image_rotation': 0, 'image_index': idx,
            },
            'meta': {'item_index': idx, 'image_index': idx},
            'item_index': idx, 'image_index': idx, 'image_rotation': 0,
        }

    # ------------------------ BBox helpers ----------------------------------
    def _is_valid_bbox(self, bbox: Any) -> bool:
        return (
            isinstance(bbox, (list, tuple))
            and len(bbox) == 4
            and all(isinstance(v, (int, float)) for v in bbox)
        )

    def _extract_image_dims(self, meta: Dict[str, Any], bbox_info: Dict[str, Any]) -> Optional[Tuple[int, int]]:
        """Try a few common places/keys for image width/height in metadata."""
        candidates = [bbox_info, meta.get('target', {}), meta.get('metadata', {}), meta.get('data', {}), meta]
        for obj in candidates:
            if not isinstance(obj, dict):
                continue
            width = obj.get('image_width') or obj.get('img_width') or obj.get('width') or obj.get('w')
            height = obj.get('image_height') or obj.get('img_height') or obj.get('height') or obj.get('h')
            if isinstance(width, (int, float)) and isinstance(height, (int, float)) and width > 0 and height > 0:
                return int(width), int(height)
            size = obj.get('image_size') or obj.get('size') or obj.get('shape')
            if isinstance(size, (list, tuple)) and len(size) >= 2 and all(isinstance(v, (int, float)) for v in size[:2]):
                h, w = size[0], size[1]  # common H, W ordering
                if w > 0 and h > 0:
                    return int(w), int(h)
            if isinstance(size, dict):
                w = size.get('width') or size.get('w')
                h = size.get('height') or size.get('h')
                if isinstance(w, (int, float)) and isinstance(h, (int, float)) and w > 0 and h > 0:
                    return int(w), int(h)
        return None

    def _find_bbox_info(self, meta: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """Locate a bounding_box block in a few common nesting patterns."""
        if not isinstance(meta, dict):
            return None
        candidates = [
            meta.get('target'),
            meta.get('data', {}).get('target') if isinstance(meta.get('data'), dict) else None,
            meta.get('data', {}),
            meta,
        ]
        for cand in candidates:
            if isinstance(cand, dict):
                bb = cand.get('bounding_box')
                if isinstance(bb, dict) and 'bounding_box' in bb:
                    return bb
        return None

    def _apply_bbox_to_prediction(
        self,
        pred: Dict[str, Any],
        meta: Dict[str, Any],
        json_path: str,
    ) -> None:
        """Populate LS-style bbox values if metadata contains usable bbox info."""
        bbox_info = self._find_bbox_info(meta) or {}
        bbox = bbox_info.get('bounding_box') if isinstance(bbox_info, dict) else None
        padded = bbox_info.get('padded_bounding_box') if isinstance(bbox_info, dict) else None

        if not self._is_valid_bbox(bbox):
            self.logger.debug(f"No usable bbox in {json_path}; leaving zeros")
            return

        # Case 1: padded bbox present (preferred)
        if self._is_valid_bbox(padded):
            ow = padded[2] - padded[0]
            oh = padded[3] - padded[1]
            if ow > 0 and oh > 0:
                try:
                    x_perc = ((bbox[0] - padded[0]) / ow) * 100.0
                    y_perc = ((bbox[1] - padded[1]) / oh) * 100.0
                    w_perc = ((bbox[2] - bbox[0]) / ow) * 100.0
                    h_perc = ((bbox[3] - bbox[1]) / oh) * 100.0
                    pred['original_width'] = ow
                    pred['original_height'] = oh
                    pred['value'].update({'x': x_perc, 'y': y_perc, 'width': w_perc, 'height': h_perc})
                    return
                except Exception as e:
                    self.logger.debug(f"Bad padded bbox in {json_path}: {e}; trying fallbacks")

        # Case 2: use image dimensions from metadata
        dims = self._extract_image_dims(meta, bbox_info)
        if dims:
            img_w, img_h = dims
            try:
                bw = bbox[2] - bbox[0]
                bh = bbox[3] - bbox[1]
                if img_w > 0 and img_h > 0 and bw > 0 and bh > 0:
                    pred['original_width'] = img_w
                    pred['original_height'] = img_h
                    pred['value'].update({
                        'x': (bbox[0] / img_w) * 100.0,
                        'y': (bbox[1] / img_h) * 100.0,
                        'width': (bw / img_w) * 100.0,
                        'height': (bh / img_h) * 100.0,
                    })
                    return
            except Exception as e:
                self.logger.debug(f"Failed bbox % calc with image dims in {json_path}: {e}; trying normalized fallback")

        # Case 3: assume bbox already normalized (0-1 or 0-100)
        try:
            vals = list(map(float, bbox))
            max_val = max(vals)
            if 0 <= min(vals) and max_val <= 1.0:
                factor = 100.0
                width_val = (vals[2] - vals[0]) * factor
                height_val = (vals[3] - vals[1]) * factor
            elif 0 <= min(vals) and max_val <= 100.0:
                factor = 1.0
                width_val = (vals[2] - vals[0]) * factor
                height_val = (vals[3] - vals[1]) * factor
            else:
                self.logger.debug(f"Cannot infer bbox scale for {json_path}; leaving zeros")
                return
            pred['original_width'] = 100
            pred['original_height'] = 100
            pred['value'].update({
                'x': vals[0] * factor,
                'y': vals[1] * factor,
                'width': width_val,
                'height': height_val,
            })
        except Exception as e:
            self.logger.debug(f"Normalized bbox fallback failed for {json_path}: {e}; leaving zeros")


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
        description='Ship UUID Uploader (no recognition, no grouping) — ADC Edition',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=(
            "Examples:\n"
            "  # First-time auth (once per user):\n"
            "  #   gcloud auth application-default login\n\n"
            "  python ship_uuid_uploader.py \\\n"
            "    --bucket ratag \\\n"
            "    --gcs-path \"recognition/bronze/raw_jsons/eilat/azimut-eilat/2025/12/01\" \\\n"
            "    --num-uuids 200\n"
        )
    )
    p.add_argument('--bucket', required=True, help='Input GCS bucket')
    p.add_argument(
        '--gcs-path',
        required=True,
        help='Base GCS path to scan for UUIDs (legacy/mission/custom supported)'
    )
    p.add_argument('--num-uuids', type=int, default=10, help='Max UUIDs to process this run')
    p.add_argument(
        '--output-bucket',
        required=True,
        help='Output GCS bucket for groups and ledger'
    )
    p.add_argument('--project', default=None, help='GCP project (optional; otherwise auto-detected from ADC)')
    p.add_argument('--log-level', default='INFO', choices=['DEBUG', 'INFO', 'WARNING', 'ERROR'])
    p.add_argument('--dry-run', default=False, action='store_true', help='Run without writing to GCS')
    p.add_argument('--test-mode', default=False, action='store_true', help='Write groups to azimut-data/recognition/silver and skip ledger updates')
    return p.parse_args()


def main() -> int:
    args = parse_arguments()
    logger = setup_logging(args.log_level)

    # Test mode writes to recognition/silver and avoids ledger updates
    test_mode = args.test_mode
    output_bucket = args.output_bucket
    group_prefix = "recognition/silver" if test_mode else "recognition/bronze/labeling/Initial_groups_phase"

    try:
        logger.info("Initializing GCS client with ADC...")
        gcs = GCSClient(project=args.project)
        builder = UUIDGroupBuilder()

        # Startup ADC/GCS reachability check against the target bucket (no project needed)
        try:
            bucket = gcs.client.bucket(args.bucket)
            if not bucket.exists():
                logger.error(
                    f"Bucket gs://{args.bucket} does not exist or is not visible. "
                    "If 403 -> grant bucket-level IAM (Storage Object Viewer/Creator/Admin)."
                )
                return 1
            logger.info(f"GCS reachable and bucket '{args.bucket}' is accessible.")
        except Exception as e:
            logger.error(
                "ADC is set but a GCS call failed at startup. "
                "If the error is 401 -> re-run `gcloud auth application-default login`. "
                "If 403 -> grant bucket-level IAM (Storage Object Creator/Admin). "
                f"Original error: {e}"
            )
            raise

        logger.info("Loading processed UUID ledger...")
        try:
            processed = gcs.load_processed_uuids(output_bucket)
        except Exception:
            # load_processed_uuids already logs auth/IAM hints
            raise

        path_display = f"gs://{args.bucket}/{args.gcs_path.lstrip('/')}"
        logger.info(f"Discovering UUIDs in {path_display}...")
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
                images_with_jsons = gcs.list_images_for_uuid(args.bucket, args.gcs_path, uid)
                if not images_with_jsons:
                    logger.warning(f"No images for UUID {uid}; skipping")
                    continue

                group = builder.build_group_for_uuid(uid, images_with_jsons, args.bucket, gcs)
                if group is None:
                    continue

                if not args.dry_run:
                    if not gcs.save_group_json(output_bucket, group, prefix=group_prefix):
                        logger.error(f"Failed to save group for UUID {uid}")
                        continue

                success.append(uid)
            except Exception as e:
                logger.error(f"Error on UUID {uid}: {e}")
                continue

        logger.info(f"Successfully emitted {len(success)} group files")

        if not args.dry_run and success and not test_mode:
            updated = processed.union(success)
            if gcs.save_processed_uuids(output_bucket, updated):
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
