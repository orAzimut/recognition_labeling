#!/usr/bin/env python3
"""
Groups Association Phase export
--------------------------------
- Pull tasks via Label Studio SDK v2 snapshot/export (robust & simple).
- Keep ONLY review-accepted tasks that are NOT already in the processed registry.
- Extract `same_vessel` from the accepted annotation.
- Preserve your original output JSON schema & ledger logic.
- NEW: Automatically triggers ship_third_stage_processing.py after successful export.

Outputs a timestamped JSON to:
  gs://{GCS_BUCKET_NAME}/{GCS_OUTPUT_PREFIX}/LS_{PROJECT_ID}_ACCEPTED_<ts>_<N>Tasks.json

Updates/creates a processed-IDs registry at:
  gs://{GCS_BUCKET_NAME}/{PROCESSED_TASKS_BLOB}
"""

import os
import sys
import json
import time
import subprocess
from typing import Dict, Any, List, Optional
from datetime import datetime, timezone

# Timezone handling (Asia/Jerusalem if tzdata/zoneinfo available)
try:
    from zoneinfo import ZoneInfo  # Python 3.9+
    TZ = ZoneInfo("Asia/Jerusalem")
except Exception:
    TZ = None

# --------- LS SDK v2 ----------
from label_studio_sdk.client import LabelStudio  # <-- SDK 2.x

# --------- GCS ----------
from google.cloud import storage
from google.oauth2 import service_account
from google.api_core.exceptions import NotFound

# =======================================
# CONFIG — fill your token / paths
# =======================================
BASE_URL   = "https://app.heartex.com"
PROJECT_ID = 185882
API_TOKEN  = "5b62611f13b4beb4d85c4b48e2cb10651a8442e9"  # <-- put your token

# Optional: if you have a saved LS View that pre-filters tasks (e.g. to 'reviewed only')
VIEW_ID: Optional[int] = None  # e.g., 123456 or None

# GCS destination for export files
GCS_CREDENTIALS_PATH = r"C:\Users\OrGil.AzureAD\OneDrive - AMPC\Desktop\Azimut.ai\recognition_labeling\resources\credentials.json"
GCS_BUCKET_NAME = "azimut_data"
GCS_OUTPUT_PREFIX = "reidentification/silver/Groups_Association_Phase_cleaned/lable_studio_exports"

# Processed tasks registry (read/write)
PROCESSED_TASKS_BLOB = "reidentification/silver/Groups_Association_Phase_cleaned/processed_tasksID_match.json"

# If True, fix paths like "gs:/bucket/..." -> "gs://bucket/..."
NORMALIZE_GS_SCHEME = False

# Polling interval for snapshot completion
SNAPSHOT_POLL_SECS = 1.8

# Third Stage Processing Configuration
THIRD_STAGE_SCRIPT_PATH = r"C:\Users\OrGil.AzureAD\OneDrive - AMPC\Desktop\Azimut.ai\recognition_labeling\postProcess\ship_third_stage_processing.py"
RECOGNITION_SERVICE_URL = "http://localhost:8080"  # Ship-Recognition-Service URL
AUTO_TRIGGER_THIRD_STAGE = True  # Set to False to disable auto-triggering

# ---------------- Acceptance helpers ----------------
def _low(x): 
    return str(x).strip().lower()

def is_review_accepted(ann: Dict[str, Any]) -> bool:
    """Decide if an annotation is review-accepted across common LS variants."""
    if ann.get("was_cancelled"):
        return False
    la = ann.get("last_action")
    if la and _low(la) in {"accepted", "fixed_and_accepted"}:
        return True
    for key in ("status", "review_result", "review_status"):
        val = ann.get(key)
        if val and _low(val) in {"accepted", "accepted_auto", "approved"}:
            return True
    if ann.get("approved") is True or ann.get("was_accepted") is True:
        return True
    for rv in ann.get("reviews") or []:
        if rv.get("accepted") is True:
            return True
        if _low(rv.get("result", "")) in {"accepted", "accepted_auto", "approved"}:
            return True
    return False

def extract_same_vessel(ann: Dict[str, Any]) -> Optional[str]:
    """Extract 'Yes'/'No' from the same_vessel choices (resilient to from_name naming)."""
    for item in ann.get("result", []) or []:
        if str(item.get("type")).lower() != "choices":
            continue
        from_name = str(item.get("from_name", "")).lower()
        to_name   = str(item.get("to_name", "")).lower()
        choices   = (item.get("value") or {}).get("choices", []) or []
        if not choices:
            continue
        # Heuristics: any 'same'/'vessel' in field names, or direct Yes/No match
        if ("same" in from_name or "vessel" in from_name or
            "same" in to_name   or "vessel" in to_name):
            return choices[0]
        if choices[0] in ("Yes", "No"):
            return choices[0]
    return None

# ---------------- gs:// & bbox helpers ----------------
def _norm_gs(paths: List[str]) -> List[str]:
    if not NORMALIZE_GS_SCHEME:
        return paths or []
    fixed = []
    for p in paths or []:
        if isinstance(p, str) and p.startswith("gs:/") and not p.startswith("gs://"):
            fixed.append("gs://" + p[4:])
        else:
            fixed.append(p)
    return fixed

def _coerce_bbox(bb: Any) -> Optional[List[float]]:
    """Ensure a single bbox is [x1, y1, x2, y2] floats."""
    if not isinstance(bb, (list, tuple)) or len(bb) != 4:
        return None
    out: List[float] = []
    for v in bb:
        try:
            out.append(float(v))
        except Exception:
            return None
    return out

def _coerce_bboxes(bbs: Any) -> List[List[float]]:
    """Coerce a list of bboxes into clean [x1, y1, x2, y2] float lists."""
    from collections.abc import Iterable as _Iterable
    if not isinstance(bbs, _Iterable) or isinstance(bbs, (str, bytes)):
        return []
    cleaned: List[List[float]] = []
    for bb in bbs:
        c = _coerce_bbox(bb)
        if c is not None:
            cleaned.append(c)
    return cleaned

# ---------------- GCS helpers ----------------
def _gcs_client() -> storage.Client:
    creds = service_account.Credentials.from_service_account_file(GCS_CREDENTIALS_PATH)
    return storage.Client(credentials=creds)

def save_json_to_gcs(data: Any, filename: str) -> str:
    """Upload JSON to GCS (export area) and return gs:// path."""
    client = _gcs_client()
    bucket = client.bucket(GCS_BUCKET_NAME)
    blob_path = f"{GCS_OUTPUT_PREFIX}/{filename}"
    blob = bucket.blob(blob_path)
    blob.upload_from_string(
        json.dumps(data, ensure_ascii=False, indent=2),
        content_type="application/json"
    )
    return f"gs://{GCS_BUCKET_NAME}/{blob_path}"

def load_processed_registry() -> Dict[str, Any]:
    """Load processed task IDs registry from GCS (or return an empty structure)."""
    client = _gcs_client()
    bucket = client.bucket(GCS_BUCKET_NAME)
    blob = bucket.blob(PROCESSED_TASKS_BLOB)
    try:
        content = blob.download_as_text()
        obj = json.loads(content)
        if not isinstance(obj, dict):
            raise ValueError("processed registry JSON is not an object")
    except NotFound:
        obj = {"processed_tasksID": [], "last_updated": None, "total_count": 0}
    except Exception as e:
        print(f"[Warn] Failed loading processed registry: {e}")
        obj = {"processed_tasksID": [], "last_updated": None, "total_count": 0}

    ids = obj.get("processed_tasksID") or []
    if not isinstance(ids, list):
        ids = []
    obj["processed_tasksID"] = ids
    obj["total_count"] = len(ids)
    return obj

def save_processed_registry(registry: Dict[str, Any]) -> str:
    """Save processed task IDs registry back to GCS and return its gs:// path."""
    client = _gcs_client()
    bucket = client.bucket(GCS_BUCKET_NAME)
    blob = bucket.blob(PROCESSED_TASKS_BLOB)
    blob.upload_from_string(
        json.dumps(registry, ensure_ascii=False, indent=2),
        content_type="application/json"
    )
    return f"gs://{GCS_BUCKET_NAME}/{PROCESSED_TASKS_BLOB}"

# ---------------- LS snapshot/export ----------------
def connect_label_studio() -> LabelStudio:
    ls = LabelStudio(base_url=BASE_URL, api_key=API_TOKEN, timeout=30.0)
    # quick sanity
    _ = ls.projects.list(page_size=1)
    return ls

def fetch_tasks_via_snapshot(ls: LabelStudio, project_id: int, view_id: Optional[int]) -> List[Dict[str, Any]]:
    """
    Uses SDK v2 snapshot/export to download tasks as JSON (stable, no pagination).
    If view_id is provided, LS applies it server-side; otherwise returns all tasks.
    """
    print(f"Creating export snapshot (project={project_id}, view_id={view_id}) …")
    export = ls.projects.exports.create(
        id=project_id,
        title=f"groups_assoc_export_{int(time.time())}",
        task_filter_options=({"view": int(view_id)} if view_id else {}),
    )
    export_pk = export.id

    # Poll for completion
    while True:
        info = ls.projects.exports.get(id=project_id, export_pk=export_pk)
        status = getattr(info, "status", None)
        if status in ("completed", "failed"):
            print(f"Export status: {status}")
            if status == "failed":
                details = getattr(info, "error", None) or getattr(info, "details", None)
                raise RuntimeError(f"Export failed: {details or info}")
            break
        time.sleep(SNAPSHOT_POLL_SECS)

    print("Downloading export JSON …")
    downloaded = ls.projects.exports.download(id=project_id, export_pk=export_pk, export_type="JSON")

    # Normalize to Python list[dict]
    if isinstance(downloaded, (bytes, bytearray)):
        payload = json.loads(downloaded.decode("utf-8"))
    elif isinstance(downloaded, str) and os.path.exists(downloaded):
        with open(downloaded, "r", encoding="utf-8") as f:
            payload = json.load(f)
    elif hasattr(downloaded, "read"):
        payload = json.load(downloaded)
    elif hasattr(downloaded, "__iter__") and not isinstance(downloaded, (str, bytes, bytearray)):
        buf = bytearray()
        for chunk in downloaded:
            if not chunk:
                continue
            if isinstance(chunk, str):
                chunk = chunk.encode("utf-8")
            buf.extend(chunk)
        payload = json.loads(buf.decode("utf-8"))
    else:
        raise TypeError(f"Unexpected download() return type: {type(downloaded)}")

    if not isinstance(payload, list):
        raise ValueError("Export payload is not a list of tasks")

    return payload

# ---------------- Third Stage Trigger ----------------
def trigger_third_stage_processing(labeled_json_path: str) -> bool:
    """
    Trigger the third stage processing script with the newly created export.
    
    Args:
        labeled_json_path: Path to the exported JSON file (with leading slash)
    
    Returns:
        True if third stage processing succeeded, False otherwise
    """
    if not labeled_json_path:
        print("WARNING: No export file to process")
        return False
    
    print("\n" + "=" * 60)
    print("TRIGGERING THIRD STAGE PROCESSING")
    print("=" * 60)
    print(f"Input file: {labeled_json_path}")
    print(f"Bucket: {GCS_BUCKET_NAME}")
    print(f"Service URL: {RECOGNITION_SERVICE_URL}")
    print()
    
    try:
        # Build command
        cmd = [
            sys.executable,  # Use the same Python interpreter
            THIRD_STAGE_SCRIPT_PATH,
            "--bucket", GCS_BUCKET_NAME,
            "--labeled-json", labeled_json_path,
            "--service-url", RECOGNITION_SERVICE_URL,
            "--credentials", GCS_CREDENTIALS_PATH,
        ]
        
        print(f"Running command: {' '.join(cmd)}\n")
        
        # Run the third stage processing script
        result = subprocess.run(
            cmd,
            check=True,
            capture_output=False,  # Show output in real-time
            text=True
        )
        
        print("\nThird stage processing completed successfully")
        return True
        
    except subprocess.CalledProcessError as e:
        print(f"\nERROR: Third stage processing failed with exit code {e.returncode}")
        return False
    except FileNotFoundError:
        print(f"\nERROR: Could not find third stage processing script: {THIRD_STAGE_SCRIPT_PATH}")
        print("Please ensure 'ship_third_stage_processing.py' is accessible at the configured path")
        return False
    except Exception as e:
        print(f"\nERROR: Unexpected error triggering third stage processing: {e}")
        return False

# ---------------- Main ----------------
def main():
    # Load processed tasks registry
    registry = load_processed_registry()
    processed_ids: set[int] = set(registry.get("processed_tasksID", []))
    print(f"Loaded registry with {len(processed_ids)} processed task IDs.")

    # Connect + snapshot
    ls = connect_label_studio()
    tasks = fetch_tasks_via_snapshot(ls, PROJECT_ID, VIEW_ID)

    outputs: List[Dict[str, Any]] = []
    new_task_ids: List[int] = []

    # ---- DEBUG TRACKING ----
    all_task_ids: List[int] = []
    accepted_task_ids: List[int] = []
    not_accepted_task_ids: List[int] = []
    no_annotations_task_ids: List[int] = []
    already_processed_task_ids: List[int] = []

    for t in tasks:
        tid = t.get("id")
        if tid is None:
            continue
        all_task_ids.append(tid)

        if tid in processed_ids:
            already_processed_task_ids.append(tid)
            continue

        anns = t.get("annotations") or []
        if not anns:
            no_annotations_task_ids.append(tid)
            continue

        accepted = [a for a in anns if is_review_accepted(a)]
        if not accepted:
            not_accepted_task_ids.append(tid)
            continue

        # Most recent accepted annotation
        accepted.sort(
            key=lambda a: (
                a.get("updated_at") or a.get("created_at") or "",
                a.get("id", 0)
            ),
            reverse=True
        )
        ann = accepted[0]

        same_vessel = extract_same_vessel(ann)

        # mark as processed and record accepted
        new_task_ids.append(tid)
        accepted_task_ids.append(tid)

        d = t.get("data", {}) or {}

        # timestamps & uuids (your additions)
        r_id_1_timestamp = d.get("r_id_1_timestamp")
        r_id_2_timestamp = d.get("r_id_2_timestamp")
        r_id_1_uuid = d.get("r_id_1_uuid", "")
        r_id_2_uuid = d.get("r_id_2_uuid", "")

        outputs.append({
            "task_id": tid,
            "annotation_id": ann.get("id"),
            "r_id_1": d.get("r_id_1"),
            "r_id_1_images": _norm_gs(d.get("r_id_1_images") or []),
            "r_id_1_jsons": _norm_gs(d.get("r_id_1_jsons") or []),
            "r_id_1_bboxes": _coerce_bboxes(d.get("r_id_1_bboxes") or []),
            "r_id_1_timestamp": r_id_1_timestamp,
            "r_id_1_uuid": r_id_1_uuid,
            "r_id_2": d.get("r_id_2"),
            "r_id_2_images": _norm_gs(d.get("r_id_2_images") or []),
            "r_id_2_jsons": _norm_gs(d.get("r_id_2_jsons") or []),
            "r_id_2_bboxes": _coerce_bboxes(d.get("r_id_2_bboxes") or []),
            "r_id_2_timestamp": r_id_2_timestamp,
            "r_id_2_uuid": r_id_2_uuid,
            "same_vessel": same_vessel,
        })

    # ---- DEBUG REPORT ----
    all_task_ids_sorted = sorted(all_task_ids)
    accepted_sorted_new = sorted(set(accepted_task_ids))
    not_accepted_sorted = sorted(set(not_accepted_task_ids))
    no_annotations_sorted = sorted(set(no_annotations_task_ids))
    already_processed_sorted = sorted(set(already_processed_task_ids))

    print("============== DEBUG / DIAGNOSTICS ==============")
    print(f"Total tasks in snapshot: {len(all_task_ids_sorted)}")
    print(f"Accepted (NEW this run): {len(accepted_sorted_new)}")
    print(f"Accepted IDs (NEW): {accepted_sorted_new}")
    print(f"Not accepted: {len(not_accepted_sorted)}  | IDs: {not_accepted_sorted}")
    print(f"No annotations: {len(no_annotations_sorted)} | IDs: {no_annotations_sorted}")
    print(f"Already processed (registry): {len(already_processed_sorted)} | IDs: {already_processed_sorted}")
    print("==================================================")

    # Timestamped filename (Asia/Jerusalem if available)
    now_local = datetime.now(TZ) if TZ else datetime.now()
    ts = now_local.strftime("%Y-%m-%d_%H-%M")
    filename = f"LS_{PROJECT_ID}_ACCEPTED_{ts}_{len(outputs)}Tasks.json"

    # Save export if any NEW accepted tasks
    exported_file_path = None
    if outputs:
        gs_path = save_json_to_gcs(outputs, filename)
        print(f"✅ Saved {len(outputs)} accepted tasks to: {gs_path}")
        # Extract path for triggering (remove gs://bucket/ prefix, keep just the path)
        exported_file_path = f"/{GCS_OUTPUT_PREFIX}/{filename}"
    else:
        print("ℹ️ No new accepted tasks to export (all processed / none accepted).")

    # Update registry (always bump timestamp)
    merged_ids = sorted(set(registry.get("processed_tasksID", [])).union(new_task_ids))
    now_utc = datetime.now(timezone.utc).isoformat(timespec="microseconds").replace("+00:00", "Z")
    registry["processed_tasksID"] = merged_ids
    registry["last_updated"] = now_utc
    registry["total_count"] = len(merged_ids)
    reg_path = save_processed_registry(registry)

    if new_task_ids:
        print(f"📝 Updated processed registry with {len(set(new_task_ids))} new IDs → {reg_path}")
        print(f"🧮 Registry total_count: {len(merged_ids)}")
    else:
        print(f"📝 Processed registry timestamp updated (no new IDs) → {reg_path}")
        print(f"🧮 Registry total_count: {len(merged_ids)}")
    
    return exported_file_path  # Return the path for triggering next stage

if __name__ == "__main__":
    # Run the main export script
    exported_file = main()
    
    # If an export was created and auto-trigger is enabled, trigger third stage processing
    if exported_file and AUTO_TRIGGER_THIRD_STAGE:
        print("\n" + "=" * 60)
        print("Stage 2 Complete - Starting Stage 3")
        print("=" * 60)
        
        # Trigger third stage processing with the newly created file
        success = trigger_third_stage_processing(exported_file)
        
        if success:
            print("\n🎉 Pipeline completed successfully!")
            sys.exit(0)
        else:
            print("\n⚠️  Pipeline completed with errors in Stage 3")
            sys.exit(1)
    elif exported_file and not AUTO_TRIGGER_THIRD_STAGE:
        print(f"\nAuto-trigger disabled. To process manually, run:")
        print(f"python {THIRD_STAGE_SCRIPT_PATH} --bucket {GCS_BUCKET_NAME} --labeled-json {exported_file}")
        sys.exit(0)
    else:
        print("\nNo new data to process - pipeline stopped at Stage 2")
        sys.exit(0)