#!/usr/bin/env python3
"""
Third Phase: Groups Association export (SDK v2 snapshot)
-------------------------------------------------------
- Pull tasks via Label Studio SDK v2 snapshot/export (robust, no pagination).
- Keep ONLY review-accepted tasks that are NOT already in the processed registry.
- Extract 'same_vessel' from the accepted annotation.
- Preserve your original output JSON schema & ledger logic.

Outputs a timestamped JSON to:
  gs://{GCS_BUCKET_NAME}/{GCS_OUTPUT_PREFIX}/LS_{PROJECT_ID}_ACCEPTED_<ts>_<N>Tasks.json

Updates/creates a processed-IDs registry at:
  gs://{GCS_BUCKET_NAME}/{PROCESSED_TASKS_BLOB}

Authentication: Uses Application Default Credentials
Run: gcloud auth application-default login
"""

import os
import json
import time
import argparse
from typing import Dict, Any, List, Optional
from datetime import datetime, timezone

# Timezone handling (Asia/Jerusalem if tzdata/zoneinfo available)
try:
    from zoneinfo import ZoneInfo  # Python 3.9+
    TZ = ZoneInfo("Asia/Jerusalem")
except Exception:
    TZ = None

# --------- Label Studio SDK v2 ----------
from label_studio_sdk.client import LabelStudio  # SDK 2.x

# --------- GCS ----------
from google.cloud import storage
from google.api_core.exceptions import NotFound

# =======================================
# CONFIG — fill your token
# =======================================
BASE_URL   = "https://app.heartex.com"
PROJECT_ID = 187693
API_TOKEN  = "34aefb3582706dd6ac6306d1b63095be6857de52"  # <- keep secure

# Optional: set a saved LS View to reduce snapshot scope (e.g., 'reviewed only')
VIEW_ID: Optional[int] = None  # e.g., 123456 or None

# GCS destination for export files
GCS_BUCKET_NAME      = "azimut_data"
GCS_OUTPUT_PREFIX    = "recognition/silver/Third_Phase_Groups_Association/lable_studio_exports"

# Processed tasks registry (global ledger)
LEDGER_BUCKET = "azimut_data"
GLOBAL_PROCESSED_BLOB_PATH = "reidentification/silver/Third_Phase_Groups_Association/processed_tasksID_Phase3.json"

# If True, fix paths like "gs:/bucket/..." -> "gs://bucket/..."
NORMALIZE_GS_SCHEME = False

# Polling interval for snapshot completion
SNAPSHOT_POLL_SECS = 1.8


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


def _extract_bucket(url: Any) -> Optional[str]:
    if not isinstance(url, str):
        return None
    if not url.startswith("gs://"):
        return None
    rest = url[5:]
    if not rest:
        return None
    return rest.split("/", 1)[0]


def _entry_buckets(entry: Dict[str, Any], fallback_bucket: str) -> List[str]:
    buckets: List[str] = []
    for key in ("r_id_1_images", "r_id_1_jsons", "r_id_2_images", "r_id_2_jsons"):
        for url in entry.get(key) or []:
            b = _extract_bucket(url)
            if b:
                buckets.append(b)
    if not buckets:
        buckets = [fallback_bucket]
    return sorted(set(buckets))

def _warn_if_misaligned(task_id: Any, who: str,
                        imgs: List[str], jsons: List[str], bboxes: List[List[float]]):
    li, lj, lb = len(imgs), len(jsons), len(bboxes)
    if not (li == lj == lb or (lb == 0 and li == lj)):
        print(f"[Warn][Task {task_id}] {who}: length mismatch "
              f"images={li}, jsons={lj}, bboxes={lb}")


# ---------------- GCS helpers ----------------
def _gcs_client() -> storage.Client:
    """Connect to GCS using Application Default Credentials (your logged-in account)."""
    return storage.Client()

def save_json_to_gcs(data: Any, filename: str, bucket_name: str) -> str:
    """Upload JSON to GCS (export area) and return gs:// path."""
    client = _gcs_client()
    bucket = client.bucket(bucket_name)
    blob_path = f"{GCS_OUTPUT_PREFIX}/{filename}"
    blob = bucket.blob(blob_path)
    blob.upload_from_string(
        json.dumps(data, ensure_ascii=False, indent=2),
        content_type="application/json"
    )
    return f"gs://{bucket_name}/{blob_path}"

def load_processed_registry(ledger_bucket: str) -> Dict[str, Any]:
    """Load processed task IDs registry from GCS (or return an empty structure)."""
    client = _gcs_client()
    bucket = client.bucket(ledger_bucket)
    blob = bucket.blob(GLOBAL_PROCESSED_BLOB_PATH)
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

def save_processed_registry(registry: Dict[str, Any], ledger_bucket: str) -> str:
    """Save processed task IDs registry back to GCS and return its gs:// path."""
    client = _gcs_client()
    bucket = client.bucket(ledger_bucket)
    blob = bucket.blob(GLOBAL_PROCESSED_BLOB_PATH)
    blob.upload_from_string(
        json.dumps(registry, ensure_ascii=False, indent=2),
        content_type="application/json"
    )
    return f"gs://{ledger_bucket}/{GLOBAL_PROCESSED_BLOB_PATH}"


# ---------------- LS snapshot/export ----------------
def connect_label_studio() -> LabelStudio:
    ls = LabelStudio(base_url=BASE_URL, api_key=API_TOKEN, timeout=30.0)
    _ = ls.projects.list(page_size=1)  # sanity
    return ls

def fetch_tasks_via_snapshot(ls: LabelStudio, project_id: int, view_id: Optional[int]) -> List[Dict[str, Any]]:
    """
    Uses SDK v2 snapshot/export to download tasks as JSON (stable, no pagination).
    If view_id is provided, LS applies it server-side; otherwise returns all tasks.
    """
    print(f"Creating export snapshot (project={project_id}, view_id={view_id}) …")
    export = ls.projects.exports.create(
        id=project_id,
        title=f"phase3_export_{int(time.time())}",
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


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Third Phase: Groups Association export (SDK v2 snapshot)")
    p.add_argument("--bucket", default=GCS_BUCKET_NAME, help="Default export bucket (used when inference is ambiguous)")
    return p.parse_args()


# ---------------- Main ----------------
def main():
    args = parse_args()
    global GCS_BUCKET_NAME
    GCS_BUCKET_NAME = args.bucket

    # Fixed ledger bucket
    ledger_bucket = LEDGER_BUCKET

    # Preload registry if ledger bucket is provided up front
    registry = {}
    raw_ids: List[Any] = []
    processed_ids: set[int] = set()
    if ledger_bucket:
        reg_init = load_processed_registry(ledger_bucket)
        registry = reg_init or {}
        raw_ids = registry.get("processed_tasksID", []) or []
        for x in raw_ids:
            try:
                processed_ids.add(int(x))
            except Exception:
                pass
        print(f"[registry] loaded {len(processed_ids)} processed IDs from {ledger_bucket}")

    # Connect + snapshot
    ls = connect_label_studio()
    tasks = fetch_tasks_via_snapshot(ls, PROJECT_ID, VIEW_ID)

    outputs: List[Dict[str, Any]] = []
    new_task_ids: List[int] = []

    # Diagnostics
    already_processed_task_ids: List[int] = []
    no_annotations_task_ids: List[int] = []
    not_accepted_task_ids: List[int] = []

    for t in tasks:
        tid = t.get("id")
        if tid is None:
            continue

        # Skip already processed tasks (if we have a registry loaded already)
        if processed_ids and tid in processed_ids:
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

        # Extract same_vessel from annotation
        same_vessel = extract_same_vessel(ann)

        d = t.get("data") or {}

        # Optional sanity check: warn if lists misalign
        _warn_if_misaligned(tid, "r_id_1",
                            _norm_gs(d.get("r_id_1_images") or []),
                            _norm_gs(d.get("r_id_1_jsons") or []),
                            _coerce_bboxes(d.get("r_id_1_bboxes") or []))
        _warn_if_misaligned(tid, "r_id_2",
                            _norm_gs(d.get("r_id_2_images") or []),
                            _norm_gs(d.get("r_id_2_jsons") or []),
                            _coerce_bboxes(d.get("r_id_2_bboxes") or []))

        # timestamps / uuids
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
        new_task_ids.append(tid)

    # ---- DIAGNOSTICS ----
    print("============== DEBUG / DIAGNOSTICS ==============")
    print(f"Accepted in snapshot    : {len(set(new_task_ids))}")
    print(f"Already processed (pre) : {len(set(already_processed_task_ids))}")
    print(f"No annotations: {len(set(no_annotations_task_ids))}")
    print(f"Not accepted: {len(set(not_accepted_task_ids))}")
    print("==================================================")

    # Determine ledger bucket if not provided
    if ledger_bucket is None:
        # infer from outputs if possible
        inferred = []
        for out in outputs:
            inferred.extend(_entry_buckets(out, GCS_BUCKET_NAME))
        ledger_bucket = inferred[0] if inferred else GCS_BUCKET_NAME
    print(f"Using global ledger bucket: {ledger_bucket}")

    # Load processed tasks registry and coerce IDs to int (global)
    registry = load_processed_registry(ledger_bucket)
    raw_ids = registry.get("processed_tasksID", []) or []
    processed_ids = set()
    for x in raw_ids:
        try:
            processed_ids.add(int(x))
        except Exception:
            pass
    print(f"[registry] loaded {len(processed_ids)} processed IDs")

    # Filter outputs against processed IDs
    fresh_outputs: List[Dict[str, Any]] = []
    for out in outputs:
        tid = out.get("task_id")
        if tid is None:
            fresh_outputs.append(out)
            continue
        if int(tid) in processed_ids:
            continue
        fresh_outputs.append(out)

    print(f"New after dedupe        : {len(fresh_outputs)}")
    if not fresh_outputs:
        print("ℹ️ No new tasks to export (all already processed or none accepted).")
    # Partition outputs by bucket
    bucket_outputs: Dict[str, List[Dict[str, Any]]] = {}
    multi_bucket_entries: List[Dict[str, Any]] = []
    for out in fresh_outputs:
        buckets = _entry_buckets(out, GCS_BUCKET_NAME)
        if len(buckets) > 1:
            multi_bucket_entries.append({"task_id": out.get("task_id"), "buckets": buckets})
        target = buckets[0]
        bucket_outputs.setdefault(target, []).append(out)

    if multi_bucket_entries:
        print("⚠ Entries with multiple buckets detected (using the first bucket listed):")
        for entry in multi_bucket_entries[:10]:
            print(f"   task_id={entry['task_id']} buckets={entry['buckets']}")
        if len(multi_bucket_entries) > 10:
            print(f"   ... and {len(multi_bucket_entries) - 10} more")

    # Save per-bucket exports
    for bucket_name, outs in bucket_outputs.items():
        if not outs:
            continue
        now_local = datetime.now(TZ) if TZ else datetime.now()
        ts = now_local.strftime("%Y-%m-%d_%H-%M")
        filename = f"LS_{PROJECT_ID}_ACCEPTED_{ts}_{len(outs)}Tasks.json"
        gs_path = save_json_to_gcs(outs, filename, bucket_name)
        print(f"✅ Saved {len(outs)} NEW accepted tasks to: {gs_path}")

    # Update global registry
    def _to_ints(seq):
        out = set()
        for x in seq or []:
            try:
                out.add(int(x))
            except Exception:
                pass
        return out

    existing_ids_int = _to_ints(raw_ids)
    added_ids_int    = _to_ints([out.get("task_id") for out in fresh_outputs])
    merged_ids_int   = sorted(existing_ids_int | added_ids_int)

    registry["processed_tasksID"] = [str(x) for x in merged_ids_int]
    now_utc = datetime.now(timezone.utc).isoformat(timespec="microseconds").replace("+00:00", "Z")
    registry["last_updated"] = now_utc
    registry["total_count"]  = len(merged_ids_int)

    reg_path = save_processed_registry(registry, ledger_bucket)

    if added_ids_int:
        print(f"📝 Updated processed registry with {len(added_ids_int)} new IDs → {reg_path}")
        print(f"🧮 Registry total_count: {len(merged_ids_int)}")
        sample = list(sorted(added_ids_int))[:5]
        print(f"🧾 Sample added IDs: {sample}")
    else:
        print(f"📝 Processed registry timestamp updated (no new IDs) → {reg_path}")
        print(f"🧮 Registry total_count: {len(merged_ids_int)}")


if __name__ == "__main__":
    main()
