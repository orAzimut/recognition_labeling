#!/usr/bin/env python3
"""
Export ONLY review-accepted (>=1) tasks that have NOT been processed before,
using Label Studio SDK v2 snapshot/export for reliability, and preserving the
output structure/logic from the original script.

- Maintains a ledger in GCS: processed_tasksID_Initial.json
- Skips duplicates on every run
- Extracts per-image Trash flags + bboxes for <Image valueList="$images">
- Summarizes bbox classes per image and per task (class_histogram)

Tested with Label Studio SDK v2.x
"""

import os
import json
import time
import re
from typing import Any, Dict, List, Optional, Tuple
from datetime import datetime, timezone
from collections import Counter

# --- Label Studio SDK v2 ---
from label_studio_sdk.client import LabelStudio  # SDK 2.x

# --- GCS ---
from google.cloud import storage
from google.oauth2 import service_account

try:
    from zoneinfo import ZoneInfo
except Exception:
    ZoneInfo = None


# =========================
# CONFIG — EDIT THESE
# =========================
# Label Studio
LS_URL = "https://app.heartex.com"
LS_API_KEY = "e3dd5c79ff9086a6b8769a35905cb249448cf3e9"  # <-- replace
PROJECT_ID = 186048

# If you have a saved View in LS that already filters to reviewed/accepted,
# set VIEW_ID; otherwise leave as None and we’ll filter client-side.
VIEW_ID: Optional[int] = None  # e.g., 123456 or None

# Labeling interface control names / values
IMAGE_TO_NAME = "images"             # <Image name="images" valueList="$images" />
BBOX_FROM_NAME = "bbox_labels"       # <RectangleLabels name="bbox_labels" ... />
FLAGS_FROM_NAME = "flags"            # <Choices name="flags" ... />
TRASH = "Trash"
ALL_TRASH = "All-Trash"

# BBox classes allowed
ALLOWED_CLASSES = {"MainGroup"} | {f"OutGroup{i}" for i in range(10)}

# GCS destination (export)
GCS_CREDENTIALS_PATH = r"C:\Users\OrGil.AzureAD\OneDrive - AMPC\Desktop\Azimut.ai\recognition_labeling\resources\credentials.json"
GCS_BUCKET = "azimut_data"
GCS_EXPORT_PREFIX = "reidentification/silver/Initial_groups_phase_cleaned/lable_studio_exports"

# GCS ledger (processed tasks)
PROCESSED_BLOB_PATH = "reidentification/silver/Initial_groups_phase_cleaned/processed_tasksID_Initial.json"

# Upload an export file even if there are 0 new tasks
UPLOAD_EMPTY_EXPORT = False

# Review acceptance guard (client-side, resilient to older payloads)
ACCEPTED_LAST_ACTIONS = {"accepted", "fixed_and_accepted"}


# =========================
# Utilities (time, GCS)
# =========================
def jerusalem_stamp_for_filename(n_tasks: int) -> str:
    tz = ZoneInfo("Asia/Jerusalem") if ZoneInfo else None
    now = datetime.now(tz) if tz else datetime.now()
    return f"{now.strftime('%Y-%m-%d_%H-%M')}_{n_tasks}Tasks.json"


def utc_now_z() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="microseconds").replace("+00:00", "Z")


def gcs_client():
    creds = service_account.Credentials.from_service_account_file(GCS_CREDENTIALS_PATH)
    return storage.Client(credentials=creds)


def gcs_download_json(bucket_name: str, blob_path: str) -> Optional[Dict[str, Any]]:
    client = gcs_client()
    blob = client.bucket(bucket_name).blob(blob_path)
    if not blob.exists():
        return None
    txt = blob.download_as_text()
    try:
        return json.loads(txt)
    except Exception:
        return None


def gcs_upload_json(bucket_name: str, blob_path: str, obj: Any):
    client = gcs_client()
    blob = client.bucket(bucket_name).blob(blob_path)
    data = json.dumps(obj, indent=2, ensure_ascii=False)
    blob.upload_from_string(data, content_type="application/json")
    print(f" Uploaded to gs://{bucket_name}/{blob_path}")


# =========================
# Acceptance / filtering helpers
# =========================
def _ann_is_accepted(ann: dict) -> bool:
    la = (ann or {}).get("last_action")
    rs = (ann or {}).get("review_status") or (ann or {}).get("review_result")
    return (isinstance(la, str) and la in ACCEPTED_LAST_ACTIONS) or (isinstance(rs, str) and rs in ACCEPTED_LAST_ACTIONS)


def _task_passes_review(task: dict) -> bool:
    if int(task.get("reviews_accepted", 0)) >= 1:
        return True
    for a in (task.get("annotations") or []):
        if isinstance(a, dict) and _ann_is_accepted(a):
            return True
    return False


# =========================
# Extraction helpers
# =========================
# BUGFIX: group(2) (the numeric part), not group(1)
ID_NUM_RE = re.compile(r'(:images(?:\$|\[)|\[\s*)(\d+)(?:\]|$)')

def _index_from_any(res: dict) -> Optional[int]:
    if not isinstance(res, dict):
        return None
    v = (res.get("value") or {}) if isinstance(res.get("value"), dict) else {}
    for k in ("item_index", "image_index", "index"):
        if isinstance(res.get(k), int):
            return res[k]
    for k in ("image_index", "index", "item_index"):
        if isinstance(v.get(k), int):
            return v[k]
    for holder in (v, res):
        if isinstance(holder, dict) and "item" in holder:
            try:
                return int(holder["item"])
            except Exception:
                pass
    for k in ("path", "$path", "item_path", "id", "to_id", "from_id"):
        s = res.get(k)
        if isinstance(s, str):
            m = ID_NUM_RE.search(s)
            if m:
                try:
                    return int(m.group(2))  # <- numeric capture
                except Exception:
                    pass
    return None


def _collect_by_choice(results: List[Dict], images: List[str], label_value: str) -> Tuple[List[str], List[dict]]:
    """Find images where Choices includes label_value (Trash/All-Trash)."""
    mapped, unbound = [], []
    if not isinstance(results, list):
        return mapped, unbound
    for res in results:
        if not isinstance(res, dict):
            continue
        if str(res.get("type")).lower() != "choices":
            continue
        if res.get("to_name") != IMAGE_TO_NAME:
            continue
        # allow lower/upper
        if str(res.get("from_name")).lower() != str(FLAGS_FROM_NAME).lower():
            continue

        v = res.get("value") or {}
        if not isinstance(v, dict):
            continue
        choices = v.get("choices") or []
        if label_value not in choices:
            continue

        # direct gs:// reference wins
        if isinstance(v.get("image"), str) and v["image"].startswith("gs://"):
            mapped.append(v["image"])
            continue

        idx = _index_from_any(res)
        if isinstance(idx, int):
            if 0 <= idx < len(images):
                mapped.append(images[idx])
                continue
            if 1 <= idx <= len(images):
                mapped.append(images[idx - 1])
                continue

        unbound.append({"which": label_value, "result": res})
    return mapped, unbound


def _has_all_trash(task: Dict) -> bool:
    images = (task.get("data") or {}).get("images", []) or task.get("images", [])
    for a in task.get("annotations") or []:
        res = a.get("result") or []
        m, u = _collect_by_choice(res, images, ALL_TRASH)
        if m or u:
            return True
    for d in task.get("drafts") or []:
        res = d.get("result") or []
        m, u = _collect_by_choice(res, images, ALL_TRASH)
        if m or u:
            return True
    return False


def _extract_bboxes(
    results: List[Dict],
    images: List[str],
    source: str = "annotation",
    parent_id: Optional[Any] = None,
) -> Dict[str, List[Dict]]:
    """
    Extract only <RectangleLabels name="bbox_labels"> to <Image name="images"> items,
    keep a single 'class' string for each bbox, and attach basics.
    """
    out = {img: [] for img in images}
    if not isinstance(results, list):
        return out

    for res in results:
        if not isinstance(res, dict):
            continue
        if str(res.get("type")).lower() != "rectanglelabels":
            continue
        if res.get("from_name") != BBOX_FROM_NAME:
            continue
        if res.get("to_name") != IMAGE_TO_NAME:
            continue

        v = res.get("value") or {}
        if not isinstance(v, dict):
            continue

        idx = _index_from_any(res)
        if idx is None:
            continue

        img_url = None
        if 0 <= idx < len(images):
            img_url = images[idx]
        elif 1 <= idx <= len(images):
            img_url = images[idx - 1]
        if not img_url:
            continue

        labels = v.get("rectanglelabels") or []
        label = labels[0] if labels else None
        if label not in ALLOWED_CLASSES:
            continue

        bbox = {
            "x": v.get("x", 0),
            "y": v.get("y", 0),
            "width": v.get("width", 0),
            "height": v.get("height", 0),
            "rotation": v.get("rotation", 0),
            "original_width": res.get("original_width"),
            "original_height": res.get("original_height"),
            "class": label,               # single string
            "source": source,             # "annotation"
        }
        if parent_id is not None:
            bbox["annotation_id"] = parent_id
        if res.get("id") is not None:
            bbox["result_id"] = res["id"]

        out[img_url].append(bbox)
    return out


def _dedupe_bboxes_per_image(bxs: List[Dict]) -> List[Dict]:
    seen = set()
    keep = []
    for b in bxs:
        key = (
            b.get("result_id")
            or (round(b.get("x", 0), 4), round(b.get("y", 0), 4),
                round(b.get("width", 0), 4), round(b.get("height", 0), 4),
                b.get("class"))
        )
        if key in seen:
            continue
        seen.add(key)
        keep.append(b)
    return keep


def _summarize_by_class(all_bboxes: Dict[str, List[Dict]]):
    task_hist = Counter()
    per_image_counts = {}
    for img, boxes in all_bboxes.items():
        c = Counter()
        for b in boxes:
            if b.get("class"):
                c[b["class"]] += 1
                task_hist[b["class"]] += 1
        per_image_counts[img] = dict(c)
    return dict(task_hist), per_image_counts


# =========================
# Processing pipeline
# =========================
def process_tasks(tasks: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    processed = []
    for t in tasks:
        data = t.get("data") or {}
        images = data.get("images", []) or t.get("images", [])
        jsons  = data.get("jsons",  []) or t.get("jsons",  [])
        r_id = data.get("r_id", "")
        created_at = data.get("created_at") or t.get("created_at", "")

        group_timestamp = data.get("group_timestamp") or t.get("group_timestamp")
        uuid = data.get("uuid") or t.get("uuid", "")

        all_bboxes: Dict[str, List[Dict]] = {img: [] for img in images}
        trash_urls: List[str] = []
        unbound_tr: List[dict] = []

        # --- Annotations: Trash + BBOXES
        for a in t.get("annotations") or []:
            if isinstance(a, dict):
                res = a.get("result") or []
                m, u = _collect_by_choice(res, images, TRASH);  trash_urls += m; unbound_tr += u
                ab = _extract_bboxes(res, images, source="annotation", parent_id=a.get("id"))
                for k, v in ab.items():
                    all_bboxes[k].extend(v)

        # --- Drafts: Trash only (no bboxes from drafts)
        for d in t.get("drafts") or []:
            if isinstance(d, dict):
                res = d.get("result") or []
                m, u = _collect_by_choice(res, images, TRASH);  trash_urls += m; unbound_tr += u

        # --- All-Trash overrides everything
        if _has_all_trash(t):
            trash_urls = images[:]

        # --- Dedupe per image & summarize
        for img in list(all_bboxes.keys()):
            all_bboxes[img] = _dedupe_bboxes_per_image(all_bboxes[img])

        task_hist, per_image_counts = _summarize_by_class(all_bboxes)

        images_with_data = []
        for idx, img in enumerate(images):
            images_with_data.append({
                "index": idx,
                "url": img,
                "json_url": jsons[idx] if idx < len(jsons) else None,
                "is_trash": img in set(trash_urls),
                "bboxes": all_bboxes.get(img, []),           # cleaned annotation boxes
                "bboxes_by_class": per_image_counts.get(img, {}),
            })

        total_bboxes = sum(len(all_bboxes[i]) for i in images)

        processed.append({
            "task_id": t.get("id"),
            "r_id": r_id,
            "uuid": uuid,
            "group_timestamp": group_timestamp,
            "num_images": len(images),
            "created_at": created_at,
            "is_labeled": bool(t.get("is_labeled")),
            "total_annotations": int(t.get("total_annotations", 0)),
            "num_annotations": len(t.get("annotations") or []),
            "num_drafts": len(t.get("drafts") or []),
            "has_predictions": bool(t.get("predictions")),
            "images_data": images_with_data,
            "num_trash": len(set(trash_urls)),
            "num_bboxes_total": total_bboxes,
            "class_histogram": task_hist,                 # e.g. {"OutGroup0": 12, ...}
            "trash_unbound": unbound_tr,                  # diagnostics (index mapping failed)
            "trash_selections": list(dict.fromkeys(trash_urls)),
        })
    return processed


# =========================
# Export/snapshot + main
# =========================
def fetch_tasks_via_snapshot(ls: LabelStudio, project_id: int, view_id: Optional[int]) -> List[Dict[str, Any]]:
    """
    Uses SDK v2 snapshot/export to download tasks as JSON (stable, no pagination headaches).
    If view_id is provided, LS will apply that filter server-side; otherwise returns all tasks,
    which we further filter client-side.
    """
    print(f"Creating export snapshot (project={project_id}, view_id={view_id}) …")
    export = ls.projects.exports.create(
        id=project_id,
        title=f"auto_export_{int(time.time())}",
        task_filter_options=({"view": int(view_id)} if view_id else {}),
    )
    export_pk = export.id
    # poll for completion
    while True:
        info = ls.projects.exports.get(id=project_id, export_pk=export_pk)
        status = getattr(info, "status", None)
        if status in ("completed", "failed"):
            print(f"Export status: {status}")
            if status == "failed":
                details = getattr(info, "error", None) or getattr(info, "details", None)
                raise RuntimeError(f"Export failed: {details or info}")
            break
        time.sleep(1.8)

    print("Downloading export JSON …")
    downloaded = ls.projects.exports.download(id=project_id, export_pk=export_pk, export_type="JSON")

    # normalize to Python object
    if isinstance(downloaded, (bytes, bytearray)):
        payload = json.loads(downloaded.decode("utf-8"))
    elif isinstance(downloaded, str) and os.path.exists(downloaded):
        with open(downloaded, "r", encoding="utf-8") as f:
            payload = json.load(f)
    elif hasattr(downloaded, "read"):
        payload = json.load(downloaded)
    elif hasattr(downloaded, "__iter__") and not isinstance(downloaded, (str, bytes, bytearray)):
        # iterable of chunks
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

    # Label Studio exports a list of tasks (dicts)
    if not isinstance(payload, list):
        raise ValueError("Export payload is not a list of tasks")

    # IMPORTANT: keep original URIs (gs://) — v2 exports preserve data as-is.
    return payload


def main():
    if not LS_API_KEY:
        raise SystemExit("Please set LS_API_KEY")

    print("=" * 60)
    print("Export: ONLY review-accepted tasks NOT processed before (via snapshot)")
    print("=" * 60)

    # 1) Load or init the processed ledger from GCS
    ledger = gcs_download_json(GCS_BUCKET, PROCESSED_BLOB_PATH) or {}
    processed_ids = set(ledger.get("processed_tasksID", []))
    print(f"Ledger has {len(processed_ids)} processed task IDs.")

    # 2) LS client
    ls = LabelStudio(base_url=LS_URL, api_key=LS_API_KEY, timeout=30.0)
    # sanity ping
    _ = ls.projects.list(page_size=1)

    # 3) Snapshot export → tasks JSON
    all_tasks = fetch_tasks_via_snapshot(ls, PROJECT_ID, VIEW_ID)

    # 4) Filter to review-accepted (client-side guard) and drop already processed
    guarded = [t for t in all_tasks if _task_passes_review(t)]
    new_tasks = [t for t in guarded if int(t.get("id")) not in processed_ids]
    print(f"✓ From {len(all_tasks)} tasks, {len(guarded)} are review-accepted; {len(new_tasks)} are NEW.")

    if not new_tasks:
        new_ledger = {
            "processed_tasksID": sorted(processed_ids),
            "last_updated": utc_now_z(),
            "total_count": len(processed_ids),
        }
        gcs_upload_json(GCS_BUCKET, PROCESSED_BLOB_PATH, new_ledger)
        print("No new tasks. Ledger timestamp updated. Exiting.")
        return

    # 5) Process only NEW tasks with your original output logic
    print("🔄 Processing new tasks …")
    processed = process_tasks(new_tasks)

    # 6) Upload the export JSON (timestamped), unless disabled for empty
    if processed or UPLOAD_EMPTY_EXPORT:
        filename = jerusalem_stamp_for_filename(len(processed))
        gcs_upload_json(GCS_BUCKET, f"{GCS_EXPORT_PREFIX}/{filename}", processed)
    else:
        print("No processed items to export; skipping export upload.")

    # 7) Update the ledger
    for t in new_tasks:
        processed_ids.add(int(t.get("id")))
    new_ledger = {
        "processed_tasksID": sorted(processed_ids),
        "last_updated": utc_now_z(),
        "total_count": len(processed_ids),
    }
    gcs_upload_json(GCS_BUCKET, PROCESSED_BLOB_PATH, new_ledger)

    # 8) Summary
    total_trash  = sum(x["num_trash"] for x in processed)
    total_bboxes = sum(x["num_bboxes_total"] for x in processed)
    print("\nSummary (NEW only):")
    print(f"  Tasks processed  : {len(processed)}")
    print(f"  Trash selections : {total_trash}")
    print(f"  Bounding boxes   : {total_bboxes}")
    print("✅ Done.")


if __name__ == "__main__":
    main()
