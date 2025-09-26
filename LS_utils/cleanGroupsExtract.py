#!/usr/bin/env python3
"""
Export ONLY review-accepted (>=1) tasks that have NOT been processed before.
- Maintains a ledger in GCS: processed_tasksID_Initial.json
- Skips duplicates on every run
- Extracts per-image Trash flags + bboxes for <Image valueList>
- Summarizes bbox classes per image and per task (class_histogram)
"""

import json
import re
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional, Tuple
from collections import Counter

# --- SDK v2 ---
from label_studio_sdk.client import LabelStudio
from label_studio_sdk.data_manager import Filters, Column, Operator, Type

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
LS_URL = "https://app.heartex.com"
LS_API_KEY = "e3dd5c79ff9086a6b8769a35905cb249448cf3e9"  # <-- put your token
PROJECT_ID = 186048

# Labeling interface control names / values
IMAGE_TO_NAME = "images"             # <Image name="images" valueList="$images" />
BBOX_FROM_NAME = "bbox_labels"       # <RectangleLabels name="bbox_labels" ... />
FLAGS_FROM_NAME = "flags"            # <Choices name="flags" ... />
TRASH = "Trash"
ALL_TRASH = "All-Trash"

# GCS destination (export)
GCS_CREDENTIALS_PATH = r"C:\Users\OrGil.AzureAD\OneDrive - AMPC\Desktop\Azimut.ai\recognition_labeling\resources\credentials.json"
GCS_BUCKET = "azimut_data"
GCS_EXPORT_PREFIX = "reidentification/silver/Initial_groups_phase_cleaned/lable_studio_exports"

# GCS ledger (processed tasks)
PROCESSED_BLOB_PATH = "reidentification/silver/Initial_groups_phase_cleaned/processed_tasksID_Initial.json"

# Pagination
PAGE_SIZE = 500

# Review acceptance guard (client-side)
ACCEPTED_LAST_ACTIONS = {"accepted", "fixed_and_accepted"}

# Skip uploading an export file if there are 0 new tasks
UPLOAD_EMPTY_EXPORT = False

ALLOWED_CLASSES = {"MainGroup"} | {f"OutGroup{i}" for i in range(10)}

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
# Build server-side filters (reviews_accepted ≥ 1)
# =========================
def build_review_accepted_filters_only_reviews_count() -> Dict[str, Any]:
    return Filters.create(
        Filters.AND,
        [
            Filters.item(
                Column.reviews_accepted,
                Operator.GREATER_OR_EQUAL,
                Type.Number,
                Filters.value(1),
            )
        ],
    )


# =========================
# Extraction helpers
# =========================
ID_NUM_RE = re.compile(r'(:images(?:\$|\[)|\[\s*)(\d+)(?:\]|$)')

def _to_dict(x):
    if hasattr(x, "model_dump"):
        return x.model_dump()
    if hasattr(x, "dict"):
        return x.dict()
    return x

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
                return int(m.group(1))
    return None

def _extract_bboxes(
    results: List[Dict],
    images: List[str],
    source: str = "annotation",          # we’ll only call with "annotation" for bboxes
    parent_id: Optional[Any] = None,
) -> Dict[str, List[Dict]]:
    out = {img: [] for img in images}
    if not isinstance(results, list):
        return out

    for res in results:
        if not isinstance(res, dict):
            continue
        if res.get("type") != "rectanglelabels":
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
        # Keep only our new classes
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
            "class": label,                 # <-- cleaner: single class string
            "source": source,               # "annotation"
        }
        if parent_id is not None:
            bbox["annotation_id"] = parent_id
        if res.get("id") is not None:
            bbox["result_id"] = res["id"]

        out[img_url].append(bbox)
    return out




def _collect_by_choice(results: List[Dict], images: List[str], label_value: str) -> Tuple[List[str], List[dict]]:
    mapped, unbound = [], []
    if not isinstance(results, list):
        return mapped, unbound
    for res in results:
        if not isinstance(res, dict):
            continue
        if res.get("type") != "choices":          continue
        if res.get("to_name") != IMAGE_TO_NAME:   continue
        if res.get("from_name") not in (FLAGS_FROM_NAME, FLAGS_FROM_NAME.lower()):
            continue

        v = res.get("value") or {}
        if not isinstance(v, dict):               continue
        choices = v.get("choices") or []
        if label_value not in choices:            continue

        if isinstance(v.get("image"), str) and v["image"].startswith("gs://"):
            mapped.append(v["image"])
            continue

        idx = _index_from_any(res)
        if isinstance(idx, int):
            if 0 <= idx < len(images):      mapped.append(images[idx]);      continue
            if 1 <= idx <= len(images):     mapped.append(images[idx - 1]);  continue

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
# Main processing
# =========================
def _summarize_bboxes(all_bboxes: Dict[str, List[Dict]]):
    """Returns (task_hist, per_image_counts) where each maps class->count"""
    per_image_counts: Dict[str, Dict[str, int]] = {}
    task_hist: Counter = Counter()
    for img, boxes in all_bboxes.items():
        c = Counter()
        for b in boxes:
            labs = b.get("rectanglelabels") or []
            if labs:
                c[labs[0]] += 1
                task_hist[labs[0]] += 1
        per_image_counts[img] = dict(c)
    return dict(task_hist), per_image_counts

def process_tasks(tasks: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    processed = []
    for t in tasks:
        data = t.get("data") or {}
        images = data.get("images", []) or t.get("images", [])
        jsons  = data.get("jsons",  []) or t.get("jsons",  [])
        r_id = data.get("r_id", "")
        created_at = data.get("created_at") or t.get("created_at", "")
        
        # Extract timestamp information - ADD THESE LINES
        group_timestamp = data.get("group_timestamp") or t.get("group_timestamp")
        uuid = data.get("uuid") or t.get("uuid", "")

        all_bboxes: Dict[str, List[Dict]] = {img: [] for img in images}
        trash_urls: List[str] = []
        unbound_tr: List[dict] = []

        # --- Annotations: collect Trash + BBOXES
        for a in t.get("annotations") or []:
            if isinstance(a, dict):
                res = a.get("result") or []
                # Trash selections from annotations
                m, u = _collect_by_choice(res, images, TRASH);  trash_urls += m; unbound_tr += u
                # Bboxes ONLY from annotations
                ab = _extract_bboxes(res, images, source="annotation", parent_id=a.get("id"))
                for k, v in ab.items():
                    all_bboxes[k].extend(v)

        # --- Drafts: collect Trash (NO bboxes from drafts)
        for d in t.get("drafts") or []:
            if isinstance(d, dict):
                res = d.get("result") or []
                m, u = _collect_by_choice(res, images, TRASH);  trash_urls += m; unbound_tr += u
                # intentionally not extracting bboxes from drafts

        # --- All-Trash overrides everything
        if _has_all_trash(t):
            trash_urls = images[:]

        # --- DEDUPE per image & summarize
        for img in list(all_bboxes.keys()):
            all_bboxes[img] = _dedupe_bboxes_per_image(all_bboxes[img])

        task_hist, per_image_counts = _summarize_by_class(all_bboxes)

        images_with_data = []
        for idx, img in enumerate(images):
            images_with_data.append({
                "index": idx,
                "url": img,
                "json_url": jsons[idx] if idx < len(jsons) else None,
                "is_trash": img in trash_urls,
                "bboxes": all_bboxes.get(img, []),           # only cleaned annotation boxes
                "bboxes_by_class": per_image_counts.get(img, {}),
            })

        total_bboxes = sum(len(all_bboxes[i]) for i in images)

        processed.append({
            "task_id": t.get("id"),
            "r_id": r_id,
            "uuid": uuid,  # ADD THIS LINE
            "group_timestamp": group_timestamp,  # ADD THIS LINE
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
            "trash_unbound": unbound_tr,                  # diagnostics if index mapping failed
            "trash_selections": list(dict.fromkeys(trash_urls)),
        })
    return processed


def _dedupe_bboxes_per_image(bxs: List[Dict]) -> List[Dict]:
    seen = set()
    keep = []
    for b in bxs:
        key = (
            b.get("result_id")
            or (round(b.get("x",0),4), round(b.get("y",0),4),
                round(b.get("width",0),4), round(b.get("height",0),4),
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


def main():
    if not LS_API_KEY:
        raise SystemExit("Please set LS_API_KEY")

    print("=" * 60)
    print("Export: ONLY review-accepted tasks NOT processed before")
    print("=" * 60)

    # 1) Load or init the processed ledger from GCS
    ledger = gcs_download_json(GCS_BUCKET, PROCESSED_BLOB_PATH) or {}
    processed_ids = set(ledger.get("processed_tasksID", []))
    print(f"Ledger has {len(processed_ids)} processed task IDs.")

    # 2) LS client + server-side filter (reviews_accepted >= 1)
    ls = LabelStudio(base_url=LS_URL, api_key=LS_API_KEY)
    filters = build_review_accepted_filters_only_reviews_count()
    query_str = json.dumps({"filters": filters})

    # Keep original gs:// URIs: resolve_uri=False
    listed = [
        _to_dict(t)
        for t in ls.tasks.list(
            project=PROJECT_ID,
            query=query_str,
            fields="all",
            page_size=PAGE_SIZE,
            resolve_uri=False
        )
    ]

    # 3) Client-side guard and DROP already-processed IDs
    guarded = [t for t in listed if _task_passes_review(t)]
    new_tasks = [t for t in guarded if int(t.get("id")) not in processed_ids]
    print(f"✓ From {len(listed)} accepted tasks, {len(new_tasks)} are NEW (not in ledger).")

    if not new_tasks:
        new_ledger = {
            "processed_tasksID": sorted(processed_ids),
            "last_updated": utc_now_z(),
            "total_count": len(processed_ids),
        }
        gcs_upload_json(GCS_BUCKET, PROCESSED_BLOB_PATH, new_ledger)
        print("No new tasks. Ledger timestamp updated. Exiting.")
        return

    # 4) Process only NEW tasks
    print("🔄 Processing new tasks …")
    processed = process_tasks(new_tasks)

    # 5) Upload the export JSON (timestamped), unless disabled for empty
    if processed or UPLOAD_EMPTY_EXPORT:
        filename = jerusalem_stamp_for_filename(len(processed))
        gcs_upload_json(GCS_BUCKET, f"{GCS_EXPORT_PREFIX}/{filename}", processed)
    else:
        print("No processed items to export; skipping export upload.")

    # 6) Update the ledger
    for t in new_tasks:
        processed_ids.add(int(t.get("id")))
    new_ledger = {
        "processed_tasksID": sorted(processed_ids),
        "last_updated": utc_now_z(),
        "total_count": len(processed_ids),
    }
    gcs_upload_json(GCS_BUCKET, PROCESSED_BLOB_PATH, new_ledger)

    # 7) Summary
    total_trash  = sum(x["num_trash"] for x in processed)
    total_bboxes = sum(x["num_bboxes_total"] for x in processed)
    print("\nSummary (NEW only):")
    print(f"  Tasks processed  : {len(processed)}")
    print(f"  Trash selections : {total_trash}")
    print(f"  Bounding boxes   : {total_bboxes}")
    print("✅ Done.")


if __name__ == "__main__":
    main()
