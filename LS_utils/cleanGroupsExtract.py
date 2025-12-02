#!/usr/bin/env python3
"""
Export ONLY review-accepted (>=1) tasks that have NOT been processed before,
using Label Studio SDK v2 snapshot/export for reliability, and preserving the
output structure/logic from the original script.

NOW USING ADC (no JSON keys). Make sure each user runs:
    gcloud auth application-default login

NEW: Automatically triggers ship_secondary_processing.py after successful export.

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
import sys
import subprocess
import argparse
from typing import Any, Dict, List, Optional, Tuple
from datetime import datetime, timezone
from collections import Counter, defaultdict

# --- Label Studio SDK v2 ---
from label_studio_sdk.client import LabelStudio  # SDK 2.x

# --- GCS (ADC) ---
from google.cloud import storage
import google.auth

try:
    from zoneinfo import ZoneInfo
except Exception:
    ZoneInfo = None


# =========================
# CONFIG
# =========================
# Label Studio
LS_URL = "https://app.heartex.com"
LS_API_KEY = "34aefb3582706dd6ac6306d1b63095be6857de52"  # <-- replace
PROJECT_ID = 186048

# If you have a saved View in LS that already filters to reviewed/accepted,
# set VIEW_ID; otherwise leave as None and we'll filter client-side.
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
GCS_EXPORT_PREFIX = "recognition/silver/Initial_groups_phase_cleaned/label_studio_exports"

# Global ledger (fixed bucket/path)
LEDGER_BUCKET = "azimut_data"
GLOBAL_PROCESSED_BLOB_PATH = "reidentification/silver/Initial_groups_phase_cleaned/processed_tasksID_Initial.json"

# Upload an export file even if there are 0 new tasks
UPLOAD_EMPTY_EXPORT = False

# Review acceptance guard (client-side, resilient to older payloads)
ACCEPTED_LAST_ACTIONS = {"accepted", "fixed_and_accepted"}

# Secondary Processing Configuration
SECONDARY_SCRIPT_PATH = "postProcess/ship_secondary_processing.py"  # Stage 2 script
RECOGNITION_SERVICE_URL = "http://localhost:8080"                   # Ship-Recognition-Service URL
AUTO_TRIGGER_SECONDARY = True                                       # Set False to disable auto-trigger


# =========================
# Utilities (time, ADC GCS)
# =========================
def _adc_storage_client(project: Optional[str] = None) -> storage.Client:
    # Resolve project: explicit -> env -> ADC default
    resolved = project or os.getenv("GOOGLE_CLOUD_PROJECT") or os.getenv("GCLOUD_PROJECT")
    if not resolved:
        _, detected = google.auth.default(scopes=["https://www.googleapis.com/auth/cloud-platform"])
        resolved = detected
    return storage.Client(project=resolved)

def jerusalem_stamp_for_filename(n_tasks: int) -> str:
    tz = ZoneInfo("Asia/Jerusalem") if ZoneInfo else None
    now = datetime.now(tz) if tz else datetime.now()
    return f"{now.strftime('%Y-%m-%d_%H-%M')}_{n_tasks}Tasks.json"

def utc_now_z() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="microseconds").replace("+00:00", "Z")

def gcs_download_json(bucket_name: str, blob_path: str) -> Optional[Dict[str, Any]]:
    client = _adc_storage_client()
    blob = client.bucket(bucket_name).blob(blob_path)
    if not blob.exists():
        return None
    txt = blob.download_as_text()
    try:
        return json.loads(txt)
    except Exception:
        return None

def gcs_upload_json(bucket_name: str, blob_path: str, obj: Any):
    client = _adc_storage_client()
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
                    return int(m.group(2))
                except Exception:
                    pass
    return None

def _collect_by_choice(results: List[Dict], images: List[str], label_value: str) -> Tuple[List[str], List[dict]]:
    mapped, unbound = [], []
    if not isinstance(results, list):
        return mapped, unbound
    for res in results:
        if not isinstance(res, dict): continue
        if str(res.get("type")).lower() != "choices": continue
        if res.get("to_name") != IMAGE_TO_NAME: continue
        if str(res.get("from_name")).lower() != str(FLAGS_FROM_NAME).lower(): continue
        v = res.get("value") or {}
        if not isinstance(v, dict): continue
        choices = v.get("choices") or []
        if label_value not in choices: continue
        if isinstance(v.get("image"), str) and v["image"].startswith("gs://"):
            mapped.append(v["image"]); continue
        idx = _index_from_any(res)
        if isinstance(idx, int):
            if 0 <= idx < len(images): mapped.append(images[idx]); continue
            if 1 <= idx <= len(images): mapped.append(images[idx - 1]); continue
        unbound.append({"which": label_value, "result": res})
    return mapped, unbound

def _has_all_trash(task: Dict) -> bool:
    images = (task.get("data") or {}).get("images", []) or task.get("images", [])
    for a in task.get("annotations") or []:
        if not isinstance(a, dict):  continue
        res = a.get("result") or []
        m, u = _collect_by_choice(res, images, ALL_TRASH)
        if m or u: return True
    for d in task.get("drafts") or []:
        if not isinstance(d, dict):  continue
        res = d.get("result") or []
        m, u = _collect_by_choice(res, images, ALL_TRASH)
        if m or u: return True
    return False

def _extract_bboxes(results: List[Dict], images: List[str], source: str = "annotation", parent_id: Optional[Any] = None) -> Dict[str, List[Dict]]:
    out = {img: [] for img in images}
    if not isinstance(results, list):
        return out
    for res in results:
        if not isinstance(res, dict): continue
        if str(res.get("type")).lower() != "rectanglelabels": continue
        if res.get("from_name") != BBOX_FROM_NAME: continue
        if res.get("to_name") != IMAGE_TO_NAME: continue
        v = res.get("value") or {}
        if not isinstance(v, dict): continue
        idx = _index_from_any(res)
        if idx is None: continue
        img_url = None
        if 0 <= idx < len(images): img_url = images[idx]
        elif 1 <= idx <= len(images): img_url = images[idx - 1]
        if not img_url: continue
        labels = v.get("rectanglelabels") or []
        label = labels[0] if labels else None
        if label not in ALLOWED_CLASSES:
            continue
        bbox = {
            "x": v.get("x", 0), "y": v.get("y", 0), "width": v.get("width", 0), "height": v.get("height", 0),
            "rotation": v.get("rotation", 0),
            "original_width": res.get("original_width"), "original_height": res.get("original_height"),
            "class": label, "source": source,
        }
        if parent_id is not None: bbox["annotation_id"] = parent_id
        if res.get("id") is not None: bbox["result_id"] = res["id"]
        out[img_url].append(bbox)
    return out

def _dedupe_bboxes_per_image(bxs: List[Dict]) -> List[Dict]:
    seen, keep = set(), []
    for b in bxs:
        key = b.get("result_id") or (round(b.get("x", 0), 4), round(b.get("y", 0), 4),
                                     round(b.get("width", 0), 4), round(b.get("height", 0), 4),
                                     b.get("class"))
        if key in seen: continue
        seen.add(key); keep.append(b)
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

        for a in t.get("annotations") or []:
            if isinstance(a, dict):
                res = a.get("result") or []
                m, u = _collect_by_choice(res, images, TRASH);  trash_urls += m; unbound_tr += u
                ab = _extract_bboxes(res, images, source="annotation", parent_id=a.get("id"))
                for k, v in ab.items(): all_bboxes[k].extend(v)

        for d in t.get("drafts") or []:
            if isinstance(d, dict):
                res = d.get("result") or []
                m, u = _collect_by_choice(res, images, TRASH);  trash_urls += m; unbound_tr += u

        if _has_all_trash(t):
            trash_urls = images[:]

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
                "bboxes": all_bboxes.get(img, []),
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
            "class_histogram": task_hist,
            "trash_unbound": unbound_tr,
            "trash_selections": list(dict.fromkeys(trash_urls)),
        })
    return processed


# =========================
# Export/snapshot + main
# =========================
def fetch_tasks_via_snapshot(ls: LabelStudio, project_id: int, view_id: Optional[int]) -> List[Dict[str, Any]]:
    print(f"Creating export snapshot (project={project_id}, view_id={view_id}) ...")
    export = ls.projects.exports.create(
        id=project_id,
        title=f"auto_export_{int(time.time())}",
        task_filter_options=({"view": int(view_id)} if view_id else {}),
    )
    export_pk = export.id
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

    print("Downloading export JSON ...")
    downloaded = ls.projects.exports.download(id=project_id, export_pk=export_pk, export_type="JSON")

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
            if not chunk: continue
            if isinstance(chunk, str): chunk = chunk.encode("utf-8")
            buf.extend(chunk)
        payload = json.loads(buf.decode("utf-8"))
    else:
        raise TypeError(f"Unexpected download() return type: {type(downloaded)}")

    if not isinstance(payload, list):
        raise ValueError("Export payload is not a list of tasks")
    return payload


def _extract_bucket(url: Any) -> Optional[str]:
    if not isinstance(url, str):
        return None
    if not url.startswith("gs://"):
        return None
    rest = url[5:]
    if not rest:
        return None
    return rest.split("/", 1)[0]


def _task_bucket_candidates(task: dict) -> List[str]:
    buckets: List[str] = []
    data = task.get("data") or {}
    for seq in (
        data.get("images") or [],
        data.get("jsons") or [],
        task.get("images") or [],
        task.get("jsons") or [],
    ):
        for url in seq or []:
            b = _extract_bucket(url)
            if b:
                buckets.append(b)
    return buckets


def trigger_secondary_processing(labeled_json_path: str, bucket: str, service_url: str, script_path: str) -> bool:
    """
    Trigger Stage 2 with the newly created export.

    Args:
        labeled_json_path: Path to the exported JSON file (leading slash ok)
        bucket: target bucket for the downstream stage
    """
    if not labeled_json_path:
        print("WARNING: No export file to process")
        return False

    print("\n" + "=" * 60)
    print("TRIGGERING SECONDARY PROCESSING")
    print("=" * 60)
    print(f"Input file: {labeled_json_path}")
    print(f"Bucket: {bucket}")
    print(f"Service URL: {service_url}\n")

    try:
        cmd = [
            sys.executable,
            script_path,
            "--bucket", bucket,
            "--labeled-json", labeled_json_path,
            "--service-url", service_url,
        ]
        print(f"Running command: {' '.join(cmd)}\n")

        subprocess.run(cmd, check=True, capture_output=False, text=True)
        print("\nSecondary processing completed successfully")
        return True

    except subprocess.CalledProcessError as e:
        print(f"\nERROR: Secondary processing failed with exit code {e.returncode}")
        return False
    except FileNotFoundError:
        print(f"\nERROR: Could not find secondary processing script: {script_path}")
        print("Please ensure 'ship_secondary_processing.py' is in the expected location")
        return False
    except Exception as e:
        print(f"\nERROR: Unexpected error triggering secondary processing: {e}")
        return False


def _startup_bucket_check(bucket_name: str) -> None:
    """Fail fast if bucket is not reachable with ADC/IAM."""
    client = _adc_storage_client()
    bucket = client.bucket(bucket_name)
    if not bucket.exists():
        raise RuntimeError(
            f"Bucket gs://{bucket_name} does not exist or is not visible. "
            "If 403 -> grant bucket-level IAM (Storage Object Viewer/Creator/Admin)."
        )


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="LS → GCS (Initial groups, multi-bucket aware)")
    p.add_argument("--secondary-script", default=SECONDARY_SCRIPT_PATH, help="Path to Stage 2 script")
    p.add_argument("--service-url", default=RECOGNITION_SERVICE_URL, help="Recognition service URL")
    p.add_argument("--auto-trigger-secondary", dest="auto_trigger_secondary", action="store_true", help="Trigger Stage 2 after export")
    p.add_argument("--no-auto-trigger-secondary", dest="auto_trigger_secondary", action="store_false", help="Skip Stage 2 trigger")
    p.set_defaults(auto_trigger_secondary=AUTO_TRIGGER_SECONDARY)
    return p.parse_args()


def main():
    if not LS_API_KEY:
        raise SystemExit("Please set LS_API_KEY")

    args = parse_args()

    print("=" * 60)
    print("Export: ONLY review-accepted tasks NOT processed before (via snapshot)")
    print("=" * 60)

    # 1) LS client
    ls = LabelStudio(base_url=LS_URL, api_key=LS_API_KEY, timeout=30.0)
    _ = ls.projects.list(page_size=1)

    # 2) Snapshot export → tasks JSON
    all_tasks = fetch_tasks_via_snapshot(ls, PROJECT_ID, VIEW_ID)

    # 3) Filter to review-accepted
    guarded = [t for t in all_tasks if _task_passes_review(t)]
    print(f"✓ From {len(all_tasks)} tasks, {len(guarded)} are review-accepted.")

    # 4) Partition tasks by bucket
    bucket_groups: Dict[str, List[dict]] = defaultdict(list)
    multi_bucket_tasks: List[dict] = []
    missing_bucket_tasks: List[Any] = []

    for t in guarded:
        buckets = _task_bucket_candidates(t)
        if not buckets:
            missing_bucket_tasks.append(t.get("id"))
            continue
        uniq = sorted(set(buckets))
        if len(uniq) > 1:
            multi_bucket_tasks.append({"task_id": t.get("id"), "buckets": uniq})
        target_bucket = uniq[0]
        bucket_groups[target_bucket].append(t)

    if missing_bucket_tasks:
        print(f"✖ {len(missing_bucket_tasks)} tasks had no gs:// URLs; please ensure InitialGroupsPhase writes bucket-qualified paths.")
        print(f"Tasks without bucket: {missing_bucket_tasks[:10]}")
        if len(missing_bucket_tasks) > 10:
            print(f"... and {len(missing_bucket_tasks) - 10} more")
        return {}

    all_buckets = sorted(bucket_groups.keys())
    print(f"Detected buckets in payload: {all_buckets}")
    if multi_bucket_tasks:
        print("⚠ Tasks with multiple buckets detected (using the first bucket listed for each):")
        for entry in multi_bucket_tasks[:10]:
            print(f"   task_id={entry['task_id']} buckets={entry['buckets']}")
        if len(multi_bucket_tasks) > 10:
            print(f"   ... and {len(multi_bucket_tasks) - 10} more")

    # 5) Process per bucket
    exported_files: Dict[str, Optional[str]] = {}
    totals = {"tasks": 0, "trash": 0, "bboxes": 0}

    # Global ledger bucket selection
    ledger_bucket = LEDGER_BUCKET
    print(f"Using global ledger: gs://{ledger_bucket}/{GLOBAL_PROCESSED_BLOB_PATH}")

    # Load global processed IDs once
    ledger = gcs_download_json(ledger_bucket, GLOBAL_PROCESSED_BLOB_PATH) or {}
    processed_ids = set(ledger.get("processed_tasksID", []))
    print(f"Global ledger has {len(processed_ids)} processed task IDs.")

    for bucket_name, tasks_for_bucket in bucket_groups.items():
        print("\n" + "-" * 60)
        print(f"Bucket: {bucket_name} | tasks in bucket: {len(tasks_for_bucket)}")
        try:
            _startup_bucket_check(bucket_name)
            print(f"GCS reachable and bucket '{bucket_name}' is accessible.")
        except Exception as e:
            print(f"Startup GCS check failed for {bucket_name}: {e}")
            continue

        new_tasks = [t for t in tasks_for_bucket if int(t.get("id")) not in processed_ids]
        print(f"→ New tasks for bucket {bucket_name}: {len(new_tasks)} / {len(tasks_for_bucket)}")

        if not new_tasks:
            exported_files[bucket_name] = None
            continue

        processed = process_tasks(new_tasks)

        exported_file_path = None
        if processed or UPLOAD_EMPTY_EXPORT:
            filename = jerusalem_stamp_for_filename(len(processed))
            full_path = f"{GCS_EXPORT_PREFIX}/{filename}"
            gcs_upload_json(bucket_name, full_path, processed)
            exported_file_path = f"/{full_path}"
            print(f"Exported to: gs://{bucket_name}{exported_file_path}")
        else:
            print("No processed items to export; skipping export upload.")

        for t in new_tasks:
            processed_ids.add(int(t.get("id")))
        bucket_trash = sum(x["num_trash"] for x in processed)
        bucket_bboxes = sum(x["num_bboxes_total"] for x in processed)
        totals["tasks"] += len(processed)
        totals["trash"] += bucket_trash
        totals["bboxes"] += bucket_bboxes

        print("Summary (NEW only):")
        print(f"  Tasks processed  : {len(processed)}")
        print(f"  Trash selections : {bucket_trash}")
        print(f"  Bounding boxes   : {bucket_bboxes}")

        exported_files[bucket_name] = exported_file_path

        if exported_file_path and args.auto_trigger_secondary:
            print("\n" + "=" * 60)
            print("Stage 1 Complete - Starting Stage 2")
            print("=" * 60)
            ok = trigger_secondary_processing(
                exported_file_path,
                bucket=bucket_name,
                service_url=args.service_url,
                script_path=args.secondary_script,
            )
            if not ok:
                print(f"⚠ Stage 2 trigger failed for bucket {bucket_name}; tasks will be retried next run.")

    print("\n" + "=" * 60)
    print("Finished Stage 1 across buckets")
    print("=" * 60)
    print(f"Buckets handled: {list(exported_files.keys())}")
    print(f"Totals → tasks={totals['tasks']} trash={totals['trash']} bboxes={totals['bboxes']}")
    # Persist global ledger once
    new_ledger = {
        "processed_tasksID": sorted(processed_ids),
        "last_updated": utc_now_z(),
        "total_count": len(processed_ids),
    }
    gcs_upload_json(ledger_bucket, GLOBAL_PROCESSED_BLOB_PATH, new_ledger)
    print(f"✓ Global ledger updated: gs://{ledger_bucket}/{GLOBAL_PROCESSED_BLOB_PATH}")

    return exported_files


if __name__ == "__main__":
    _ = main()
    sys.exit(0)
