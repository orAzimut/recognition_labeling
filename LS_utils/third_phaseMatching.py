#!/usr/bin/env python3
import requests
import json
from typing import Dict, Any, List, Optional
from datetime import datetime, timezone

# Timezone handling (Asia/Jerusalem if tzdata/zoneinfo available)
try:
    from zoneinfo import ZoneInfo  # Python 3.9+
    TZ = ZoneInfo("Asia/Jerusalem")
except Exception:
    TZ = None

# --------- GCS ----------
from google.cloud import storage
from google.oauth2 import service_account
from google.api_core.exceptions import NotFound

# =======================================
# CONFIG — fill your token
# =======================================
BASE_URL   = "https://app.heartex.com"
PROJECT_ID = 187693
API_TOKEN  = "e3dd5c79ff9086a6b8769a35905cb249448cf3e9"  # <- keep secure

# GCS destination for export files
GCS_CREDENTIALS_PATH = r"C:\Users\OrGil.AzureAD\OneDrive - AMPC\Desktop\Azimut.ai\recognition_labeling\resources\credentials.json"
GCS_BUCKET_NAME      = "azimut_data"
GCS_OUTPUT_PREFIX    = "reidentification/silver/Third_Phase_Groups_Association/lable_studio_exports"

# Processed tasks registry (read/write)
PROCESSED_TASKS_BLOB = "reidentification/silver/Third_Phase_Groups_Association/processed_tasksID_Phase3.json"

# API behavior
PAGE_SIZE  = 200
TIMEOUT    = 60

# If True, fix paths like "gs:/bucket/..." -> "gs://bucket/..."
NORMALIZE_GS_SCHEME = False

session = requests.Session()
session.headers.update({
    "Authorization": f"Token {API_TOKEN}",
    "Content-Type": "application/json"
})

# ---------------- API helpers ----------------

def list_tasks(project_id: int) -> List[Dict[str, Any]]:
    """
    Paginate ALL tasks for the project (robust to server caps and 404/400-after-last-page).
    Do NOT stop on len(batch) < PAGE_SIZE because the server may cap page_size (e.g., to 100).
    """
    tasks: List[Dict[str, Any]] = []
    page = 1
    while True:
        params = {
            "project": project_id,
            "page": page,
            "page_size": PAGE_SIZE,   # server may cap to 100
            "fields": "all",
            "resolve_uri": "false",
        }
        r = session.get(f"{BASE_URL}/api/tasks/", params=params, timeout=TIMEOUT)
        if r.status_code in (404, 400):
            print(f"[list_tasks] page={page} -> {r.status_code}; assuming end.")
            break
        r.raise_for_status()
        obj = r.json()

        if isinstance(obj, dict):
            batch = obj.get("tasks") or obj.get("results") or obj.get("items") or []
        elif isinstance(obj, list):
            batch = obj
        else:
            batch = []

        print(f"[list_tasks] page={page} got {len(batch)} tasks")
        if not batch:
            break

        tasks.extend(batch)
        page += 1

    print(f"[list_tasks] TOTAL pulled: {len(tasks)}")
    return tasks

def list_annotations_for_task(task_id: int) -> List[Dict[str, Any]]:
    """Fetch full annotations for a task (includes 'result' and review fields)."""
    r = session.get(f"{BASE_URL}/api/tasks/{task_id}/annotations", timeout=TIMEOUT)
    if r.status_code == 404:
        # fallback on some LS variants
        r = session.get(f"{BASE_URL}/api/annotations", params={"task": task_id}, timeout=TIMEOUT)
    r.raise_for_status()
    data = r.json()
    if isinstance(data, dict) and "results" in data:
        return data["results"]
    return data if isinstance(data, list) else []

def is_review_accepted(ann: Dict[str, Any]) -> bool:
    """Decide if an annotation is review-accepted across common LS variants."""
    def low(x): return str(x).strip().lower()
    if ann.get("was_cancelled"):
        return False
    la = ann.get("last_action")
    if la and low(la) in {"accepted", "fixed_and_accepted"}:
        return True
    for key in ("status", "review_result"):
        val = ann.get(key)
        if val and low(val) in {"accepted", "accepted_auto", "approved"}:
            return True
    if ann.get("approved") is True or ann.get("was_accepted") is True:
        return True
    for rv in ann.get("reviews") or []:
        if rv.get("accepted") is True:
            return True
        if low(rv.get("result", "")) in {"accepted", "accepted_auto", "approved"}:
            return True
    return False

def extract_same_vessel(ann: Dict[str, Any]) -> Optional[str]:
    """Extract the same_vessel choice from annotation results."""
    for item in ann.get("result", []):
        if item.get("type") == "choices":
            # Check if this is the same_vessel field (by from_name or other identifier)
            # Different Label Studio configs might use different field names
            from_name = item.get("from_name", "").lower()
            to_name = item.get("to_name", "").lower()
            
            # Try to identify if this is the same_vessel choice
            # Adjust these conditions based on your Label Studio configuration
            if "same" in from_name or "vessel" in from_name or "same" in to_name or "vessel" in to_name:
                choices = item.get("value", {}).get("choices", [])
                if choices:
                    return choices[0]  # Should be "Yes" or "No"
            
            # Fallback: if there's only one choices field, assume it's same_vessel
            choices = item.get("value", {}).get("choices", [])
            if choices and choices[0] in ["Yes", "No"]:
                return choices[0]
    
    return None

def _norm_gs(paths: List[str]) -> List[str]:
    if not NORMALIZE_GS_SCHEME:
        return paths
    fixed = []
    for p in paths or []:
        if isinstance(p, str) and p.startswith("gs:/") and not p.startswith("gs://"):
            fixed.append("gs://" + p[4:])
        else:
            fixed.append(p)
    return fixed

# ---------------- BBox helpers ----------------

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

def _warn_if_misaligned(task_id: Any, who: str,
                        imgs: List[str], jsons: List[str], bboxes: List[List[float]]):
    li, lj, lb = len(imgs), len(jsons), len(bboxes)
    if not (li == lj == lb or (lb == 0 and li == lj)):
        print(f"[Warn][Task {task_id}] {who}: length mismatch "
              f"images={li}, jsons={lj}, bboxes={lb}")

# --------------- GCS helpers -----------------

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

# --------------- Main ---------------------

def main():
    # Load processed tasks registry and coerce IDs to int
    registry = load_processed_registry()
    raw_ids = registry.get("processed_tasksID", []) or []
    processed_ids: set[int] = set()
    for x in raw_ids:
        try:
            processed_ids.add(int(x))
        except Exception:
            pass
    print(f"[registry] loaded {len(processed_ids)} processed IDs")

    # Pull tasks from Label Studio
    tasks = list_tasks(PROJECT_ID)

    outputs: List[Dict[str, Any]] = []
    new_task_ids: List[int] = []

    # DEBUG buckets (optional diagnostics)
    already_processed_task_ids: List[int] = []
    no_annotations_task_ids: List[int] = []
    not_accepted_task_ids: List[int] = []

    for t in tasks:
        tid = t.get("id")
        if tid is None:
            continue

        # Skip already processed tasks
        if tid in processed_ids:
            already_processed_task_ids.append(tid)
            continue

        anns = list_annotations_for_task(tid)
        if not anns:
            no_annotations_task_ids.append(tid)
            continue

        accepted = [a for a in anns if is_review_accepted(a)]
        if not accepted:
            not_accepted_task_ids.append(tid)
            continue

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

        outputs.append({
            "task_id": tid,
            "annotation_id": ann.get("id"),
            "r_id_1": d.get("r_id_1"),
            "r_id_1_images": _norm_gs(d.get("r_id_1_images") or []),
            "r_id_1_jsons": _norm_gs(d.get("r_id_1_jsons") or []),
            "r_id_1_bboxes": _coerce_bboxes(d.get("r_id_1_bboxes") or []),
            "r_id_2": d.get("r_id_2"),
            "r_id_2_images": _norm_gs(d.get("r_id_2_images") or []),
            "r_id_2_jsons": _norm_gs(d.get("r_id_2_jsons") or []),
            "r_id_2_bboxes": _coerce_bboxes(d.get("r_id_2_bboxes") or []),
            "same_vessel": same_vessel,  # Added same_vessel field
        })
        new_task_ids.append(tid)

    # ---- DIAGNOSTICS ----
    print("============== DEBUG / DIAGNOSTICS ==============")
    print(f"Accepted (NEW this run): {len(set(new_task_ids))}")
    print(f"Already processed: {len(set(already_processed_task_ids))}")
    print(f"No annotations: {len(set(no_annotations_task_ids))}")
    print(f"Not accepted: {len(set(not_accepted_task_ids))}")
    print("==================================================")

    # Timestamped filename (Asia/Jerusalem if available)
    now_local = datetime.now(TZ) if TZ else datetime.now()
    ts = now_local.strftime("%Y-%m-%d_%H-%M")
    filename = f"LS_{PROJECT_ID}_ACCEPTED_{ts}_{len(outputs)}Tasks.json"

    # Save export only if there are new tasks
    if outputs:
        gs_path = save_json_to_gcs(outputs, filename)
        print(f"✅ Saved {len(outputs)} NEW accepted tasks to: {gs_path}")
    else:
        print("ℹ️ No new tasks to export (all already processed or none accepted).")

    # Always bump last_updated; add new IDs if any; then save registry
    # Normalize registry IDs -> int; ignore anything non-numeric just in case
    def _to_ints(seq):
        out = set()
        for x in seq or []:
            try:
                out.add(int(x))
            except Exception:
                pass
        return out

    existing_ids_int = _to_ints(raw_ids)
    new_ids_int      = _to_ints(new_task_ids)

    merged_ids_int = sorted(existing_ids_int | new_ids_int)   # all ints now

    # Write back as strings (stable format if your JSON had strings)
    registry["processed_tasksID"] = [str(x) for x in merged_ids_int]
    now_utc = datetime.now(timezone.utc).isoformat(timespec="microseconds").replace("+00:00", "Z")
    registry["last_updated"] = now_utc
    registry["total_count"]  = len(merged_ids_int)

    reg_path = save_processed_registry(registry)

    if new_ids_int:
        print(f"📝 Updated processed registry with {len(new_ids_int)} new IDs → {reg_path}")
        print(f"🧮 Registry total_count: {len(merged_ids_int)}")
    else:
        print(f"📝 Processed registry timestamp updated (no new IDs) → {reg_path}")
        print(f"🧮 Registry total_count: {len(merged_ids_int)}")


if __name__ == "__main__":
    main()