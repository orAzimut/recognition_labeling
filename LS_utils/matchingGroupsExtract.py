#!/usr/bin/env python3
"""
Groups Association Phase export (robust, full pagination + DNS/connect retry)
-----------------------------------------------------------------------------

- Pull ALL tasks from Label Studio:
    1) Try SDK v2 snapshot/export (with explicit 'select all')
    2) If snapshot fails, fall back to REST pagination (include_annotations=true)
       with support for BOTH LS response styles:
         A) {"results": [...], "next": "..."}  (cursor)
         B) [...] (list) + page=? pagination

- Adds robust retries:
    * DNS preflight with retries
    * requests.Session with urllib3 Retry for connect/read/status
    * manual per-request retry loop (catches NameResolutionError/ConnectionError)

- Keep ONLY review-accepted tasks that are NOT already in the processed registry.
- Extract `same_vessel` from the accepted annotation.
- Preserve your original output JSON schema & ledger logic.
- Uses ADC for GCS (gcloud auth application-default login).

Outputs:
  gs://{GCS_BUCKET_NAME}/{GCS_OUTPUT_PREFIX}/LS_{PROJECT_ID}_ACCEPTED_<ts>_<N>Tasks.json
Ledger:
  gs://{GCS_BUCKET_NAME}/{PROCESSED_TASKS_BLOB}
"""

import os
import sys
import json
import time
import socket
import subprocess
import argparse
from typing import Dict, Any, List, Optional
from datetime import datetime, timezone

# Timezone handling (Asia/Jerusalem if tzdata/zoneinfo available)
try:
    from zoneinfo import ZoneInfo  # Python 3.9+
    TZ = ZoneInfo("Asia/Jerusalem")
except Exception:
    TZ = None

# --------- LS SDK v2 ----------
from label_studio_sdk.client import LabelStudio  # SDK 2.x

# --------- HTTP (fallback REST) ----------
import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

# --------- GCS (ADC) ----------
from google.cloud import storage
from google.api_core.exceptions import NotFound
import google.auth

# =======================================
# CONFIG — fill your token / paths
# =======================================
BASE_URL   = "https://app.heartex.com"
PROJECT_ID = 185882
API_TOKEN  = "34aefb3582706dd6ac6306d1b63095be6857de52"

# No View: we want ALL tasks; leave as None
VIEW_ID: Optional[int] = None

# GCS destination for export files
GCS_BUCKET_NAME   = "azimut_data"
GCS_OUTPUT_PREFIX = "recognition/silver/Groups_Association_Phase_cleaned/lable_studio_exports"

# Processed tasks registry (global ledger)
LEDGER_BUCKET = "azimut_data"
GLOBAL_PROCESSED_BLOB_PATH = "reidentification/silver/Groups_Association_Phase_cleaned/processed_tasksID_match.json"

# If True, fix paths like "gs:/bucket/..." -> "gs://bucket/..."
NORMALIZE_GS_SCHEME = False

# Polling interval for snapshot completion
SNAPSHOT_POLL_SECS = 1.8

# Third Stage Processing Configuration
THIRD_STAGE_SCRIPT_PATH = r"C:\Users\OrGil.AzureAD\OneDrive - AMPC\Desktop\Azimut.ai\recognition_labeling\postProcess\ship_third_stage_processing.py"
RECOGNITION_SERVICE_URL = "http://localhost:8080"  # Ship-Recognition-Service URL
AUTO_TRIGGER_THIRD_STAGE = True  # Set to False to disable auto-triggering

# -------- retrier config --------
CONNECT_RETRIES = 6           # quick retries for DNS/connect
READ_RETRIES    = 3
STATUS_RETRIES  = 3
BACKOFF_FACTOR  = 1.0         # 1s, 2s, 4s, ...

PER_REQUEST_MAX_RETRIES = 5   # additional manual attempts per page, on ANY RequestException
PER_REQUEST_SLEEP_BASE  = 1.0 # seconds; grows exponentially


# ---------------- ADC helpers ----------------
def _adc_storage_client(project: Optional[str] = None) -> storage.Client:
    resolved = project or os.getenv("GOOGLE_CLOUD_PROJECT") or os.getenv("GCLOUD_PROJECT")
    if not resolved:
        _, detected = google.auth.default(scopes=["https://www.googleapis.com/auth/cloud-platform"])
        resolved = detected
    return storage.Client(project=resolved)

def _startup_bucket_check(bucket_name: str) -> None:
    """Fail fast if bucket is not reachable with ADC/IAM."""
    client = _adc_storage_client()
    bucket = client.bucket(bucket_name)
    if not bucket.exists():
        raise RuntimeError(
            f"Bucket gs://{bucket_name} does not exist or is not visible. "
            "If 403 -> grant bucket-level IAM (Storage Object Viewer/Creator/Admin)."
        )


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
        if choices and choices[0] in ("Yes", "No"):
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


def _output_buckets(output: Dict[str, Any]) -> List[str]:
    buckets: List[str] = []
    for key in ("r_id_1_images", "r_id_1_jsons", "r_id_2_images", "r_id_2_jsons"):
        for url in output.get(key) or []:
            b = _extract_bucket(url)
            if b:
                buckets.append(b)
    return sorted(set(buckets))


# ---------------- GCS helpers (ADC) ----------------
def save_json_to_gcs(data: Any, filename: str, bucket_name: str) -> str:
    """Upload JSON to GCS (export area) and return gs:// path."""
    client = _adc_storage_client()
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
    client = _adc_storage_client()
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
    client = _adc_storage_client()
    bucket = client.bucket(ledger_bucket)
    blob = bucket.blob(GLOBAL_PROCESSED_BLOB_PATH)
    blob.upload_from_string(
        json.dumps(registry, ensure_ascii=False, indent=2),
        content_type="application/json"
    )
    return f"gs://{ledger_bucket}/{GLOBAL_PROCESSED_BLOB_PATH}"


# ---------------- HTTP session & DNS preflight ----------------
def _make_retrying_session() -> requests.Session:
    """
    Create a requests.Session with robust retry/backoff on connect/read/status.
    """
    sess = requests.Session()
    retry = Retry(
        total=None,                    # we'll control counts via specific args
        connect=CONNECT_RETRIES,
        read=READ_RETRIES,
        status=STATUS_RETRIES,
        backoff_factor=BACKOFF_FACTOR,
        status_forcelist=[429, 500, 502, 503, 504],
        allowed_methods=frozenset(["GET", "POST", "PUT", "DELETE", "HEAD", "OPTIONS", "TRACE", "PATCH"]),
        raise_on_status=False,
        respect_retry_after_header=True,
    )
    adapter = HTTPAdapter(max_retries=retry, pool_connections=10, pool_maxsize=10)
    sess.mount("https://", adapter)
    sess.mount("http://", adapter)
    return sess

def _dns_preflight(hostname: str, attempts: int = 6, sleep_sec: float = 1.0) -> None:
    """
    Try to resolve hostname a few times before starting REST pagination.
    """
    last_err = None
    for i in range(1, attempts + 1):
        try:
            socket.gethostbyname(hostname)
            return
        except OSError as e:
            last_err = e
            print(f"[dns] attempt {i}/{attempts} failed: {e}; retrying in {sleep_sec}s")
            time.sleep(sleep_sec)
            sleep_sec *= 1.5
    raise RuntimeError(f"DNS resolution failed for {hostname}: {last_err}")


# ---------------- LS connections & fetchers ----------------
def connect_label_studio() -> LabelStudio:
    ls = LabelStudio(base_url=BASE_URL, api_key=API_TOKEN, timeout=30.0)
    _ = ls.projects.list(page_size=1)  # sanity call
    return ls

def _rest_fetch_all_tasks(project_id: int, api_token: str, base_url: str, page_size: int = 500):
    """
    Fetch ALL tasks via REST with pagination and include annotations.
    Handles both LS styles:
      A) dict: {"results": [...], "next": "..."}  -> follow "next"
      B) list: [...] (~100)                        -> iterate page=1,2,3 via ?page=
    Adds manual retry per page to survive DNS/connect blips.
    """
    _dns_preflight(hostname=socket.getfqdn(base_url.replace("https://", "").replace("http://", "")) or "app.heartex.com")

    sess = _make_retrying_session()
    headers = {
        "Authorization": f"Token {api_token}",
        "Content-Type": "application/json",
    }
    base = f"{base_url.rstrip('/')}/api/projects/{project_id}/tasks"

    out: List[Dict[str, Any]] = []

    def _get_with_retries(url: str, *, params: Optional[dict] = None) -> requests.Response:
        delay = PER_REQUEST_SLEEP_BASE
        last_exc = None
        for attempt in range(1, PER_REQUEST_MAX_RETRIES + 1):
            try:
                resp = sess.get(url, headers=headers, params=params, timeout=60)
                resp.raise_for_status()
                return resp
            except requests.exceptions.RequestException as e:
                last_exc = e
                print(f"[http] GET failed (attempt {attempt}/{PER_REQUEST_MAX_RETRIES}) for {url} "
                      f"params={params} -> {e}; retrying in {delay:.1f}s")
                time.sleep(delay)
                delay *= 2.0
        raise last_exc  # out of attempts

    # ---- First call (detect style) ----
    params = {"page_size": page_size, "include_annotations": "true", "ordering": "-id"}
    r = _get_with_retries(base, params=params)
    payload = r.json()

    # ---- STYLE A: dict with results/next ----
    if isinstance(payload, dict) and "results" in payload:
        while True:
            if not isinstance(payload, dict):
                break
            out.extend(payload.get("results", []))
            next_url = payload.get("next")
            if not next_url:
                break
            r = _get_with_retries(next_url, params=None)  # next already encoded
            payload = r.json()

    # ---- STYLE B: list (no cursor/next) -> page-by-page fallback ----
    elif isinstance(payload, list):
        out.extend(payload)  # page 1
        page = 2
        while True:
            r = _get_with_retries(base, params={
                "page": page,
                "page_size": page_size,
                "include_annotations": "true",
                "ordering": "-id",
            })
            chunk = r.json()
            if not isinstance(chunk, list) or len(chunk) == 0:
                break
            out.extend(chunk)
            page += 1

    print(f"[REST] total fetched across pages: {len(out)}")
    return out

def fetch_tasks_via_snapshot_or_rest(ls: LabelStudio, project_id: int, view_id: Optional[int]) -> List[Dict[str, Any]]:
    """
    Try snapshot/export first (with explicit 'select all' when no view).
    If it fails, fall back to REST pagination (include_annotations=true).
    """
    print(f"Creating export snapshot (project={project_id}, view_id={view_id}) …")

    # Explicit selection prevents “empty selection” converter failures.
    task_filter_options = ({"view": int(view_id)}
                           if view_id
                           else {"selectedItems": {"all": True}})

    try:
        export = ls.projects.exports.create(
            id=project_id,
            title=f"groups_assoc_export_{int(time.time())}",
            task_filter_options=task_filter_options,
        )
        export_pk = export.id

        # Poll for completion
        while True:
            info = ls.projects.exports.get(id=project_id, export_pk=export_pk)
            status = getattr(info, "status", None)
            if status in ("completed", "failed"):
                print(f"Export status: {status}")
                if status == "failed":
                    err  = getattr(info, "error", None)
                    det  = getattr(info, "details", None)
                    try:
                        raw = info.__dict__
                    except Exception:
                        raw = str(info)
                    print("=== Export Failure Diagnostics ===")
                    print(f"error:   {err}")
                    print(f"details: {det}")
                    print(f"raw:     {raw}")
                    print("==================================")
                    raise RuntimeError("Label Studio snapshot export failed")
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

        print(f"[snapshot] got {len(payload)} tasks")
        return payload

    except Exception as e:
        print(f"[WARN] Snapshot export failed: {e}")
        print("[INFO] Falling back to REST pagination (include_annotations=true)…")
        tasks = _rest_fetch_all_tasks(project_id=project_id, api_token=API_TOKEN, base_url=BASE_URL)
        return tasks


# ---------------- Third Stage Trigger ----------------
def trigger_third_stage_processing(labeled_json_path: str, bucket_name: str, service_url: str, script_path: str) -> bool:
    """
    Trigger the third stage processing script with the newly created export.
    """
    if not labeled_json_path:
        print("WARNING: No export file to process")
        return False

    print("\n" + "=" * 60)
    print("TRIGGERING THIRD STAGE PROCESSING")
    print("=" * 60)
    print(f"Input file: {labeled_json_path}")
    print(f"Bucket: {bucket_name}")
    print(f"Service URL: {service_url}")
    print()

    try:
        cmd = [
            sys.executable,  # Use the same Python interpreter
            script_path,
            "--bucket", bucket_name,
            "--labeled-json", labeled_json_path,
            "--service-url", service_url,
        ]
        print(f"Running command: {' '.join(cmd)}\n")

        subprocess.run(
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
        print(f"\nERROR: Could not find third stage processing script: {script_path}")
        print("Please ensure 'ship_third_stage_processing.py' is accessible at the configured path")
        return False
    except Exception as e:
        print(f"\nERROR: Unexpected error triggering third stage processing: {e}")
        return False


# ---------------- Main ----------------
def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Stage 2 LS extractor (Groups Association) with bucket override")
    p.add_argument("--bucket", default=GCS_BUCKET_NAME, help="Default export bucket (used when inference is ambiguous)")
    p.add_argument("--service-url", default=RECOGNITION_SERVICE_URL, help="Recognition service URL")
    p.add_argument("--auto-trigger-third", dest="auto_trigger_third", action="store_true", help="Trigger Stage 3 after export")
    p.add_argument("--no-auto-trigger-third", dest="auto_trigger_third", action="store_false", help="Skip Stage 3 trigger")
    p.set_defaults(auto_trigger_third=AUTO_TRIGGER_THIRD_STAGE)
    return p.parse_args()


def main():
    args = parse_args()
    global GCS_BUCKET_NAME, RECOGNITION_SERVICE_URL, AUTO_TRIGGER_THIRD_STAGE
    GCS_BUCKET_NAME = args.bucket
    RECOGNITION_SERVICE_URL = args.service_url
    AUTO_TRIGGER_THIRD_STAGE = args.auto_trigger_third

    # Fast ADC/bucket reachability check
    try:
        _startup_bucket_check(GCS_BUCKET_NAME)
        print(f"GCS reachable and bucket '{GCS_BUCKET_NAME}' is accessible.")
    except Exception as e:
        print(f"Startup GCS check failed: {e}")
        sys.exit(1)

    ledger_bucket = LEDGER_BUCKET
    print(f"Using global ledger: gs://{ledger_bucket}/{GLOBAL_PROCESSED_BLOB_PATH}")

    # Load processed tasks registry and coerce IDs to int
    registry = load_processed_registry(ledger_bucket)
    raw_ids = registry.get("processed_tasksID", []) or []
    processed_ids: set[int] = set()
    for x in raw_ids:
        try:
            processed_ids.add(int(x))
        except Exception:
            pass
    print(f"Loaded registry with {len(processed_ids)} processed task IDs.")

    # Connect + sanity
    ls = connect_label_studio()
    try:
        proj = ls.projects.get(id=PROJECT_ID)
        print(f"Project OK: id={proj.id}, title={getattr(proj, 'title', '')}")
    except Exception as e:
        print(f"[WARN] Could not read project meta (continuing): {e}")

    # Fetch ALL tasks (snapshot with select-all; fallback to REST)
    tasks = fetch_tasks_via_snapshot_or_rest(ls, PROJECT_ID, VIEW_ID)

    # Sanity: how many are review-accepted overall now?
    accepted_overall = 0
    for t in tasks:
        anns = t.get("annotations") or []
        if any(is_review_accepted(a) for a in anns):
            accepted_overall += 1
    print(f"[sanity] Review-accepted tasks in project now: {accepted_overall}")

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

        d = t.get("data", {}) or {}

        # timestamps & uuids
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
        accepted_task_ids.append(tid)

    # Update global processed IDs with newly accepted task IDs
    processed_ids.update(new_task_ids)

    # ---- DEBUG / DIAGNOSTICS ----
    print("============== DEBUG / DIAGNOSTICS ==============")
    print(f"Total tasks fetched     : {len(all_task_ids)}")
    print(f"Accepted (NEW this run) : {len(set(accepted_task_ids))}")
    print(f"Already processed       : {len(set(already_processed_task_ids))}")
    print(f"No annotations          : {len(set(no_annotations_task_ids))}")
    print(f"Not accepted            : {len(set(not_accepted_task_ids))}")
    print("==================================================")

    # Partition outputs by bucket (infer from paths)
    bucket_outputs: Dict[str, List[Dict[str, Any]]] = {}
    multi_bucket_outputs: List[Dict[str, Any]] = []
    fallback_bucket = GCS_BUCKET_NAME
    for out in outputs:
        buckets = _output_buckets(out)
        if not buckets:
            buckets = [fallback_bucket]
        if len(buckets) > 1:
            multi_bucket_outputs.append({"task_id": out.get("task_id"), "buckets": buckets})
        target_bucket = buckets[0]
        bucket_outputs.setdefault(target_bucket, []).append(out)

    if multi_bucket_outputs:
        print("⚠ Entries with multiple buckets detected (using the first bucket listed):")
        for entry in multi_bucket_outputs[:10]:
            print(f"   task_id={entry['task_id']} buckets={entry['buckets']}")
        if len(multi_bucket_outputs) > 10:
            print(f"   ... and {len(multi_bucket_outputs) - 10} more")

    exported_file_path = None
    for bucket_name, outs in bucket_outputs.items():
        if not outs:
            continue
        now_local = datetime.now(TZ) if TZ else datetime.now()
        ts = now_local.strftime("%Y-%m-%d_%H-%M")
        fname = f"LS_{PROJECT_ID}_ACCEPTED_{ts}_{len(outs)}Tasks.json"
        gs_path = save_json_to_gcs(outs, fname, bucket_name)
        print(f"✅ Saved {len(outs)} accepted tasks to: {gs_path}")
        exported_file_path = f"/{GCS_OUTPUT_PREFIX}/{fname}"

        if AUTO_TRIGGER_THIRD_STAGE:
            print("\n" + "=" * 60)
            print("Stage 2 Complete - Starting Stage 3")
            print("=" * 60)
            success = trigger_third_stage_processing(
                exported_file_path,
                bucket_name=bucket_name,
                service_url=RECOGNITION_SERVICE_URL,
                script_path=THIRD_STAGE_SCRIPT_PATH,
            )
            if not success:
                print(f"⚠️  Stage 3 trigger failed for bucket {bucket_name}")

    # Update registry (always bump timestamp)
    merged_ids = sorted(set(processed_ids))
    now_utc = datetime.now(timezone.utc).isoformat(timespec="microseconds").replace("+00:00", "Z")
    registry["processed_tasksID"] = merged_ids            # keep as ints
    registry["last_updated"] = now_utc
    registry["total_count"] = len(merged_ids)
    reg_path = save_processed_registry(registry, ledger_bucket)

    if new_task_ids:
        print(f"📝 Updated processed registry with {len(set(new_task_ids))} new IDs → {reg_path}")
        print(f"🧮 Registry total_count: {len(merged_ids)}")
    else:
        print(f"📝 Processed registry timestamp updated (no new IDs) → {reg_path}")
        print(f"🧮 Registry total_count: {len(merged_ids)}")

    return exported_file_path  # Return last export path (if any)


if __name__ == "__main__":
    exported_file = main()

    # If an export was created and auto-trigger is enabled, trigger third stage processing
    if exported_file and AUTO_TRIGGER_THIRD_STAGE:
        print("\n" + "=" * 60)
        print("Stage 2 Complete - Starting Stage 3")
        print("=" * 60)

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
