# inplace_transform_all_imo_preview_cap.py
# pip install google-cloud-storage pillow

import io
import json
import os
import re
from dataclasses import dataclass
from typing import Tuple, Optional, List, Iterable, Dict, Any

from google.cloud import storage
from google.oauth2 import service_account
from PIL import Image

# ----------------------------
# USER SETTINGS (EDIT THESE)
# ----------------------------
GOOGLE_APPLICATION_CREDENTIALS = r"C:\Users\OrGil.AzureAD\OneDrive - AMPC\Desktop\Azimut.ai\recognition_labeling\resources\credentials.json"  # <— set this

# Parent console URL (works with /browser/... or /browser/_details/...)
ROOT_CONSOLE_URL = (
    "https://console.cloud.google.com/storage/browser/_details/"
    "outsource_data/reidentification/bronze/json_lables/ship_spotting"
)

# Local backup base folder for original (untransformed) JSONs.
# Originals are saved under <LOCAL_UNTRANSFORMED_BASE>\<IMO>\original-filename.json
LOCAL_UNTRANSFORMED_BASE = r"C:\Users\OrGil.AzureAD\OneDrive - AMPC\Desktop\Azimut.ai\recognition_labeling\untransformed jsons"

# Optional: add a prelabel rectangle based on the bbox in the source JSON.
# NOTE: Turning this on reads image dimensions from GCS, which is slower.
ADD_PREDICTED_BOX = True
BBOX_FORMAT_HINT = "auto"  # "auto" | "xyxy" | "xywh"
DEFAULT_RECTANGLE_LABEL = "Merchant"

# Names from your Label Studio View
RECT_LABELS_FROM_NAME = "bbox_labels"
TO_NAME_IMAGE = "image"

# Limit how many files to preview (printing full sample JSON) before confirmation
MAX_PREVIEW_FILES = 40

# ----------------------------
# Helpers
# ----------------------------

def normalize_scheme(url: str) -> str:
    """Fix 'jttp' typo."""
    return re.sub(r"^jttp", "http", url, flags=re.IGNORECASE)

def http_to_gs(url: str) -> Optional[str]:
    """Convert common GCS HTTP(S) URLs to gs://bucket/key."""
    url = normalize_scheme(url).strip()

    m = re.match(r"^https?://storage\.googleapis\.com/([^/]+)/(.+)$", url, flags=re.I)
    if m:
        return f"gs://{m.group(1)}/{m.group(2).split('?',1)[0]}"

    m = re.match(r"^https?://storage\.cloud\.google\.com/([^/]+)/(.+)$", url, flags=re.I)
    if m:
        return f"gs://{m.group(1)}/{m.group(2).split('?',1)[0]}"

    m = re.match(r"^https?://console\.cloud\.google\.com/storage/browser/_details/([^/]+)/(.+)$", url, flags=re.I)
    if m:
        return f"gs://{m.group(1)}/{m.group(2).split('?',1)[0]}"

    m = re.match(r"^https?://console\.cloud\.google\.com/storage/browser/([^/]+)(/.*)?$", url, flags=re.I)
    if m:
        bucket = m.group(1)
        tail = (m.group(2) or "").lstrip("/")
        return f"gs://{bucket}/{tail}".rstrip("/")

    return None

def console_folder_to_bucket_prefix(url: str) -> Tuple[str, str]:
    """Convert console folder URL (with/without _details) to (bucket, prefix)."""
    url = url.strip()
    gs = http_to_gs(url)
    if gs and gs.startswith("gs://"):
        without = gs[len("gs://"):]
        parts = without.split("/", 1)
        bucket = parts[0]
        prefix = parts[1] if len(parts) > 1 else ""
        if prefix and not prefix.endswith("/"):
            prefix += "/"
        return bucket, prefix

    # Fallback regexes
    m = re.search(r"/storage/browser/_details/([^/]+)/(.+)$", url)
    if m:
        bucket, tail = m.group(1), m.group(2).split("?", 1)[0]
        if tail and not tail.endswith("/"):
            tail += "/"
        return bucket, tail

    m = re.search(r"/storage/browser/([^/]+)(/.*)?$", url)
    if m:
        bucket = m.group(1)
        tail = (m.group(2) or "").lstrip("/")
        if tail and not tail.endswith("/"):
            tail += "/"
        return bucket, tail

    raise ValueError(f"Could not parse console folder URL: {url}")

@dataclass
class GsPath:
    bucket: str
    key: str

def parse_gs_uri(gs_uri: str) -> GsPath:
    if not gs_uri.startswith("gs://"):
        raise ValueError(f"Expected gs:// URI, got: {gs_uri}")
    without = gs_uri[len("gs://"):]
    parts = without.split("/", 1)
    if len(parts) == 1:
        return GsPath(parts[0], "")
    return GsPath(parts[0], parts[1])

def make_client(creds_path: str) -> storage.Client:
    creds = service_account.Credentials.from_service_account_file(creds_path)
    return storage.Client(credentials=creds, project=creds.project_id)

def gs_read_bytes(client: storage.Client, gs_uri: str) -> bytes:
    p = parse_gs_uri(gs_uri)
    blob = client.bucket(p.bucket).blob(p.key)
    return blob.download_as_bytes()

def gs_write_text(client: storage.Client, gs_uri: str, text: str):
    p = parse_gs_uri(gs_uri)
    blob = client.bucket(p.bucket).blob(p.key)
    blob.upload_from_string(text, content_type="application/json")

def image_size_from_gs(client: storage.Client, gs_uri: str) -> Tuple[int, int]:
    b = gs_read_bytes(client, gs_uri)
    with Image.open(io.BytesIO(b)) as im:
        return im.size  # (width, height)

def safe_get(dct, path, default=None):
    cur = dct
    for key in path:
        if not isinstance(cur, dict) or key not in cur:
            return default
        cur = cur[key]
    return cur

def infer_bbox_format(bbox: List[float], img_w: int, img_h: int, hint: str = "auto") -> str:
    if hint in ("xyxy", "xywh"):
        return hint
    if len(bbox) != 4:
        return "xyxy"
    x1, y1, a, b = bbox
    if (a >= 0 and b >= 0) and (x1 + a <= img_w + 2) and (y1 + b <= img_h + 2):
        return "xywh"
    return "xyxy"

def bbox_pixels_to_percent(bbox: List[float], img_w: int, img_h: int, fmt: str):
    if len(bbox) != 4:
        raise ValueError(f"bbox must have 4 numbers, got {bbox}")
    W = max(img_w, 1)
    H = max(img_h, 1)
    if fmt == "xywh":
        x, y, w, h = bbox
        return (x / W * 100.0, y / H * 100.0, w / W * 100.0, h / H * 100.0)
    x1, y1, x2, y2 = bbox
    return (x1 / W * 100.0, y1 / H * 100.0, (x2 - x1) / W * 100.0, (y2 - y1) / H * 100.0)

def map_label(original: dict, default_label: str) -> str:
    cls = safe_get(original, ["target", "classification"])
    if isinstance(cls, str):
        mapping = {
            "boat": "Merchant",
            "merchant": "Merchant",
            "tug": "Tug",
            "containers": "Containers",
            "fishing": "Fishing",
            "military": "Military",
            "bulk": "Bulk",
            "tanker": "Tanker",
            "pilot": "Pilot",
            "rubber": "Rubber",
            "patrol-boat": "Patrol-Boat",
            "support": "Support",
            "general-cargo": "General-Cargo",
            "ro-ro": "Ro-Ro",
        }
        norm = cls.strip().lower()
        return mapping.get(norm, default_label)
    return default_label

def normalize_image_path(img: Optional[str]) -> Optional[str]:
    """Ensure image path is gs://..., converting from http(s)/console links if needed."""
    if not img or not isinstance(img, str):
        return img
    if img.startswith("gs://"):
        return img
    gs = http_to_gs(img)
    return gs or img

def prune_duplicates_in_outsource(out_src: dict, data_image: str, data_imo: str) -> dict:
    """Remove duplicates from outsource_json that already live under data."""
    src = json.loads(json.dumps(out_src))  # deep copy

    # root IMO
    if "IMO" in src and src["IMO"] == data_imo:
        del src["IMO"]

    # target.IMO
    tgt = src.get("target")
    if isinstance(tgt, dict) and tgt.get("IMO") == data_imo:
        del tgt["IMO"]

    # target.bounding_box.image
    bb = safe_get(src, ["target", "bounding_box"])
    if isinstance(bb, dict):
        img_in_bb = bb.get("image")
        if isinstance(img_in_bb, str):
            norm_img_in_bb = http_to_gs(img_in_bb) or img_in_bb
            if norm_img_in_bb == data_image:
                del bb["image"]

    return src

# ---------- Robust parsing ----------

def parse_container(raw: bytes):
    """
    Return (container_type, payload)
      - ("dict", dict)
      - ("list", [dict, ...])
      - ("ndjson", [dict, ...])  # lines
    """
    text = raw.decode("utf-8", errors="replace").strip()
    # Try dict/list JSON first
    try:
        parsed = json.loads(text)
        if isinstance(parsed, dict):
            return "dict", parsed
        if isinstance(parsed, list):
            parsed = [x for x in parsed if isinstance(x, dict)]
            return "list", parsed
    except Exception:
        pass
    # Fallback NDJSON
    lines = []
    for line in text.splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            obj = json.loads(line)
            if isinstance(obj, dict):
                lines.append(obj)
        except Exception:
            continue
    return "ndjson", lines

# ---------- Transform one record ----------

def transform_one_record(
    original: dict,
    client: storage.Client,
    add_pred_box: bool,
    bbox_hint: str
) -> Optional[dict]:
    """Transform one outsource-json record to LS single-task JSON; return None if invalid."""
    imo = original.get("IMO") or safe_get(original, ["target", "IMO"]) or ""

    image_path = safe_get(original, ["target", "bounding_box", "image"])
    image_path = normalize_image_path(image_path)

    if not image_path:
        image_path = normalize_image_path(original.get("image") or original.get("img") or safe_get(original, ["data", "image"]))

    if not image_path:
        return None

    bbox_px = safe_get(original, ["target", "bounding_box", "bounding_box"])
    conf = safe_get(original, ["target", "bounding_box", "conf"])

    pruned_outsource = prune_duplicates_in_outsource(original, data_image=image_path, data_imo=imo)

    out = {
        "data": {
            "image": image_path,
            "IMO": imo,
            "outsource_json": pruned_outsource
        },
        "predictions": [],
        "annotations": []
    }

    if add_pred_box and isinstance(bbox_px, list) and len(bbox_px) == 4:
        try:
            img_w, img_h = image_size_from_gs(client, image_path)
            fmt = infer_bbox_format(bbox_px, img_w, img_h, hint=bbox_hint)
            x, y, w, h = bbox_pixels_to_percent(bbox_px, img_w, img_h, fmt)
            if w > 0 and h > 0:
                out["predictions"] = [
                    {
                        "model_version": "outsource_json",
                        "score": conf,
                        "result": [
                            {
                                "from_name": RECT_LABELS_FROM_NAME,
                                "to_name": TO_NAME_IMAGE,
                                "type": "rectanglelabels",
                                "original_width": img_w,
                                "original_height": img_h,
                                "value": {
                                    "x": x,
                                    "y": y,
                                    "width": w,
                                    "height": h,
                                    "rectanglelabels": [map_label(original, DEFAULT_RECTANGLE_LABEL)]
                                }
                            }
                        ]
                    }
                ]
        except Exception as e:
            print(f"  ⚠ prelabel skipped (could not read image or compute bbox): {e}")

    return out

# ----------------------------
# Main
# ----------------------------

def extract_imo_from_blobname(blob_name: str) -> str:
    # Try to parse pattern like ".../IMO_1000588/..."
    m = re.search(r"IMO_(\d{7})", blob_name)
    return m.group(1) if m else "unknown_IMO"

def save_local_untransformed(imo: str, blob_base: str, raw: bytes):
    out_dir = os.path.join(LOCAL_UNTRANSFORMED_BASE, imo)
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, blob_base)
    with open(out_path, "wb") as f:
        f.write(raw)
    print(f"  • Saved local original → {out_path}")

def main():
    bucket_name, prefix = console_folder_to_bucket_prefix(ROOT_CONSOLE_URL)
    client = make_client(GOOGLE_APPLICATION_CREDENTIALS)

    # Gather all JSON blobs recursively under root prefix, restricted to IMO_* subfolders
    print(f"Scanning gs://{bucket_name}/{prefix} ...")
    all_blobs = [b for b in client.list_blobs(bucket_name, prefix=prefix) if b.name.lower().endswith(".json")]
    blobs = [b for b in all_blobs if re.search(r"/IMO_\d{7}/", b.name)]

    if not blobs:
        print("No JSON files found under IMO_* folders.")
        return

    # 1) PREVIEW (cap at MAX_PREVIEW_FILES)
    print("\n========== PREVIEW (first {} files) ==========".format(MAX_PREVIEW_FILES))
    previewed = 0
    for blob in blobs:
        if previewed >= MAX_PREVIEW_FILES:
            break

        try:
            raw = blob.download_as_bytes()
            container, payload = parse_container(raw)
        except Exception as e:
            print(f"✖ Failed to read/parse {blob.name}: {e}")
            continue

        transformed_records = []
        if container == "dict":
            out = transform_one_record(payload, client, ADD_PREDICTED_BOX, BBOX_FORMAT_HINT)
            if out:
                transformed_records.append(out)
        else:
            for rec in payload:
                out = transform_one_record(rec, client, ADD_PREDICTED_BOX, BBOX_FORMAT_HINT)
                if out:
                    transformed_records.append(out)

        sample_text = "(no valid records found)"
        if transformed_records:
            sample_text = json.dumps(transformed_records[0], ensure_ascii=False, indent=2)

        print(f"\n--- {blob.name} ---")
        print(f"Container: {container}, records in: {1 if container=='dict' else len(payload)}, records out: {len(transformed_records)}")
        print("Sample transformed record (first):")
        print(sample_text)

        previewed += 1

    if len(blobs) > MAX_PREVIEW_FILES:
        print(f"\n(…and {len(blobs) - MAX_PREVIEW_FILES} more files not shown in preview)")

    print("\nNo changes have been made yet.")

    # 2) CONFIRMATION (single global prompt)
    resp = input("\nType YES to overwrite ALL listed JSONs in place (and save local originals). Anything else to abort: ").strip()
    if resp != "YES":
        print("Aborted. No changes were made.")
        return

    # 3) WRITE PHASE (overwrite in GCS; save local original per file under IMO folder)
    total_overwritten = 0
    for blob in blobs:
        try:
            raw = blob.download_as_bytes()
            container, payload = parse_container(raw)
        except Exception as e:
            print(f"✖ Re-read failed {blob.name}: {e}")
            continue

        transformed_records = []
        if container == "dict":
            out = transform_one_record(payload, client, ADD_PREDICTED_BOX, BBOX_FORMAT_HINT)
            if out:
                transformed_records.append(out)
        else:
            for rec in payload:
                out = transform_one_record(rec, client, ADD_PREDICTED_BOX, BBOX_FORMAT_HINT)
                if out:
                    transformed_records.append(out)

        if not transformed_records:
            print(f"⚠ Skipping {blob.name}: produced 0 transformed records.")
            continue

        # Save local original
        base = os.path.basename(blob.name)
        imo = transformed_records[0]["data"].get("IMO") or extract_imo_from_blobname(blob.name)
        try:
            save_local_untransformed(imo, base, raw)
        except Exception as e:
            print(f"  ⚠ Could not save local original for {blob.name}: {e}")

        # Prepare output text in the SAME container format as source
        if container == "dict":
            out_text = json.dumps(transformed_records[0], ensure_ascii=False, indent=2)
        elif container == "list":
            out_text = json.dumps(transformed_records, ensure_ascii=False, indent=2)
        else:  # ndjson
            out_text = "\n".join(json.dumps(r, ensure_ascii=False) for r in transformed_records) + "\n"

        # Overwrite in place
        gs_uri = f"gs://{blob.bucket.name}/{blob.name}"
        try:
            gs_write_text(client, gs_uri, out_text)
            print(f"✓ Overwrote {gs_uri}")
            total_overwritten += 1
        except Exception as e:
            print(f"✖ Failed to write {gs_uri}: {e}")

    print(f"\n✅ Done. Overwritten files: {total_overwritten}")

if __name__ == "__main__":
    main()
