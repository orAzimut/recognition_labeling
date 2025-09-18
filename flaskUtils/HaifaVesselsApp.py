#!/usr/bin/env python3
"""
Enhanced Haifa Vessel Gallery System with Shipspotting Fallback and YOLO Detection
Combines real-time IMO extraction, cloud storage sync, shipspotting scraper, 
Flask gallery viewer, and YOLO model inference for vessel detection
"""

import os
import re
import sys
import time
import json
import shutil
import logging
import threading
import uuid
from pathlib import Path
from datetime import datetime
from collections import defaultdict
from typing import List, Dict, Set, Tuple, Optional
from io import BytesIO
from urllib.parse import quote
import requests
import yaml
from flask import Flask, render_template_string, send_file, jsonify
from google.cloud import storage
from google.oauth2 import service_account
import cloudscraper
from bs4 import BeautifulSoup
from PIL import Image
from requests.exceptions import RequestException
import numpy as np

# YOLO imports
try:
    from ultralytics import YOLO
    YOLO_AVAILABLE = True
except ImportError:
    YOLO_AVAILABLE = False
    print("Warning: ultralytics not installed. YOLO detection features will be disabled.")
    print("Install with: pip install ultralytics")

# ====================== Configuration ======================
# You can modify these paths as needed
CREDENTIALS_PATH = r"C:\Users\OrGil.AzureAD\OneDrive - AMPC\Desktop\Azimut.ai\recognition_gallery\resources\credentials.json"
BUCKET_NAME = "outsource_data"
BASE_PATH = "reidentification/bronze/raw_crops/ship_spotting"
JSON_LABELS_PATH = "reidentification/bronze/json_lables/ship_spotting"  # Note: keeping the typo "lables" as in original
LOCAL_GALLERY_PATH = r"C:\Users\OrGil.AzureAD\OneDrive - AMPC\Desktop\Azimut.ai\recognition_gallery\Haifa_Gallery"

# YOLO Model Configuration
YOLO_MODEL_PATH = r"C:\Users\OrGil.AzureAD\OneDrive - AMPC\Desktop\Azimut.ai\recognition_labeling\Albatross\platform\ML\models\Albatross-v0.5.pt"
YOLO_CONFIDENCE_THRESHOLD = 0.25  # Adjust as needed
YOLO_IOU_THRESHOLD = 0.45  # Adjust as needed

# API Configuration
API_KEY = 'b123dc58-4c18-4b0c-9f04-82a06be63ff9'
PORT_LAT = 32.8154
PORT_LON = 35.0043
SEARCH_RADIUS = 10  # km

# Flask Configuration
FLASK_PORT = 5000
FLASK_HOST = '0.0.0.0'

# Sync Configuration
SYNC_INTERVAL = 3600  # seconds (1 hour)
AUTO_SYNC = False  # Set to False to disable automatic syncing

# Shipspotting Configuration
SHIPSPOTTING_BASE = "https://www.shipspotting.com"
SHIPSPOTTING_GALLERY = SHIPSPOTTING_BASE + "/photos/gallery"
SHIPSPOTTING_PHOTO = SHIPSPOTTING_BASE + "/photos/{pid}"
SCRAPE_STEP_DELAY = 0.35  # seconds between photo downloads
SCRAPE_SEARCH_DELAY = 1.0  # seconds between searches
MAX_PHOTOS_PER_IMO = 150  # Limit photos per vessel to avoid excessive scraping

# ====================== Logging Setup ======================
log_dir = Path("logs")
log_dir.mkdir(exist_ok=True)
log_filename = log_dir / f"vessel_gallery_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(log_filename),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# ====================== HTML Template (same as before) ======================
HTML_TEMPLATE = """
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Haifa Bay Vessel Gallery</title>
    <style>
        * {
            margin: 0;
            padding: 0;
            box-sizing: border-box;
        }
        
        body {
            font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            min-height: 100vh;
            padding: 20px;
        }
        
        .container {
            max-width: 1400px;
            margin: 0 auto;
            background: white;
            border-radius: 20px;
            box-shadow: 0 20px 60px rgba(0,0,0,0.3);
            overflow: hidden;
        }
        
        .header {
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            color: white;
            padding: 30px;
            text-align: center;
            position: relative;
        }
        
        .header h1 {
            font-size: 2.5em;
            margin-bottom: 10px;
            text-shadow: 2px 2px 4px rgba(0,0,0,0.2);
        }
        
        .header p {
            font-size: 1.2em;
            opacity: 0.95;
        }
        
        .sync-status {
            position: absolute;
            right: 30px;
            top: 30px;
            background: rgba(255,255,255,0.2);
            padding: 10px 20px;
            border-radius: 20px;
            font-size: 0.9em;
        }
        
        .sync-status.syncing {
            animation: pulse 1.5s infinite;
        }
        
        @keyframes pulse {
            0% { opacity: 0.6; }
            50% { opacity: 1; }
            100% { opacity: 0.6; }
        }
        
        .back-button {
            position: absolute;
            left: 30px;
            top: 50%;
            transform: translateY(-50%);
            background: rgba(255,255,255,0.2);
            border: 2px solid white;
            color: white;
            padding: 10px 20px;
            border-radius: 25px;
            cursor: pointer;
            font-size: 16px;
            transition: all 0.3s;
            text-decoration: none;
            display: inline-block;
        }
        
        .back-button:hover {
            background: white;
            color: #667eea;
            transform: translateY(-50%) scale(1.05);
        }
        
        .stats {
            background: #f8f9fa;
            padding: 20px 30px;
            display: flex;
            justify-content: space-around;
            flex-wrap: wrap;
            gap: 20px;
        }
        
        .stat-card {
            background: white;
            padding: 15px 25px;
            border-radius: 10px;
            box-shadow: 0 2px 10px rgba(0,0,0,0.1);
            text-align: center;
            min-width: 150px;
        }
        
        .stat-number {
            font-size: 2em;
            font-weight: bold;
            color: #667eea;
        }
        
        .stat-label {
            color: #666;
            margin-top: 5px;
        }
        
        .search-bar {
            padding: 20px 30px;
            background: white;
            border-bottom: 1px solid #eee;
        }
        
        .search-input {
            width: 100%;
            padding: 12px 20px;
            font-size: 16px;
            border: 2px solid #e0e0e0;
            border-radius: 25px;
            outline: none;
            transition: border-color 0.3s;
        }
        
        .search-input:focus {
            border-color: #667eea;
        }
        
        .gallery {
            padding: 30px;
            display: grid;
            grid-template-columns: repeat(auto-fill, minmax(250px, 1fr));
            gap: 25px;
        }
        
        .imo-card {
            background: white;
            border-radius: 15px;
            overflow: hidden;
            box-shadow: 0 5px 20px rgba(0,0,0,0.1);
            transition: all 0.3s;
            cursor: pointer;
            position: relative;
        }
        
        .imo-card:hover {
            transform: translateY(-5px);
            box-shadow: 0 10px 30px rgba(0,0,0,0.2);
        }
        
        .imo-card img {
            width: 100%;
            height: 200px;
            object-fit: cover;
            display: block;
        }
        
        .imo-info {
            padding: 15px;
            background: white;
        }
        
        .imo-number {
            font-size: 1.2em;
            font-weight: bold;
            color: #333;
            margin-bottom: 5px;
        }
        
        .vessel-name {
            color: #666;
            font-size: 0.9em;
            margin-bottom: 3px;
        }
        
        .photo-count {
            color: #999;
            font-size: 0.85em;
        }
        
        .photo-badge {
            position: absolute;
            top: 10px;
            right: 10px;
            background: rgba(102, 126, 234, 0.9);
            color: white;
            padding: 5px 10px;
            border-radius: 15px;
            font-size: 0.85em;
            font-weight: bold;
        }
        
        .detail-gallery {
            padding: 30px;
            display: grid;
            grid-template-columns: repeat(auto-fill, minmax(300px, 1fr));
            gap: 20px;
        }
        
        .detail-photo {
            background: white;
            border-radius: 10px;
            overflow: hidden;
            box-shadow: 0 3px 15px rgba(0,0,0,0.1);
            transition: transform 0.3s;
        }
        
        .detail-photo:hover {
            transform: scale(1.05);
        }
        
        .detail-photo img {
            width: 100%;
            height: 250px;
            object-fit: cover;
            display: block;
            cursor: pointer;
        }
        
        .photo-name {
            padding: 10px;
            text-align: center;
            background: #f8f9fa;
            font-size: 0.9em;
            color: #666;
        }
        
        .modal {
            display: none;
            position: fixed;
            top: 0;
            left: 0;
            width: 100%;
            height: 100%;
            background: rgba(0,0,0,0.9);
            z-index: 1000;
            cursor: pointer;
        }
        
        .modal img {
            position: absolute;
            top: 50%;
            left: 50%;
            transform: translate(-50%, -50%);
            max-width: 90%;
            max-height: 90%;
            border-radius: 10px;
        }
        
        .modal-close {
            position: absolute;
            top: 20px;
            right: 40px;
            color: white;
            font-size: 40px;
            cursor: pointer;
            z-index: 1001;
        }
    </style>
</head>
<body>
    <div class="container">
        {% if view == 'main' %}
        <div class="header">
            <div class="sync-status" id="syncStatus">
                Last sync: {{ last_sync }}
            </div>
            <h1>🚢 Haifa Bay Vessel Gallery</h1>
            <p>Real-time vessel monitoring with automatic updates</p>
        </div>
        
        <div class="stats">
            <div class="stat-card">
                <div class="stat-number">{{ total_imos }}</div>
                <div class="stat-label">Active Vessels</div>
            </div>
            <div class="stat-card">
                <div class="stat-number">{{ total_photos }}</div>
                <div class="stat-label">Total Photos</div>
            </div>
            <div class="stat-card">
                <div class="stat-number">{{ avg_photos }}</div>
                <div class="stat-label">Avg Photos/Vessel</div>
            </div>
        </div>
        
        <div class="search-bar">
            <input type="text" class="search-input" id="searchInput" 
                   placeholder="Search by IMO number or vessel name..." 
                   onkeyup="filterGallery()">
        </div>
        
        <div class="gallery" id="gallery">
            {% for imo, data in imos.items() %}
            <div class="imo-card" 
                 data-imo="{{ imo }}" 
                 data-name="{{ data.name|lower }}">
                <div class="photo-badge">{{ data.photos|length }} photos</div>
                <img src="/image/{{ imo }}/{{ data.photos[0] }}" alt="IMO {{ imo }}" loading="lazy">
                <div class="imo-info">
                    <div class="imo-number">IMO {{ imo }}</div>
                    <div class="vessel-name">{{ data.name }}</div>
                    <div class="photo-count">{{ data.photos|length }} photo{% if data.photos|length > 1 %}s{% endif %}</div>
                </div>
            </div>
            {% endfor %}
        </div>
        
        {% elif view == 'detail' %}
        <div class="header">
            <a href="/" class="back-button">← Back to Gallery</a>
            <h1>🚢 {{ vessel_name }}</h1>
            <p>IMO {{ imo }} - {{ photos|length }} photos available</p>
        </div>
        
        <div class="detail-gallery">
            {% for photo in photos %}
            <div class="detail-photo">
                <img src="/image/{{ imo }}/{{ photo }}" alt="{{ photo }}" 
                     onclick="openModal('/image/{{ imo }}/{{ photo }}')">
                <div class="photo-name">{{ photo }}</div>
            </div>
            {% endfor %}
        </div>
        {% endif %}
    </div>
    
    <div id="imageModal" class="modal" onclick="closeModal()">
        <span class="modal-close">&times;</span>
        <img id="modalImage" src="" alt="">
    </div>
    
    <script>
        // Add event listeners to all IMO cards for both left and middle click
        document.addEventListener('DOMContentLoaded', function() {
            const imoCards = document.querySelectorAll('.imo-card');
            
            imoCards.forEach(card => {
                // Handle mouse down events for all buttons
                card.addEventListener('mousedown', function(e) {
                    const imo = this.dataset.imo;
                    const url = '/imo/' + imo;
                    
                    if (e.button === 0) { // Left click
                        // Navigate in same tab
                        window.location.href = url;
                    } else if (e.button === 1) { // Middle click (scroll wheel click)
                        // Open in new tab
                        e.preventDefault();
                        window.open(url, '_blank');
                    }
                });
                
                // Prevent default middle click behavior
                card.addEventListener('auxclick', function(e) {
                    if (e.button === 1) {
                        e.preventDefault();
                    }
                });
            });
        });
        
        function filterGallery() {
            const searchInput = document.getElementById('searchInput');
            const filter = searchInput.value.toLowerCase();
            const cards = document.querySelectorAll('.imo-card');
            
            cards.forEach(card => {
                const imo = card.dataset.imo.toLowerCase();
                const name = card.dataset.name;
                if (imo.includes(filter) || name.includes(filter)) {
                    card.style.display = '';
                } else {
                    card.style.display = 'none';
                }
            });
        }
        
        function openModal(imageSrc) {
            const modal = document.getElementById('imageModal');
            const modalImg = document.getElementById('modalImage');
            modal.style.display = 'block';
            modalImg.src = imageSrc;
        }
        
        function closeModal() {
            const modal = document.getElementById('imageModal');
            modal.style.display = 'none';
        }
        
        document.addEventListener('keydown', function(event) {
            if (event.key === 'Escape') {
                closeModal();
            }
        });
        
        setTimeout(() => {
            location.reload();
        }, 300000);
    </script>
</body>
</html>
"""

# ====================== YOLO Detector Class ======================
class VesselDetector:
    """YOLO-based vessel detector for generating Label Studio JSON annotations"""
    
    def __init__(self):
        self.model = None
        self.model_loaded = False
        
        if YOLO_AVAILABLE:
            self.load_model()
    
    def load_model(self):
        """Load the YOLO model"""
        try:
            if Path(YOLO_MODEL_PATH).exists():
                self.model = YOLO(YOLO_MODEL_PATH)
                self.model_loaded = True
                logger.info(f"YOLO model loaded from {YOLO_MODEL_PATH}")
            else:
                logger.warning(f"YOLO model not found at {YOLO_MODEL_PATH}")
                self.model_loaded = False
        except Exception as e:
            logger.error(f"Failed to load YOLO model: {e}")
            self.model_loaded = False
    
    def detect_vessels(self, image_path: Path) -> List[Dict]:
        """Run YOLO detection on an image and return detections"""
        if not self.model_loaded or not self.model:
            return []
        
        try:
            # Run inference
            results = self.model(
                str(image_path),
                conf=YOLO_CONFIDENCE_THRESHOLD,
                iou=YOLO_IOU_THRESHOLD,
                verbose=False
            )
            
            detections = []
            if results and len(results) > 0:
                result = results[0]
                
                if result.boxes is not None and len(result.boxes) > 0:
                    # Get image dimensions
                    img = Image.open(image_path)
                    img_width, img_height = img.size
                    
                    for box in result.boxes:
                        # Get box coordinates (xyxy format)
                        x1, y1, x2, y2 = box.xyxy[0].tolist()
                        confidence = float(box.conf[0])
                        
                        # Get class (assuming 0=Merchant, 1=Military, etc. - adjust based on your model)
                        class_id = int(box.cls[0])
                        class_name = self.get_class_name(class_id)
                        
                        detections.append({
                            'bbox': [x1, y1, x2, y2],
                            'bbox_normalized': {
                                'x': (x1 / img_width) * 100,  # Convert to percentage
                                'y': (y1 / img_height) * 100,
                                'width': ((x2 - x1) / img_width) * 100,
                                'height': ((y2 - y1) / img_height) * 100
                            },
                            'confidence': confidence,
                            'class': class_name,
                            'class_id': class_id,
                            'img_width': img_width,
                            'img_height': img_height
                        })
            
            return detections
            
        except Exception as e:
            logger.error(f"Error detecting vessels in {image_path}: {e}")
            return []
    
    def get_class_name(self, class_id: int) -> str:
        """Map class ID to vessel type name"""
        # Adjust these mappings based on your model's classes
        class_map = {
            0: "Merchant",
            1: "Military",
            2: "Fishing",
            3: "Passenger",
            4: "Tanker",
            5: "Container",
            6: "Bulk Carrier",
            7: "General Cargo",
            8: "Unknown"
        }
        return class_map.get(class_id, "Unknown")
    
    def create_label_studio_json(self, image_filename: str, imo: str, detections: List[Dict], vessel_details: Dict) -> Dict:
        """Create a Label Studio compatible JSON for the detections"""
        
        # Construct the GCS path for the image
        gcs_image_path = f"gs://{BUCKET_NAME}/{BASE_PATH}/IMO_{imo}/{image_filename}"
        
        # Get vessel info
        vessel_info = vessel_details.get(imo, {})
        
        # Generate UUIDs for consistency
        annotation_uuid = str(uuid.uuid4())
        target_uuid = str(uuid.uuid4())
        
        # Current timestamp
        current_time = datetime.now()
        timestamp_str = current_time.strftime("%Y-%m-%d %H:%M:%S")
        timestamp_float = current_time.timestamp()
        
        # Create predictions list
        predictions_list = []
        
        for idx, detection in enumerate(detections):
            # Create the result value for Label Studio format
            result_value = {
                "from_name": "bbox_labels",
                "to_name": "image",
                "type": "rectanglelabels",
                "original_width": detection['img_width'],
                "original_height": detection['img_height'],
                "value": {
                    "x": detection['bbox_normalized']['x'],
                    "y": detection['bbox_normalized']['y'],
                    "width": detection['bbox_normalized']['width'],
                    "height": detection['bbox_normalized']['height'],
                    "rectanglelabels": [detection['class']]
                }
            }
            
            predictions_list.append(result_value)
        
        # Build the complete JSON structure
        label_json = {
            "data": {
                "image": gcs_image_path,
                "IMO": imo,
                "outsource_json": {
                    "target": {
                        "SOURCE": "shipspotting",
                        "TYPE": "EO",
                        "FIRST_DETECTION": timestamp_float,
                        "last_update_time": timestamp_float,
                        "ID": int(image_filename.split('_')[-1].split('.')[0]) if '_' in image_filename else idx,
                        "uuid": target_uuid,
                        "tracker_ID": int(image_filename.split('_')[-1].split('.')[0]) if '_' in image_filename else idx,
                        "threat_level": "UNKNOWN",
                        "quality": 10,
                        "is_child": False,
                        "state": "Observing",
                        "current_coordinate": [
                            vessel_info.get("lat"),
                            vessel_info.get("lon")
                        ],
                        "long_term_history": [],
                        "speed": vessel_info.get("speed"),
                        "course": vessel_info.get("course"),
                        "distance_from_platform": None,
                        "max_distance_from_platform": None,
                        "aspect": None,
                        "size_m": None,
                        "alerts": [],
                        "classification": "boat",
                        "identification": None,
                        "bounding_box": {
                            "bounding_box": detections[0]['bbox'] if detections else [0, 0, 100, 100],
                            "padded_bounding_box": None,
                            "identification": None,
                            "conf": detections[0]['confidence'] if detections else 0.0,
                            "id": int(image_filename.split('_')[-1].split('.')[0]) if '_' in image_filename else idx,
                            "frame_count": None,
                            "start_frame": None,
                            "end_frame": None,
                            "reid_count": None,
                            "is_static": None,
                            "emb_dist": None,
                            "smooth_mean": None,
                            "smooth_embedding_update": None
                        },
                        "frame_number": None,
                        "MMSI": int(vessel_info.get("mmsi")) if vessel_info.get("mmsi", "").isdigit() else None,
                        "SHIP_TYPE": vessel_info.get("vessel_type"),
                        "NAME": vessel_info.get("name"),
                        "CALL_SIGN": None,
                        "DIMENSIONS": None,
                        "add_counter": 0,
                        "delete_counter": 0,
                        "uncertainty_area": [],
                        "children": [],
                        "manual_not_suspicious": False,
                        "manual_not_suspicious_for_db": False
                    },
                    "platform": {
                        "platform_name": "SHIP-SPOTTING",
                        "platform_type": "static",
                        "location": {
                            "latitude": None,
                            "longitude": None,
                            "altitude": None
                        },
                        "attitude": {
                            "yaw": None,
                            "roll": None,
                            "pitch": None
                        },
                        "is_flying": False,
                        "remote_location": {
                            "latitude": 0,
                            "longitude": 0
                        },
                        "simulator_data": 0,
                        "camera_data": {
                            "gimbal": {
                                "pitch": None,
                                "yaw": None,
                                "roll": None
                            },
                            "gimbal_q": {},
                            "gimbal_j": {},
                            "focal_lengths": {
                                "zoom": None
                            },
                            "zoom_limits": {
                                "zoom": [None, None]
                            },
                            "focal_length_limits": {
                                "zoom": [None, None]
                            },
                            "fov_polygons": {
                                "zoom": []
                            },
                            "real_data_received": True,
                            "center_point": [None, None],
                            "horizon_offset": {
                                "zoom": 0
                            },
                            "horizon_line_list": []
                        },
                        "frame_metadata": None
                    },
                    "uuid": annotation_uuid,
                    "timestamp": timestamp_str,
                    "created_at": timestamp_str
                }
            },
            "predictions": [
                {
                    "model_version": "outsource_json",
                    "score": detections[0]['confidence'] if detections else 0.0,
                    "result": predictions_list
                }
            ] if predictions_list else [],
            "annotations": []
        }
        
        return label_json

# ====================== Haifa Bay API Client (unchanged) ======================
class HaifaBayTracker:
    """Client for fetching vessel data from Datalastic API"""
    
    def __init__(self):
        self.api_key = API_KEY
        self.api_base_url = "https://api.datalastic.com/api/v0"
        self.session = requests.Session()
        self.port_lat = PORT_LAT
        self.port_lon = PORT_LON

    def get_haifa_vessels(self, radius: int = SEARCH_RADIUS) -> List[Dict]:
        """Get all vessels in Haifa Bay area"""
        endpoint = f"{self.api_base_url}/vessel_inradius"
        params = {
            "api-key": self.api_key,
            "lat": self.port_lat,
            "lon": self.port_lon,
            "radius": radius,
        }
        try:
            resp = self.session.get(endpoint, params=params, timeout=30)
            resp.raise_for_status()
            data = resp.json()
            if not data.get("meta", {}).get("success", False):
                logger.warning("API request was not successful: %s", data.get("meta"))
                return []
            return data.get("data", {}).get("vessels", [])
        except Exception as e:
            logger.warning("Error fetching vessels: %s", e)
            return []

    def get_imo_numbers_with_details(self) -> Tuple[List[str], Dict[str, Dict]]:
        """Get IMO numbers with vessel details"""
        vessels = self.get_haifa_vessels()
        imo_list: List[str] = []
        vessel_details: Dict[str, Dict] = {}

        for vessel in vessels:
            imo = vessel.get("imo")
            if imo and imo.strip() and imo.strip().lower() not in {"null", "n/a", "none", "0"}:
                imo_clean = imo.strip()
                imo_list.append(imo_clean)
                vessel_details[imo_clean] = {
                    "name": vessel.get("name", "Unknown"),
                    "vessel_type": vessel.get("type", "Unknown"),
                    "mmsi": vessel.get("mmsi", ""),
                    "lat": vessel.get("lat", 0),
                    "lon": vessel.get("lon", 0),
                    "destination": vessel.get("destination", ""),
                    "speed": vessel.get("speed", 0),
                    "course": vessel.get("course", 0),
                    "timestamp": vessel.get("last_position_time", ""),
                    "extracted_at": datetime.now().isoformat(),
                }

        unique_imos = sorted(set(imo_list))
        return unique_imos, vessel_details

# ====================== Shipspotting Scraper (Enhanced) ======================
class ShipspottingScraper:
    """Web scraper for shipspotting.com with YOLO detection support"""
    
    def __init__(self):
        self.session = cloudscraper.create_scraper()
        self.session.get(SHIPSPOTTING_BASE)  # Get cf_clearance cookie
        self.imo_regex = re.compile(r"\bIMO[:\s#]*(\d{7})\b")
        self.mmsi_regex = re.compile(r"\bMMSI[:\s#]*(\d{9})\b")
        self.detector = VesselDetector()  # Initialize YOLO detector
        
    def fetch(self, url: str, binary: bool = False, retry: int = 2):
        """Fetch URL with retry logic"""
        for _ in range(retry + 1):
            try:
                r = self.session.get(url, timeout=15)
                if r.ok:
                    return r.content if binary else r.text
                if r.status_code == 404:
                    return None
                if r.status_code in (403, 503):
                    self.session.get(SHIPSPOTTING_BASE, timeout=15)
                    time.sleep(1.5)
                    continue
            except RequestException:
                return None
        return None
    
    def search_vessel_by_imo(self, imo: str, max_pages: int = 3) -> List[str]:
        """Search for photos of a vessel using IMO"""
        logger.info(f"Searching shipspotting.com for IMO {imo}...")
        
        all_photo_ids = []
        page = 1
        
        while page <= max_pages:
            gallery_url = f"{SHIPSPOTTING_GALLERY}?imo={imo}&page={page}&viewType=normal&sortBy=newest"
            
            html = self.fetch(gallery_url)
            if not html:
                break
            
            soup = BeautifulSoup(html, "lxml")
            page_photo_ids = []
            
            # Look for photo links
            photo_links = soup.find_all("a", href=re.compile(r"/photos/(\d+)"))
            
            for link in photo_links:
                href = link.get("href", "")
                match = re.search(r"/photos/(\d+)", href)
                if match:
                    photo_id = match.group(1)
                    if photo_id not in all_photo_ids:
                        page_photo_ids.append(photo_id)
                        all_photo_ids.append(photo_id)
            
            if not page_photo_ids:
                break
                
            page += 1
            time.sleep(0.5)
        
        # Limit to MAX_PHOTOS_PER_IMO
        return all_photo_ids[:MAX_PHOTOS_PER_IMO]
    
    def parse_photo_page(self, html: str, photo_id: str) -> Optional[Dict]:
        """Parse photo page to extract metadata"""
        soup = BeautifulSoup(html, "lxml")
        og = soup.find("meta", {"property": "og:image"})
        if not og:
            return None
            
        img_url = og["content"].replace("/small/", "/large/")
        full = soup.get_text(" ", strip=True)
        name = soup.find("h1").get_text(strip=True) if soup.find("h1") else None
        
        # Extract vessel type
        vessel_type = None
        vt_match = re.search(r"Vessel\s+Type\s*:?\s*([^\n\r•]+?)(?=\s+(?:Gross|Length|Built|IMO|MMSI|\n|\r|$))", full, re.I)
        if vt_match:
            vessel_type = vt_match.group(1).strip()
            vessel_type = re.sub(r'\s+', ' ', vessel_type).strip()
            vessel_type = re.split(r'[,;•\-–]', vessel_type)[0].strip()
            if len(vessel_type) > 50:
                vessel_type = vessel_type[:50].strip()
        
        imo_match = self.imo_regex.search(full)
        mmsi_match = self.mmsi_regex.search(full)
        
        return {
            "photo_id": photo_id,
            "page": SHIPSPOTTING_PHOTO.format(pid=photo_id),
            "name": name,
            "imo": imo_match.group(1) if imo_match else None,
            "mmsi": mmsi_match.group(1) if mmsi_match else None,
            "vessel_type": vessel_type,
            "image_url": img_url,
        }
    
    def process_and_upload_detections(self, imo: str, local_imo_path: Path, vessel_details: Dict, gcs_manager):
        """Process all images with YOLO and upload detection JSONs to GCS"""
        if not self.detector.model_loaded:
            logger.warning("YOLO model not loaded, skipping detection processing")
            return 0
        
        # Get all image files in the IMO folder
        image_extensions = ['.jpg', '.jpeg', '.png']
        image_files = [f for f in local_imo_path.iterdir() 
                      if f.is_file() and f.suffix.lower() in image_extensions]
        
        if not image_files:
            logger.info(f"No images found for detection in {local_imo_path}")
            return 0
        
        logger.info(f"Processing {len(image_files)} images with YOLO model for IMO {imo}")
        
        json_count = 0
        for image_file in image_files:
            try:
                # Run YOLO detection
                detections = self.detector.detect_vessels(image_file)
                
                # Create Label Studio JSON
                label_json = self.detector.create_label_studio_json(
                    image_file.name,
                    imo,
                    detections,
                    vessel_details
                )
                
                # Save JSON locally (temporary)
                json_filename = f"{image_file.stem}.json"
                local_json_path = local_imo_path / json_filename
                
                with open(local_json_path, 'w') as f:
                    json.dump(label_json, f, indent=2)
                
                # Upload JSON to GCS
                json_blob_path = f"{JSON_LABELS_PATH}/IMO_{imo}/{json_filename}"
                blob = gcs_manager.bucket.blob(json_blob_path)
                blob.upload_from_filename(str(local_json_path))
                
                # Remove local JSON file to save space
                local_json_path.unlink()
                
                json_count += 1
                logger.info(f"Uploaded detection JSON for {image_file.name}")
                
            except Exception as e:
                logger.error(f"Failed to process detections for {image_file.name}: {e}")
        
        if json_count > 0:
            logger.info(f"Uploaded {json_count} detection JSONs to GCS for IMO {imo}")
        
        return json_count
    
    def download_photos_for_imo(self, imo: str, local_imo_path: Path, vessel_details: Dict, gcs_manager=None) -> int:
        """Download photos from shipspotting for a specific IMO and process with YOLO"""
        photo_ids = self.search_vessel_by_imo(imo)
        
        if not photo_ids:
            logger.warning(f"No photos found on shipspotting.com for IMO {imo}")
            return 0
        
        logger.info(f"Found {len(photo_ids)} photos on shipspotting.com for IMO {imo}")
        
        success_count = 0
        for photo_id in photo_ids:
            html = self.fetch(SHIPSPOTTING_PHOTO.format(pid=photo_id))
            if not html:
                continue
                
            meta = self.parse_photo_page(html, photo_id)
            if not meta:
                continue
            
            # Download image
            img_bytes = self.fetch(meta["image_url"], binary=True)
            if not img_bytes:
                continue
            
            try:
                # Save image
                jpg_file = local_imo_path / f"shipspotting_{photo_id}.jpg"
                Image.open(BytesIO(img_bytes)).convert("RGB").save(jpg_file, quality=92)
                success_count += 1
                logger.info(f"✓ Downloaded {jpg_file.name} for IMO {imo}")
            except Exception as e:
                logger.error(f"Failed to save photo {photo_id}: {e}")
            
            time.sleep(SCRAPE_STEP_DELAY)
        
        # Create/Update metadata JSON
        if success_count > 0:
            self.update_imo_metadata(imo, local_imo_path, vessel_details, success_count, "shipspotting")
            
            # Upload images to GCS if manager provided
            if gcs_manager:
                upload_count = gcs_manager.upload_imo_photos(imo, local_imo_path)
                logger.info(f"Uploaded {upload_count} photos to GCS for IMO {imo}")
                
                # Process images with YOLO and upload detection JSONs
                detection_count = self.process_and_upload_detections(imo, local_imo_path, vessel_details, gcs_manager)
                logger.info(f"Processed and uploaded {detection_count} detection JSONs for IMO {imo}")

        return success_count
    
    def update_imo_metadata(self, imo: str, local_imo_path: Path, vessel_details: Dict, photo_count: int, source: str):
        """Create or update IMO metadata JSON file with sync_timestamp as a list"""
        json_file = local_imo_path / f"vessel_metadata.json"
        current_timestamp = datetime.now().isoformat()
        
        # Load existing metadata if it exists
        if json_file.exists():
            try:
                with open(json_file, 'r') as f:
                    metadata = json.load(f)
                    
                # Handle backward compatibility: convert old format to new format
                if "sync_timestamp" in metadata:
                    if isinstance(metadata["sync_timestamp"], str):
                        # Convert single string to list
                        metadata["sync_timestamp"] = [metadata["sync_timestamp"]]
                    elif not isinstance(metadata["sync_timestamp"], list):
                        # If it's neither string nor list, start fresh
                        metadata["sync_timestamp"] = []
                else:
                    metadata["sync_timestamp"] = []
            except:
                metadata = {"sync_timestamp": []}
        else:
            metadata = {"sync_timestamp": []}
        
        # Update metadata
        vessel_info = vessel_details.get(imo, {})
        metadata.update({
            "imo": imo,
            "vessel_name": vessel_info.get("name", "Unknown"),
            "vessel_type": vessel_info.get("vessel_type", "Unknown"),
            "mmsi": vessel_info.get("mmsi", ""),
            "last_position": {
                "lat": vessel_info.get("lat", 0),
                "lon": vessel_info.get("lon", 0),
                "timestamp": vessel_info.get("timestamp", "")
            },
            "destination": vessel_info.get("destination", ""),
            "download_info": {
                "source": source,
                "photo_count": photo_count,
                "downloaded_at": datetime.now().isoformat(),
                "max_photos_limit": MAX_PHOTOS_PER_IMO if source == "shipspotting" else None,
                "yolo_processing": YOLO_AVAILABLE and self.detector.model_loaded
            },
            "gallery_path": str(local_imo_path)
        })
        
        # Add current timestamp to the list if not already present (within same minute)
        if not metadata["sync_timestamp"] or not any(
            current_timestamp[:16] == ts[:16] for ts in metadata["sync_timestamp"]
        ):
            metadata["sync_timestamp"].append(current_timestamp)
        
        # Save updated metadata
        with open(json_file, 'w') as f:
            json.dump(metadata, f, indent=2)

# ====================== GCS Manager (unchanged) ======================
class GCSManager:
    """Manager for Google Cloud Storage operations"""
    
    def __init__(self):
        self.client = None
        self.bucket = None
        self.initialize_client()
        
    def initialize_client(self):
        """Initialize GCS client"""
        try:
            credentials = service_account.Credentials.from_service_account_file(
                CREDENTIALS_PATH
            )
            self.client = storage.Client(credentials=credentials)
            self.bucket = self.client.bucket(BUCKET_NAME)
            logger.info("Successfully initialized GCS client")
        except Exception as e:
            logger.error(f"Failed to initialize GCS client: {str(e)}")
            raise
    
    def download_imo_photos(self, imo_number: str, local_imo_path: Path, vessel_details: Dict) -> int:
        """Download all photos for a specific IMO to its own folder"""
        folder_name = f"IMO_{imo_number}"
        prefix = f"{BASE_PATH}/{folder_name}/"
        
        # Create the IMO subfolder
        local_imo_path.mkdir(parents=True, exist_ok=True)
        
        # List all files in the IMO folder
        blobs = list(self.bucket.list_blobs(prefix=prefix))
        
        # Filter for image files
        image_extensions = ['.jpg', '.jpeg', '.png', '.bmp', '.gif', '.tiff', '.webp']
        image_blobs = [
            blob for blob in blobs 
            if not blob.name.endswith('/') and 
            any(blob.name.lower().endswith(ext) for ext in image_extensions)
        ]
        
        if not image_blobs:
            logger.warning(f"No images found in GCS for IMO {imo_number}")
            return 0
        
        success_count = 0
        for idx, blob in enumerate(image_blobs, 1):
            try:
                # Keep original filename structure
                filename = os.path.basename(blob.name)
                local_file_path = local_imo_path / filename
                
                # Download the file
                blob.download_to_filename(str(local_file_path))
                success_count += 1
                
            except Exception as e:
                logger.error(f"Failed to download {blob.name}: {str(e)}")
        
        if success_count > 0:
            # Update metadata JSON with the new list format
            scraper = ShipspottingScraper()
            scraper.update_imo_metadata(imo_number, local_imo_path, vessel_details, success_count, "gcs")
            logger.info(f"IMO {imo_number}: Downloaded {success_count}/{len(image_blobs)} photos from GCS")
        
        return success_count
    
    def upload_imo_photos(self, imo_number: str, local_imo_path: Path) -> int:
        """Upload local IMO photos to GCS"""
        folder_name = f"IMO_{imo_number}"
        prefix = f"{BASE_PATH}/{folder_name}/"
        
        uploaded_count = 0
        image_extensions = ['.jpg', '.jpeg', '.png', '.bmp', '.gif', '.tiff', '.webp']
        
        for file_path in local_imo_path.iterdir():
            if file_path.is_file() and any(file_path.suffix.lower() == ext for ext in image_extensions):
                try:
                    # Create blob path in GCS
                    blob_name = f"{prefix}{file_path.name}"
                    blob = self.bucket.blob(blob_name)
                    
                    # Upload the file
                    blob.upload_from_filename(str(file_path))
                    uploaded_count += 1
                    logger.info(f"Uploaded {file_path.name} to GCS for IMO {imo_number}")
                    
                except Exception as e:
                    logger.error(f"Failed to upload {file_path.name}: {e}")
        
        return uploaded_count

# ====================== Gallery Synchronizer (unchanged) ======================
class GallerySynchronizer:
    """Handles synchronization between API, GCS, and local storage"""
    
    def __init__(self):
        self.tracker = HaifaBayTracker()
        self.gcs_manager = GCSManager()
        self.scraper = ShipspottingScraper()
        self.gallery_path = Path(LOCAL_GALLERY_PATH)
        self.vessel_details = {}
        self.last_sync_time = None
        self.current_imos_in_bay = set()  # Initialize this attribute
        
    def ensure_gallery_directory(self):
        """Create gallery directory if it doesn't exist"""
        self.gallery_path.mkdir(parents=True, exist_ok=True)
        
    def get_current_local_imos(self) -> Set[str]:
        """Get list of IMO folders currently in local gallery"""
        if not self.gallery_path.exists():
            return set()
        
        imo_folders = set()
        for item in self.gallery_path.iterdir():
            if item.is_dir() and item.name.isdigit():
                imo_folders.add(item.name)
        return imo_folders
    
    def cleanup_old_imos(self, current_imos: Set[str]):
        """Remove IMO folders that are no longer in Haifa Bay"""
        local_imos = self.get_current_local_imos()
        to_remove = local_imos - current_imos
        
        for imo in to_remove:
            imo_path = self.gallery_path / imo
            try:
                shutil.rmtree(imo_path)
                logger.info(f"Removed old IMO folder: {imo}")
            except Exception as e:
                logger.error(f"Failed to remove {imo}: {e}")
    
    def check_imo_has_photos(self, imo_path: Path) -> bool:
        """Check if an IMO folder has any photos"""
        if not imo_path.exists():
            return False
        
        image_extensions = ['.jpg', '.jpeg', '.png', '.bmp', '.gif', '.tiff', '.webp']
        for file in imo_path.iterdir():
            if file.is_file() and any(file.suffix.lower() == ext for ext in image_extensions):
                return True
        return False
    
    def update_sync_timestamp_for_imo(self, imo: str, vessel_details: Dict):
        """Update only the sync_timestamp for an IMO that already has photos"""
        local_imo_path = self.gallery_path / imo
        json_file = local_imo_path / f"vessel_metadata.json"
        current_timestamp = datetime.now().isoformat()
        
        if json_file.exists():
            try:
                with open(json_file, 'r') as f:
                    metadata = json.load(f)
                    
                # Handle backward compatibility
                if "sync_timestamp" in metadata:
                    if isinstance(metadata["sync_timestamp"], str):
                        metadata["sync_timestamp"] = [metadata["sync_timestamp"]]
                    elif not isinstance(metadata["sync_timestamp"], list):
                        metadata["sync_timestamp"] = []
                else:
                    metadata["sync_timestamp"] = []
                    
                # Update vessel info with latest data
                vessel_info = vessel_details.get(imo, {})
                if vessel_info:
                    metadata.update({
                        "vessel_name": vessel_info.get("name", metadata.get("vessel_name", "Unknown")),
                        "vessel_type": vessel_info.get("vessel_type", metadata.get("vessel_type", "Unknown")),
                        "mmsi": vessel_info.get("mmsi", metadata.get("mmsi", "")),
                        "last_position": {
                            "lat": vessel_info.get("lat", 0),
                            "lon": vessel_info.get("lon", 0),
                            "timestamp": vessel_info.get("timestamp", "")
                        },
                        "destination": vessel_info.get("destination", ""),
                    })
                    
            except:
                # If file exists but can't be read, create fresh metadata
                metadata = {"sync_timestamp": []}
        else:
            # Create new metadata file if it doesn't exist
            metadata = {"sync_timestamp": []}
            vessel_info = vessel_details.get(imo, {})
            metadata.update({
                "imo": imo,
                "vessel_name": vessel_info.get("name", "Unknown"),
                "vessel_type": vessel_info.get("vessel_type", "Unknown"),
                "mmsi": vessel_info.get("mmsi", ""),
                "last_position": {
                    "lat": vessel_info.get("lat", 0),
                    "lon": vessel_info.get("lon", 0),
                    "timestamp": vessel_info.get("timestamp", "")
                },
                "destination": vessel_info.get("destination", ""),
                "gallery_path": str(local_imo_path)
            })
        
        # Add current timestamp to the list
        if not metadata["sync_timestamp"] or not any(
            current_timestamp[:16] == ts[:16] for ts in metadata["sync_timestamp"]
        ):
            metadata["sync_timestamp"].append(current_timestamp)
            logger.info(f"Updated sync timestamp for IMO {imo}")
        
        # Save updated metadata
        with open(json_file, 'w') as f:
            json.dump(metadata, f, indent=2)
    
    def sync_gallery(self) -> bool:
        """Main synchronization method"""
        logger.info("=" * 60)
        logger.info("Starting gallery synchronization...")
        
        try:
            # Ensure gallery directory exists
            self.ensure_gallery_directory()
            
            # Get current IMOs from API
            logger.info("Fetching current vessels in Haifa Bay...")
            imo_list, vessel_details = self.tracker.get_imo_numbers_with_details()
            self.vessel_details = vessel_details
            self.current_imos_in_bay = set(imo_list)  # Store current IMOs

            if not imo_list:
                logger.warning("No vessels found in Haifa Bay")
                return False
            
            logger.info(f"Found {len(imo_list)} vessels in Haifa Bay")
            
            # Convert to set for easier comparison
            current_imos = set(imo_list)
            local_imos = self.get_current_local_imos()
            
            # Update sync timestamps for ALL IMOs found in API call
            for imo in current_imos:
                local_imo_path = self.gallery_path / imo
                if local_imo_path.exists():
                    # IMO folder exists, update its timestamp
                    self.update_sync_timestamp_for_imo(imo, self.vessel_details)
                else:
                    # IMO folder doesn't exist, it will be created when downloading photos
                    logger.info(f"New IMO {imo} detected, will create folder when downloading photos")
            
            # Find what needs to be downloaded
            to_download = current_imos - local_imos
            
            # Also check existing IMO folders that might be empty
            imos_needing_photos = []
            for imo in current_imos:
                local_imo_path = self.gallery_path / imo
                if not self.check_imo_has_photos(local_imo_path):
                    imos_needing_photos.append(imo)
            
            logger.info(f"Status: {len(local_imos)} existing, {len(to_download)} new IMOs, {len(imos_needing_photos)} need photos")
            
            # Process IMOs needing photos
            for imo in imos_needing_photos:
                local_imo_path = self.gallery_path / imo
                local_imo_path.mkdir(parents=True, exist_ok=True)
                
                # Try GCS first
                logger.info(f"Attempting to download IMO {imo} from GCS...")
                photo_count = self.gcs_manager.download_imo_photos(imo, local_imo_path, self.vessel_details)
                
                # If no photos from GCS, try shipspotting
                if photo_count == 0:
                    logger.info(f"No GCS photos for IMO {imo}, trying shipspotting.com...")
                    photo_count = self.scraper.download_photos_for_imo(imo, local_imo_path, self.vessel_details, self.gcs_manager)
                    
                    if photo_count == 0:
                        logger.warning(f"No photos found anywhere for IMO {imo}")
                        # Still create/update metadata even if no photos found
                        self.update_sync_timestamp_for_imo(imo, self.vessel_details)
                    else:
                        logger.info(f"Downloaded {photo_count} photos from shipspotting for IMO {imo}")
                else:
                    logger.info(f"Downloaded {photo_count} photos from GCS for IMO {imo}")
                
                time.sleep(0.5)  # Rate limiting
            
            # Clean up old IMOs (optional - commented out for safety)
            # self.cleanup_old_imos(current_imos)
            
            self.last_sync_time = datetime.now()
            logger.info(f"Synchronization complete at {self.last_sync_time}")
            return True
            
        except Exception as e:
            logger.error(f"Synchronization failed: {e}")
            return False

# ====================== Flask Application (unchanged) ======================
app = Flask(__name__)
synchronizer = GallerySynchronizer()

def get_imo_photos() -> Dict:
    """Scan gallery directory and group photos by IMO folder"""
    imo_data = {}
    gallery_path = Path(LOCAL_GALLERY_PATH)
    
    if not gallery_path.exists():
        return imo_data
    
    # Scan IMO folders
    for imo_folder in gallery_path.iterdir():
        if imo_folder.is_dir() and imo_folder.name.isdigit():
            imo_number = imo_folder.name
            
            # Get all image files in the folder
            image_extensions = ['.jpg', '.jpeg', '.png', '.bmp', '.gif', '.tiff', '.webp']
            photos = []
            
            for file in imo_folder.iterdir():
                if file.is_file() and any(file.suffix.lower() == ext for ext in image_extensions):
                    photos.append(file.name)
            
            if photos:
                photos.sort()
                
                # Try to get vessel name from metadata or vessel_details
                vessel_name = "Unknown Vessel"
                
                # First try metadata JSON
                metadata_file = imo_folder / "vessel_metadata.json"
                if metadata_file.exists():
                    try:
                        with open(metadata_file, 'r') as f:
                            metadata = json.load(f)
                            vessel_name = metadata.get("vessel_name", "Unknown Vessel")
                    except:
                        pass
                
                # Fall back to vessel_details
                if vessel_name == "Unknown Vessel" and imo_number in synchronizer.vessel_details:
                    vessel_name = synchronizer.vessel_details[imo_number].get("name", "Unknown Vessel")
                
                imo_data[imo_number] = {
                    "photos": photos,
                    "name": vessel_name
                }
    
    return imo_data

@app.route('/')
def index():
    """Main gallery page"""
    all_imo_data = get_imo_photos()  # Get ALL local IMOs
    
    # Filter to show only vessels currently in Haifa Bay
    current_imo_data = {}
    for imo, data in all_imo_data.items():
        if imo in synchronizer.current_imos_in_bay:
            current_imo_data[imo] = data
    
    # If no sync has happened yet, show all (or none)
    if not synchronizer.current_imos_in_bay:
        current_imo_data = all_imo_data  # Or {} to show none
    
    total_photos = sum(len(data["photos"]) for data in current_imo_data.values())
    avg_photos = round(total_photos / len(current_imo_data)) if current_imo_data else 0
    
    last_sync = "Never"
    if synchronizer.last_sync_time:
        last_sync = synchronizer.last_sync_time.strftime("%Y-%m-%d %H:%M:%S")
    
    return render_template_string(
        HTML_TEMPLATE,
        view='main',
        imos=current_imo_data,  # Only show current vessels
        total_imos=len(current_imo_data),
        total_photos=total_photos,
        avg_photos=avg_photos,
        last_sync=last_sync
    )

@app.route('/imo/<imo_number>')
def imo_detail(imo_number):
    """Detail page for specific IMO"""
    imo_data = get_imo_photos()
    
    if imo_number not in imo_data:
        return "IMO not found", 404
    
    vessel_name = imo_data[imo_number]["name"]
    
    return render_template_string(
        HTML_TEMPLATE,
        view='detail',
        imo=imo_number,
        vessel_name=vessel_name,
        photos=imo_data[imo_number]["photos"]
    )

@app.route('/image/<imo>/<filename>')
def serve_image(imo, filename):
    """Serve image file from IMO subfolder"""
    file_path = Path(LOCAL_GALLERY_PATH) / imo / filename
    if file_path.exists():
        return send_file(str(file_path), mimetype='image/jpeg')
    return "Image not found", 404

@app.route('/api/sync', methods=['POST'])
def trigger_sync():
    """Manual sync trigger endpoint"""
    success = synchronizer.sync_gallery()
    return jsonify({"success": success, "timestamp": datetime.now().isoformat()})

@app.route('/api/status')
def api_status():
    """Get current system status"""
    all_imo_data = get_imo_photos()
    current_imos = {imo: data for imo, data in all_imo_data.items() 
                    if imo in synchronizer.current_imos_in_bay}
    
    return jsonify({
        "total_cached_imos": len(all_imo_data),
        "currently_in_bay": len(current_imos),
        "total_cached_photos": sum(len(data["photos"]) for data in all_imo_data.values()),
        "current_photos": sum(len(data["photos"]) for data in current_imos.values()),
        "last_sync": synchronizer.last_sync_time.isoformat() if synchronizer.last_sync_time else None,
        "auto_sync": AUTO_SYNC,
        "sync_interval": SYNC_INTERVAL,
        "yolo_enabled": YOLO_AVAILABLE and VesselDetector().model_loaded
    })

# ====================== Background Sync Thread (unchanged) ======================
def background_sync():
    """Background thread for automatic synchronization"""
    while AUTO_SYNC:
        try:
            synchronizer.sync_gallery()
            time.sleep(SYNC_INTERVAL)
        except Exception as e:
            logger.error(f"Background sync error: {e}")
            time.sleep(60)  # Wait a minute before retrying

# ====================== Main Entry Point (updated) ======================
def main():
    """Main application entry point"""
    print("\n" + "="*60)
    print(" Haifa Bay Vessel Gallery System")
    print("="*60)
    print(f"Gallery Path: {LOCAL_GALLERY_PATH}")
    print(f"GCS Bucket: {BUCKET_NAME}")
    print(f"📍 API Endpoint: Haifa Bay ({PORT_LAT}, {PORT_LON})")
    print(f"Search Radius: {SEARCH_RADIUS} km")
    print(f"Auto-sync: {'Enabled' if AUTO_SYNC else 'Disabled'}")
    print(f"Shipspotting fallback: Enabled (max {MAX_PHOTOS_PER_IMO} photos/vessel)")
    
    # Check YOLO status
    if YOLO_AVAILABLE:
        detector = VesselDetector()
        if detector.model_loaded:
            print(f"YOLO Detection: Enabled (Model: {Path(YOLO_MODEL_PATH).name})")
        else:
            print(f"YOLO Detection: Model not found at {YOLO_MODEL_PATH}")
    else:
        print("YOLO Detection: Disabled (ultralytics not installed)")
    
    if AUTO_SYNC:
        print(f"Sync Interval: {SYNC_INTERVAL} seconds")
    print("="*60)
    
    # Initial synchronization
    print("\nPerforming initial synchronization...")
    success = synchronizer.sync_gallery()
    
    if not success:
        print("Initial sync failed, but continuing with existing data...")
    
    # Start background sync thread if enabled
    if AUTO_SYNC:
        sync_thread = threading.Thread(target=background_sync, daemon=True)
        sync_thread.start()
        print("Background sync thread started")
    
    # Start Flask server
    print("\n" + "="*60)
    print(" Starting web server...")
    print(f"Open your browser and go to: http://localhost:{FLASK_PORT}")
    print("Press Ctrl+C to stop the server")
    print("="*60 + "\n")
    
    try:
        app.run(host=FLASK_HOST, port=FLASK_PORT, debug=False)
    except KeyboardInterrupt:
        print("\n\n👋 Shutting down...")
        sys.exit(0)

if __name__ == "__main__":
    main()