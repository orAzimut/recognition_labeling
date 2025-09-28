#!/usr/bin/env python3
"""
Retrofit Script: Add captured dates to existing gallery photos
Fetches dates from ShipSpotting and renames files
"""

import os
import re
import time
import logging
from pathlib import Path
from datetime import datetime
import cloudscraper
from bs4 import BeautifulSoup
from typing import Optional, Dict, List

# ====================== Configuration ======================
# TEST MODE - Set to False to process all IMOs
TEST_MODE = False
TEST_IMO = ""  # Leave empty to test first found IMO, or specify like "9808986"

# Paths
GALLERY_PATH = r"C:\Users\OrGil.AzureAD\OneDrive - AMPC\Desktop\Azimut.ai\recognition_gallery\Haifa_Gallery"

# ShipSpotting Configuration
SHIPSPOTTING_BASE = "https://www.shipspotting.com"
SHIPSPOTTING_PHOTO = SHIPSPOTTING_BASE + "/photos/{pid}"
FETCH_DELAY = 0.5  # seconds between requests to be respectful

# Logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class PhotoDateRetrofitter:
    """Retrofits existing photos with capture dates from ShipSpotting"""
    
    def __init__(self):
        self.session = cloudscraper.create_scraper()
        self.session.get(SHIPSPOTTING_BASE)  # Get cf_clearance cookie
        self.processed_count = 0
        self.skipped_count = 0
        self.failed_count = 0
        
    def fetch_page(self, url: str, retry: int = 2):
        """Fetch URL with retry logic"""
        for _ in range(retry + 1):
            try:
                r = self.session.get(url, timeout=15)
                if r.ok:
                    return r.text
                if r.status_code == 404:
                    return None
                if r.status_code in (403, 503):
                    self.session.get(SHIPSPOTTING_BASE, timeout=15)
                    time.sleep(1.5)
                    continue
            except Exception as e:
                logger.warning(f"Request failed: {e}")
                return None
        return None
    
    def extract_captured_date(self, photo_id: str) -> Optional[str]:
        """Extract captured date for a specific photo ID"""
        url = SHIPSPOTTING_PHOTO.format(pid=photo_id)
        logger.debug(f"Fetching {url}")
        
        html = self.fetch_page(url)
        if not html:
            logger.warning(f"Failed to fetch page for photo {photo_id}")
            return None
        
        # Parse the page
        soup = BeautifulSoup(html, "lxml")
        full_text = soup.get_text(" ", strip=True)
        
        # Extract captured date
        captured_match = re.search(r"Captured\s*:\s*([A-Za-z]+\s+\d{1,2},\s+\d{4})", full_text, re.I)
        if captured_match:
            try:
                date_str = captured_match.group(1)
                parsed_date = datetime.strptime(date_str, "%b %d, %Y")
                formatted_date = parsed_date.strftime("%Y%m%d")
                logger.info(f"Photo {photo_id}: Found date {date_str} -> {formatted_date}")
                return formatted_date
            except Exception as e:
                logger.error(f"Failed to parse date '{date_str}': {e}")
                return "19000101"  # Default fallback date
        else:
            logger.warning(f"Photo {photo_id}: No capture date found, using default")
            return "19000101"
    
    def extract_photo_id(self, filename: str) -> Optional[str]:
        """Extract photo ID from filename"""
        # Handle different naming patterns
        patterns = [
            r'^(\d+)\.jpg$',  # Just number: 12345.jpg
            r'^shipspotting_(\d+)\.jpg$',  # shipspotting_12345.jpg
            r'^(\d+)_\d{8}\.jpg$',  # Already has date: 12345_20250920.jpg
            r'^shipspotting_(\d+)_\d{8}\.jpg$',  # shipspotting_12345_20250920.jpg
        ]
        
        for pattern in patterns:
            match = re.match(pattern, filename, re.IGNORECASE)
            if match:
                # Check if already has date
                if '_' in filename and filename.split('_')[-1].split('.')[0].isdigit() and len(filename.split('_')[-1].split('.')[0]) == 8:
                    logger.info(f"Skipping {filename} - already has date")
                    return None
                return match.group(1)
        
        return None
    
    def process_imo_folder(self, imo_path: Path):
        """Process all photos in an IMO folder"""
        imo = imo_path.name
        logger.info(f"\n{'='*60}")
        logger.info(f"Processing IMO {imo}")
        logger.info(f"Path: {imo_path}")
        
        # Get all image files
        image_extensions = ['.jpg', '.jpeg', '.png']
        image_files = [f for f in imo_path.iterdir() 
                      if f.is_file() and f.suffix.lower() in image_extensions]
        
        if not image_files:
            logger.warning(f"No images found in {imo_path}")
            return
        
        logger.info(f"Found {len(image_files)} image files")
        
        # Process each file
        for idx, image_file in enumerate(image_files, 1):
            filename = image_file.name
            logger.info(f"\n[{idx}/{len(image_files)}] Processing: {filename}")
            
            # Extract photo ID
            photo_id = self.extract_photo_id(filename)
            if not photo_id:
                logger.info(f"  → Skipping (no valid photo ID or already has date)")
                self.skipped_count += 1
                continue
            
            logger.info(f"  → Photo ID: {photo_id}")
            
            # Get captured date from ShipSpotting
            captured_date = self.extract_captured_date(photo_id)
            if not captured_date:
                logger.error(f"  → Failed to get date")
                self.failed_count += 1
                continue
            
            # Create new filename
            new_filename = f"{photo_id}_{captured_date}.jpg"
            new_path = image_file.parent / new_filename
            
            # Rename file
            try:
                if new_path.exists():
                    logger.warning(f"  → Target file already exists: {new_filename}")
                    self.skipped_count += 1
                else:
                    image_file.rename(new_path)
                    logger.success(f"  ✓ Renamed to: {new_filename}")
                    self.processed_count += 1
            except Exception as e:
                logger.error(f"  → Failed to rename: {e}")
                self.failed_count += 1
            
            # Be respectful to the server
            time.sleep(FETCH_DELAY)
    
    def run(self):
        """Main execution"""
        gallery_path = Path(GALLERY_PATH)
        
        if not gallery_path.exists():
            logger.error(f"Gallery path not found: {GALLERY_PATH}")
            return
        
        # Get all IMO folders
        imo_folders = [d for d in gallery_path.iterdir() 
                      if d.is_dir() and d.name.isdigit()]
        
        if not imo_folders:
            logger.error("No IMO folders found")
            return
        
        logger.info(f"Found {len(imo_folders)} IMO folders")
        
        if TEST_MODE:
            # Test mode - process only one IMO
            if TEST_IMO:
                # Specific IMO requested
                test_folder = gallery_path / TEST_IMO
                if test_folder.exists():
                    logger.info(f"\nTEST MODE: Processing only IMO {TEST_IMO}")
                    self.process_imo_folder(test_folder)
                else:
                    logger.error(f"Test IMO folder not found: {TEST_IMO}")
            else:
                # Process first IMO found
                logger.info(f"\nTEST MODE: Processing only first IMO ({imo_folders[0].name})")
                self.process_imo_folder(imo_folders[0])
        else:
            # Process all IMOs
            logger.info(f"\nProcessing ALL {len(imo_folders)} IMO folders...")
            for idx, imo_folder in enumerate(imo_folders, 1):
                logger.info(f"\n[{idx}/{len(imo_folders)}] IMO Folder")
                self.process_imo_folder(imo_folder)
                
                # Small delay between IMOs
                if idx < len(imo_folders):
                    time.sleep(1)
        
        # Final summary
        logger.info(f"\n{'='*60}")
        logger.info("RETROFIT COMPLETE")
        logger.info(f"Processed: {self.processed_count} photos")
        logger.info(f"Skipped: {self.skipped_count} photos")
        logger.info(f"Failed: {self.failed_count} photos")
        logger.info(f"{'='*60}")

# Add success logging color (green)
class ColoredFormatter(logging.Formatter):
    """Custom formatter with colors"""
    
    COLORS = {
        'DEBUG': '\033[36m',    # Cyan
        'INFO': '\033[0m',      # Default
        'SUCCESS': '\033[32m',  # Green
        'WARNING': '\033[33m',  # Yellow
        'ERROR': '\033[31m',    # Red
        'RESET': '\033[0m'      # Reset
    }
    
    def format(self, record):
        if record.levelname in self.COLORS:
            record.levelname = f"{self.COLORS[record.levelname]}{record.levelname}{self.COLORS['RESET']}"
        return super().format(record)

# Add custom success level
logging.SUCCESS = 25  # Between INFO and WARNING
logging.addLevelName(logging.SUCCESS, 'SUCCESS')
def success(self, message, *args, **kwargs):
    if self.isEnabledFor(logging.SUCCESS):
        self._log(logging.SUCCESS, message, args, **kwargs)
logging.Logger.success = success

# Apply colored formatter
handler = logging.StreamHandler()
handler.setFormatter(ColoredFormatter('%(asctime)s - %(levelname)s - %(message)s'))
logging.root.handlers = [handler]

if __name__ == "__main__":
    print("\n" + "="*60)
    print(" PHOTO DATE RETROFITTER")
    print("="*60)
    print(f"Gallery Path: {GALLERY_PATH}")
    print(f"Test Mode: {TEST_MODE}")
    if TEST_MODE:
        print(f"Test IMO: {TEST_IMO if TEST_IMO else 'First found'}")
    print("="*60 + "\n")
    
    retrofitter = PhotoDateRetrofitter()
    
    try:
        retrofitter.run()
    except KeyboardInterrupt:
        print("\n\nProcess interrupted by user")
    except Exception as e:
        logger.error(f"Unexpected error: {e}", exc_info=True)