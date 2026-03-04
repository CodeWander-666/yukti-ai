"""
Configuration for knowledge updater – defines paths and data sources.
Sources are loaded from JSON files to allow dynamic updates.
"""

import os
import json
import logging
from pathlib import Path

logger = logging.getLogger(__name__)

BASE_DIR = Path(__file__).parent.parent.absolute()

# Paths
DATASET_PATH = BASE_DIR / "dataset" / "dataset.csv"
UPLOADS_PATH = BASE_DIR / "data" / "uploads"
VECTORDB_PATH = Path(os.getenv("FAISS_PATH", BASE_DIR / "faiss_index"))

# Ensure uploads directory exists
UPLOADS_PATH.mkdir(parents=True, exist_ok=True)

# JSON files for user‑defined sources
SOURCES_FILE = BASE_DIR / "knowledge_updater" / "sources.json"
WEBSITES_FILE = BASE_DIR / "knowledge_updater" / "web_sources.json"

# Default sources
SOURCES = {
    "rss": [],
    "api": [],
    "websites": []
}

# Load user sources if they exist
if SOURCES_FILE.exists():
    try:
        with open(SOURCES_FILE, 'r') as f:
            user_sources = json.load(f)
            SOURCES.update(user_sources)
        logger.info(f"Loaded user sources from {SOURCES_FILE}")
    except Exception as e:
        logger.error(f"Failed to load {SOURCES_FILE}: {e}")

# Load websites (crawling targets) if they exist
if WEBSITES_FILE.exists():
    try:
        with open(WEBSITES_FILE, 'r') as f:
            SOURCES["websites"] = json.load(f).get("websites", [])
        logger.info(f"Loaded websites from {WEBSITES_FILE}")
    except Exception as e:
        logger.error(f"Failed to load {WEBSITES_FILE}: {e}")

# Update interval for scheduler (not used in auto‑updater)
UPDATE_INTERVAL = 3600  # 1 hour
