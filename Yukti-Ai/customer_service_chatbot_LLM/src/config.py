"""
Configuration for knowledge updater – defines data sources and paths.
"""

import os
from pathlib import Path

# Project root (assumes this file is in knowledge_updater/)
BASE_DIR = Path(__file__).parent.parent.absolute()

# Paths
DATASET_PATH = BASE_DIR / "dataset" / "dataset.csv"
UPLOADS_PATH = BASE_DIR / "data" / "uploads"
VECTORDB_PATH = Path(os.getenv("FAISS_PATH", BASE_DIR / "faiss_index"))

# Ensure uploads directory exists
UPLOADS_PATH.mkdir(parents=True, exist_ok=True)

# Web sources – can be extended via admin UI
SOURCES = {
    "rss": [
        {"name": "Example RSS", "url": "https://example.com/rss", "enabled": True},
    ],
    "api": [
        {"name": "Example API", "url": "https://api.example.com/posts", "enabled": False},
    ],
    # Local uploads are always enabled
}

# Update interval (seconds) – for scheduler
UPDATE_INTERVAL = 3600  # 1 hour
