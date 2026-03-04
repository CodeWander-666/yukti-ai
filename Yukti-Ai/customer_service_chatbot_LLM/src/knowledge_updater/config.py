"""
Configuration for knowledge updater – defines paths and data sources.
"""
import os
import sys
from pathlib import Path

# Add project root to path to allow imports from src
BASE_DIR = Path(__file__).parent.parent.parent.absolute()
if str(BASE_DIR) not in sys.path:
    sys.path.insert(0, str(BASE_DIR))

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
    # Websites are loaded from web_sources.json (created by admin)
}

UPDATE_INTERVAL = 3600  # 1 hour
