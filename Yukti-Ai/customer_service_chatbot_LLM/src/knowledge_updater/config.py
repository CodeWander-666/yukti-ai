"""
Configuration for the knowledge updater.
Shares paths with main app but can run independently.
"""

import os
import sys
from pathlib import Path

# ----------------------------------------------------------------------
# Try to import from main config, but fallback to local definitions
# ----------------------------------------------------------------------
try:
    # When run from the project root with PYTHONPATH set correctly
    from config import SOURCES, VECTORDB_PATH, BASE_DIR
except ImportError:
    # Fallback: define minimal config for standalone operation
    BASE_DIR = Path(__file__).parent.parent.absolute()
    VECTORDB_PATH = Path(os.getenv("FAISS_PATH", BASE_DIR / "faiss_index"))
    
    # Default sources – can be overridden by environment or file
    SOURCES = [
        {
            "type": "csv",
            "name": "Original Dataset",
            "path": BASE_DIR / "dataset" / "dataset.csv",
            "columns": ["prompt", "response"],
            "content_template": "Q: {prompt}\nA: {response}",
            "enabled": True,
        },
        # Additional sources can be added via environment or config file
    ]
    
    # Optionally load from a JSON config file if present
    config_file = BASE_DIR / "updater_config.json"
    if config_file.exists():
        import json
        try:
            with open(config_file, 'r') as f:
                user_config = json.load(f)
                if 'SOURCES' in user_config:
                    SOURCES = user_config['SOURCES']
                if 'VECTORDB_PATH' in user_config:
                    VECTORDB_PATH = Path(user_config['VECTORDB_PATH'])
        except Exception as e:
            logging.error(f"Failed to load updater_config.json: {e}")

# Ensure VECTORDB_PATH is a Path object
if not isinstance(VECTORDB_PATH, Path):
    VECTORDB_PATH = Path(VECTORDB_PATH)
