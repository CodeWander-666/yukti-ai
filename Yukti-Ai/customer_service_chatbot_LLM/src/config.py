"""
Central configuration for Yukti AI.
All paths, constants, and environment‑based settings are defined here.
This module should be imported by every other module.
"""

import os
from pathlib import Path

# ----------------------------------------------------------------------
# Base directories
# ----------------------------------------------------------------------
# The root directory of the project (two levels up from src/)
BASE_DIR = Path(__file__).parent.parent.absolute()

# Paths for datasets and indices
DATASET_PATH = BASE_DIR / "dataset" / "dataset.csv"
VECTORDB_PATH = Path(os.getenv("FAISS_PATH", BASE_DIR / "faiss_index"))

# ----------------------------------------------------------------------
# API keys (read from environment or Streamlit secrets)
# ----------------------------------------------------------------------
def get_env_or_secret(key: str) -> str:
    """Try to get a value from environment variable; if not, fall back to Streamlit secrets."""
    value = os.getenv(key)
    if value:
        return value
    try:
        import streamlit as st
        return st.secrets.get(key, "")
    except ImportError:
        return ""

ZHIPU_API_KEY = get_env_or_secret("ZHIPU_API_KEY")
GOOGLE_API_KEY = get_env_or_secret("GOOGLE_API_KEY")

# ----------------------------------------------------------------------
# Model concurrency limits
# ----------------------------------------------------------------------
MODEL_CONCURRENCY = {
    "glm-4-flash": 200,
    "glm-4-plus": 20,
    "glm-5": 3,
    "cogview-3-flash": 10,
    "cogview-4": 5,
    "cogvideox-3": 5,
    "cogvideox-flash": 3,
    "glm-4-voice": 5,
    "gemini": 60,
}

# ----------------------------------------------------------------------
# Cache and polling settings
# ----------------------------------------------------------------------
RETRIEVAL_CACHE_TTL = int(os.getenv("RETRIEVAL_CACHE_TTL", "3600"))   # seconds
TASK_POLL_INTERVAL = int(os.getenv("TASK_POLL_INTERVAL", "5"))         # seconds

# ----------------------------------------------------------------------
# Logging configuration
# ----------------------------------------------------------------------
LOG_LEVEL = os.getenv("LOG_LEVEL", "INFO")
LOG_FORMAT = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'

# ----------------------------------------------------------------------
# Feature flags
# ----------------------------------------------------------------------
ENABLE_TRANSFORMER = os.getenv("ENABLE_TRANSFORMER", "false").lower() == "true"

# ----------------------------------------------------------------------
# Database path (shared with task queue)
# ----------------------------------------------------------------------
DB_PATH = BASE_DIR / "yukti_tasks.db"
