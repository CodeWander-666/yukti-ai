import os
from pathlib import Path

# Base directory of the project
BASE_DIR = Path(__file__).parent.parent.absolute()

# FAISS index location (configurable via environment variable)
VECTORDB_PATH = Path(os.getenv("FAISS_PATH", BASE_DIR / "faiss_index"))

# Data sources for knowledge base
SOURCES = [
    {
        "type": "csv",
        "name": "Original Dataset",
        "path": BASE_DIR / "dataset" / "dataset.csv",
        "columns": ["prompt", "response"],
        "content_template": "Q: {prompt}\nA: {response}",
    },
    # Additional sources can be added here (e.g., RSS, API)
]

# Cache TTL for document retrieval (seconds)
CACHE_TTL = int(os.getenv("RETRIEVAL_CACHE_TTL", "3600"))

# Task poll interval (seconds)
TASK_POLL_INTERVAL = int(os.getenv("TASK_POLL_INTERVAL", "5"))
