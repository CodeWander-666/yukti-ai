import os
from pathlib import Path

# Base directory of the project
BASE_DIR = Path(__file__).parent.parent.absolute()

# Knowledge base paths
VECTORDB_PATH = Path(os.getenv("FAISS_PATH", BASE_DIR / "faiss_index"))
DATASET_PATH = BASE_DIR / "dataset" / "dataset.csv"

# Sources for dynamic knowledge updater
SOURCES = [
    {
        "type": "csv",
        "path": str(DATASET_PATH),
        "name": "Original Dataset",
        "columns": ["prompt", "response"],
        "content_template": "Q: {prompt}\nA: {response}"
    }
    # Add more sources (rss, api) here as needed
]

# Cache TTL for document retrieval (seconds)
RETRIEVAL_CACHE_TTL = int(os.getenv("RETRIEVAL_CACHE_TTL", "3600"))

# Task polling interval (seconds)
TASK_POLL_INTERVAL = int(os.getenv("TASK_POLL_INTERVAL", "5"))

# Concurrency limits (can be overridden by env)
ASYNC_TASK_CONCURRENCY = int(os.getenv("ASYNC_TASK_CONCURRENCY", "5"))

# Logging level
LOG_LEVEL = os.getenv("LOG_LEVEL", "INFO")
