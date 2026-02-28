import os
from pathlib import Path

# ----------------------------------------------------------------------
# Determine project root and data paths (matches langchain_helper.py)
# ----------------------------------------------------------------------
# __file__ is /.../src/knowledge_updater/config.py
BASE_DIR = Path(__file__).parent.parent.parent.absolute()  # customer_service_chatbot_LLM/
DATASET_DIR = BASE_DIR / "dataset"
VECTORDB_PATH = BASE_DIR / "faiss_index"

# ----------------------------------------------------------------------
# Source definitions – add all your knowledge sources here
# ----------------------------------------------------------------------
SOURCES = [
    {
        "type": "csv",
        "path": str(DATASET_DIR / "dataset.csv"),
        "name": "Original Dataset",
        "columns": ["prompt", "response"],
        "content_template": "Q: {prompt}\nA: {response}",
    },
    # Example additional CSV source (uncomment after adding the file)
    # {
    #     "type": "csv",
    #     "path": str(DATASET_DIR / "faq.csv"),
    #     "name": "FAQ",
    #     "columns": ["prompt", "response"],
    #     "content_template": "Q: {prompt}\nA: {response}",
    # },
    # Future RSS source example:
    # {
    #     "type": "rss",
    #     "url": "https://example.com/feed.xml",
    #     "name": "Company Blog",
    # },
]

# ----------------------------------------------------------------------
# Update interval (used only if you embed a scheduler inside the process)
# ----------------------------------------------------------------------
UPDATE_INTERVAL_HOURS = 1   # not used in standalone cron mode
