#!/usr/bin/env python3
"""
Standalone script to build the medical FAISS index.
Run manually or via cron.
"""

import sys
from pathlib import Path

# Add project root to path
BASE_DIR = Path(__file__).parent.parent.parent.absolute()
sys.path.insert(0, str(BASE_DIR))

from src.config import MEDICAL_DATASET_PATH, MEDICAL_VECTORDB_PATH
from src.embeddings import get_embeddings
from src.knowledge_updater.medical_loader import load_medquad
from langchain_community.vectorstores import FAISS
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def build_medical_index():
    """Load medical documents and build FAISS index."""
    logger.info("Building medical knowledge base...")
    docs = load_medquad(MEDICAL_DATASET_PATH)
    if not docs:
        logger.error("No documents loaded. Aborting.")
        return False

    embeddings = get_embeddings()
    vectordb = FAISS.from_documents(docs, embeddings)
    vectordb.save_local(str(MEDICAL_VECTORDB_PATH))
    logger.info(f"Medical index saved to {MEDICAL_VECTORDB_PATH} with {len(docs)} documents.")
    return True

if __name__ == "__main__":
    build_medical_index()