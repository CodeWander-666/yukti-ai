"""
Core builder for FAISS index – fetches sources, deduplicates, and atomically replaces index.
Includes robust backup handling.
"""

import logging
import shutil
import sys
import tempfile
from pathlib import Path
from typing import List

from langchain_community.vectorstores import FAISS
from langchain_core.documents import Document

from .connectors import fetch_all_sources
from .config import VECTORDB_PATH

# Add project root to path to import src.embeddings
BASE_DIR = Path(__file__).parent.parent.absolute()
if str(BASE_DIR) not in sys.path:
    sys.path.insert(0, str(BASE_DIR))

from src.embeddings import get_embeddings

logger = logging.getLogger(__name__)

def deduplicate_documents(docs: List[Document]) -> List[Document]:
    """Simple deduplication based on page_content hash."""
    seen = set()
    unique = []
    for doc in docs:
        h = hash(doc.page_content)
        if h not in seen:
            seen.add(h)
            unique.append(doc)
    logger.info(f"Deduplicated: {len(docs)} -> {len(unique)} documents")
    return unique

def rebuild_index() -> bool:
    """
    Fetch all documents, build a new FAISS index in a temporary directory,
    then atomically replace the old index. Removes any existing backup first.
    """
    logger.info("Starting knowledge base rebuild...")

    # Fetch documents
    docs = fetch_all_sources()
    if not docs:
        logger.warning("No documents fetched. Index unchanged.")
        return False

    # Deduplicate
    docs = deduplicate_documents(docs)

    # Load embeddings
    embeddings = get_embeddings()

    # Build index in temporary directory
    with tempfile.TemporaryDirectory() as tmpdir:
        tmp_path = Path(tmpdir) / "faiss_index"
        try:
            vectordb = FAISS.from_documents(docs, embeddings)
            vectordb.save_local(str(tmp_path))
            logger.info(f"Temporary index built at {tmp_path}")
        except Exception as e:
            logger.exception("FAISS build failed")
            return False

        # Atomically replace the old index
        if VECTORDB_PATH.exists():
            backup = VECTORDB_PATH.with_suffix(".bak")
            # Remove any existing backup
            if backup.exists():
                shutil.rmtree(backup)
                logger.info(f"Removed old backup {backup}")
            shutil.move(str(VECTORDB_PATH), str(backup))
            logger.info(f"Backed up old index to {backup}")

        shutil.move(str(tmp_path), str(VECTORDB_PATH))
        logger.info(f"New index deployed to {VECTORDB_PATH}")

    return True
