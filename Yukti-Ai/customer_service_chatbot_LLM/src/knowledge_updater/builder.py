"""
Core builder for FAISS index – fetches sources, deduplicates, and atomically replaces index.
Includes robust backup handling and cross-device move fix.
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
BASE_DIR = Path(__file__).parent.parent.parent.absolute()
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
    then atomically replace the old index. Handles cross-device moves by using copytree.
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

        # Atomically replace the old index (handle cross-device by using copytree)
        if VECTORDB_PATH.exists():
            backup = VECTORDB_PATH.with_suffix(".bak")
            if backup.exists():
                shutil.rmtree(backup)
                logger.info(f"Removed old backup {backup}")
            # Move current index to backup (using move, which may fail cross-device; fallback to copytree+rmtree)
            try:
                shutil.move(str(VECTORDB_PATH), str(backup))
            except OSError as e:
                # Cross-device move not possible; copy then delete
                logger.warning(f"Move failed ({e}), using copy+remove")
                if backup.exists():
                    shutil.rmtree(backup)
                shutil.copytree(str(VECTORDB_PATH), str(backup))
                shutil.rmtree(str(VECTORDB_PATH))
            logger.info(f"Backed up old index to {backup}")

        # Deploy new index (use copytree with dirs_exist_ok=True for cross-device)
        if VECTORDB_PATH.exists():
            shutil.rmtree(VECTORDB_PATH)
        shutil.copytree(str(tmp_path), str(VECTORDB_PATH), dirs_exist_ok=True)
        logger.info(f"New index deployed to {VECTORDB_PATH}")

    return True
