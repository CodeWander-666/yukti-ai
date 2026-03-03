"""
Core builder for FAISS index – fetches sources, deduplicates, and atomically replaces index.
"""

import logging
import shutil
import tempfile
from pathlib import Path
from typing import List

from langchain_community.vectorstores import FAISS
from langchain_core.documents import Document

from .connectors import fetch_all_sources
from .config import VECTORDB_PATH
from src.embeddings import get_embeddings  # reuse embedding loader

logger = logging.getLogger(__name__)

def deduplicate_documents(docs: List[Document]) -> List[Document]:
    """Simple deduplication based on page_content hash."""
    seen = set()
    unique = []
    for doc in docs:
        # Use content hash as key
        h = hash(doc.page_content)
        if h not in seen:
            seen.add(h)
            unique.append(doc)
    logger.info(f"Deduplicated: {len(docs)} -> {len(unique)} documents")
    return unique

def rebuild_index() -> bool:
    """
    Fetch all documents, build a new FAISS index in a temporary directory,
    then atomically replace the old index.
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
            shutil.move(str(VECTORDB_PATH), str(backup))
            logger.info(f"Backed up old index to {backup}")

        shutil.move(str(tmp_path), str(VECTORDB_PATH))
        logger.info(f"New index deployed to {VECTORDB_PATH}")

    # Clean up uploads folder (optional) – move processed files to archive
    # For simplicity, we keep them; you may implement archiving later.

    return True
