"""
Rebuild FAISS index from all configured sources.
Includes deduplication, atomic replacement, and comprehensive logging.
"""

import os
import time
import logging
import tempfile
import shutil
from pathlib import Path
from typing import List, Optional

import numpy as np
from langchain_community.vectorstores import FAISS
from langchain_core.documents import Document

from embeddings import get_embeddings
from .config import SOURCES, VECTORDB_PATH
from . import connectors

logger = logging.getLogger(__name__)

# Deduplication threshold (cosine similarity)
DEDUP_THRESHOLD = 0.9

def cosine_similarity(a: List[float], b: List[float]) -> float:
    """Compute cosine similarity between two vectors."""
    a = np.array(a)
    b = np.array(b)
    norm_a = np.linalg.norm(a)
    norm_b = np.linalg.norm(b)
    if norm_a == 0 or norm_b == 0:
        return 0.0
    return float(np.dot(a, b) / (norm_a * norm_b))

def is_duplicate(
    new_doc: Document,
    existing_docs: List[Document],
    embeddings,
    threshold: float = DUP_THRESHOLD
) -> bool:
    """Check if new_doc is too similar to any existing document."""
    if not existing_docs:
        return False
    try:
        new_emb = embeddings.embed_query(new_doc.page_content)
        for doc in existing_docs:
            emb = embeddings.embed_query(doc.page_content)
            if cosine_similarity(new_emb, emb) > threshold:
                return True
    except Exception as e:
        logger.warning(f"Error during deduplication check: {e}")
        # If error, assume not duplicate to avoid losing data
        return False
    return False

def rebuild_index() -> bool:
    """
    Rebuild the FAISS index from all configured sources.
    Returns True on success, False on failure.
    """
    start_time = time.time()
    logger.info("Starting knowledge base rebuild...")
    all_docs = []

    # ------------------------------------------------------------------
    # 1. Collect documents from enabled sources
    # ------------------------------------------------------------------
    for source in SOURCES:
        if not source.get("enabled", True):
            logger.info(f"Skipping disabled source: {source.get('name')}")
            continue

        src_type = source.get("type")
        try:
            if src_type == "csv":
                docs = connectors.fetch_csv(source)
            elif src_type == "rss":
                docs = connectors.fetch_rss(source)
            elif src_type == "api":
                docs = connectors.fetch_api(source)
            else:
                logger.warning(f"Unknown source type '{src_type}' – skipping")
                continue

            all_docs.extend(docs)
            logger.info(f"Added {len(docs)} documents from {source.get('name')}")
        except Exception as e:
            logger.exception(f"Error fetching from {source.get('name', 'unknown')}")
            # Continue with other sources; don't let one source break the whole rebuild
            continue

    if not all_docs:
        logger.warning("No documents fetched – index not updated.")
        return False

    logger.info(f"Total documents fetched: {len(all_docs)}")

    # ------------------------------------------------------------------
    # 2. Load existing index (if any) for deduplication
    # ------------------------------------------------------------------
    embeddings = get_embeddings()
    existing_docs = []
    if VECTORDB_PATH.exists():
        try:
            old_db = FAISS.load_local(
                str(VECTORDB_PATH),
                embeddings,
                allow_dangerous_deserialization=True
            )
            # Extract all documents from the index
            # Note: This assumes docstore is accessible; if not, we skip dedup
            if hasattr(old_db, 'docstore') and hasattr(old_db.docstore, '_dict'):
                existing_docs = list(old_db.docstore._dict.values())
                logger.info(f"Loaded {len(existing_docs)} existing documents for deduplication.")
            else:
                logger.warning("Could not access docstore; deduplication against existing docs disabled.")
        except Exception as e:
            logger.warning(f"Could not load existing index for deduplication: {e}")
            # Proceed without deduplication against existing docs

    # ------------------------------------------------------------------
    # 3. Deduplicate new documents (against existing and among themselves)
    # ------------------------------------------------------------------
    unique_new_docs = []
    duplicate_count = 0
    for doc in all_docs:
        if is_duplicate(doc, existing_docs + unique_new_docs, embeddings):
            duplicate_count += 1
            logger.debug(f"Duplicate skipped: {doc.page_content[:50]}...")
        else:
            unique_new_docs.append(doc)

    if duplicate_count > 0:
        logger.info(f"Skipped {duplicate_count} duplicate documents.")
    if not unique_new_docs:
        logger.info("No new unique documents to add.")
        return True

    logger.info(f"After deduplication: {len(unique_new_docs)} new documents.")

    # ------------------------------------------------------------------
    # 4. Build index in temporary directory, then atomically replace
    # ------------------------------------------------------------------
    with tempfile.TemporaryDirectory() as tmpdir:
        tmp_path = Path(tmpdir) / "index"
        try:
            if existing_docs:
                # Combine old and new, then rebuild entire index
                combined = existing_docs + unique_new_docs
                vectordb = FAISS.from_documents(combined, embeddings)
                logger.info(f"Rebuilt index with {len(combined)} total documents.")
            else:
                # First build or no existing index
                vectordb = FAISS.from_documents(unique_new_docs, embeddings)
                logger.info(f"Built new index with {len(unique_new_docs)} documents.")

            vectordb.save_local(str(tmp_path))
            logger.info(f"Index built in temporary location: {tmp_path}")

            # Atomic replace: move temp dir to final destination
            if VECTORDB_PATH.exists():
                backup = VECTORDB_PATH.with_suffix(".bak")
                if backup.exists():
                    shutil.rmtree(backup)
                VECTORDB_PATH.rename(backup)
                logger.info(f"Existing index backed up to {backup}")

            shutil.move(str(tmp_path), str(VECTORDB_PATH))
            logger.info(f"Index atomically moved to {VECTORDB_PATH}")

            # Optionally remove backup if everything succeeded
            backup = VECTORDB_PATH.with_suffix(".bak")
            if backup.exists():
                shutil.rmtree(backup)
                logger.info("Backup removed after successful update.")

            elapsed = time.time() - start_time
            logger.info(f"Knowledge base rebuild completed in {elapsed:.2f} seconds.")
            return True

        except Exception as e:
            logger.exception("Failed to build or save FAISS index.")
            # Attempt to restore from backup if it exists
            backup = VECTORDB_PATH.with_suffix(".bak")
            if backup.exists() and not VECTORDB_PATH.exists():
                shutil.move(str(backup), str(VECTORDB_PATH))
                logger.info("Restored from backup after failure.")
            return False
