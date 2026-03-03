import os
import tempfile
import logging
import shutil
from pathlib import Path
from langchain_community.vectorstores import FAISS

from embeddings import get_embeddings  # shared loader
from config import SOURCES, VECTORDB_PATH
from . import connectors

logger = logging.getLogger(__name__)

def rebuild_index() -> bool:
    logger.info("Starting knowledge base rebuild...")
    all_docs = []

    for source in SOURCES:
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
        except Exception as e:
            logger.exception(f"Error fetching from {source.get('name', 'unknown')}")
            continue

    if not all_docs:
        logger.warning("No documents fetched – index not updated.")
        return False

    logger.info(f"Total documents: {len(all_docs)}")

    # Build index in a temporary directory
    with tempfile.TemporaryDirectory() as tmpdir:
        tmp_path = Path(tmpdir) / "index"
        try:
            embeddings = get_embeddings()
            vectordb = FAISS.from_documents(all_docs, embeddings)
            vectordb.save_local(str(tmp_path))
            logger.info(f"Index built in temporary location: {tmp_path}")

            # Atomic replace: move tmp to final destination
            if VECTORDB_PATH.exists():
                backup = VECTORDB_PATH.with_suffix(".bak")
                if backup.exists():
                    shutil.rmtree(backup)
                VECTORDB_PATH.rename(backup)
            shutil.move(str(tmp_path), str(VECTORDB_PATH))
            logger.info(f"Index atomically moved to {VECTORDB_PATH}")
            return True
        except Exception as e:
            logger.exception("Failed to build or save FAISS index.")
            return False
