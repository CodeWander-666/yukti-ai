"""
Pure embedding loader – no Streamlit dependencies.
Used by both the main app and the knowledge updater.
"""

import logging
from functools import lru_cache

from langchain_community.embeddings import HuggingFaceEmbeddings

logger = logging.getLogger(__name__)

# ----------------------------------------------------------------------
# Embedding model – cached to avoid reloading
# ----------------------------------------------------------------------
@lru_cache(maxsize=1)
def get_embeddings():
    """
    Return a HuggingFace embedding model (all-MiniLM-L6-v2).
    The result is cached using lru_cache to ensure only one instance is created.
    This function can be called from both Streamlit and non‑Streamlit environments.
    """
    try:
        logger.info("Loading embedding model: sentence-transformers/all-MiniLM-L6-v2")
        embeddings = HuggingFaceEmbeddings(
            model_name="sentence-transformers/all-MiniLM-L6-v2",
            model_kwargs={'device': 'cpu'},          # force CPU to avoid GPU issues
            encode_kwargs={'normalize_embeddings': True}
        )
        # Quick test to ensure model loads
        _ = embeddings.embed_query("test")
        logger.info("Embedding model loaded successfully.")
        return embeddings
    except Exception as e:
        logger.exception("Failed to load embedding model")
        raise RuntimeError(f"Embedding model unavailable: {e}") from e
