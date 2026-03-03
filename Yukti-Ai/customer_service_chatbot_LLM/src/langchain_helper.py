"""
Handles all vector store operations – embeddings, FAISS index creation,
retrieval, and optional re‑ranking. Includes comprehensive error handling
and detailed diagnostics.
"""

import os
import logging
from pathlib import Path
from typing import List, Optional, Dict, Any

import pandas as pd
from langchain_community.vectorstores import FAISS
from langchain_community.document_loaders import CSVLoader
from langchain_core.documents import Document

from config import DATASET_PATH, VECTORDB_PATH
from embeddings import get_embeddings

logger = logging.getLogger(__name__)

# Try multiple encodings for CSV
ENCODINGS = ['utf-8', 'cp1252', 'latin-1', 'iso-8859-1']

# ----------------------------------------------------------------------
# Core functions
# ----------------------------------------------------------------------
def create_vector_db() -> bool:
    """
    Build FAISS index from the main dataset CSV.
    Tries multiple encodings and returns success status.
    """
    if not DATASET_PATH.exists():
        logger.error(f"Dataset not found at {DATASET_PATH}")
        return False

    # Try each encoding until one works
    data = None
    used_encoding = None
    for enc in ENCODINGS:
        try:
            loader = CSVLoader(
                file_path=str(DATASET_PATH),
                source_column="prompt",
                encoding=enc
            )
            data = loader.load()
            used_encoding = enc
            logger.info(f"Loaded {len(data)} documents with encoding {enc}")
            break
        except UnicodeDecodeError:
            continue
        except Exception as e:
            logger.warning(f"Failed with encoding {enc}: {e}")
            continue
    if data is None:
        logger.error("Could not read CSV with any supported encoding.")
        return False

    try:
        embeddings = get_embeddings()
        vectordb = FAISS.from_documents(data, embeddings)
        vectordb.save_local(str(VECTORDB_PATH))
        logger.info(f"Index saved to {VECTORDB_PATH} (encoding: {used_encoding})")
        return True
    except Exception as e:
        logger.exception("FAISS build failed")
        return False

def load_vectorstore() -> Optional[FAISS]:
    """Load the FAISS index from disk."""
    if not VECTORDB_PATH.exists():
        logger.warning("FAISS index not found.")
        return None
    try:
        embeddings = get_embeddings()
        return FAISS.load_local(
            str(VECTORDB_PATH),
            embeddings,
            allow_dangerous_deserialization=True
        )
    except Exception as e:
        logger.exception("Failed to load vector store")
        return None

def retrieve_documents(query: str, k: int = 5) -> List[Document]:
    """Retrieve top k documents using similarity search."""
    vectordb = load_vectorstore()
    if vectordb is None:
        raise FileNotFoundError("Knowledge base not found. Please build it first.")
    try:
        return vectordb.similarity_search(query, k=k)
    except Exception as e:
        logger.exception("Similarity search failed")
        raise RuntimeError(f"Retrieval failed: {e}") from e

def check_kb_status() -> bool:
    """Return True if the FAISS index exists."""
    return VECTORDB_PATH.exists()

def get_document_count() -> Optional[int]:
    """Return the number of documents in the index, or None if not available."""
    vectordb = load_vectorstore()
    if vectordb:
        return vectordb.index.ntotal
    return None

# ----------------------------------------------------------------------
# Detailed status for admin dashboard
# ----------------------------------------------------------------------
def get_kb_detailed_status() -> Dict[str, Any]:
    """
    Returns a dictionary with detailed status of the knowledge base.
    Keys:
        - ready: bool (True if index exists and can be loaded)
        - error: str or None (if any error occurred during check)
        - path_exists: bool
        - index_loadable: bool (if FAISS can load without error)
        - document_count: int or None
        - dataset_exists: bool
        - dataset_readable: bool (if CSV can be read)
        - encoding_used: str or None
    """
    result = {
        "ready": False,
        "error": None,
        "path_exists": VECTORDB_PATH.exists(),
        "index_loadable": False,
        "document_count": None,
        "dataset_exists": DATASET_PATH.exists(),
        "dataset_readable": False,
        "encoding_used": None,
    }

    # Check dataset readability
    if result["dataset_exists"]:
        for enc in ENCODINGS:
            try:
                pd.read_csv(DATASET_PATH, encoding=enc, nrows=1)
                result["dataset_readable"] = True
                result["encoding_used"] = enc
                break
            except Exception:
                continue
        if not result["dataset_readable"]:
            result["error"] = f"Dataset exists but cannot be read with any tried encoding: {ENCODINGS}"

    # Try to load the index
    if result["path_exists"]:
        try:
            vectordb = load_vectorstore()
            if vectordb:
                result["index_loadable"] = True
                result["document_count"] = vectordb.index.ntotal
                result["ready"] = True
        except Exception as e:
            result["error"] = f"Index exists but cannot be loaded: {e}"
    else:
        if result["error"] is None:
            result["error"] = "Index directory does not exist."

    return result

# ----------------------------------------------------------------------
# Optional re‑ranking (if cross‑encoder available)
# ----------------------------------------------------------------------
try:
    from sentence_transformers import CrossEncoder
    CROSS_ENCODER_AVAILABLE = True
except ImportError:
    CROSS_ENCODER_AVAILABLE = False

def get_cross_encoder() -> Optional[object]:
    """Return a cross‑encoder model for re‑ranking, or None if not available."""
    if not CROSS_ENCODER_AVAILABLE:
        logger.debug("sentence-transformers not fully installed; re‑ranking disabled.")
        return None
    try:
        # Cache the model at module level (simple LRU)
        if not hasattr(get_cross_encoder, "_model"):
            logger.info("Loading cross‑encoder: cross-encoder/ms-marco-MiniLM-L-6-v2")
            get_cross_encoder._model = CrossEncoder("cross-encoder/ms-marco-MiniLM-L-6-v2")
        return get_cross_encoder._model
    except Exception as e:
        logger.warning(f"Failed to load cross‑encoder: {e}")
        return None

def retrieve_and_rerank(
    query: str,
    k: int = 5,
    rerank_top: int = 3,
    use_rerank: bool = True
) -> List[Document]:
    """Retrieve more documents and optionally re‑rank."""
    docs = retrieve_documents(query, k=k * 2)
    if not docs:
        return docs

    if not use_rerank:
        return docs[:k]

    cross_encoder = get_cross_encoder()
    if cross_encoder is None:
        return docs[:k]

    try:
        pairs = [[query, doc.page_content] for doc in docs]
        scores = cross_encoder.predict(pairs)
        scored_docs = sorted(zip(docs, scores), key=lambda x: x[1], reverse=True)
        return [doc for doc, _ in scored_docs[:rerank_top]]
    except Exception as e:
        logger.exception("Re‑ranking failed; falling back to basic retrieval.")
        return docs[:k]

# ----------------------------------------------------------------------
# Exports (explicit)
# ----------------------------------------------------------------------
__all__ = [
    "create_vector_db",
    "load_vectorstore",
    "retrieve_documents",
    "check_kb_status",
    "get_document_count",
    "get_kb_detailed_status",
    "retrieve_and_rerank",
]
