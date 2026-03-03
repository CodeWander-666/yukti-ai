"""
Handles all vector store operations – embeddings, FAISS index creation,
retrieval, and optional re‑ranking. Includes robust error handling and
encoding fallbacks.
"""

import os
import logging
from pathlib import Path
from typing import List, Optional

import streamlit as st
import pandas as pd
from langchain_community.vectorstores import FAISS
from langchain_community.document_loaders import CSVLoader
from langchain_core.documents import Document

# Use shared embedding loader (if available), otherwise fallback
try:
    from embeddings import get_embeddings
except ImportError:
    # Fallback: define a simple embedding loader (should not happen in normal setup)
    from langchain_community.embeddings import HuggingFaceEmbeddings
    def get_embeddings():
        return HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

from config import VECTORDB_PATH, BASE_DIR

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
    csv_path = BASE_DIR / "dataset" / "dataset.csv"
    if not csv_path.exists():
        st.error(f"❌ Dataset not found at {csv_path}")
        logger.error(f"Dataset missing: {csv_path}")
        return False

    # Try each encoding until one works
    data = None
    for enc in ENCODINGS:
        try:
            loader = CSVLoader(
                file_path=str(csv_path),
                source_column="prompt",
                encoding=enc
            )
            data = loader.load()
            logger.info(f"Loaded {len(data)} documents with encoding {enc}")
            break
        except UnicodeDecodeError:
            continue
        except Exception as e:
            logger.warning(f"Failed with encoding {enc}: {e}")
            continue
    else:
        st.error("❌ Could not read CSV with any supported encoding.")
        logger.error("All encodings failed for dataset.csv")
        return False

    try:
        embeddings = get_embeddings()
        vectordb = FAISS.from_documents(data, embeddings)
        vectordb.save_local(str(VECTORDB_PATH))
        logger.info(f"Index saved to {VECTORDB_PATH}")
        st.success(f"✅ Knowledge base rebuilt with {len(data)} documents!")
        return True
    except Exception as e:
        st.error(f"❌ Failed to build index: {e}")
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
        st.error(f"❌ Failed to load knowledge base: {e}")
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
        raise RuntimeError(f"Retrieval failed: {e}")

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
# Optional re‑ranking (if cross‑encoder available)
# ----------------------------------------------------------------------
try:
    from sentence_transformers import CrossEncoder
    CROSS_ENCODER_AVAILABLE = True
except ImportError:
    CROSS_ENCODER_AVAILABLE = False

@st.cache_resource(show_spinner=False)
def get_cross_encoder() -> Optional[object]:
    """Return a cached cross‑encoder model for re‑ranking."""
    if not CROSS_ENCODER_AVAILABLE:
        logger.warning("sentence-transformers not fully installed; re‑ranking disabled.")
        return None
    try:
        logger.info("Loading cross‑encoder: cross-encoder/ms-marco-MiniLM-L-6-v2")
        return CrossEncoder("cross-encoder/ms-marco-MiniLM-L-6-v2")
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
# Exported symbols
# ----------------------------------------------------------------------
__all__ = [
    "create_vector_db",
    "load_vectorstore",
    "retrieve_documents",
    "check_kb_status",
    "get_document_count",
    "get_embeddings",
    "retrieve_and_rerank",
    "VECTORDB_PATH",
    "BASE_DIR",
]
