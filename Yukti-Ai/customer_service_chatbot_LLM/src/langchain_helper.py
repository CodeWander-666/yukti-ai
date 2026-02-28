"""
Yukti AI – LangChain Helper (Ultimate Edition)
Industry‑grade knowledge base engine with caching, re‑ranking, and exhaustive error handling.
"""

import os
import logging
from functools import lru_cache
from typing import List, Optional, Union
from pathlib import Path

import streamlit as st
import pandas as pd
from dotenv import load_dotenv

# LangChain imports
from langchain_community.vectorstores import FAISS
from langchain_community.document_loaders import CSVLoader
from langchain_community.embeddings import HuggingFaceInstructEmbeddings
from langchain_core.documents import Document

# Optional: sentence-transformers for cross-encoder
try:
    from sentence_transformers import CrossEncoder
    CROSS_ENCODER_AVAILABLE = True
except ImportError:
    CROSS_ENCODER_AVAILABLE = False

load_dotenv()
logger = logging.getLogger(__name__)

# ----------------------------------------------------------------------
# Paths (absolute – works on Streamlit Cloud)
# ----------------------------------------------------------------------
BASE_DIR = Path(__file__).parent.parent.absolute()
DATASET_PATH = BASE_DIR / "dataset" / "dataset.csv"
VECTORDB_PATH = BASE_DIR / "faiss_index"

# ----------------------------------------------------------------------
# Cached Embedding Model
# ----------------------------------------------------------------------
@st.cache_resource(show_spinner=False)
def get_embeddings():
    """Return a cached embedding model (lightning fast)."""
    try:
        logger.info("Loading embedding model: all-MiniLM-L6-v2")
        return HuggingFaceInstructEmbeddings(
            model_name="sentence-transformers/all-MiniLM-L6-v2",
            encode_kwargs={'normalize_embeddings': True}
        )
    except Exception as e:
        logger.exception("Failed to load embedding model")
        st.error(f"❌ Embedding model unavailable: {e}")
        raise RuntimeError(f"Embedding model unavailable: {e}")

# ----------------------------------------------------------------------
# Cached Cross‑Encoder for Re‑ranking (optional)
# ----------------------------------------------------------------------
@st.cache_resource(show_spinner=False)
def get_cross_encoder() -> Optional[object]:
    """
    Return a cached cross‑encoder model for re‑ranking.
    Returns None if not available (graceful fallback).
    """
    if not CROSS_ENCODER_AVAILABLE:
        logger.warning("sentence-transformers not fully installed; re‑ranking disabled.")
        return None
    try:
        logger.info("Loading cross‑encoder: cross-encoder/ms-marco-MiniLM-L-6-v2")
        return CrossEncoder("cross-encoder/ms-marco-MiniLM-L-6-v2")
    except Exception as e:
        logger.warning(f"Failed to load cross‑encoder: {e}")
        return None

# ----------------------------------------------------------------------
# Vector Database Operations
# ----------------------------------------------------------------------
def create_vector_db() -> bool:
    """
    Build FAISS index from CSV with exhaustive error handling.
    Returns True on success, False on failure (user already sees error messages).
    """
    if not DATASET_PATH.exists():
        st.error(f"❌ Dataset not found at {DATASET_PATH}")
        logger.error(f"Dataset missing: {DATASET_PATH}")
        return False

    logger.info(f"Loading CSV from {DATASET_PATH}")
    try:
        loader = CSVLoader(
            file_path=str(DATASET_PATH),
            source_column="prompt",
            encoding="utf-8"
        )
        data = loader.load()
        logger.info(f"Loaded {len(data)} documents")
    except Exception as e:
        st.error(f"❌ Failed to load CSV: {e}")
        logger.exception("CSV loading failed")
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
    """
    Load the FAISS index from disk.
    Returns None if index does not exist or cannot be loaded.
    """
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

# ----------------------------------------------------------------------
# Basic Retrieval
# ----------------------------------------------------------------------
def retrieve_documents(query: str, k: int = 5) -> List[Document]:
    """
    Retrieve top k documents using similarity search.
    Raises FileNotFoundError if index missing.
    """
    vectordb = load_vectorstore()
    if vectordb is None:
        raise FileNotFoundError("Knowledge base not found. Please build it first.")
    try:
        return vectordb.similarity_search(query, k=k)
    except Exception as e:
        logger.exception("Similarity search failed")
        raise RuntimeError(f"Retrieval failed: {e}")

# ----------------------------------------------------------------------
# Advanced Retrieval with Re‑ranking
# ----------------------------------------------------------------------
def retrieve_and_rerank(
    query: str,
    k: int = 5,
    rerank_top: int = 3,
    use_rerank: bool = True
) -> List[Document]:
    """
    Retrieve more documents (k * 2) and optionally re‑rank using a cross‑encoder.
    Returns top `rerank_top` documents after re‑ranking, or top k if re‑rank disabled/not available.
    """
    # First, retrieve more documents (k * 2) for better coverage
    docs = retrieve_documents(query, k=k * 2)
    if not docs:
        return docs

    if not use_rerank:
        return docs[:k]

    cross_encoder = get_cross_encoder()
    if cross_encoder is None:
        logger.info("Cross‑encoder not available; falling back to basic retrieval.")
        return docs[:k]

    try:
        # Prepare pairs for cross‑encoder
        pairs = [[query, doc.page_content] for doc in docs]
        scores = cross_encoder.predict(pairs)
        # Sort documents by score descending
        scored_docs = sorted(zip(docs, scores), key=lambda x: x[1], reverse=True)
        # Return top rerank_top
        return [doc for doc, _ in scored_docs[:rerank_top]]
    except Exception as e:
        logger.exception("Re‑ranking failed; falling back to basic retrieval.")
        return docs[:k]

# ----------------------------------------------------------------------
# Utility Functions
# ----------------------------------------------------------------------
def check_kb_status() -> bool:
    """Return True if FAISS index exists."""
    return VECTORDB_PATH.exists()

def get_document_count() -> Optional[int]:
    """Return number of documents in the index, or None if not loaded."""
    vectordb = load_vectorstore()
    if vectordb:
        return vectordb.index.ntotal
    return None

# ----------------------------------------------------------------------
# For standalone testing (optional)
# ----------------------------------------------------------------------
if __name__ == "__main__":
    print("Testing knowledge base status...")
    if check_kb_status():
        print(f"Index exists at {VECTORDB_PATH}")
        count = get_document_count()
        if count is not None:
            print(f"Document count: {count}")
    else:
        print("No index found. You can create one by calling create_vector_db()")
