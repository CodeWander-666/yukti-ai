"""
Yukti AI – LangChain Helper (Ultimate Edition)
Handles embeddings, FAISS vector store, and document retrieval with optional re‑ranking.
"""

import os
import logging
from functools import lru_cache
from typing import List, Optional
from pathlib import Path

import streamlit as st
import pandas as pd
from dotenv import load_dotenv

from langchain_community.vectorstores import FAISS
from langchain_community.document_loaders import CSVLoader
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_core.documents import Document

try:
    from sentence_transformers import CrossEncoder
    CROSS_ENCODER_AVAILABLE = True
except ImportError:
    CROSS_ENCODER_AVAILABLE = False

load_dotenv()
logger = logging.getLogger(__name__)

BASE_DIR = Path(__file__).parent.parent.absolute()
DATASET_PATH = BASE_DIR / "dataset" / "dataset.csv"
VECTORDB_PATH = BASE_DIR / "faiss_index"

@st.cache_resource(show_spinner=False)
def get_embeddings():
    """Return a cached embedding model (lightning fast)."""
    try:
        logger.info("Loading embedding model: all-MiniLM-L6-v2")
        return HuggingFaceEmbeddings(
            model_name="sentence-transformers/all-MiniLM-L6-v2",
            encode_kwargs={'normalize_embeddings': True}
        )
    except Exception as e:
        logger.exception("Failed to load embedding model")
        st.error(f"❌ Embedding model unavailable: {e}")
        raise RuntimeError(f"Embedding model unavailable: {e}")

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

def create_vector_db() -> bool:
    """Build FAISS index from CSV with exhaustive error handling."""
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

def check_kb_status() -> bool:
    return VECTORDB_PATH.exists()

def get_document_count() -> Optional[int]:
    vectordb = load_vectorstore()
    if vectordb:
        return vectordb.index.ntotal
    return None
