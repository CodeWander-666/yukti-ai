import os
import logging
from pathlib import Path
from typing import List, Optional

import streamlit as st
import pandas as pd
from langchain_community.vectorstores import FAISS
from langchain_community.document_loaders import CSVLoader
from langchain_core.documents import Document

from embeddings import get_embeddings  # shared embedding loader
from config import VECTORDB_PATH, BASE_DIR

logger = logging.getLogger(__name__)

# Try multiple encodings for CSVLoader
ENCODINGS = ['utf-8', 'cp1252', 'latin-1', 'iso-8859-1']

def create_vector_db() -> bool:
    """Build FAISS index from CSV with encoding fallback."""
    csv_path = BASE_DIR / "dataset" / "dataset.csv"
    if not csv_path.exists():
        st.error(f"❌ Dataset not found at {csv_path}")
        logger.error(f"Dataset missing: {csv_path}")
        return False

    # Try each encoding until one works
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

# Optional: cross‑encoder remains unchanged (omitted for brevity)
# ... (rest of the file stays the same)
