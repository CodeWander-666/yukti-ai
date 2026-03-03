import logging
import sys
from functools import lru_cache

from langchain_community.embeddings import HuggingFaceEmbeddings

logger = logging.getLogger(__name__)

def get_embeddings(use_streamlit_cache: bool = False):
    """
    Load the sentence transformer embeddings model.
    If use_streamlit_cache is True and streamlit is available, cache the model.
    Otherwise, load normally (for cron jobs).
    """
    model_name = "sentence-transformers/all-MiniLM-L6-v2"
    try:
        if use_streamlit_cache and 'streamlit' in sys.modules:
            import streamlit as st
            @st.cache_resource
            def _load():
                logger.info("Loading embedding model with Streamlit cache")
                return HuggingFaceEmbeddings(
                    model_name=model_name,
                    encode_kwargs={'normalize_embeddings': True}
                )
            return _load()
        else:
            logger.info("Loading embedding model without cache")
            return HuggingFaceEmbeddings(
                model_name=model_name,
                encode_kwargs={'normalize_embeddings': True}
            )
    except Exception as e:
        logger.exception("Failed to load embedding model")
        raise RuntimeError(f"Embedding model unavailable: {e}")
