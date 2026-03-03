import os
import time
import hashlib
import logging
from typing import List, Dict, Any

from langchain_core.documents import Document
from langchain_helper import retrieve_documents
from model_manager import load_model, get_model_config
from language_detector import detect_language
from language_utils import get_language_name
from config import CACHE_TTL

logger = logging.getLogger(__name__)

_retrieval_cache = {}

def _cache_key(query: str) -> str:
    return hashlib.md5(query.encode()).hexdigest()

def retrieve_cached(query: str, k: int = 3) -> List[Document]:
    cache_key = _cache_key(query)
    cached = _retrieval_cache.get(cache_key)
    if cached and (time.time() - cached['timestamp']) < CACHE_TTL:
        logger.debug("Retrieval cache hit")
        return cached['docs']
    try:
        docs = retrieve_documents(query, k=k)
        _retrieval_cache[cache_key] = {'docs': docs, 'timestamp': time.time()}
        logger.debug("Retrieval cache miss, stored")
        return docs
    except Exception as e:
        logger.exception("Retrieval error")
        raise

def detect_emotion(text: str) -> str:
    # Simple keyword expansion; could be replaced with a small model
    text_lower = text.lower()
    if any(w in text_lower for w in ["sad","unhappy","depressed","upset","heartbroken"]): return "sad"
    if any(w in text_lower for w in ["happy","excited","great","wonderful","joy","delighted"]): return "happy"
    if any(w in text_lower for w in ["angry","frustrated","annoyed","mad","irritated"]): return "angry"
    if any(w in text_lower for w in ["confused","lost","unsure","puzzled"]): return "confused"
    return "neutral"

def think(
    user_query: str,
    conversation_history: List[Dict[str, str]],
    model_key: str,
    language: str = None
) -> Dict[str, Any]:
    # ... (existing code, but modify fallback prompt)
    # ...
    try:
        response = llm.invoke(prompt, language=target_lang)
        # ...
    except Exception as e:
        logger.exception("LLM invocation failed, trying fallback prompt")
        # Enhanced fallback with language and emotion
        if target_lang == 'hinglish':
            lang_instr = "Hinglish mein jawab do."
        elif target_lang != 'en':
            lang_instr = f"Answer in {get_language_name(target_lang)}."
        else:
            lang_instr = ""
        fallback_prompt = f"{lang_instr}\nUser mood: {emotion}\nUser query: {user_query}\nAnswer concisely:"
        try:
            response = llm.invoke(fallback_prompt, language=target_lang)
            full_text = response.content if hasattr(response, 'content') else str(response)
        except Exception as e2:
            logger.exception("Fallback also failed")
            return {
                "type": "sync",
                "answer": f"Sorry, I couldn't generate an answer: {e2}",
                "monologue": "",
                "sources": docs,
                "thinking_time": time.time() - start_time,
                "emotion": emotion
            }
