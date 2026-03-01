"""
Yukti AI – Fast Thinking Engine
Single LLM call, emotion detection, retrieval caching, and conversation history.
"""

import logging
import time
import hashlib
from typing import List, Dict, Any

from langchain_core.documents import Document
from langchain_helper import retrieve_documents
from model_manager import load_model, get_model_config

logger = logging.getLogger(__name__)

# Simple retrieval cache (1 hour TTL)
_retrieval_cache = {}
CACHE_TTL = 3600

def _cache_key(query: str) -> str:
    return hashlib.md5(query.encode()).hexdigest()

def retrieve_cached(query: str, k: int = 3) -> List[Document]:
    cache_key = _cache_key(query)
    cached = _retrieval_cache.get(cache_key)
    if cached and (time.time() - cached['timestamp']) < CACHE_TTL:
        logger.info("Retrieval cache hit")
        return cached['docs']
    try:
        docs = retrieve_documents(query, k=k)
        _retrieval_cache[cache_key] = {'docs': docs, 'timestamp': time.time()}
        return docs
    except Exception as e:
        logger.exception("Retrieval error")
        raise

def detect_emotion(text: str) -> str:
    text_lower = text.lower()
    if any(w in text_lower for w in ["sad","unhappy","depressed","upset"]): return "sad"
    if any(w in text_lower for w in ["happy","excited","great","wonderful"]): return "happy"
    if any(w in text_lower for w in ["angry","frustrated","annoyed","mad"]): return "angry"
    if any(w in text_lower for w in ["confused","lost","unsure"]): return "confused"
    return "neutral"

def think(
    user_query: str,
    conversation_history: List[Dict[str, str]],
    model_key: str
) -> Dict[str, Any]:
    config = get_model_config(model_key)
    if not config or config["type"] != "sync":
        raise ValueError("think() requires a sync model")

    start_time = time.time()
    llm = load_model(model_key)

    emotion = detect_emotion(user_query)

    try:
        docs = retrieve_cached(user_query, k=3)
    except FileNotFoundError:
        return {
            "type": "sync",
            "answer": "Knowledge base not ready. Please update first.",
            "monologue": "",
            "sources": [],
            "thinking_time": 0.0,
            "emotion": emotion
        }
    except Exception as e:
        return {
            "type": "sync",
            "answer": f"Retrieval error: {e}",
            "monologue": "",
            "sources": [],
            "thinking_time": time.time() - start_time,
            "emotion": emotion
        }

    context = "\n\n".join([doc.page_content for doc in docs])
    history_str = "\n".join(
        f"{'User' if t['role']=='user' else 'Yukti'}: {t['content']}"
        for t in conversation_history[-5:]
    )

    # Single prompt combining all tasks
    prompt = f"""You are Yukti AI, a helpful assistant with emotional awareness.
Conversation history:
{history_str}

Relevant information:
{context}

User mood: {emotion}
User query: {user_query}

Think step by step, then answer. Format:
MONOLOGUE:
(Your reasoning)
ANSWER:
(Your final answer)"""

    try:
        response = llm.invoke(prompt)
        full_text = response.content if hasattr(response, 'content') else str(response)
    except Exception as e:
        logger.exception("LLM invocation failed")
        return {
            "type": "sync",
            "answer": f"Sorry, I couldn't answer: {e}",
            "monologue": "",
            "sources": docs,
            "thinking_time": time.time() - start_time,
            "emotion": emotion
        }

    # Parse monologue and answer
    monologue, answer = "", full_text
    if "MONOLOGUE:" in full_text and "ANSWER:" in full_text:
        parts = full_text.split("MONOLOGUE:", 1)[1]
        if "ANSWER:" in parts:
            monologue, answer = parts.split("ANSWER:", 1)
            monologue = monologue.strip()
            answer = answer.strip()

    return {
        "type": "sync",
        "answer": answer,
        "monologue": monologue,
        "sources": docs,
        "thinking_time": time.time() - start_time,
        "emotion": emotion
    }
