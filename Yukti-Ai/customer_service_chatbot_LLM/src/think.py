"""
Core reasoning engine – combines retrieval, emotion detection, language awareness,
and LLM invocation in a single call. Includes caching and comprehensive error handling.
"""

import os
import time
import hashlib
import logging
from typing import List, Dict, Any, Optional

from langchain_core.documents import Document

# Local imports
from langchain_helper import retrieve_documents
from model_manager import load_model, get_model_config
from language_detector import detect_language
from language_utils import get_language_name
from config import CACHE_TTL

logger = logging.getLogger(__name__)

# ----------------------------------------------------------------------
# Retrieval Cache (in‑memory with TTL)
# ----------------------------------------------------------------------
_retrieval_cache: Dict[str, Dict] = {}

def _cache_key(query: str) -> str:
    """Generate a cache key from the query string."""
    return hashlib.md5(query.encode('utf-8')).hexdigest()

def retrieve_cached(query: str, k: int = 3) -> List[Document]:
    """
    Retrieve documents with caching. Returns cached results if fresh.
    Raises FileNotFoundError if knowledge base not built.
    """
    cache_key = _cache_key(query)
    cached = _retrieval_cache.get(cache_key)
    now = time.time()

    if cached and (now - cached['timestamp']) < CACHE_TTL:
        logger.debug(f"Retrieval cache hit for query: {query[:50]}...")
        return cached['docs']

    try:
        docs = retrieve_documents(query, k=k)
        _retrieval_cache[cache_key] = {'docs': docs, 'timestamp': now}
        logger.debug(f"Retrieval cache miss, stored for query: {query[:50]}...")
        return docs
    except FileNotFoundError:
        # Re-raise to let caller handle missing knowledge base
        raise
    except Exception as e:
        logger.exception(f"Retrieval error for query: {query[:50]}...")
        raise RuntimeError(f"Document retrieval failed: {e}") from e

# ----------------------------------------------------------------------
# Emotion Detection (simple keyword‑based; can be upgraded later)
# ----------------------------------------------------------------------
def detect_emotion(text: str) -> str:
    """
    Detect user emotion based on keywords.
    Returns one of: sad, happy, angry, confused, neutral.
    """
    text_lower = text.lower()
    if any(w in text_lower for w in ["sad", "unhappy", "depressed", "upset", "heartbroken"]):
        return "sad"
    if any(w in text_lower for w in ["happy", "excited", "great", "wonderful", "joy", "delighted"]):
        return "happy"
    if any(w in text_lower for w in ["angry", "frustrated", "annoyed", "mad", "irritated"]):
        return "angry"
    if any(w in text_lower for w in ["confused", "lost", "unsure", "puzzled"]):
        return "confused"
    return "neutral"

# ----------------------------------------------------------------------
# Main Thinking Function
# ----------------------------------------------------------------------
def think(
    user_query: str,
    conversation_history: List[Dict[str, str]],
    model_key: str,
    language: Optional[str] = None
) -> Dict[str, Any]:
    """
    Generate a response using retrieval‑augmented generation.

    Args:
        user_query: The current user input.
        conversation_history: List of previous messages (role, content).
        model_key: The service name (e.g., "Yukti‑Flash").
        language: Optional override; if None, auto‑detect from query.

    Returns:
        Dict with keys:
            type: "sync"
            answer: The generated answer.
            monologue: Reasoning steps (if any).
            sources: List of retrieved documents.
            thinking_time: Time taken in seconds.
            emotion: Detected emotion.
    """
    start_time = time.time()

    # ------------------------------------------------------------------
    # 1. Validate model and get configuration
    # ------------------------------------------------------------------
    config = get_model_config(model_key)
    if not config:
        error_msg = f"Unknown or unavailable model key: {model_key}"
        logger.error(error_msg)
        return {
            "type": "sync",
            "answer": f"Error: {error_msg}",
            "monologue": "",
            "sources": [],
            "thinking_time": 0.0,
            "emotion": "neutral"
        }

    if config.get("type") != "sync":
        error_msg = f"think() requires a sync model, got {model_key} (type: {config.get('type')})"
        logger.error(error_msg)
        return {
            "type": "sync",
            "answer": f"Error: {error_msg}",
            "monologue": "",
            "sources": [],
            "thinking_time": 0.0,
            "emotion": "neutral"
        }

    # ------------------------------------------------------------------
    # 2. Detect emotion
    # ------------------------------------------------------------------
    emotion = detect_emotion(user_query)
    logger.debug(f"Detected emotion: {emotion}")

    # ------------------------------------------------------------------
    # 3. Determine target language
    # ------------------------------------------------------------------
    if language:
        target_lang = language
        method = 'provided'
    else:
        try:
            lang_info = detect_language(user_query)
            target_lang = lang_info['language']
            method = lang_info['method']
        except Exception as e:
            logger.warning(f"Language detection failed, defaulting to English: {e}")
            target_lang = 'en'
            method = 'fallback'
    logger.info(f"Target language: {target_lang} (method: {method})")

    # ------------------------------------------------------------------
    # 4. Retrieve relevant documents (with caching)
    # ------------------------------------------------------------------
    try:
        docs = retrieve_cached(user_query, k=3)
    except FileNotFoundError:
        return {
            "type": "sync",
            "answer": "Knowledge base not ready. Please update it first.",
            "monologue": "",
            "sources": [],
            "thinking_time": time.time() - start_time,
            "emotion": emotion
        }
    except Exception as e:
        logger.exception("Retrieval error")
        return {
            "type": "sync",
            "answer": f"Sorry, I encountered an error while searching: {e}",
            "monologue": "",
            "sources": [],
            "thinking_time": time.time() - start_time,
            "emotion": emotion
        }

    # ------------------------------------------------------------------
    # 5. Build context and prompt
    # ------------------------------------------------------------------
    context = "\n\n".join([doc.page_content for doc in docs])

    # Format conversation history (last 5 exchanges)
    history_str = "\n".join(
        f"{'User' if msg['role'] == 'user' else 'Yukti'}: {msg['content']}"
        for msg in conversation_history[-5:]
    )

    # Language instruction
    if target_lang == 'hinglish':
        lang_instruction = "Answer in Hinglish (mix of Hindi and English)."
    elif target_lang != 'en':
        lang_name = get_language_name(target_lang)
        lang_instruction = f"Answer in {lang_name}."
    else:
        lang_instruction = ""

    prompt = f"""You are Yukti AI, a helpful assistant.
{lang_instruction}
Conversation history:
{history_str}

Relevant information:
{context}

User mood: {emotion}
User query: {user_query}

Think step by step, then answer. Format your response as:
MONOLOGUE:
(Your reasoning)
ANSWER:
(Your final answer)"""

    # ------------------------------------------------------------------
    # 6. Invoke the language model
    # ------------------------------------------------------------------
    llm = load_model(model_key)
    answer = ""
    monologue = ""
    full_text = ""

    try:
        # Pass language info to model via kwargs
        response = llm.invoke(prompt, language=target_lang)
        # The invoke method may return a string directly or an object with .content
        if hasattr(response, 'content'):
            full_text = response.content
        else:
            full_text = str(response)
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
            response2 = llm.invoke(fallback_prompt, language=target_lang)
            if hasattr(response2, 'content'):
                full_text = response2.content
            else:
                full_text = str(response2)
        except Exception as e2:
            logger.exception("Fallback also failed")
            return {
                "type": "sync",
                "answer": f"Sorry, I couldn't generate an answer due to a technical issue.",
                "monologue": "",
                "sources": docs,
                "thinking_time": time.time() - start_time,
                "emotion": emotion
            }

    # ------------------------------------------------------------------
    # 7. Parse monologue and answer (if format is followed)
    # ------------------------------------------------------------------
    if "MONOLOGUE:" in full_text and "ANSWER:" in full_text:
        try:
            parts = full_text.split("MONOLOGUE:", 1)[1]
            if "ANSWER:" in parts:
                monologue_part, answer_part = parts.split("ANSWER:", 1)
                monologue = monologue_part.strip()
                answer = answer_part.strip()
        except Exception as e:
            logger.warning(f"Failed to parse monologue/answer: {e}")
            answer = full_text.strip()
    else:
        answer = full_text.strip()

    if not answer:
        answer = "(No response generated)"
        logger.warning("Empty answer from LLM")

    # ------------------------------------------------------------------
    # 8. Return result
    # ------------------------------------------------------------------
    thinking_time = time.time() - start_time
    logger.info(f"Thought for {thinking_time:.2f}s, emotion: {emotion}, lang: {target_lang}")

    return {
        "type": "sync",
        "answer": answer,
        "monologue": monologue,
        "sources": docs,
        "thinking_time": thinking_time,
        "emotion": emotion
    }
