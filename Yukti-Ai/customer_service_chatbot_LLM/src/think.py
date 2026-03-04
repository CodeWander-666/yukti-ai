"""
Core reasoning engine – combines retrieval, emotion detection, language awareness,
and LLM invocation in a single call. Also supports on‑demand web scraping for
real‑time information (e.g., product searches, news).
Includes caching and comprehensive error handling.
"""

import re
import time
import hashlib
import logging
from typing import List, Dict, Any, Optional

from langchain_core.documents import Document

# Local imports
from langchain_helper import retrieve_documents
from model_manager import load_model, get_model_config
from language_detector import detect_language, get_response_language
from language_utils import get_language_name
from config import RETRIEVAL_CACHE_TTL

# Import the scraper – will be None if not available
try:
    from knowledge_updater.connectors import scrape_url
    SCRAPER_AVAILABLE = True
except ImportError:
    SCRAPER_AVAILABLE = False
    def scrape_url(*args, **kwargs): return None

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

    if cached and (now - cached['timestamp']) < RETRIEVAL_CACHE_TTL:
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
# URL detection (simple regex)
# ----------------------------------------------------------------------
def extract_url(text: str) -> Optional[str]:
    """Return the first URL found in text, or None."""
    url_pattern = r'(https?://[^\s]+)'
    match = re.search(url_pattern, text)
    return match.group(0) if match else None

def is_scrape_intent(text: str) -> bool:
    """
    Heuristic to decide if the user wants to scrape a website.
    Looks for a URL and keywords like 'scrape', 'get content', 'fetch', 'read', etc.
    """
    url = extract_url(text)
    if not url:
        return False
    keywords = ['scrape', 'extract', 'fetch', 'get content', 'read', 'show me', 'what is on', 'content of']
    text_lower = text.lower()
    return any(kw in text_lower for kw in keywords)

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
    Generate a response using retrieval‑augmented generation, or
    if the user requests real‑time web data, scrape the URL and return content.

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
            sources: List of retrieved documents or URLs.
            thinking_time: Time taken in seconds.
            emotion: Detected emotion.
    """
    start_time = time.time()

    # ------------------------------------------------------------------
    # 0. Check for web scraping intent (real‑time data)
    # ------------------------------------------------------------------
    if SCRAPER_AVAILABLE and is_scrape_intent(user_query):
        url = extract_url(user_query)
        logger.info(f"Detected scrape intent for URL: {url}")
        use_js = "javascript" in user_query.lower() or "dynamic" in user_query.lower()
        try:
            scraped_text = scrape_url(url, use_js)
            if scraped_text:
                # Truncate if too long (e.g., 2000 chars) to avoid token limits
                if len(scraped_text) > 2000:
                    scraped_text = scraped_text[:2000] + "... (truncated)"
                answer = f"Here's the content I scraped from {url}:\n\n{scraped_text}"
                return {
                    "type": "sync",
                    "answer": answer,
                    "monologue": f"Scraped {url} (JS={use_js})",
                    "sources": [url],
                    "thinking_time": time.time() - start_time,
                    "emotion": "neutral"
                }
            else:
                answer = f"Failed to scrape {url}. The site may be blocking automated requests or requires JavaScript."
                return {
                    "type": "sync",
                    "answer": answer,
                    "monologue": f"Scraping failed for {url}",
                    "sources": [],
                    "thinking_time": time.time() - start_time,
                    "emotion": "neutral"
                }
        except Exception as e:
            logger.exception("Scraping error")
            answer = f"An error occurred while scraping {url}: {e}"
            return {
                "type": "sync",
                "answer": answer,
                "monologue": "",
                "sources": [],
                "thinking_time": time.time() - start_time,
                "emotion": "neutral"
            }

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
            "answer": "Knowledge base not ready. Please wait for auto‑update to complete.",
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

    # Enhanced prompt for e‑commerce/product queries (optional)
    # If the query seems product‑related, we can add a hint.
    product_keywords = ['price', 'buy', 'product', 'amazon', 'shop', 'trending', 'sneakers', 'shoes']
    product_hint = ""
    if any(kw in user_query.lower() for kw in product_keywords):
        product_hint = "If the information comes from an e‑commerce site, include product name, price, and a link if available. "

    prompt = f"""You are Yukti AI, a helpful assistant.
{lang_instruction}
Conversation history:
{history_str}

Relevant information:
{context}

User mood: {emotion}
User query: {user_query}

{product_hint}Think step by step, then answer. Format your response as:
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
                "answer": "Sorry, I couldn't generate an answer due to a technical issue.",
                "monologue": "",
                "sources": [doc.metadata.get('source', 'Unknown') for doc in docs],
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
    # 8. Extract source URLs from retrieved documents
    # ------------------------------------------------------------------
    sources = []
    for doc in docs:
        source = doc.metadata.get('source')
        if source:
            sources.append(source)
        else:
            sources.append("Unknown source")

    # ------------------------------------------------------------------
    # 9. Return result
    # ------------------------------------------------------------------
    thinking_time = time.time() - start_time
    logger.info(f"Thought for {thinking_time:.2f}s, emotion: {emotion}, lang: {target_lang}")

    return {
        "type": "sync",
        "answer": answer,
        "monologue": monologue,
        "sources": sources,
        "thinking_time": thinking_time,
        "emotion": emotion
    }
