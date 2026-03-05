"""
Core reasoning engine – combines retrieval, emotion detection, language awareness,
LLM invocation, and optional live web search.
"""

import re
import time
import hashlib
import logging
from typing import List, Dict, Any, Optional

from langchain_core.documents import Document

from langchain_helper import retrieve_documents
from model_manager import load_model, get_model_config
from language_detector import detect_language, get_response_language
from language_utils import get_language_name
from config import RETRIEVAL_CACHE_TTL

try:
    from duckduckgo_search import DDGS
    WEB_SEARCH_AVAILABLE = True
except ImportError:
    WEB_SEARCH_AVAILABLE = False

try:
    from knowledge_updater.connectors import scrape_url
    SCRAPER_AVAILABLE = True
except ImportError:
    SCRAPER_AVAILABLE = False
    def scrape_url(*args, **kwargs): return None

logger = logging.getLogger(__name__)

# Retrieval cache
_retrieval_cache: Dict[str, Dict] = {}

def _cache_key(query: str) -> str:
    return hashlib.md5(query.encode('utf-8')).hexdigest()

def retrieve_cached(query: str, k: int = 3) -> List[Document]:
    cache_key = _cache_key(query)
    cached = _retrieval_cache.get(cache_key)
    now = time.time()
    if cached and (now - cached['timestamp']) < RETRIEVAL_CACHE_TTL:
        return cached['docs']
    try:
        docs = retrieve_documents(query, k=k)
        _retrieval_cache[cache_key] = {'docs': docs, 'timestamp': now}
        return docs
    except FileNotFoundError:
        raise
    except Exception as e:
        logger.exception(f"Retrieval error: {e}")
        raise RuntimeError(f"Document retrieval failed: {e}") from e

def detect_emotion(text: str) -> str:
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

def extract_url(text: str) -> Optional[str]:
    url_pattern = r'(https?://[^\s]+)'
    match = re.search(url_pattern, text)
    return match.group(0) if match else None

def is_scrape_intent(text: str) -> bool:
    url = extract_url(text)
    if not url:
        return False
    keywords = ['scrape', 'extract', 'fetch', 'get content', 'read', 'show me', 'what is on', 'content of']
    text_lower = text.lower()
    return any(kw in text_lower for kw in keywords)

def web_search(query: str, max_results: int = 5) -> List[Dict[str, str]]:
    """Perform a web search using DuckDuckGo and return a list of result dicts (title, link, snippet)."""
    if not WEB_SEARCH_AVAILABLE:
        return []
    try:
        with DDGS() as ddgs:
            results = []
            for r in ddgs.text(query, max_results=max_results):
                results.append({
                    "title": r.get("title", ""),
                    "link": r.get("href", ""),
                    "snippet": r.get("body", "")
                })
            return results
    except Exception as e:
        logger.error(f"Web search error: {e}")
        return []

def think(
    user_query: str,
    conversation_history: List[Dict[str, str]],
    model_key: str,
    language: Optional[str] = None,
    web_search: bool = False
) -> Dict[str, Any]:
    start_time = time.time()

    # 0. Web search if enabled and no explicit URL scrape
    if web_search and not is_scrape_intent(user_query) and WEB_SEARCH_AVAILABLE:
        logger.info("Performing web search for: %s", user_query)
        results = web_search(user_query, max_results=3)
        if results:
            # Format results as context
            context = "Web search results:\n\n"
            sources = []
            for r in results:
                context += f"Title: {r['title']}\nURL: {r['link']}\nSnippet: {r['snippet']}\n\n"
                sources.append(r['link'])
            # Build prompt
            lang_instruction = ""
            if language:
                if language == 'hinglish':
                    lang_instruction = "Answer in Hinglish (mix of Hindi and English)."
                elif language != 'en':
                    lang_instruction = f"Answer in {get_language_name(language)}."
            prompt = f"""You are Yukti AI. Use the following web search results to answer the user's query.
{lang_instruction}
Web search results:
{context}
User query: {user_query}
Answer concisely, citing sources if possible."""
            # Invoke LLM
            llm = load_model(model_key)
            try:
                response = llm.invoke(prompt, language=language)
                if hasattr(response, 'content'):
                    answer = response.content
                else:
                    answer = str(response)
                return {
                    "type": "sync",
                    "answer": answer,
                    "monologue": "Used web search results.",
                    "sources": sources,
                    "thinking_time": time.time() - start_time,
                    "emotion": detect_emotion(user_query)
                }
            except Exception as e:
                logger.error("LLM failed after web search: %s", e)
                # fallback to plain results
                ans = "Here are some web results:\n\n"
                for r in results:
                    ans += f"- {r['title']}: {r['link']}\n  {r['snippet']}\n\n"
                return {
                    "type": "sync",
                    "answer": ans,
                    "monologue": "",
                    "sources": [r['link'] for r in results],
                    "thinking_time": time.time() - start_time,
                    "emotion": detect_emotion(user_query)
                }

    # 1. Check for scrape intent (URL)
    if SCRAPER_AVAILABLE and is_scrape_intent(user_query):
        url = extract_url(user_query)
        use_js = "javascript" in user_query.lower() or "dynamic" in user_query.lower()
        scraped_text = scrape_url(url, use_js)
        if scraped_text:
            if len(scraped_text) > 2000:
                scraped_text = scraped_text[:2000] + "... (truncated)"
            answer = f"Here's the content I scraped from {url}:\n\n{scraped_text}"
            return {
                "type": "sync",
                "answer": answer,
                "monologue": f"Scraped {url}",
                "sources": [url],
                "thinking_time": time.time() - start_time,
                "emotion": detect_emotion(user_query)
            }
        else:
            answer = f"Failed to scrape {url}."
            return {
                "type": "sync",
                "answer": answer,
                "monologue": "",
                "sources": [],
                "thinking_time": time.time() - start_time,
                "emotion": detect_emotion(user_query)
            }

    # 2. Normal RAG flow
    config = get_model_config(model_key)
    if not config or config.get("type") != "sync":
        return {"type": "sync", "answer": f"Model {model_key} not available or not sync."}

    emotion = detect_emotion(user_query)

    # Language
    if language:
        target_lang = language
    else:
        lang_info = detect_language(user_query)
        target_lang = lang_info['language']

    # Retrieve
    try:
        docs = retrieve_cached(user_query, k=3)
    except FileNotFoundError:
        return {
            "type": "sync",
            "answer": "Knowledge base not ready. Please wait for auto‑update.",
            "monologue": "",
            "sources": [],
            "thinking_time": time.time() - start_time,
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
        f"{'User' if msg['role'] == 'user' else 'Yukti'}: {msg['content']}"
        for msg in conversation_history[-5:]
    )

    if target_lang == 'hinglish':
        lang_instruction = "Answer in Hinglish (mix of Hindi and English)."
    elif target_lang != 'en':
        lang_instruction = f"Answer in {get_language_name(target_lang)}."
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

    llm = load_model(model_key)
    answer = ""
    monologue = ""
    full_text = ""

    try:
        response = llm.invoke(prompt, language=target_lang)
        full_text = response.content if hasattr(response, 'content') else str(response)
    except Exception as e:
        logger.exception("LLM failed, using fallback")
        fallback_prompt = f"{lang_instruction}\nUser mood: {emotion}\nUser query: {user_query}\nAnswer concisely:"
        try:
            response2 = llm.invoke(fallback_prompt, language=target_lang)
            full_text = response2.content if hasattr(response2, 'content') else str(response2)
        except Exception as e2:
            return {
                "type": "sync",
                "answer": "Sorry, I couldn't generate an answer.",
                "monologue": "",
                "sources": [doc.metadata.get('source', 'Unknown') for doc in docs],
                "thinking_time": time.time() - start_time,
                "emotion": emotion
            }

    if "MONOLOGUE:" in full_text and "ANSWER:" in full_text:
        try:
            parts = full_text.split("MONOLOGUE:", 1)[1]
            if "ANSWER:" in parts:
                monologue_part, answer_part = parts.split("ANSWER:", 1)
                monologue = monologue_part.strip()
                answer = answer_part.strip()
        except Exception:
            answer = full_text.strip()
    else:
        answer = full_text.strip()

    if not answer:
        answer = "(No response)"

    sources = []
    for doc in docs:
        source = doc.metadata.get('source')
        sources.append(source if source else "Unknown")

    return {
        "type": "sync",
        "answer": answer,
        "monologue": monologue,
        "sources": sources,
        "thinking_time": time.time() - start_time,
        "emotion": emotion
    }
