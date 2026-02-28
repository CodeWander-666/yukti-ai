"""
Yukti AI – Advanced Thinking Engine
Human-like reasoning with associative memory, self-questioning, and emotional awareness.
Optimized for speed: caching, parallel retrieval, concise prompts.
"""

import logging
import hashlib
import time
from typing import List, Dict, Any, Optional, Generator
from concurrent.futures import ThreadPoolExecutor, as_completed
from langchain_core.documents import Document
from langchain_helper import get_llm, retrieve_documents

logger = logging.getLogger(__name__)

# ----------------------------------------------------------------------
# Cache for retrieval results (speeds up repeated queries)
# ----------------------------------------------------------------------
_retrieval_cache = {}
CACHE_TTL = 3600  # 1 hour

def _cache_key(query: str) -> str:
    return hashlib.md5(query.encode()).hexdigest()

def get_cached_retrieval(query: str) -> Optional[List[Document]]:
    key = _cache_key(query)
    entry = _retrieval_cache.get(key)
    if entry and (time.time() - entry["timestamp"]) < CACHE_TTL:
        return entry["docs"]
    return None

def set_cached_retrieval(query: str, docs: List[Document]):
    key = _cache_key(query)
    _retrieval_cache[key] = {"docs": docs, "timestamp": time.time()}

# ----------------------------------------------------------------------
# Query expansion – generate multiple related questions
# ----------------------------------------------------------------------
def expand_query(original_query: str, llm) -> List[str]:
    """Generate 3 alternative phrasings of the query to broaden retrieval."""
    prompt = f"""Given the user's question, generate 3 different ways to ask the same question.
These will be used to search a knowledge base. Return each on a new line, no numbering.

Original question: {original_query}

Alternative phrasings:"""
    try:
        response = llm.invoke(prompt)
        alternatives = [line.strip() for line in response.split("\n") if line.strip()]
        return [original_query] + alternatives[:3]
    except Exception as e:
        logger.warning(f"Query expansion failed: {e}")
        return [original_query]

# ----------------------------------------------------------------------
# Retrieve and merge documents from multiple queries (parallel)
# ----------------------------------------------------------------------
def retrieve_diverse(query: str, k: int = 3) -> List[Document]:
    """
    Expand query, retrieve for each variant in parallel, deduplicate.
    """
    # Check cache
    cached = get_cached_retrieval(query)
    if cached is not None:
        logger.info("Retrieval cache hit")
        return cached[:k*2]

    llm = get_llm()
    expanded = expand_query(query, llm)

    all_docs = []
    seen = set()
    with ThreadPoolExecutor(max_workers=3) as executor:
        future_to_q = {
            executor.submit(retrieve_documents, q, k*2): q
            for q in expanded
        }
        for future in as_completed(future_to_q):
            try:
                docs = future.result()
                for doc in docs:
                    # Simple dedup using first 200 chars
                    key = doc.page_content[:200]
                    if key not in seen:
                        seen.add(key)
                        all_docs.append(doc)
            except Exception as e:
                logger.error(f"Retrieval failed for a variant: {e}")

    result = all_docs[:k*3]
    set_cached_retrieval(query, result)
    return result

# ----------------------------------------------------------------------
# Organize retrieved documents into a structured mental model
# ----------------------------------------------------------------------
def organize_knowledge(docs: List[Document], llm) -> str:
    """Ask the LLM to summarize and categorize the retrieved information."""
    if not docs:
        return "No relevant information found."
    combined = "\n\n".join([f"- {doc.page_content}" for doc in docs])
    prompt = f"""You are Yukti AI's internal organizer. Review the following retrieved information and create a concise, structured summary that groups related facts. This summary will be used for reasoning.

Retrieved information:
{combined}

Structured summary (use bullet points and categories):"""
    try:
        return llm.invoke(prompt)
    except Exception as e:
        logger.error(f"Organization failed: {e}")
        return combined  # fallback

# ----------------------------------------------------------------------
# Main thinking function – returns structured output
# ----------------------------------------------------------------------
def think(
    user_query: str,
    conversation_history: List[Dict[str, str]]
) -> Dict[str, Any]:
    """
    Generate a human‑like answer with transparent reasoning.
    Returns dict with keys: 'answer', 'monologue', 'sources', 'thinking_time'.
    """
    start_time = time.time()
    llm = get_llm()

    # Step 1: Retrieve memories (fast, cached)
    logger.info("Retrieving memories...")
    try:
        docs = retrieve_diverse(user_query, k=3)
    except FileNotFoundError:
        return {
            "answer": "The knowledge base is not ready. Please update it first.",
            "monologue": "",
            "sources": [],
            "thinking_time": 0.0
        }
    except Exception as e:
        logger.exception("Retrieval error")
        return {
            "answer": f"Sorry, I encountered an error accessing my memory: {e}",
            "monologue": "",
            "sources": [],
            "thinking_time": 0.0
        }

    # Step 2: Organize knowledge (one LLM call)
    logger.info("Organizing knowledge...")
    knowledge_summary = organize_knowledge(docs, llm)

    # Step 3: Format conversation history
    history_str = ""
    for turn in conversation_history[-5:]:
        role = "User" if turn["role"] == "user" else "Yukti"
        history_str += f"{role}: {turn['content']}\n"

    # Step 4: Generate reasoning and final answer (single LLM call)
    reasoning_prompt = f"""You are Yukti AI, a deeply thoughtful assistant. You have access to your memories (organized below) and the conversation history. Now, think step by step:

1. **Review memories**: What facts, examples, or procedures are relevant?
2. **Self-question**: What else do I need to know? What might the user really be asking? Generate 1-2 internal questions and answer them using the memories.
3. **Consider context**: How does the conversation history affect my response?
4. **Formulate answer**: Write a final answer that is empathetic, informative, and natural.

Conversation history:
{history_str}

Organized memories:
{knowledge_summary}

User's current query: "{user_query}"

Now, write your internal monologue (questions, thoughts) and then your final answer. Format as:

MONOLOGUE:
(Your inner thoughts, questions, and reasoning here)

ANSWER:
(Your final answer to the user)
"""
    try:
        full_response = llm.invoke(reasoning_prompt)
    except Exception as e:
        logger.exception("Reasoning failed")
        return {
            "answer": f"Sorry, I couldn't think through that: {e}",
            "monologue": "",
            "sources": docs,
            "thinking_time": time.time() - start_time
        }

    # Extract monologue and answer
    monologue = ""
    answer = full_response
    if "MONOLOGUE:" in full_response and "ANSWER:" in full_response:
        parts = full_response.split("MONOLOGUE:", 1)[1]
        if "ANSWER:" in parts:
            monologue, answer = parts.split("ANSWER:", 1)
            monologue = monologue.strip()
            answer = answer.strip()

    return {
        "answer": answer,
        "monologue": monologue,
        "sources": docs,
        "thinking_time": time.time() - start_time
    }
