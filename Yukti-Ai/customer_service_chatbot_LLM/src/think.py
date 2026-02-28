"""
Yukti AI – Advanced Thinking Engine (The Brain)
Human‑like reasoning with emotional awareness, self‑questioning, memory, and structured output.
"""

import logging
import time
import hashlib
from typing import List, Dict, Any, Optional
from concurrent.futures import ThreadPoolExecutor, as_completed

from langchain_core.documents import Document
from langchain_helper import retrieve_documents, get_cross_encoder
from model_manager import load_model, get_model_config

logger = logging.getLogger(__name__)

# ----------------------------------------------------------------------
# Conversation Memory (stores recent exchanges and a running summary)
# ----------------------------------------------------------------------
class ConversationMemory:
    def __init__(self, max_history: int = 10):
        self.history = []
        self.max_history = max_history
        self.summary = ""

    def add_exchange(self, user: str, assistant: str):
        self.history.append({"user": user, "assistant": assistant})
        if len(self.history) > self.max_history:
            self.history.pop(0)

    def get_recent(self, n: int = 5) -> str:
        recent = self.history[-n:]
        lines = []
        for turn in recent:
            lines.append(f"User: {turn['user']}")
            lines.append(f"Yukti: {turn['assistant']}")
        return "\n".join(lines)

    def update_summary(self, llm, query: str, answer: str):
        """Generate or update a running summary of the conversation (optional)."""
        # For now, we skip to keep speed; could be implemented if needed.
        pass

# ----------------------------------------------------------------------
# Emotional tone detection (simple keyword‑based)
# ----------------------------------------------------------------------
def detect_emotion(text: str) -> str:
    """Return a likely user emotion based on keywords."""
    text_lower = text.lower()
    if any(word in text_lower for word in ["sad", "unhappy", "depressed", "upset"]):
        return "sad"
    if any(word in text_lower for word in ["happy", "excited", "great", "wonderful"]):
        return "happy"
    if any(word in text_lower for word in ["angry", "frustrated", "annoyed", "mad"]):
        return "angry"
    if any(word in text_lower for word in ["confused", "lost", "unsure"]):
        return "confused"
    return "neutral"

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
# Retrieve and merge documents from multiple queries with optional re‑ranking
# ----------------------------------------------------------------------
def retrieve_diverse(query: str, llm, k: int = 3) -> List[Document]:
    """
    Expand query, retrieve for each variant in parallel, deduplicate, and re‑rank.
    Returns a diverse set of relevant documents.
    """
    expanded = expand_query(query, llm)

    all_docs = []
    seen = set()
    with ThreadPoolExecutor(max_workers=3) as executor:
        future_to_q = {executor.submit(retrieve_documents, q, k*3): q for q in expanded}
        for future in as_completed(future_to_q):
            try:
                docs = future.result()
                for doc in docs:
                    # Deduplicate using first 200 characters as a simple key
                    key = doc.page_content[:200]
                    if key not in seen:
                        seen.add(key)
                        all_docs.append(doc)
            except Exception as e:
                logger.error(f"Retrieval failed for a variant: {e}")

    # Optional re‑ranking with cross‑encoder
    cross_encoder = get_cross_encoder()
    if cross_encoder and all_docs:
        pairs = [[query, doc.page_content] for doc in all_docs]
        scores = cross_encoder.predict(pairs)
        scored = sorted(zip(all_docs, scores), key=lambda x: x[1], reverse=True)
        all_docs = [doc for doc, _ in scored[:k*2]]
    else:
        all_docs = all_docs[:k*2]

    return all_docs

# ----------------------------------------------------------------------
# Self‑questioning – generate and answer internal questions
# ----------------------------------------------------------------------
def self_question(query: str, context: str, llm) -> str:
    """
    Generate 1‑2 sub‑questions about the query, answer them using context,
    and return a consolidated reasoning.
    """
    prompt = f"""You are Yukti AI's internal reasoner. Based on the user's query and the provided context,
think about what additional information might be needed to answer well.
Generate 1‑2 specific questions that you would need to answer first, and then answer them using the context.

User query: {query}

Context:
{context}

Format:
Question 1: ...
Answer 1: ...
Question 2: ...
Answer 2: ...
(if only one question, just do one)
"""
    try:
        response = llm.invoke(prompt)
        return response.strip()
    except Exception as e:
        logger.warning(f"Self‑questioning failed: {e}")
        return ""

# ----------------------------------------------------------------------
# Main think function – the brain of Yukti AI
# ----------------------------------------------------------------------
def think(
    user_query: str,
    conversation_history: List[Dict[str, str]],
    model_key: str,
    memory: Optional[ConversationMemory] = None
) -> Dict[str, Any]:
    """
    Generate a human‑like answer using:
    - Query expansion and diverse retrieval
    - Re‑ranking (if cross‑encoder available)
    - Self‑questioning
    - Conversation memory
    - Emotional awareness
    - Internal monologue and final answer
    """
    config = get_model_config(model_key)
    if not config:
        raise ValueError(f"Unknown model: {model_key}")

    # For async models (video), just submit task and return
    if config["type"] == "async":
        llm = load_model(model_key)
        task_id = llm.invoke(user_query)
        return {
            "type": "async",
            "task_id": task_id,
            "variant": model_key,
            "model": config["model"]
        }

    # --- Sync models: reasoning begins ---
    start_time = time.time()
    llm = load_model(model_key)

    # 1. Detect user emotion
    emotion = detect_emotion(user_query)

    # 2. Retrieve diverse documents
    try:
        docs = retrieve_diverse(user_query, llm, k=3)
    except FileNotFoundError:
        return {
            "type": "sync",
            "answer": "The knowledge base is not ready. Please update it first.",
            "monologue": "",
            "sources": [],
            "thinking_time": 0.0
        }
    except Exception as e:
        logger.exception("Retrieval error")
        return {
            "type": "sync",
            "answer": f"Sorry, I encountered an error accessing my memory: {e}",
            "monologue": "",
            "sources": [],
            "thinking_time": 0.0
        }

    # Build context from retrieved docs
    context = "\n\n".join([doc.page_content for doc in docs])

    # 3. Self‑questioning
    self_qa = self_question(user_query, context, llm)

    # 4. Format conversation history (last 5 exchanges)
    history_str = ""
    for turn in conversation_history[-5:]:
        role = "User" if turn["role"] == "user" else "Yukti"
        history_str += f"{role}: {turn['content']}\n"

    # 5. Build the reasoning prompt with all components
    reasoning_prompt = f"""You are Yukti AI, a deeply thoughtful assistant with emotional awareness and memory.
You have access to:
- Relevant memories (context)
- Your own internal questions and answers (self‑QA)
- The conversation history
- The user's detected mood: {emotion}

Now, think step by step:

1. **Review memories and self‑QA**: What facts, examples, or reasoning are most relevant?
2. **Consider the user's emotion**: How should you adjust your tone and content?
3. **Connect to conversation**: How does the history influence your response?
4. **Formulate answer**: Write a final answer that is empathetic, informative, and natural.

Conversation history:
{history_str}

Context:
{context}

Self‑QA:
{self_qa}

User's current query: "{user_query}"

Now, write your internal monologue (your thoughts, questions, and reasoning) and then your final answer. Format as:

MONOLOGUE:
(Your inner thoughts, questions, and reasoning here)

ANSWER:
(Your final answer to the user)
"""
    try:
        full_response = llm.invoke(reasoning_prompt)
    except Exception as e:
        logger.exception("LLM invocation failed")
        return {
            "type": "sync",
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

    # Optionally update memory (if memory object provided)
    if memory:
        memory.add_exchange(user_query, answer)

    return {
        "type": "sync",
        "answer": answer,
        "monologue": monologue,
        "sources": docs,
        "thinking_time": time.time() - start_time,
        "emotion": emotion
    }
