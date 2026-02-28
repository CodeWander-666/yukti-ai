"""
Yukti AI – Thinking Engine
Handles both sync reasoning and async generation tasks.
"""

import logging
import time
from typing import List, Dict, Any

# Import model manager (which now uses LLM package)
from model_manager import load_model, get_model_config
from langchain_helper import retrieve_documents  # unchanged

logger = logging.getLogger(__name__)

def think(
    user_query: str,
    conversation_history: List[Dict[str, str]],
    model_key: str
) -> Dict[str, Any]:
    """
    Generate a response. For sync models, returns answer + monologue + sources.
    For async models, returns a task ID.
    """
    config = get_model_config(model_key)
    if not config:
        raise ValueError(f"Unknown model: {model_key}")

    if config["type"] == "async":
        # Async generation task – submit and return task info
        llm = load_model(model_key)
        task_id = llm.invoke(user_query)
        return {
            "type": "async",
            "task_id": task_id,
            "variant": model_key,
            "model": config["model"]
        }
    else:
        # Sync reasoning – proceed with retrieval and thinking
        start_time = time.time()
        llm = load_model(model_key)

        # Retrieve relevant documents (same as before)
        try:
            docs = retrieve_documents(user_query, k=3)
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
                "answer": f"Sorry, I encountered an error: {e}",
                "monologue": "",
                "sources": [],
                "thinking_time": 0.0
            }

        # Build context from docs
        context = "\n\n".join([doc.page_content for doc in docs])

        # Format conversation history
        history_str = ""
        for turn in conversation_history[-5:]:
            role = "User" if turn["role"] == "user" else "Yukti"
            history_str += f"{role}: {turn['content']}\n"

        # Construct prompt for the LLM (with reasoning instructions)
        reasoning_prompt = f"""You are Yukti AI, a thoughtful assistant. Use the context and conversation to answer.

Context:
{context}

Conversation:
{history_str}

User: {user_query}

Think step by step, then write your answer. Format as:

MONOLOGUE:
(Your reasoning)

ANSWER:
(Your final answer)
"""
        try:
            full_response = llm.invoke(reasoning_prompt)
        except Exception as e:
            logger.exception("LLM invocation failed")
            return {
                "type": "sync",
                "answer": f"Sorry, I couldn't generate an answer: {e}",
                "monologue": "",
                "sources": docs,
                "thinking_time": time.time() - start_time
            }

        # Parse monologue and answer
        monologue = ""
        answer = full_response
        if "MONOLOGUE:" in full_response and "ANSWER:" in full_response:
            parts = full_response.split("MONOLOGUE:", 1)[1]
            if "ANSWER:" in parts:
                monologue, answer = parts.split("ANSWER:", 1)
                monologue = monologue.strip()
                answer = answer.strip()

        return {
            "type": "sync",
            "answer": answer,
            "monologue": monologue,
            "sources": docs,
            "thinking_time": time.time() - start_time
        }
