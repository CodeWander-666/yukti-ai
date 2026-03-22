"""
Medical module for Yukti‑Doctor.
Handles retrieval, dialogue, file uploads, and response generation.
"""

import logging
import tempfile
from pathlib import Path
from typing import List, Dict, Any, Optional
import re

# For OCR
try:
    import pytesseract
    from PIL import Image
    TESSERACT_AVAILABLE = True
except ImportError:
    TESSERACT_AVAILABLE = False

# For optional image classification (stub)
try:
    import torch
    # You would load your medical vision model here
    VISION_AVAILABLE = False
except ImportError:
    VISION_AVAILABLE = False

from langchain_core.documents import Document
from src.langchain_helper import load_medical_vectorstore
from src.model_manager import load_model  # We'll use the base LLM for synthesis

logger = logging.getLogger(__name__)

# ----------------------------------------------------------------------
# Medical Retriever
# ----------------------------------------------------------------------
class MedicalRetriever:
    def __init__(self):
        self.db = load_medical_vectorstore()
        if self.db is None:
            logger.warning("Medical vector store not available. Yukti‑Doctor will have limited knowledge.")

    def retrieve(self, query: str, k: int = 5) -> List[Document]:
        """Retrieve top k medical documents."""
        if self.db is None:
            return []
        try:
            return self.db.similarity_search(query, k=k)
        except Exception as e:
            logger.exception("Medical retrieval failed")
            return []

# ----------------------------------------------------------------------
# Dialogue State Machine
# ----------------------------------------------------------------------
class MedicalDialogue:
    """
    Tracks conversation state for symptom checking.
    In a full implementation, this would use a clinical decision tree.
    Here we provide a simple example.
    """
    def __init__(self):
        self.chief_complaint = None
        self.asked_questions = set()
        self.answers = {}
        self.red_flag = False

    def update(self, user_input: str):
        """Process user input and update state."""
        # Simple red flag detection (example)
        red_flags = ["chest pain", "difficulty breathing", "severe bleeding", "suicidal"]
        if any(flag in user_input.lower() for flag in red_flags):
            self.red_flag = True

        # For demonstration, we just store the input as the chief complaint if not set
        if self.chief_complaint is None:
            self.chief_complaint = user_input

    def next_question(self) -> Optional[str]:
        """Determine the next question to ask based on state."""
        if self.red_flag:
            return None  # Will trigger emergency message
        # Simple logic: ask a few generic follow-ups
        if "duration" not in self.asked_questions:
            self.asked_questions.add("duration")
            return "How long have you been experiencing this?"
        if "severity" not in self.asked_questions:
            self.asked_questions.add("severity")
            return "On a scale of 1-10, how severe is it?"
        return None  # No more questions

# ----------------------------------------------------------------------
# File Processing
# ----------------------------------------------------------------------
class FileProcessor:
    @staticmethod
    def process_upload(uploaded_file) -> Dict[str, Any]:
        """
        Process an uploaded file (image or PDF).
        Returns a dict with extracted text and any image analysis.
        """
        result = {"text": "", "image_analysis": None}
        if uploaded_file is None:
            return result

        # Save temporarily
        suffix = Path(uploaded_file.name).suffix
        with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
            tmp.write(uploaded_file.getbuffer())
            tmp_path = tmp.name

        # Handle images
        if suffix.lower() in ['.jpg', '.jpeg', '.png']:
            # OCR
            if TESSERACT_AVAILABLE:
                try:
                    img = Image.open(tmp_path)
                    result["text"] = pytesseract.image_to_string(img)
                except Exception as e:
                    logger.error(f"OCR failed: {e}")

            # Optional image classification (stub)
            if VISION_AVAILABLE:
                # Here you would call your medical vision model
                result["image_analysis"] = "Possible pneumonia detected (example)."

        # Handle PDFs (simple text extraction)
        elif suffix.lower() == '.pdf':
            try:
                import PyPDF2
                with open(tmp_path, 'rb') as f:
                    reader = PyPDF2.PdfReader(f)
                    text = ""
                    for page in reader.pages:
                        text += page.extract_text()
                result["text"] = text
            except Exception as e:
                logger.error(f"PDF extraction failed: {e}")

        # Clean up
        Path(tmp_path).unlink(missing_ok=True)
        return result

# ----------------------------------------------------------------------
# Main Medical Think Function
# ----------------------------------------------------------------------
def think_medical(
    user_query: str,
    conversation_history: List[Dict[str, str]],
    uploaded_file=None,
    model_key: str = "Yukti‑Doctor"
) -> Dict[str, Any]:
    """
    Process a medical query with optional file upload.
    Returns a response dict similar to think.py.
    """
    # Initialize dialogue state (in production, store in session)
    dialogue = MedicalDialogue()
    dialogue.update(user_query)

    # Check for red flag
    if dialogue.red_flag:
        return {
            "type": "sync",
            "answer": "⚠️ **If you are experiencing a medical emergency, please call your local emergency services immediately.**\n\nBased on your description, this could be serious. Do not wait for an AI response.",
            "monologue": "Red flag detected",
            "sources": [],
            "thinking_time": 0,
            "emotion": "urgent"
        }

    # Process uploaded file if any
    file_info = {}
    if uploaded_file:
        file_info = FileProcessor.process_upload(uploaded_file)
        # Append extracted info to query for context
        if file_info["text"]:
            user_query += f"\n\n[Extracted from uploaded file: {file_info['text'][:500]}]"
        if file_info["image_analysis"]:
            user_query += f"\n\n[Image analysis: {file_info['image_analysis']}]"

    # Retrieve relevant medical documents
    retriever = MedicalRetriever()
    docs = retriever.retrieve(user_query, k=5)
    context = "\n\n".join([doc.page_content for doc in docs])
    sources = list(set([doc.metadata.get('source', 'Unknown') for doc in docs]))

    # Determine if we need to ask a follow-up question
    next_q = dialogue.next_question()
    if next_q and len(conversation_history) < 3:  # limit follow-ups
        # If we have a follow-up, we return it as the answer and let the user respond
        return {
            "type": "sync",
            "answer": next_q,
            "monologue": "Asking follow-up question",
            "sources": sources,
            "thinking_time": 0,
            "emotion": "neutral"
        }

    # Otherwise, generate a final answer using the LLM
    # Use the base model (Yukti‑Flash) for synthesis, or a dedicated medical model
    llm = load_model("Yukti‑Flash")  # or a fine-tuned medical model

    prompt = f"""You are Yukti‑Doctor, an experienced medical AI assistant with 100 years of clinical knowledge. Use the following medical references to answer the user's question. If the question is about symptoms, provide a list of possible causes with brief explanations, and always recommend consulting a real doctor.

Medical references:
{context}

User: {user_query}

Provide a clear, compassionate, and evidence‑based answer. Include a disclaimer. If the information is insufficient, say so and suggest what additional information would help.
"""
    try:
        response = llm.invoke(prompt)
        answer = response if isinstance(response, str) else response.get('content', str(response))
    except Exception as e:
        logger.exception("Medical LLM invocation failed")
        answer = "I'm sorry, I encountered an error processing your medical question. Please try again later."

    return {
        "type": "sync",
        "answer": answer,
        "monologue": "Retrieved medical documents and synthesized response.",
        "sources": sources,
        "thinking_time": 0,  # we don't track here
        "emotion": "neutral"
    }