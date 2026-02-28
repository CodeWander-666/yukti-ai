"""
Yukti AI – Main Application (The Brain)
Orchestrates all components with bulletproof error handling, ultra‑fast performance,
and a clean, professional UI.
"""

import os
import sys
import time
import logging
import traceback
from pathlib import Path

# Add project root to path (ensures imports work even on Streamlit Cloud)
PROJECT_ROOT = Path(__file__).parent.parent.parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import streamlit as st
import pandas as pd
from langchain_core.documents import Document
from langchain_community.vectorstores import FAISS

# Local imports (must be in same directory)
from langchain_helper import get_embeddings, VECTORDB_PATH, BASE_DIR, create_vector_db
from think import think
from model_manager import (
    get_available_models,
    MODELS,
    get_active_tasks,
    get_task_status,
    ZHIPU_AVAILABLE,
)

# Configure logging (for debugging; logs appear in Streamlit Cloud logs)
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# ----------------------------------------------------------------------
# Page configuration (MUST be first Streamlit command)
# ----------------------------------------------------------------------
st.set_page_config(
    page_title="Yukti AI",
    page_icon="✨",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ----------------------------------------------------------------------
# Custom CSS for a clean, professional look
# ----------------------------------------------------------------------
st.markdown("""
<style>
    .main > div { padding: 0 2rem; }
    .stChatMessage { border-radius: 12px; padding: 1rem; margin-bottom: 0.5rem; }
    .stChatMessage[data-testid="user-message"] { background-color: #f0f2f6; }
    .stChatMessage[data-testid="assistant-message"] { background-color: #e3f2fd; border-left: 4px solid #1976d2; }
    .css-1d391kg { background-color: #f8f9fa; }
    .stButton > button { border-radius: 8px; font-weight: 500; transition: all 0.2s; }
    .stButton > button:hover { transform: translateY(-1px); box-shadow: 0 4px 8px rgba(0,0,0,0.1); }
    .stTextInput > div > div > input { border-radius: 24px; border: 1px solid #ddd; padding: 12px 20px; font-size: 16px; }
    h1, h2, h3 { font-family: 'Inter', sans-serif; font-weight: 500; }
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
</style>
""", unsafe_allow_html=True)

# ----------------------------------------------------------------------
# Title and subtitle
# ----------------------------------------------------------------------
st.title("Yukti AI")
st.caption("Your Intelligent Assistant – Powered by Zhipu GLM & Gemini")

# ----------------------------------------------------------------------
# Session state initialization
# ----------------------------------------------------------------------
if "messages" not in st.session_state:
    st.session_state.messages = [
        {"role": "assistant", "content": "Hello! I'm Yukti AI. How can I help you today?"}
    ]
if "knowledge_base_ready" not in st.session_state:
    try:
        st.session_state.knowledge_base_ready = os.path.exists(VECTORDB_PATH)
    except Exception as e:
        logger.error(f"Failed to check knowledge base status: {e}")
        st.session_state.knowledge_base_ready = False
if "show_thinking" not in st.session_state:
    st.session_state.show_thinking = True
if "tasks" not in st.session_state:
    st.session_state.tasks = {}  # task_id -> {variant, status, progress, result_url, error}

# ----------------------------------------------------------------------
# Data sources configuration
# ----------------------------------------------------------------------
SOURCES = [{
    "type": "csv",
    "path": os.path.join(BASE_DIR, "dataset", "dataset.csv"),
    "name": "Original Dataset",
    "columns": ["prompt", "response"],
    "content_template": "Q: {prompt}\nA: {response}"
}]

# ----------------------------------------------------------------------
# Helper Functions
# ----------------------------------------------------------------------
def load_all_documents():
    """Load documents from all sources with robust error handling."""
    docs = []
    for src in SOURCES:
        if src["type"] != "csv":
            continue
        path = src["path"]
        try:
            if not os.path.exists(path):
                st.sidebar.warning(f"File not found: {path}")
                continue

            encodings = ['utf-8', 'cp1252', 'latin-1', 'iso-8859-1']
            df = None
            for enc in encodings:
                try:
                    df = pd.read_csv(path, encoding=enc, on_bad_lines='skip')
                    logger.info(f"Successfully read {path} with encoding {enc}")
                    break
                except UnicodeDecodeError:
                    continue
                except pd.errors.EmptyDataError:
                    st.sidebar.error(f"File {path} is empty.")
                    return None
                except pd.errors.ParserError as e:
                    st.sidebar.error(f"CSV parsing error in {path}: {e}")
                    return None
                except Exception as e:
                    logger.warning(f"Unexpected error reading {path} with {enc}: {e}")
                    continue

            if df is None:
                st.sidebar.error(f"Cannot read {path} with any tried encoding.")
                return None

            missing = [c for c in src["columns"] if c not in df.columns]
            if missing:
                st.sidebar.error(f"Missing columns {missing} in {path}")
                return None

            for idx, row in df.iterrows():
                try:
                    content = src["content_template"].format(**{c: row[c] for c in src["columns"]})
                    docs.append(Document(
                        page_content=content,
                        metadata={"source": path, "row": idx}
                    ))
                except KeyError as e:
                    logger.warning(f"Row {idx} missing key {e}, skipping")
                    continue
                except Exception as e:
                    logger.warning(f"Unexpected error formatting row {idx}: {e}")
                    continue

            st.sidebar.success(f"Loaded {len(df)} rows from {src['name']}")

        except PermissionError:
            st.sidebar.error(f"Permission denied: cannot read {path}")
            return None
        except MemoryError:
            st.sidebar.error(f"Out of memory while reading {path}")
            return None
        except Exception as e:
            st.sidebar.error(f"Unexpected error reading {path}: {e}")
            logger.exception("load_all_documents fatal error")
            return None
    return docs

def rebuild_knowledge_base():
    """Rebuild FAISS index from all sources."""
    try:
        with st.spinner("Loading documents..."):
            docs = load_all_documents()
            if docs is None:
                st.error("Failed to load documents. Check logs.")
                return False
            if not docs:
                st.warning("No documents found. Please add data to your CSV.")
                return False

        with st.spinner("Generating embeddings and building index..."):
            try:
                embeddings = get_embeddings()
            except Exception as e:
                st.error(f"Failed to load embedding model: {e}")
                logger.exception("Embedding model load failed")
                return False

            try:
                vectordb = FAISS.from_documents(docs, embeddings)
            except ValueError as e:
                st.error(f"FAISS creation error (maybe empty documents?): {e}")
                return False
            except Exception as e:
                st.error(f"Unexpected error during index build: {e}")
                logger.exception("FAISS.from_documents failed")
                return False

            try:
                vectordb.save_local(VECTORDB_PATH)
            except PermissionError:
                st.error(f"Permission denied: cannot write to {VECTORDB_PATH}")
                return False
            except OSError as e:
                st.error(f"Disk write error: {e}")
                return False
            except Exception as e:
                st.error(f"Failed to save index: {e}")
                return False

            st.session_state.knowledge_base_ready = True
            st.success(f"Knowledge base rebuilt with {len(docs)} documents!")
            return True
    except Exception as e:
        st.error(f"Unexpected error in rebuild_knowledge_base: {e}")
        logger.exception("rebuild_knowledge_base fatal error")
        return False

def stream_response(placeholder, full_text, delay=0.02):
    """Simulate letter‑by‑letter streaming."""
    if not full_text or not isinstance(full_text, str):
        placeholder.markdown("")
        return
    try:
        words = full_text.split()
        current = ""
        for word in words:
            current += word + " "
            placeholder.markdown(current + "▌")
            time.sleep(delay)
        placeholder.markdown(current)
    except Exception as e:
        logger.warning(f"Streaming error: {e}")
        placeholder.markdown(full_text)  # fallback to full text

# ----------------------------------------------------------------------
# Sidebar
# ----------------------------------------------------------------------
with st.sidebar:
    st.header("Knowledge Base")
    if st.button("Update Knowledge Base", use_container_width=True):
        rebuild_knowledge_base()
    st.divider()

    st.subheader("Status")
    if st.session_state.knowledge_base_ready:
        st.markdown("**Active**")
    else:
        st.markdown("**Not built** – click update above.")
    st.divider()

    st.subheader("Brain")
    try:
        model_options = get_available_models()
    except Exception as e:
        st.error(f"Failed to load models: {e}")
        model_options = []
    selected_model = st.selectbox(
        "Choose thinking style",
        options=model_options,
        format_func=lambda x: f"{x} – {MODELS[x]['description']}" if x in MODELS else x,
        key="model_selector"
    )
    st.session_state.selected_model = selected_model
    st.divider()

    st.subheader("Display")
    st.session_state.show_thinking = st.checkbox("Show thinking process", value=True)
    st.divider()

    if ZHIPU_AVAILABLE:
        st.subheader("📋 Active Tasks")
        if st.button("Refresh All Tasks", use_container_width=True):
            st.rerun()
        try:
            active = get_active_tasks()
            for task_id, variant, status, progress in active:
                with st.container():
                    st.markdown(f"**{variant}** ({task_id[:8]})")
                    if status == "processing":
                        st.progress(progress/100, text=f"{progress}%")
                    elif status in ("submitted", "pending"):
                        st.text("⏳ Queued")
                    else:
                        st.text(f"Status: {status}")
        except Exception as e:
            st.warning(f"Could not fetch active tasks: {e}")
        st.divider()

    st.subheader("Sources")
    for src in SOURCES:
        st.markdown(f"- **{src['name']}**")
    st.divider()

    if st.button("Clear Conversation", use_container_width=True):
        st.session_state.messages = [
            {"role": "assistant", "content": "Hello! I'm Yukti AI. How can I help you today?"}
        ]
        st.rerun()

# ----------------------------------------------------------------------
# Main chat interface
# ----------------------------------------------------------------------
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

# Chat input
if prompt := st.chat_input("Ask me anything..."):
    # Add user message
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    # Generate response
    with st.chat_message("assistant"):
        response_placeholder = st.empty()
        result = None
        answer = ""
        full_response = ""

        try:
            if not st.session_state.knowledge_base_ready:
                full_response = "The knowledge base is not ready. Please click 'Update Knowledge Base' in the sidebar first."
                response_placeholder.markdown(full_response)
                result = {"type": "sync", "answer": full_response}
            else:
                history = [
                    {"role": msg["role"], "content": msg["content"]}
                    for msg in st.session_state.messages[-10:]
                ]

                try:
                    with st.spinner("Thinking..."):
                        result = think(prompt, history, st.session_state.selected_model)
                except ValueError as e:
                    st.error(f"Model error: {e}")
                    result = {"type": "sync", "answer": f"Model error: {e}"}
                except TimeoutError:
                    st.error("Request timed out. Please try again.")
                    result = {"type": "sync", "answer": "Request timed out."}
                except Exception as e:
                    st.error(f"Unexpected error: {e}")
                    logger.exception("think() failed")
                    result = {"type": "sync", "answer": f"An error occurred: {e}"}

                # Handle response type
                if result.get("type") == "async":
                    task_id = result["task_id"]
                    variant = result["variant"]
                    # Store task in session state
                    st.session_state.tasks[task_id] = {
                        "variant": variant,
                        "status": "submitted",
                        "progress": 0,
                        "result_url": None,
                        "error": None
                    }
                    # Display task container with progress
                    task_container = st.container()
                    with task_container:
                        st.markdown(f"**{variant} task started**")
                        st.code(f"Task ID: {task_id}")
                        progress_placeholder = st.empty()
                        status_placeholder = st.empty()
                        refresh_btn = st.button("Refresh Progress", key=f"refresh_{task_id}")
                        if refresh_btn:
                            task_info = get_task_status(task_id)
                            if task_info:
                                st.session_state.tasks[task_id].update(task_info)
                        # Show current progress from session state
                        task_info = st.session_state.tasks[task_id]
                        if task_info["status"] == "processing":
                            progress_placeholder.progress(task_info["progress"]/100, text=f"{task_info['progress']}%")
                        elif task_info["status"] == "completed":
                            progress_placeholder.success("✅ Completed!")
                            if task_info["result_url"]:
                                status_placeholder.markdown(f"[View result]({task_info['result_url']})")
                        elif task_info["status"] == "failed":
                            progress_placeholder.error(f"❌ Failed: {task_info['error']}")
                        else:
                            progress_placeholder.info("⏳ Queued...")
                    full_response = f"**{variant} task started** – use the refresh button above to check progress."
                else:
                    # Sync response
                    if st.session_state.show_thinking and result.get("monologue"):
                        with st.expander("Show thinking process"):
                            st.markdown(result["monologue"])
                    answer = result.get("answer", "")
                    stream_response(response_placeholder, answer, delay=0.03)

                    if result.get("sources"):
                        with st.expander("View source documents"):
                            for i, doc in enumerate(result["sources"][:3]):
                                st.markdown(f"**Source {i+1}:**")
                                st.write(doc.page_content[:500] + "..." if len(doc.page_content) > 500 else doc.page_content)
                                if i < len(result["sources"]) - 1:
                                    st.divider()
                    st.caption(f"Thought for {result.get('thinking_time', 0):.2f}s")

        except Exception as e:
            st.error(f"An unexpected error occurred: {e}")
            logger.exception("Fatal error in main chat loop")
            full_response = ""
            result = {"type": "sync", "answer": full_response}

    # Save assistant message to history (for async, we store a short message; full details are in the container)
    if result and result.get("type") == "async":
        st.session_state.messages.append({"role": "assistant", "content": f"**{variant} task started** – use the refresh button to check progress."})
    else:
        st.session_state.messages.append({"role": "assistant", "content": answer if answer else full_response})

# ----------------------------------------------------------------------
# End of file
# ----------------------------------------------------------------------
