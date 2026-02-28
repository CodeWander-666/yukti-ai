"""
Yukti AI – Main Application (Ultimate Edition)
Orchestrates all models with real‑time progress, futuristic UI, and bulletproof error handling.
"""

import os
import sys
import time
import logging
import tempfile
from pathlib import Path

# Add project root to path
PROJECT_ROOT = Path(__file__).parent.parent.parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import streamlit as st
import pandas as pd
import requests
from langchain_core.documents import Document
from langchain_community.vectorstores import FAISS

from langchain_helper import get_embeddings, VECTORDB_PATH, BASE_DIR, create_vector_db
from think import think
from model_manager import (
    get_available_models,
    MODELS,
    get_active_tasks,
    get_task_status,
    ZHIPU_AVAILABLE,
    load_model,
)

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# ----------------------------------------------------------------------
# Page configuration (MUST be first)
# ----------------------------------------------------------------------
st.set_page_config(
    page_title="Yukti AI",
    page_icon="✨",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ----------------------------------------------------------------------
# Futuristic UI with glass morphism and gradients
# ----------------------------------------------------------------------
st.markdown("""
<style>
    /* Global theme */
    .stApp {
        background: linear-gradient(135deg, #0b0c1e 0%, #1a1b2f 100%);
        color: #e0e0ff;
    }
    /* Chat messages */
    .stChatMessage {
        border-radius: 20px;
        padding: 1rem;
        margin-bottom: 1rem;
        backdrop-filter: blur(10px);
        border: 1px solid rgba(255,255,255,0.1);
        box-shadow: 0 8px 32px rgba(0,0,0,0.3);
    }
    .stChatMessage[data-testid="user-message"] {
        background: rgba(255,255,255,0.05);
        border-left: 4px solid #00f2fe;
    }
    .stChatMessage[data-testid="assistant-message"] {
        background: rgba(20,40,80,0.3);
        border-left: 4px solid #ff6a88;
    }
    /* Sidebar */
    .css-1d391kg {
        background: rgba(10,10,20,0.8);
        backdrop-filter: blur(10px);
        border-right: 1px solid rgba(255,255,255,0.05);
    }
    /* Buttons */
    .stButton > button {
        background: linear-gradient(90deg, #667eea 0%, #764ba2 100%);
        color: white;
        border: none;
        border-radius: 50px;
        padding: 0.5rem 2rem;
        font-weight: 600;
        transition: all 0.3s;
        box-shadow: 0 4px 15px rgba(102,126,234,0.4);
    }
    .stButton > button:hover {
        transform: translateY(-2px);
        box-shadow: 0 6px 20px rgba(102,126,234,0.6);
    }
    /* Input field */
    .stTextInput > div > div > input {
        background: rgba(255,255,255,0.05);
        border: 1px solid rgba(255,255,255,0.1);
        border-radius: 50px;
        color: white;
        padding: 0.75rem 1.5rem;
        font-size: 1.1rem;
    }
    .stTextInput > div > div > input:focus {
        border-color: #667eea;
        box-shadow: 0 0 0 2px rgba(102,126,234,0.3);
    }
    /* Headers */
    h1, h2, h3 {
        font-family: 'Inter', sans-serif;
        background: linear-gradient(90deg, #f0f0ff, #a0b0ff);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        font-weight: 600;
    }
    /* Task containers */
    .task-container {
        background: rgba(20,20,40,0.6);
        backdrop-filter: blur(10px);
        border-radius: 15px;
        padding: 1rem;
        margin: 0.5rem 0;
        border: 1px solid rgba(255,255,255,0.1);
    }
    /* Progress bar */
    .stProgress > div > div > div {
        background: linear-gradient(90deg, #00f2fe, #ff6a88);
    }
    /* Hide Streamlit branding */
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
</style>
""", unsafe_allow_html=True)

# ----------------------------------------------------------------------
# Title and subtitle
# ----------------------------------------------------------------------
st.title("Yukti AI")
st.caption("Your Futuristic Cognitive Companion")

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
if "task_refresh_counter" not in st.session_state:
    st.session_state.task_refresh_counter = 0

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
# Helper functions
# ----------------------------------------------------------------------
def load_all_documents():
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
                except Exception as e:
                    logger.warning(f"Row {idx} formatting error: {e}")
                    continue

            st.sidebar.success(f"Loaded {len(df)} rows from {src['name']}")

        except Exception as e:
            st.sidebar.error(f"Error reading {path}: {e}")
            logger.exception("load_all_documents fatal")
            return None
    return docs

def rebuild_knowledge_base():
    with st.spinner("Loading documents..."):
        docs = load_all_documents()
        if docs is None:
            st.error("Failed to load documents. Check logs.")
            return False
        if not docs:
            st.warning("No documents found.")
            return False
    with st.spinner("Generating embeddings and building index..."):
        try:
            embeddings = get_embeddings()
            vectordb = FAISS.from_documents(docs, embeddings)
            vectordb.save_local(VECTORDB_PATH)
            st.session_state.knowledge_base_ready = True
            st.success(f"Knowledge base rebuilt with {len(docs)} documents!")
            return True
        except Exception as e:
            st.error(f"Build failed: {e}")
            logger.exception("rebuild_knowledge_base")
            return False

def stream_response(placeholder, full_text, delay=0.02):
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
        placeholder.markdown(full_text)

def render_task(task_id):
    """Display a single task container with progress and controls."""
    task_info = st.session_state.tasks.get(task_id)
    if not task_info:
        return

    with st.container():
        st.markdown(f"<div class='task-container'>", unsafe_allow_html=True)
        cols = st.columns([3, 1])
        with cols[0]:
            st.markdown(f"**{task_info['variant']}**  \n`{task_id[:8]}`")
            if task_info['status'] == 'processing':
                st.progress(task_info['progress']/100, text=f"{task_info['progress']}%")
            elif task_info['status'] == 'completed':
                st.success("✅ Completed")
            elif task_info['status'] == 'failed':
                st.error(f"❌ Failed: {task_info['error']}")
            else:
                st.info("⏳ Queued...")
        with cols[1]:
            if task_info['status'] == 'processing':
                if st.button("Refresh", key=f"refresh_{task_id}"):
                    updated = get_task_status(task_id)
                    if updated:
                        st.session_state.tasks[task_id].update(updated)
                    st.rerun()
            elif task_info['status'] == 'completed' and task_info.get('result_url'):
                # Provide download button
                if task_info['variant'] == 'Yukti‑Video':
                    st.markdown(f"[🎬 Watch Video]({task_info['result_url']})")
                    st.download_button("Download", data=requests.get(task_info['result_url']).content,
                                       file_name=f"yukti_video_{task_id[:8]}.mp4")
                else:
                    st.image(task_info['result_url'], use_container_width=True)
                    st.download_button("Download Image", data=requests.get(task_info['result_url']).content,
                                       file_name=f"yukti_image_{task_id[:8]}.png")
        st.markdown("</div>", unsafe_allow_html=True)

# ----------------------------------------------------------------------
# Sidebar
# ----------------------------------------------------------------------
with st.sidebar:
    st.markdown("## 🧠 Control Panel")

    st.markdown("### Knowledge Base")
    if st.button("🔄 Update Knowledge Base", use_container_width=True):
        rebuild_knowledge_base()
    st.divider()

    st.markdown("### Status")
    if st.session_state.knowledge_base_ready:
        st.markdown("✅ **Active**")
    else:
        st.markdown("⚠️ **Not built** – click update above.")

    st.divider()

    st.markdown("### Brain")
    try:
        model_options = get_available_models()
    except Exception as e:
        st.error(f"Failed to load models: {e}")
        model_options = []
    selected_model = st.selectbox(
        "Choose thinking style",
        options=model_options,
        format_func=lambda x: f"{x} – {MODELS[x]['description']}",
        key="model_selector"
    )
    st.session_state.selected_model = selected_model

    # Additional parameters based on model
    if selected_model == "Yukti‑Audio":
        voice = st.selectbox("Voice", ["female", "male", "jam", "kazi", "douji"], index=0)
        st.session_state.voice = voice
    elif selected_model == "Yukti‑Video":
        quality = st.selectbox("Quality", ["quality", "speed"], index=0)
        with_audio = st.checkbox("Include audio", value=True)
        size = st.selectbox("Resolution", ["1920x1080", "1280x720", "3840x2160"], index=0)
        fps = st.selectbox("FPS", [30, 60], index=0)
        st.session_state.video_params = {
            "quality": quality,
            "with_audio": with_audio,
            "size": size,
            "fps": fps
        }

    st.divider()

    st.markdown("### Display")
    st.session_state.show_thinking = st.checkbox("Show thinking process", value=True)

    st.divider()

    if ZHIPU_AVAILABLE:
        st.markdown("### 📋 Active Tasks")
        if st.button("🔄 Refresh All Tasks", use_container_width=True):
            st.rerun()
        # Fetch fresh task list
        try:
            active_tasks = get_active_tasks()
            # Update session state tasks
            for task_id, variant, status, progress in active_tasks:
                if task_id not in st.session_state.tasks:
                    st.session_state.tasks[task_id] = {
                        "variant": variant,
                        "status": status,
                        "progress": progress,
                        "result_url": None,
                        "error": None
                    }
                else:
                    st.session_state.tasks[task_id].update({
                        "status": status,
                        "progress": progress
                    })
        except Exception as e:
            st.warning(f"Could not fetch tasks: {e}")

        # Render all active tasks
        for task_id in list(st.session_state.tasks.keys()):
            render_task(task_id)
        st.divider()

    st.markdown("### Sources")
    for src in SOURCES:
        st.markdown(f"- **{src['name']}**")

    st.divider()

    if st.button("🗑️ Clear Conversation", use_container_width=True):
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
            model_key = st.session_state.selected_model
            config = MODELS.get(model_key, {})

            # Check if knowledge base is needed (for text models)
            if config.get("type") == "sync" and config.get("api") == "chat":
                if not st.session_state.knowledge_base_ready:
                    full_response = "The knowledge base is not ready. Please click 'Update Knowledge Base' in the sidebar first."
                    response_placeholder.markdown(full_response)
                    result = {"type": "sync", "answer": full_response}
                else:
                    history = [
                        {"role": msg["role"], "content": msg["content"]}
                        for msg in st.session_state.messages[-10:]
                    ]
                    with st.spinner("Thinking..."):
                        result = think(prompt, history, model_key)
            else:
                # Generation models (image, video, audio) – bypass think
                model = load_model(model_key)
                with st.spinner("Generating..."):
                    if model_key == "Yukti‑Audio":
                        result = model.invoke(prompt, voice=st.session_state.get("voice", "female"))
                        # result is a file path
                        with open(result, "rb") as f:
                            audio_bytes = f.read()
                        st.audio(audio_bytes, format="audio/wav")
                        st.download_button("Download Audio", data=audio_bytes, file_name="yukti_audio.wav")
                        full_response = "Audio generated."
                        result = {"type": "sync", "format": "audio"}
                    elif model_key == "Yukti‑Image":
                        image_url = model.invoke(prompt)
                        st.image(image_url, use_container_width=True)
                        # Download button
                        img_data = requests.get(image_url).content
                        st.download_button("Download Image", data=img_data, file_name="yukti_image.png")
                        full_response = "Image generated."
                        result = {"type": "sync", "format": "image"}
                    elif model_key == "Yukti‑Video":
                        params = st.session_state.get("video_params", {})
                        task_id = model.invoke(prompt, **params)
                        st.session_state.tasks[task_id] = {
                            "variant": "Yukti‑Video",
                            "status": "submitted",
                            "progress": 0,
                            "result_url": None,
                            "error": None
                        }
                        full_response = f"Video task started: `{task_id}`"
                        result = {"type": "async", "task_id": task_id}
                    else:
                        # Fallback to think for other sync models (like Gemini)
                        history = [
                            {"role": msg["role"], "content": msg["content"]}
                            for msg in st.session_state.messages[-10:]
                        ]
                        with st.spinner("Thinking..."):
                            result = think(prompt, history, model_key)

            # Handle result from think (if any)
            if result and result.get("type") == "async":
                # Already handled above, but we have task_id
                task_id = result.get("task_id")
                # Optionally show a mini progress indicator in chat
                st.info(f"Task {task_id} submitted. Check sidebar for progress.")
            elif result and result.get("type") == "sync" and result.get("answer"):
                answer = result.get("answer", "")
                if st.session_state.show_thinking and result.get("monologue"):
                    with st.expander("Show thinking process"):
                        st.markdown(result["monologue"])
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

    # Save assistant message to history (only for text responses)
    if result and result.get("type") == "sync" and result.get("answer"):
        st.session_state.messages.append({"role": "assistant", "content": answer})
    elif result and result.get("type") == "async":
        st.session_state.messages.append({"role": "assistant", "content": full_response})
    elif full_response:
        st.session_state.messages.append({"role": "assistant", "content": full_response})
