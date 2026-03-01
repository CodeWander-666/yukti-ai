"""
Yukti AI – Main Application (Cyberpunk Ultimate Edition)
Neon‑themed UI, 3D model selector with press effects, persistent media,
dynamic language adaptation (Hindi, English, Hinglish, explicit instructions, tone).
"""

import os
import sys
import time
import logging
import tempfile
from pathlib import Path

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
# NEW: Import language detector
from language_detector import detect_language

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# ----------------------------------------------------------------------
# Page configuration – no sparkle, we'll use custom HTML for the title
# ----------------------------------------------------------------------
st.set_page_config(
    page_title="Yukti AI",
    page_icon=" ",   # empty (we'll add custom neon in the title)
    layout="wide",
    initial_sidebar_state="expanded"
)

# ----------------------------------------------------------------------
# Cyberpunk CSS with 3D selectbox press effects
# ----------------------------------------------------------------------
st.markdown("""
<style>
    /* Global cyberpunk theme */
    .stApp {
        background: linear-gradient(135deg, #0d0b1a 0%, #1a1a2f 100%);
        color: #e0e0ff;
    }
    /* Chat messages with neon glow */
    .stChatMessage {
        border-radius: 20px;
        padding: 1rem;
        margin-bottom: 1rem;
        backdrop-filter: blur(10px);
        border: 1px solid rgba(255,0,255,0.3);
        box-shadow: 0 0 20px rgba(255,0,255,0.3), 0 8px 32px rgba(0,0,0,0.5);
    }
    .stChatMessage[data-testid="user-message"] {
        background: rgba(255,0,255,0.05);
        border-left: 4px solid #ff00ff;
    }
    .stChatMessage[data-testid="assistant-message"] {
        background: rgba(0,255,255,0.05);
        border-left: 4px solid #00ffff;
    }
    /* Sidebar */
    .css-1d391kg {
        background: rgba(10,10,20,0.9);
        backdrop-filter: blur(10px);
        border-right: 1px solid #ff00ff;
        box-shadow: -5px 0 20px rgba(255,0,255,0.3);
    }
    /* 3D buttons */
    .stButton > button {
        background: linear-gradient(145deg, #1e1e3f, #2a2a5a);
        border: none;
        border-radius: 15px;
        padding: 0.6rem 1.5rem;
        color: #fff;
        font-weight: 600;
        font-size: 1rem;
        box-shadow: 0 5px 0 #0b0b1a, 0 10px 20px rgba(255,0,255,0.4);
        transition: all 0.1s ease;
        transform: translateY(0);
        margin: 0.5rem 0;
        width: 100%;
        cursor: pointer;
        border-bottom: 2px solid #ff00ff;
    }
    .stButton > button:hover {
        transform: translateY(-2px);
        box-shadow: 0 7px 0 #0b0b1a, 0 15px 25px rgba(0,255,255,0.5);
        background: linear-gradient(145deg, #2a2a5a, #3a3a7a);
    }
    .stButton > button:active {
        transform: translateY(4px);
        box-shadow: 0 2px 0 #0b0b1a, 0 8px 15px rgba(255,0,255,0.4);
    }
    /* File uploader */
    .stFileUploader {
        background: rgba(30,30,60,0.5);
        border-radius: 15px;
        padding: 1rem;
        border: 1px dashed #ff00ff;
        box-shadow: 0 0 15px rgba(255,0,255,0.3);
    }
    /* Compact images */
    .stImage {
        max-width: 300px;
        max-height: 300px;
        border-radius: 12px;
        border: 2px solid #00ffff;
        box-shadow: 0 0 15px #00ffff;
        margin: 0.5rem 0;
    }
    /* Progress bar */
    .stProgress > div > div > div {
        background: linear-gradient(90deg, #ff00ff, #00ffff);
    }
    /* Neon URL Sticker */
    .url-wrapper {
        display: flex;
        align-items: center;
        gap: 12px;
        background: #1a1a1a;
        padding: 10px 20px;
        border-radius: 8px;
        font-family: 'Inter', sans-serif;
        margin-bottom: 1rem;
        border: 1px solid #00ffff;
        box-shadow: 0 0 15px #00ffff;
    }
    .neon-sticker {
        font-size: 0.7rem;
        font-weight: 900;
        color: #fff;
        background: #000;
        padding: 2px 8px;
        border-radius: 4px;
        text-transform: uppercase;
        letter-spacing: 1px;
        border: 1.5px solid #ff00ff;
        box-shadow: 0 0 5px #ff00ff, inset 0 0 5px #ff00ff;
        text-shadow: 0 0 2px #fff, 0 0 8px #ff00ff;
        user-select: none;
        animation: neon-pulse 1.5s infinite alternate;
    }
    .url-text {
        color: #888;
        text-decoration: none;
        font-size: 0.9rem;
        transition: color 0.3s ease;
    }
    .url-wrapper:hover .url-text {
        color: #fff;
    }
    @keyframes neon-pulse {
        from { opacity: 1; }
        to { opacity: 0.8; box-shadow: 0 0 15px #ff00ff, inset 0 0 8px #ff00ff; }
    }
    /* Model selector – styled as a 3D button with press effect */
    .stSelectbox > div > div {
        background: linear-gradient(145deg, #1e1e3f, #2a2a5a) !important;
        border: 1px solid #ff00ff !important;
        border-radius: 15px !important;
        color: white !important;
        box-shadow: 0 5px 0 #0b0b1a, 0 10px 20px rgba(255,0,255,0.3) !important;
        transition: all 0.1s ease;
        position: relative !important;
        padding: 0.6rem 1rem !important;
        margin: 0.5rem 0 !important;
        cursor: pointer;
        font-weight: 600;
        text-align: center;
        border-bottom: 2px solid #ff00ff !important;
        transform: translateY(0);
    }
    .stSelectbox > div > div:hover {
        transform: translateY(-2px);
        box-shadow: 0 7px 0 #0b0b1a, 0 15px 25px rgba(0,255,255,0.4) !important;
        background: linear-gradient(145deg, #2a2a5a, #3a3a7a) !important;
    }
    .stSelectbox > div > div:active {
        transform: translateY(4px);
        box-shadow: 0 2px 0 #0b0b1a, 0 8px 15px rgba(255,0,255,0.4) !important;
    }
    .stSelectbox > div > div > div {
        color: white !important;
    }
    /* Hide Streamlit branding */
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
</style>
""", unsafe_allow_html=True)

# ----------------------------------------------------------------------
# Custom title: "Yukti" followed by neon "AI"
# ----------------------------------------------------------------------
st.markdown("""
<div style="display: flex; align-items: center; justify-content: center; gap: 10px;">
    <span style="font-size: 2.5rem; font-weight: 600; color: #e0e0ff; text-shadow: 0 0 10px #00ffff;">Yukti</span>
    <span style="
        font-size: 2.5rem;
        font-weight: 900;
        color: #fff;
        background: #000;
        padding: 5px 15px;
        border-radius: 8px;
        text-transform: uppercase;
        letter-spacing: 2px;
        border: 2px solid #ff00ff;
        box-shadow: 0 0 15px #ff00ff, inset 0 0 10px #ff00ff;
        text-shadow: 0 0 5px #fff, 0 0 15px #ff00ff;
        animation: neon-pulse 1.5s infinite alternate;
    ">AI</span>
</div>
<p style='text-align: center; color: #aaa;'>Your Futuristic Cognitive Companion</p>
""", unsafe_allow_html=True)

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
if "tasks" not in st.session_state:
    st.session_state.tasks = {}
if "uploaded_files" not in st.session_state:
    st.session_state.uploaded_files = {}
# NEW: Track conversation language for consistency
if "conversation_language" not in st.session_state:
    st.session_state.conversation_language = None  # will be set after first message

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
# Helper functions (unchanged)
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
    task_info = st.session_state.tasks.get(task_id)
    if not task_info:
        return
    with st.container():
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
                if st.button("⟳", key=f"refresh_{task_id}"):
                    updated = get_task_status(task_id)
                    if updated:
                        st.session_state.tasks[task_id].update(updated)
                    st.rerun()
            elif task_info['status'] == 'completed' and task_info.get('result_url'):
                if task_info['variant'] == 'Yukti‑Video':
                    st.markdown(f"[🎬 Watch Video]({task_info['result_url']})")
                    st.download_button("📥 Download", data=requests.get(task_info['result_url']).content,
                                       file_name=f"yukti_video_{task_id[:8]}.mp4")
                else:
                    st.image(task_info['result_url'], use_container_width=False, width=300)
                    st.download_button("📥 Download Image", data=requests.get(task_info['result_url']).content,
                                       file_name=f"yukti_image_{task_id[:8]}.png")

# ----------------------------------------------------------------------
# Sidebar (unchanged)
# ----------------------------------------------------------------------
with st.sidebar:
    st.markdown("""
    <div class="url-wrapper">
      <span class="neon-sticker">YuktiAI</span>
      <a href="https://yukti.ai" target="_blank" class="url-text">https://yukti.ai</a>
    </div>
    """, unsafe_allow_html=True)
    st.divider()

    st.markdown("## 🧠 Brain")
    model_options = get_available_models()
    display_names = [
        f"{m} – {MODELS.get(m, {}).get('description', 'No description')}"
        for m in model_options
    ]
    selected_display = st.selectbox(
        label="Select model",
        options=display_names,
        index=0,
        key="model_display",
        label_visibility="collapsed"
    )
    selected_model = model_options[display_names.index(selected_display)]
    st.session_state.selected_model = selected_model

    uploaded_file = None
    if selected_model in ["Yukti‑Video", "Yukti‑Image", "Yukti‑Audio"]:
        st.markdown("### 📎 Attach File")
        if selected_model == "Yukti‑Video":
            uploaded_file = st.file_uploader("Upload image (optional)", type=["png", "jpg", "jpeg"])
        elif selected_model == "Yukti‑Image":
            uploaded_file = st.file_uploader("Upload reference image (optional)", type=["png", "jpg", "jpeg"])
        elif selected_model == "Yukti‑Audio":
            uploaded_file = st.file_uploader("Upload audio (optional)", type=["mp3", "wav"])

    st.divider()
    st.markdown("### 📚 Knowledge Base")
    if st.button("🔄 Update", use_container_width=True):
        rebuild_knowledge_base()
    if st.session_state.knowledge_base_ready:
        st.markdown("✅ **Active**")
    else:
        st.markdown("⚠️ **Not built**")
    st.divider()

    if ZHIPU_AVAILABLE:
        st.markdown("### 📋 Tasks")
        if st.button("⟳ Refresh Tasks", use_container_width=True):
            st.rerun()
        try:
            active_tasks = get_active_tasks()
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

        for task_id in list(st.session_state.tasks.keys()):
            render_task(task_id)
        st.divider()

    if st.button("🗑️ Clear Chat", use_container_width=True):
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
        if "media" in msg:
            for media in msg["media"]:
                if media["type"] == "image":
                    st.image(media["url"], use_container_width=False, width=300)
                elif media["type"] == "audio":
                    st.audio(media["url"])
                elif media["type"] == "video":
                    st.video(media["url"])

if prompt := st.chat_input("Ask me anything..."):
    # Detect language and tone
    lang_info = detect_language(prompt)
    target_lang = lang_info['language']
    explicit = lang_info['explicit_instruction']
    logger.info(f"Detected language: {target_lang} (method: {lang_info['method']}, explicit: {explicit})")
    
    # Update conversation language if explicit instruction or first message
    if explicit:
        st.session_state.conversation_language = explicit
    elif st.session_state.conversation_language is None:
        st.session_state.conversation_language = target_lang
    # else keep previous conversation language

    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message("assistant"):
        response_placeholder = st.empty()
        result = None
        answer = ""
        full_response = ""
        media = []

        try:
            model_key = st.session_state.selected_model
            config = MODELS.get(model_key, {})
            extra_kwargs = {}
            if uploaded_file is not None:
                with tempfile.NamedTemporaryFile(delete=False, suffix=Path(uploaded_file.name).suffix) as tmp:
                    tmp.write(uploaded_file.getvalue())
                    extra_kwargs["image_url"] = tmp.name

            # For text models, pass language as extra kwargs
            if model_key in ["Yukti‑Flash", "Yukti‑Quantum"]:
                if not st.session_state.knowledge_base_ready:
                    full_response = "The knowledge base is not ready. Please click 'Update' in the sidebar first."
                    response_placeholder.markdown(full_response)
                    result = {"type": "sync", "answer": full_response}
                else:
                    history = [
                        {"role": msg["role"], "content": msg["content"]}
                        for msg in st.session_state.messages[-10:]
                    ]
                    with st.spinner("Thinking..."):
                        # Pass the conversation language (or explicit if any) to think()
                        # The language will be used by the model via kwargs
                        result = think(prompt, history, model_key, language=st.session_state.conversation_language)
            else:
                # Generation models – may or may not use language; pass it anyway
                model = load_model(model_key)
                with st.spinner("Generating..."):
                    if model_key == "Yukti‑Audio":
                        voice = "female"
                        # Pass language hint (though GLM-4-Voice may not use it directly)
                        audio_path = model.invoke(prompt, voice=voice, language=st.session_state.conversation_language, **extra_kwargs)
                        with open(audio_path, "rb") as f:
                            audio_bytes = f.read()
                        st.audio(audio_bytes, format="audio/wav")
                        st.download_button("📥 Download Audio", data=audio_bytes, file_name="yukti_audio.wav")
                        full_response = "Audio generated."
                        result = {"type": "sync", "format": "audio"}
                        media.append({"type": "audio", "url": audio_path})
                    elif model_key == "Yukti‑Image":
                        image_url = model.invoke(prompt, **extra_kwargs)
                        st.image(image_url, use_container_width=False, width=300)
                        img_data = requests.get(image_url).content
                        st.download_button("📥 Download Image", data=img_data, file_name="yukti_image.png")
                        full_response = "Image generated."
                        result = {"type": "sync", "format": "image"}
                        media.append({"type": "image", "url": image_url})
                    elif model_key == "Yukti‑Video":
                        task_id = model.invoke(prompt, **extra_kwargs)
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
                        history = [
                            {"role": msg["role"], "content": msg["content"]}
                            for msg in st.session_state.messages[-10:]
                        ]
                        with st.spinner("Thinking..."):
                            result = think(prompt, history, model_key, language=st.session_state.conversation_language)

            if result and result.get("type") == "async":
                task_id = result.get("task_id")
                st.info(f"Task {task_id} submitted. Check sidebar for progress.")
            elif result and result.get("type") == "sync":
                answer = result.get("answer", "")
                if not answer:
                    answer = "(No response generated)"
                    st.warning("The model returned an empty response. Please try again.")
                if result.get("monologue"):
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

    if result and result.get("type") == "sync" and result.get("answer"):
        st.session_state.messages.append({"role": "assistant", "content": answer, "media": media})
    elif result and result.get("type") == "async":
        st.session_state.messages.append({"role": "assistant", "content": full_response})
    elif full_response:
        st.session_state.messages.append({"role": "assistant", "content": full_response})
