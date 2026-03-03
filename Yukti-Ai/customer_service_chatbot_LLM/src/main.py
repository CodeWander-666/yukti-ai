"""
Streamlit frontend – provides cyberpunk UI, model selection, chat,
knowledge base management, and async task display (video/image/audio).
Integrates language detection and ensures smooth user experience.
"""

import os
import sys
import time
import logging
import tempfile
from pathlib import Path

# Ensure project root is in path (for cron compatibility, but not strictly needed here)
PROJECT_ROOT = Path(__file__).parent.parent.absolute()
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

# Third‑party imports (with graceful fallbacks)
try:
    import streamlit as st
except ImportError:
    print("Streamlit not installed. Please install it.")
    sys.exit(1)

import pandas as pd
import requests

# Local imports – each may raise ImportError if missing; we catch and show friendly message.
try:
    from langchain_helper import get_embeddings, VECTORDB_PATH, BASE_DIR, create_vector_db, check_kb_status
except ImportError as e:
    st.error(f"Failed to import langchain_helper: {e}")
    st.stop()

try:
    from think import think
except ImportError as e:
    st.error(f"Failed to import think: {e}")
    st.stop()

try:
    from model_manager import (
        get_available_models,
        MODELS,
        get_active_tasks,
        get_task_status,
        ZHIPU_AVAILABLE,
        load_model,
    )
except ImportError as e:
    st.error(f"Failed to import model_manager: {e}")
    st.stop()

try:
    from language_detector import detect_language
except ImportError as e:
    st.warning(f"Language detector not available: {e}. Responses will default to English.")
    # Define a fallback
    def detect_language(text):
        return {"language": "en", "method": "fallback", "explicit_instruction": None}

try:
    from ui_helpers import render_task
except ImportError:
    # Define a simple fallback if ui_helpers not present
    def render_task(task_id, task_info):
        st.write(f"Task {task_id}: {task_info.get('status')}")

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# ----------------------------------------------------------------------
# Page configuration – must be first Streamlit command
# ----------------------------------------------------------------------
st.set_page_config(
    page_title="Yukti AI",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ----------------------------------------------------------------------
# Cyberpunk CSS with 3D effects and media thumbnails
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
    /* Media thumbnails */
    .stImage, .stVideo, .stAudio {
        border-radius: 12px;
        border: 2px solid #00ffff;
        box-shadow: 0 0 15px #00ffff;
        margin: 0.5rem 0;
        max-width: 300px;
    }
    /* Progress bar */
    .stProgress > div > div > div {
        background: linear-gradient(90deg, #ff00ff, #00ffff);
    }
    /* URL sticker */
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
    /* Model selector – styled as 3D button */
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
# Custom title: "Yukti" + neon "AI"
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
        st.session_state.knowledge_base_ready = check_kb_status()
    except Exception as e:
        logger.error(f"Failed to check knowledge base status: {e}")
        st.session_state.knowledge_base_ready = False

if "tasks" not in st.session_state:
    st.session_state.tasks = {}

if "uploaded_files" not in st.session_state:
    st.session_state.uploaded_files = {}

if "conversation_language" not in st.session_state:
    st.session_state.conversation_language = None  # Will be set after first message

# ----------------------------------------------------------------------
# Helper functions
# ----------------------------------------------------------------------
def rebuild_knowledge_base():
    """Wrapper for create_vector_db with error handling and user feedback."""
    try:
        with st.spinner("Rebuilding knowledge base..."):
            success = create_vector_db()
        if success:
            st.session_state.knowledge_base_ready = True
            st.success("✅ Knowledge base updated!")
        else:
            st.error("❌ Knowledge base update failed. Check logs.")
    except Exception as e:
        st.error(f"❌ Unexpected error: {e}")
        logger.exception("rebuild_knowledge_base")

def stream_response(placeholder, full_text, delay=0.02):
    """Simulate streaming by gradually revealing words."""
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

# ----------------------------------------------------------------------
# Sidebar
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
    try:
        model_options = get_available_models()
        if not model_options:
            st.warning("No models available. Check API keys.")
            model_options = ["Yukti‑Flash"]  # fallback
    except Exception as e:
        st.error(f"Failed to fetch models: {e}")
        logger.exception("get_available_models")
        model_options = ["Yukti‑Flash"]

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

    # Async tasks section
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
            logger.exception("get_active_tasks")

        for task_id in list(st.session_state.tasks.keys()):
            render_task(task_id, st.session_state.tasks[task_id])
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
                    # Download button
                    try:
                        img_data = requests.get(media["url"]).content
                        st.download_button(
                            "📥 Download Image",
                            data=img_data,
                            file_name=f"yukti_image_{hash(media['url'])}.png",
                            key=f"dl_img_{hash(media['url'])}"
                        )
                    except Exception as e:
                        st.error(f"Download failed: {e}")
                        logger.exception("Image download")
                elif media["type"] == "audio":
                    st.audio(media["url"])
                    # Download button
                    try:
                        with open(media["url"], "rb") as f:
                            audio_data = f.read()
                        st.download_button(
                            "📥 Download Audio",
                            data=audio_data,
                            file_name=f"yukti_audio_{hash(media['url'])}.wav",
                            key=f"dl_audio_{hash(media['url'])}"
                        )
                    except Exception as e:
                        st.error(f"Download failed: {e}")
                        logger.exception("Audio download")
                elif media["type"] == "video":
                    st.video(media["url"])
                    # Download button (streaming to avoid memory issues)
                    try:
                        with st.spinner("Preparing download..."):
                            response = requests.get(media["url"], stream=True)
                            response.raise_for_status()
                            st.download_button(
                                "📥 Download Video",
                                data=response.iter_content(chunk_size=8192),
                                file_name=f"yukti_video_{hash(media['url'])}.mp4",
                                key=f"dl_video_{hash(media['url'])}"
                            )
                    except Exception as e:
                        st.error(f"Download failed: {e}")
                        logger.exception("Video download")

if prompt := st.chat_input("Ask me anything..."):
    # Detect language and tone
    try:
        lang_info = detect_language(prompt)
        target_lang = lang_info['language']
        explicit = lang_info.get('explicit_instruction')
        logger.info(f"Detected language: {target_lang} (method: {lang_info.get('method')}, explicit: {explicit})")
    except Exception as e:
        logger.warning(f"Language detection failed: {e}")
        target_lang = 'en'
        explicit = None

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
                # Save uploaded file to temp location
                with tempfile.NamedTemporaryFile(delete=False, suffix=Path(uploaded_file.name).suffix) as tmp:
                    tmp.write(uploaded_file.getvalue())
                    extra_kwargs["image_url"] = tmp.name  # for video/image
                    # For audio, we might need to handle differently; but video/image only accept image_url

            # For text models, call think()
            if model_key in ["Yukti‑Flash", "Yukti‑Quantum"]:
                if not st.session_state.knowledge_base_ready:
                    full_response = "The knowledge base is not ready. Please click 'Update' in the sidebar first."
                    response_placeholder.markdown(full_response)
                    result = {"type": "sync", "answer": full_response}
                else:
                    history = [
                        {"role": msg["role"], "content": msg["content"]}
                        for msg in st.session_state.messages[-10:]  # last 10 messages
                    ]
                    with st.spinner("Thinking..."):
                        result = think(prompt, history, model_key, language=st.session_state.conversation_language)
            else:
                # Generation models
                model = load_model(model_key)
                with st.spinner("Generating..."):
                    if model_key == "Yukti‑Audio":
                        voice = "female"  # could be made configurable
                        audio_path = model.invoke(prompt, voice=voice, language=st.session_state.conversation_language, **extra_kwargs)
                        with open(audio_path, "rb") as f:
                            audio_bytes = f.read()
                        st.audio(audio_bytes, format="audio/wav")
                        st.download_button(
                            "📥 Download Audio",
                            data=audio_bytes,
                            file_name=f"yukti_audio_{int(time.time())}.wav"
                        )
                        full_response = "Audio generated."
                        result = {"type": "sync", "format": "audio"}
                        media.append({"type": "audio", "url": audio_path})
                    elif model_key == "Yukti‑Image":
                        image_url = model.invoke(prompt, **extra_kwargs)
                        st.image(image_url, use_container_width=False, width=300)
                        try:
                            img_data = requests.get(image_url).content
                            st.download_button(
                                "📥 Download Image",
                                data=img_data,
                                file_name=f"yukti_image_{int(time.time())}.png"
                            )
                        except Exception as e:
                            st.error(f"Download failed: {e}")
                        full_response = "Image generated."
                        result = {"type": "sync", "format": "image"}
                        media.append({"type": "image", "url": image_url})
                    elif model_key == "Yukti‑Video":
                        task_id = model.invoke(prompt, language=st.session_state.conversation_language, **extra_kwargs)
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
                        # Fallback to think for any other model
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

    # Append assistant message to session state
    if result and result.get("type") == "sync" and result.get("answer"):
        st.session_state.messages.append({"role": "assistant", "content": answer, "media": media})
    elif result and result.get("type") == "async":
        st.session_state.messages.append({"role": "assistant", "content": full_response})
    elif full_response:
        st.session_state.messages.append({"role": "assistant", "content": full_response})
