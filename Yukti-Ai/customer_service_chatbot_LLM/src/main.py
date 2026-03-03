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
from language_detector import detect_language

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# ----------------------------------------------------------------------
# Page configuration
# ----------------------------------------------------------------------
st.set_page_config(
    page_title="Yukti AI",
    page_icon=" ",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ----------------------------------------------------------------------
# Cyberpunk CSS (unchanged, plus styles for new fixed bar)
# ----------------------------------------------------------------------
st.markdown("""
<style>
    /* Global cyberpunk theme (kept as is) */
    .stApp {
        background: linear-gradient(135deg, #0d0b1a 0%, #1a1a2f 100%);
        color: #e0e0ff;
    }
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
    .css-1d391kg {
        background: rgba(10,10,20,0.9);
        backdrop-filter: blur(10px);
        border-right: 1px solid #ff00ff;
        box-shadow: -5px 0 20px rgba(255,0,255,0.3);
    }
    /* 3D buttons */
    .stButton > button, .custom-3d-button {
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
    .stButton > button:hover, .custom-3d-button:hover {
        transform: translateY(-2px);
        box-shadow: 0 7px 0 #0b0b1a, 0 15px 25px rgba(0,255,255,0.5);
        background: linear-gradient(145deg, #2a2a5a, #3a3a7a);
    }
    .stButton > button:active, .custom-3d-button:active {
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
    .stImage {
        max-width: 300px;
        max-height: 300px;
        border-radius: 12px;
        border: 2px solid #00ffff;
        box-shadow: 0 0 15px #00ffff;
        margin: 0.5rem 0;
    }
    .stProgress > div > div > div {
        background: linear-gradient(90deg, #ff00ff, #00ffff);
    }
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
    /* Make room for fixed bottom bar */
    .main > div {
        padding-bottom: 120px !important;
    }
    /* Fixed bottom bar */
    .fixed-bottom-bar {
        position: fixed;
        bottom: 0;
        left: 0;
        right: 0;
        background: rgba(10,10,20,0.95);
        backdrop-filter: blur(10px);
        border-top: 1px solid #ff00ff;
        padding: 0.5rem 1rem;
        z-index: 100;
        margin-left: 21rem;
    }
    @media (max-width: 992px) {
        .fixed-bottom-bar {
            margin-left: 0;
        }
    }
    .input-row {
        display: flex;
        gap: 10px;
        align-items: center;
        margin-bottom: 8px;
    }
    .button-row {
        display: flex;
        gap: 15px;
        justify-content: flex-start;
        margin-top: 5px;
    }
    .small-3d-button {
        background: linear-gradient(145deg, #1e1e3f, #2a2a5a);
        border: none;
        border-radius: 50%;
        width: 36px;
        height: 36px;
        color: #fff;
        font-size: 1.1rem;
        display: inline-flex;
        align-items: center;
        justify-content: center;
        cursor: pointer;
        box-shadow: 0 3px 0 #0b0b1a, 0 5px 10px rgba(255,0,255,0.4);
        transition: all 0.1s ease;
        transform: translateY(0);
        border-bottom: 2px solid #ff00ff;
    }
    .small-3d-button:hover {
        transform: translateY(-2px);
        box-shadow: 0 5px 0 #0b0b1a, 0 8px 15px rgba(0,255,255,0.5);
        background: linear-gradient(145deg, #2a2a5a, #3a3a7a);
    }
    .small-3d-button:active {
        transform: translateY(2px);
        box-shadow: 0 1px 0 #0b0b1a, 0 3px 8px rgba(255,0,255,0.4);
    }
    .send-button {
        background: linear-gradient(145deg, #ff00ff, #ff66ff);
        border: none;
        border-radius: 50%;
        width: 42px;
        height: 42px;
        color: white;
        font-size: 1.2rem;
        display: inline-flex;
        align-items: center;
        justify-content: center;
        cursor: pointer;
        box-shadow: 0 3px 0 #0b0b1a, 0 5px 10px rgba(255,0,255,0.6);
        transition: all 0.1s ease;
        transform: translateY(0);
        border-bottom: 2px solid #ff00ff;
    }
    .send-button:hover {
        transform: translateY(-2px);
        box-shadow: 0 5px 0 #0b0b1a, 0 8px 15px rgba(0,255,255,0.6);
    }
    .send-button:active {
        transform: translateY(2px);
        box-shadow: 0 1px 0 #0b0b1a, 0 3px 8px rgba(255,0,255,0.5);
    }
    .send-button:disabled {
        opacity: 0.5;
        cursor: not-allowed;
    }
    .stTextInput > div > input {
        background: rgba(30,30,60,0.8);
        border: 1px solid #ff00ff;
        border-radius: 24px;
        color: #e0e0ff;
        padding: 10px 15px;
        font-size: 0.95rem;
        width: 100%;
    }
    .stTextInput > div > input:focus {
        outline: none;
        box-shadow: 0 0 0 2px #00ffff;
    }
    .stTextInput > label {
        display: none;
    }
    /* Hide default chat input */
    .stChatInput {
        display: none !important;
    }
</style>
""", unsafe_allow_html=True)

# ----------------------------------------------------------------------
# Custom title
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
if "conversation_language" not in st.session_state:
    st.session_state.conversation_language = None
if "clear_input" not in st.session_state:
    st.session_state.clear_input = False
if "processing" not in st.session_state:
    st.session_state.processing = False
if "uploaded_file" not in st.session_state:
    st.session_state.uploaded_file = None

# ----------------------------------------------------------------------
# Data sources configuration (unchanged)
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
# JavaScript for voice, file, and auto‑scroll
# ----------------------------------------------------------------------
st.markdown("""
<script>
function waitForStreamlit() {
    if (window.parent.document.querySelector('input[data-testid="stTextInput"]')) {
        initializeFeatures();
    } else {
        setTimeout(waitForStreamlit, 100);
    }
}

function initializeFeatures() {
    let fileInput = document.getElementById('hidden-file-input');
    if (!fileInput) {
        fileInput = document.createElement('input');
        fileInput.type = 'file';
        fileInput.id = 'hidden-file-input';
        fileInput.style.display = 'none';
        document.body.appendChild(fileInput);
    }

    fileInput.addEventListener('change', function(e) {
        if (e.target.files.length > 0) {
            const file = e.target.files[0];
            const reader = new FileReader();
            reader.onload = function(ev) {
                const data = ev.target.result;
                const fileReceiver = window.parent.document.querySelector('input[data-testid="stTextInput"][key="file_receiver"]');
                if (fileReceiver) {
                    fileReceiver.value = file.name + ',' + data.split(',')[1];
                    fileReceiver.dispatchEvent(new Event('input', { bubbles: true }));
                }
            };
            reader.readAsDataURL(file);
        }
    });

    window.startVoiceRecognition = function() {
        if (!('webkitSpeechRecognition' in window) && !('SpeechRecognition' in window)) {
            alert('Voice recognition not supported in this browser. Please use Chrome or Edge.');
            return;
        }
        const recognition = new (window.SpeechRecognition || window.webkitSpeechRecognition)();
        recognition.lang = 'en-US';
        recognition.interimResults = false;
        recognition.maxAlternatives = 1;

        recognition.start();

        recognition.onresult = function(event) {
            const transcript = event.results[0][0].transcript;
            const textInput = window.parent.document.querySelector('input[data-testid="stTextInput"][key="message_input"]');
            if (textInput) {
                textInput.value = transcript;
                textInput.dispatchEvent(new Event('input', { bubbles: true }));
            }
        };

        recognition.onerror = function(event) {
            alert('Voice error: ' + event.error);
        };
    };

    window.triggerFileUpload = function() {
        document.getElementById('hidden-file-input').click();
    };
}

// Auto‑scroll
function scrollToBottom() {
    const chatContainer = window.parent.document.querySelector('.main > div');
    if (chatContainer) {
        chatContainer.scrollTop = chatContainer.scrollHeight;
    }
}
scrollToBottom();
const observer = new MutationObserver(scrollToBottom);
if (window.parent.document.querySelector('.main > div')) {
    observer.observe(window.parent.document.querySelector('.main > div'), { childList: true, subtree: true });
}

waitForStreamlit();
</script>
""", unsafe_allow_html=True)

# Hidden receiver for file data
file_receiver = st.text_input("file_receiver", key="file_receiver", label_visibility="collapsed", value="", placeholder="")

if file_receiver and not st.session_state.uploaded_file:
    try:
        parts = file_receiver.split(",", 1)
        if len(parts) == 2:
            file_name, b64data = parts
            file_bytes = base64.b64decode(b64data)
            st.session_state.uploaded_file = (file_name, file_bytes)
            st.session_state.file_receiver = ""
    except Exception as e:
        st.error(f"Failed to process file: {e}")
        logger.exception("File decode error")
        st.session_state.uploaded_file = None

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
# Main chat area – display messages
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

# ----------------------------------------------------------------------
# Custom fixed bottom bar (replaces st.chat_input)
# ----------------------------------------------------------------------
with st.container():
    st.markdown('<div class="fixed-bottom-bar">', unsafe_allow_html=True)

    # First row: text input + send button
    col1, col2 = st.columns([8, 1])
    with col1:
        # Handle input clearing
        if st.session_state.clear_input:
            default_value = ""
            st.session_state.clear_input = False
        else:
            default_value = None
        message_input = st.text_input(
            "Message",
            key="message_input",
            value=default_value,
            label_visibility="collapsed",
            placeholder="Type a message",
            disabled=st.session_state.processing
        )
    with col2:
        send_clicked = st.button(
            "➤",
            key="send_button",
            disabled=st.session_state.processing or not message_input,
            help="Send message"
        )

    # Second row: voice and file buttons
    col_voice, col_file, _ = st.columns([1, 1, 10])
    with col_voice:
        st.markdown('<button id="voice-button" class="small-3d-button">🎤</button>', unsafe_allow_html=True)
    with col_file:
        st.markdown('<button id="file-button" class="small-3d-button">📎</button>', unsafe_allow_html=True)

    st.markdown('</div>', unsafe_allow_html=True)

# ----------------------------------------------------------------------
# Process user message when send button is clicked
# ----------------------------------------------------------------------
if send_clicked and message_input and not st.session_state.processing:
    st.session_state.processing = True
    st.session_state.clear_input = True
    current_prompt = message_input

    # Get uploaded file if any (from the file button)
    uploaded_file_obj = None
    if st.session_state.uploaded_file:
        file_name, file_bytes = st.session_state.uploaded_file
        uploaded_file_obj = BytesIO(file_bytes)
        uploaded_file_obj.name = file_name
        st.session_state.uploaded_file = None

    # Detect language
    lang_info = detect_language(current_prompt)
    target_lang = lang_info['language']
    explicit = lang_info['explicit_instruction']
    logger.info(f"Detected language: {target_lang} (method: {lang_info['method']}, explicit: {explicit})")

    if explicit:
        st.session_state.conversation_language = explicit
    elif st.session_state.conversation_language is None:
        st.session_state.conversation_language = target_lang

    st.session_state.messages.append({"role": "user", "content": current_prompt})
    with st.chat_message("user"):
        st.markdown(current_prompt)

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
            if uploaded_file_obj is not None:
                with tempfile.NamedTemporaryFile(delete=False, suffix=Path(uploaded_file_obj.name).suffix) as tmp:
                    tmp.write(uploaded_file_obj.getvalue())
                    extra_kwargs["image_url"] = tmp.name

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
                        result = think(current_prompt, history, model_key, language=st.session_state.conversation_language)
            else:
                model = load_model(model_key)
                with st.spinner("Generating..."):
                    if model_key == "Yukti‑Audio":
                        voice = "female"
                        audio_path = model.invoke(current_prompt, voice=voice, language=st.session_state.conversation_language, **extra_kwargs)
                        with open(audio_path, "rb") as f:
                            audio_bytes = f.read()
                        st.audio(audio_bytes, format="audio/wav")
                        st.download_button("📥 Download Audio", data=audio_bytes, file_name="yukti_audio.wav")
                        full_response = "Audio generated."
                        result = {"type": "sync", "format": "audio"}
                        media.append({"type": "audio", "url": audio_path})
                    elif model_key == "Yukti‑Image":
                        image_url = model.invoke(current_prompt, **extra_kwargs)
                        st.image(image_url, use_container_width=False, width=300)
                        img_data = requests.get(image_url).content
                        st.download_button("📥 Download Image", data=img_data, file_name="yukti_image.png")
                        full_response = "Image generated."
                        result = {"type": "sync", "format": "image"}
                        media.append({"type": "image", "url": image_url})
                    elif model_key == "Yukti‑Video":
                        task_id = model.invoke(current_prompt, language=st.session_state.conversation_language, **extra_kwargs)
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
                            result = think(current_prompt, history, model_key, language=st.session_state.conversation_language)

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
        st.session_state.messages.append({"role": "assistant", "content": full_response, "task_id": task_id})
    elif full_response:
        st.session_state.messages.append({"role": "assistant", "content": full_response})

    st.session_state.processing = False
    st.rerun()

# ----------------------------------------------------------------------
# Real‑time video progress update (poll every 2 seconds)
# ----------------------------------------------------------------------
if st.session_state.tasks:
    any_pending = False
    for task_id, task_info in list(st.session_state.tasks.items()):
        if task_info['status'] in ('submitted', 'pending', 'processing'):
            any_pending = True
            try:
                updated = get_task_status(task_id)
                if updated:
                    task_info.update(updated)
                    # Find the message that contains this task
                    for idx, msg in enumerate(st.session_state.messages):
                        if msg.get('task_id') == task_id:
                            if task_info['status'] == 'completed':
                                msg['content'] = "✅ Video generated!"
                                msg['media'] = [{"type": "video", "url": task_info['result_url']}]
                                del msg['task_id']
                                del st.session_state.tasks[task_id]
                            elif task_info['status'] == 'failed':
                                msg['content'] = f"❌ Video failed: {task_info['error']}"
                                del msg['task_id']
                                del st.session_state.tasks[task_id]
                            else:
                                progress = task_info.get('progress', 0)
                                msg['content'] = f"⏳ Generating video... {progress}%"
                            break
            except Exception as e:
                logger.error(f"Error polling task {task_id}: {e}")
    if any_pending:
        time.sleep(2)
        st.rerun()
