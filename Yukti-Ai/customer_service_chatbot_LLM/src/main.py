import os
import sys
import time
import logging
import tempfile
import base64
from io import BytesIO
from pathlib import Path

# Fix path for custom modules
PROJECT_ROOT = Path(__file__).parent.parent.parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import streamlit as st
import pandas as pd
import requests
from langchain_core.documents import Document
from langchain_community.vectorstores import FAISS

# Custom Yukti Modules
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

# Logging setup
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# ----------------------------------------------------------------------
# Page configuration
# ----------------------------------------------------------------------
st.set_page_config(
    page_title="Yukti AI",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ----------------------------------------------------------------------
# Cyber-Gemini CSS: Gemini Layout + Cyberpunk Skin
# ----------------------------------------------------------------------
st.markdown("""
<style>
    /* Gemini Layout Framework */
    .stApp {
        background: #0d0b1a; /* Dark Cyberpunk Base */
        color: #e0e0ff;
        font-family: 'Inter', sans-serif;
    }

    /* Constrained Chat Area (Gemini Style) */
    .main .block-container {
        max-width: 850px;
        padding-top: 2rem;
        padding-bottom: 150px;
    }

    /* Sidebar - Futuristic Console */
    section[data-testid="stSidebar"] {
        background-color: rgba(10, 10, 20, 0.95) !important;
        border-right: 1px solid #ff00ff !important;
        box-shadow: 5px 0 15px rgba(255, 0, 255, 0.2);
    }

    /* Centered Chat Messages */
    .stChatMessage {
        background-color: transparent !important;
        border: none !important;
        margin-bottom: 2rem !important;
    }
    
    .stChatMessage[data-testid="user-message"] {
        background: rgba(255, 0, 255, 0.05) !important;
        border-radius: 24px !important;
        border: 1px solid rgba(255, 0, 255, 0.2) !important;
        padding: 1.2rem !important;
    }

    /* The Gemini Pill Input Bar (Floating) */
    .fixed-bottom-bar {
        position: fixed;
        bottom: 30px;
        left: 50%;
        transform: translateX(-50%);
        width: 90%;
        max-width: 800px;
        background: rgba(20, 20, 35, 0.9);
        backdrop-filter: blur(20px);
        border: 1px solid rgba(0, 255, 255, 0.4);
        border-radius: 35px;
        padding: 10px 20px;
        display: flex;
        align-items: center;
        gap: 12px;
        z-index: 1000;
        box-shadow: 0 0 30px rgba(0, 255, 255, 0.2);
    }

    /* Icon Buttons (Cyberpunk Neon) */
    .bar-btn {
        background: transparent;
        border: none;
        color: #00ffff;
        font-size: 1.4rem;
        cursor: pointer;
        display: flex;
        align-items: center;
        justify-content: center;
        width: 44px;
        height: 44px;
        border-radius: 50%;
        transition: all 0.2s;
    }
    .bar-btn:hover {
        background: rgba(0, 255, 255, 0.1);
        text-shadow: 0 0 10px #00ffff;
        transform: scale(1.1);
    }
    .bar-btn.recording {
        color: #ff0055;
        animation: pulse-red 1.5s infinite;
    }

    @keyframes pulse-red {
        0% { box-shadow: 0 0 0 0 rgba(255, 0, 85, 0.4); }
        70% { box-shadow: 0 0 0 15px rgba(255, 0, 85, 0); }
        100% { box-shadow: 0 0 0 0 rgba(255, 0, 85, 0); }
    }

    /* Text Input Overrides */
    .stTextInput input {
        background: transparent !important;
        border: none !important;
        color: #fff !important;
        font-size: 1rem !important;
        padding: 10px 0 !important;
    }
    .stTextInput div[data-baseweb="input"] {
        background: transparent !important;
        border: none !important;
    }

    /* Hide Default Elements */
    #MainMenu, footer, header {visibility: hidden;}
    .stChatInput {display: none !important;}
</style>
""", unsafe_allow_html=True)

# ----------------------------------------------------------------------
# Session State & Logic Hooks
# ----------------------------------------------------------------------
if "messages" not in st.session_state:
    st.session_state.messages = [{"role": "assistant", "content": "Hello! I'm Yukti AI. How can I help you today?"}]
if "tasks" not in st.session_state: st.session_state.tasks = {}
if "processing" not in st.session_state: st.session_state.processing = False
if "uploaded_file" not in st.session_state: st.session_state.uploaded_file = None
if "knowledge_base_ready" not in st.session_state:
    st.session_state.knowledge_base_ready = os.path.exists(VECTORDB_PATH)

# Helper: Rebuild knowledge base
def rebuild_kb():
    # ... (Your existing logic for loading docs and FAISS)
    st.session_state.knowledge_base_ready = True

# ----------------------------------------------------------------------
# Side Menu (Gemini Style Sidebar)
# ----------------------------------------------------------------------
with st.sidebar:
    st.markdown("<h2 style='color:#ff00ff; text-shadow:0 0 10px #ff00ff;'>YUKTI CONSOLE</h2>", unsafe_allow_html=True)
    
    # Model Selection (Settings Button-like placement)
    model_options = get_available_models()
    selected_model = st.selectbox("Intelligence Core", model_options, index=0)
    st.session_state.selected_model = selected_model
    
    st.divider()
    
    # Knowledge Base Controls
    if st.button("🔄 Sync Knowledge", use_container_width=True):
        rebuild_kb()
    
    # Task Management
    if ZHIPU_AVAILABLE:
        st.markdown("### 📡 Active Downlinks")
        for task_id in list(st.session_state.tasks.keys()):
            # Using your existing render_task logic
            task = st.session_state.tasks[task_id]
            st.caption(f"{task['variant']} - {task['status']}")

    st.spacer = st.empty()
    if st.button("🗑️ Reset Neural Link", use_container_width=True):
        st.session_state.messages = [{"role": "assistant", "content": "Neural link reset. Ready."}]
        st.rerun()

# ----------------------------------------------------------------------
# Chat Screen (Centered flow)
# ----------------------------------------------------------------------
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])
        if "media" in msg:
            for m in msg["media"]:
                if m["type"] == "image": st.image(m["url"], width=300)
                elif m["type"] == "video": st.video(m["url"])

# ----------------------------------------------------------------------
# The Gemini Chat Bar (HTML Interface)
# ----------------------------------------------------------------------
# Invisible file receiver
file_receiver = st.text_input("file_receiver", key="file_receiver", label_visibility="collapsed")

# Floating Pill UI
st.markdown('<div class="fixed-bottom-bar">', unsafe_allow_html=True)

col_plus, col_txt, col_actions = st.columns([1, 10, 2])

with col_plus:
    st.markdown('<button id="file-button" class="bar-btn" title="Upload Media" onclick="document.getElementById(\'hidden-file-input\').click()">＋</button>', unsafe_allow_html=True)

with col_txt:
    message_input = st.text_input(
        "prompt",
        key="message_input",
        placeholder="Type your command here...",
        label_visibility="collapsed",
        disabled=st.session_state.processing
    )

with col_actions:
    st.markdown('<div style="display:flex; gap:10px;">', unsafe_allow_html=True)
    st.markdown('<button id="voice-button" class="bar-btn">🎙️</button>', unsafe_allow_html=True)
    # The actual send logic button (styled as an arrow)
    send_clicked = st.button("➤", key="send_btn_logic", help="Execute command")
    st.markdown('</div>', unsafe_allow_html=True)

st.markdown('</div>', unsafe_allow_html=True)

# ----------------------------------------------------------------------
# JavaScript Bridge (Voice, File, and Auto-Submit)
# ----------------------------------------------------------------------
st.markdown("""
<script>
    // Hidden File Input
    if(!document.getElementById('hidden-file-input')){
        const input = document.createElement('input');
        input.type = 'file';
        input.id = 'hidden-file-input';
        input.style.display = 'none';
        document.body.appendChild(input);
        
        input.onchange = (e) => {
            const file = e.target.files[0];
            const reader = new FileReader();
            reader.onload = (ev) => {
                const receiver = window.parent.document.querySelector('input[aria-label="file_receiver"]');
                receiver.value = file.name + ',' + ev.target.result.split(',')[1];
                receiver.dispatchEvent(new Event('input', { bubbles: true }));
            };
            reader.readAsDataURL(file);
        };
    }

    // Voice Recognition (Your existing logic)
    const voiceBtn = document.getElementById('voice-button');
    if(voiceBtn){
        voiceBtn.onclick = () => {
            const recognition = new (window.SpeechRecognition || window.webkitSpeechRecognition)();
            recognition.onstart = () => voiceBtn.classList.add('recording');
            recognition.onresult = (e) => {
                const transcript = e.results[0][0].transcript;
                const txtInput = window.parent.document.querySelector('input[aria-label="prompt"]');
                txtInput.value = transcript;
                txtInput.dispatchEvent(new Event('input', { bubbles: true }));
            };
            recognition.onend = () => voiceBtn.classList.remove('recording');
            recognition.start();
        };
    }
</script>
""", unsafe_allow_html=True)

# ----------------------------------------------------------------------
# Message Processing Logic
# ----------------------------------------------------------------------
if send_clicked and message_input:
    st.session_state.processing = True
    user_prompt = message_input
    
    # Add User message
    st.session_state.messages.append({"role": "user", "content": user_prompt})
    
    with st.chat_message("assistant"):
        with st.spinner("Cyber-Thinking..."):
            # ... (Your logic for thinking/model.invoke goes here)
            # This is where you call think() or load_model() based on your session state
            time.sleep(1) # Simulating logic
            response = "Command received. Processing through " + st.session_state.selected_model
            st.session_state.messages.append({"role": "assistant", "content": response})
    
    st.session_state.processing = False
    st.rerun()
