import os
import sys
import time
import logging
import base64
from pathlib import Path
from io import BytesIO

# --- Environment Setup ---
PROJECT_ROOT = Path(__file__).parent.parent.parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import streamlit as st
import pandas as pd
from langchain_helper import VECTORDB_PATH, BASE_DIR
from model_manager import get_available_models, ZHIPU_AVAILABLE, MODELS

# --- Page Config ---
st.set_page_config(page_title="Yukti AI", page_icon="⚡", layout="wide")

# --- Cyber-Gemini CSS (Centered Flow + Neon) ---
st.markdown("""
<style>
    .stApp { background: #0d0b1a; color: #e0e0ff; }
    
    /* Centered Chat Screen */
    .main .block-container {
        max-width: 850px;
        padding-top: 2rem;
        padding-bottom: 160px;
    }

    /* Message Bubbles */
    .stChatMessage { background: transparent !important; border: none !important; margin-bottom: 1.5rem !important; }
    .stChatMessage[data-testid="user-message"] {
        background: rgba(255, 0, 255, 0.05) !important;
        border-radius: 20px !important;
        border-left: 3px solid #ff00ff !important;
    }

    /* Floating Gemini-Style Input Bar */
    .fixed-bottom-bar {
        position: fixed;
        bottom: 25px;
        left: 50%;
        transform: translateX(-50%);
        width: 90%;
        max-width: 800px;
        background: rgba(20, 20, 35, 0.95);
        backdrop-filter: blur(15px);
        border: 1px solid rgba(0, 255, 255, 0.3);
        border-radius: 35px;
        padding: 5px 15px;
        display: flex;
        align-items: center;
        z-index: 1000;
        box-shadow: 0 0 25px rgba(0, 255, 255, 0.1);
    }

    /* Buttons */
    .cyber-btn {
        background: transparent; border: none; color: #00ffff;
        font-size: 1.5rem; cursor: pointer; width: 50px; height: 50px;
        display: flex; align-items: center; justify-content: center;
        border-radius: 50%; transition: 0.2s;
    }
    .cyber-btn:hover { background: rgba(0, 255, 255, 0.1); text-shadow: 0 0 10px #00ffff; }
    
    .recording-active { color: #ff0055 !important; animation: blink 1s infinite; }
    @keyframes blink { 50% { opacity: 0.5; } }

    /* Input Overrides */
    .stTextInput input { background: transparent !important; border: none !important; color: white !important; }
    div[data-baseweb="input"] { background: transparent !important; border: none !important; }
    
    /* Hide Defaults */
    #MainMenu, footer, header { visibility: hidden; }
    .stChatInput { display: none !important; }
</style>
""", unsafe_allow_html=True)

# --- State Management ---
if "messages" not in st.session_state:
    st.session_state.messages = [{"role": "assistant", "content": "Neural Link Established. How can I assist, User?"}]
if "uploaded_file_data" not in st.session_state:
    st.session_state.uploaded_file_data = None

# --- Sidebar ---
with st.sidebar:
    st.markdown("<h2 style='color:#ff00ff;'>YUKTI AI</h2>", unsafe_allow_html=True)
    selected_model = st.selectbox("Intelligence Core", get_available_models(), key="model_select")
    
    st.divider()
    if st.button("🗑️ Clear Memory", use_container_width=True):
        st.session_state.messages = []
        st.rerun()

# --- Chat Display ---
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

# --- The Interface Bar ---
# Hidden file receiver that catches Base64 data from JS
file_val = st.text_input("file_bridge", key="file_bridge", label_visibility="collapsed")
if file_val and not st.session_state.uploaded_file_data:
    st.session_state.uploaded_file_data = file_val
    st.toast(f"File Linked: {file_val.split(',')[0]}", icon="📎")

st.markdown('<div class="fixed-bottom-bar">', unsafe_allow_html=True)
col_plus, col_input, col_mic, col_send = st.columns([1, 8, 1, 1])

with col_plus:
    st.markdown('<button id="file-btn" class="cyber-btn" onclick="triggerFile()">＋</button>', unsafe_allow_html=True)

with col_input:
    # Use a unique label to help JS find it
    user_input = st.text_input("chat_input", key="user_input", placeholder="Ask anything...", label_visibility="collapsed")

with col_mic:
    st.markdown('<button id="mic-btn" class="cyber-btn" onclick="startVoice()">🎙️</button>', unsafe_allow_html=True)

with col_send:
    send_btn = st.button("➤", key="actual_send_logic")

st.markdown('</div>', unsafe_allow_html=True)

# --- JavaScript Logic (The Bridge) ---
st.markdown("""
<script>
    // 1. File Upload Logic
    const fileInput = window.parent.document.createElement('input');
    fileInput.type = 'file';
    fileInput.id = 'hidden-file-upload';
    window.parent.document.body.appendChild(fileInput);

    window.triggerFile = function() {
        fileInput.click();
    };

    fileInput.onchange = function(e) {
        const file = e.target.files[0];
        const reader = new FileReader();
        reader.onload = function(event) {
            const bridge = window.parent.document.querySelector('input[aria-label="file_bridge"]');
            bridge.value = file.name + "," + event.target.result;
            bridge.dispatchEvent(new Event('input', { bubbles: true }));
        };
        reader.readAsDataURL(file);
    };

    // 2. Voice Recognition Logic
    window.startVoice = function() {
        const recognition = new (window.SpeechRecognition || window.webkitSpeechRecognition)();
        const micBtn = document.getElementById('mic-btn');
        
        recognition.onstart = () => micBtn.classList.add('recording-active');
        
        recognition.onresult = (event) => {
            const transcript = event.results[0][0].transcript;
            const inputField = window.parent.document.querySelector('input[aria-label="chat_input"]');
            inputField.value = transcript;
            inputField.dispatchEvent(new Event('input', { bubbles: true }));
        };
        
        recognition.onend = () => micBtn.classList.remove('recording-active');
        recognition.start();
    };
</script>
""", unsafe_allow_html=True)

# --- Backend Processing ---
if send_btn and user_input:
    st.session_state.messages.append({"role": "user", "content": user_input})
    
    # Logic Routing based on file presence
    with st.chat_message("assistant"):
        if st.session_state.uploaded_file_data:
            st.info(f"Processing command with attached media...")
            # Reset after use
            st.session_state.uploaded_file_data = None
        
        # Placeholder for your 'think' or 'model_manager' calls
        response = f"I am analyzing your request via {st.session_state.model_select}..."
        st.markdown(response)
        st.session_state.messages.append({"role": "assistant", "content": response})
    
    st.rerun()
