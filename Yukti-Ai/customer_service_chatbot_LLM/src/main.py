"""
Original WhatsApp‑style chat interface (restored) with database persistence,
visible sidebar toggle, and detailed knowledge base error reporting.
"""

import os
import sys
import time
import logging
import tempfile
import base64
from pathlib import Path
from datetime import datetime
from io import BytesIO

import streamlit as st
import requests

# Ensure project root is in path
PROJECT_ROOT = Path(__file__).parent.parent.absolute()
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

# Local imports with graceful fallbacks
try:
    from langchain_helper import (
        create_vector_db,
        get_kb_detailed_status,
        VECTORDB_PATH,
        BASE_DIR,
    )
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
    def detect_language(text):
        return {"language": "en", "method": "fallback", "explicit_instruction": None}

# Import database for chat persistence (if available)
try:
    from database import save_message, load_messages
    DATABASE_AVAILABLE = True
except ImportError:
    DATABASE_AVAILABLE = False
    logging.warning("database.py not found; chat persistence disabled.")
    def save_message(role, content, timestamp): pass
    def load_messages(): return []

# Configure logging
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
# WhatsApp‑style CSS (header is NOT hidden – sidebar toggle visible)
# ----------------------------------------------------------------------
st.markdown("""
<style>
    .stApp {
        background: #0b141a;
    }
    /* Keep the Streamlit header visible for sidebar toggle */
    /* Only hide the default menu and footer */
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    /* Main content area – make room for fixed input bar */
    .main > div {
        padding-bottom: 80px !important;
    }
    /* Chat message container */
    .stChatMessage {
        padding: 0.5rem 1rem;
        margin-bottom: 0.5rem;
        border-radius: 18px;
        max-width: 70%;
        word-wrap: break-word;
        box-shadow: 0 1px 2px rgba(0,0,0,0.1);
        position: relative;
    }
    /* User message (right side) */
    .stChatMessage[data-testid="user-message"] {
        background: #005c4b;
        color: white;
        align-self: flex-end;
        margin-left: auto;
        border-bottom-right-radius: 4px;
    }
    /* Assistant message (left side) */
    .stChatMessage[data-testid="assistant-message"] {
        background: #202c33;
        color: #e9edef;
        align-self: flex-start;
        border-bottom-left-radius: 4px;
    }
    /* Timestamp */
    .message-timestamp {
        font-size: 0.7rem;
        color: rgba(255,255,255,0.6);
        margin-top: 0.2rem;
        text-align: right;
    }
    /* Chat input bar container – fixed at bottom */
    .chat-input-container {
        position: fixed;
        bottom: 0;
        left: 0;
        right: 0;
        background: #1f2c33;
        border-top: 1px solid #2a3942;
        padding: 0.5rem 1rem;
        display: flex;
        align-items: center;
        gap: 10px;
        z-index: 100;
        margin-left: 21rem; /* sidebar width */
    }
    @media (max-width: 992px) {
        .chat-input-container {
            margin-left: 0;
        }
    }
    /* 3D buttons */
    .chat-bar-button {
        background: #2a3942;
        border: none;
        border-radius: 50%;
        width: 40px;
        height: 40px;
        color: #8696a0;
        font-size: 1.3rem;
        display: flex;
        align-items: center;
        justify-content: center;
        cursor: pointer;
        transition: all 0.2s;
        box-shadow: 0 2px 5px rgba(0,0,0,0.3);
        border: none;
        outline: none;
    }
    .chat-bar-button:hover {
        background: #3b4a54;
        color: #e9edef;
        transform: scale(1.05);
    }
    .chat-bar-button:active {
        transform: scale(0.95);
    }
    /* Text input */
    .stTextInput > div > input {
        background: #2a3942;
        border: none;
        border-radius: 24px;
        color: #e9edef;
        padding: 10px 15px;
        font-size: 0.95rem;
        width: 100%;
    }
    .stTextInput > div > input:focus {
        outline: none;
        box-shadow: 0 0 0 2px #00a884;
    }
    /* Hide Streamlit input label */
    .stTextInput > label {
        display: none;
    }
    /* Video container */
    .stVideo {
        margin: 0.5rem 0;
    }
    /* Sidebar styling */
    .css-1d391kg {
        background: #1f2c33;
        border-right: 1px solid #2a3942;
    }
    /* Hide default chat input (we use custom) */
    .stChatInput {
        display: none !important;
    }
</style>
""", unsafe_allow_html=True)

# ----------------------------------------------------------------------
# Custom title
# ----------------------------------------------------------------------
st.markdown("""
<div style="display: flex; align-items: center; justify-content: center; gap: 5px; margin-bottom: 10px;">
    <span style="font-size: 1.5rem; font-weight: 600; color: #e9edef;">Yukti</span>
    <span style="
        font-size: 1.2rem;
        font-weight: 900;
        color: #00a884;
        background: #1f2c33;
        padding: 2px 8px;
        border-radius: 12px;
        border: 1px solid #00a884;
    ">AI</span>
</div>
""", unsafe_allow_html=True)

# ----------------------------------------------------------------------
# Session state initialization
# ----------------------------------------------------------------------
if "messages" not in st.session_state:
    # Load from database if available
    if DATABASE_AVAILABLE:
        st.session_state.messages = load_messages()
        if not st.session_state.messages:
            st.session_state.messages = [
                {"role": "assistant", "content": "Hello! I'm Yukti AI. How can I help you today?", "timestamp": datetime.now().strftime("%H:%M")}
            ]
    else:
        st.session_state.messages = [
            {"role": "assistant", "content": "Hello! I'm Yukti AI. How can I help you today?", "timestamp": datetime.now().strftime("%H:%M")}
        ]

if "kb_status" not in st.session_state:
    st.session_state.kb_status = get_kb_detailed_status()

if "tasks" not in st.session_state:
    st.session_state.tasks = {}

if "conversation_language" not in st.session_state:
    st.session_state.conversation_language = None

if "uploaded_file" not in st.session_state:
    st.session_state.uploaded_file = None

# NEW: flag to clear input on next run
if "clear_input" not in st.session_state:
    st.session_state.clear_input = False

# ----------------------------------------------------------------------
# JavaScript for voice and file input (unchanged, working)
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
                const fileReceiver = window.parent.document.querySelector('input[data-testid="stTextInput"][aria-label="file_receiver"]');
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
            const textInput = window.parent.document.querySelector('input[data-testid="stTextInput"][aria-label="Message"]');
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
# Sidebar (with knowledge base diagnostics and task list)
# ----------------------------------------------------------------------
with st.sidebar:
    st.markdown("## 🧠 Model")
    try:
        model_options = get_available_models()
        if not model_options:
            st.warning("No models available. Check API keys.")
            model_options = ["Yukti‑Flash"]
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

    st.divider()

    # ------------------------------------------------------------------
    # Knowledge Base section with detailed error reporting
    # ------------------------------------------------------------------
    st.markdown("### 📚 Knowledge Base")
    kb_status = st.session_state.kb_status

    if kb_status["ready"]:
        st.markdown(f"✅ **Active** ({kb_status['document_count']} documents)")
    else:
        st.markdown("❌ **Not Ready**")
        if kb_status["error"]:
            st.error(f"Error: {kb_status['error']}")
        else:
            # Show detailed diagnostics
            if not kb_status["dataset_exists"]:
                st.error("Dataset file not found at dataset/dataset.csv")
            elif not kb_status["dataset_readable"]:
                st.error(f"Dataset exists but cannot be read. Tried encodings: {kb_status.get('encoding_used', 'none')}")
            elif not kb_status["path_exists"]:
                st.info("Index directory does not exist. Click 'Update' to build it.")
            elif not kb_status["index_loadable"]:
                st.error("Index exists but cannot be loaded. Try rebuilding.")

    if st.button("🔄 Update", use_container_width=True):
        with st.spinner("Rebuilding knowledge base..."):
            success = create_vector_db()
            if success:
                st.session_state.kb_status = get_kb_detailed_status()
                st.success("Knowledge base updated!")
                st.rerun()
            else:
                st.error("Update failed. Check logs.")
                st.session_state.kb_status = get_kb_detailed_status()

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
            logger.exception("get_active_tasks")

        for task_id in list(st.session_state.tasks.keys()):
            task_info = st.session_state.tasks[task_id]
            st.markdown(f"**{task_info['variant']}** `{task_id[:8]}`")
            if task_info['status'] == 'completed' and task_info.get('result_url'):
                st.success("✅ Completed")
                if task_info['variant'] == 'Yukti‑Video':
                    st.video(task_info['result_url'])
                else:
                    st.image(task_info['result_url'], width=200)
            elif task_info['status'] == 'failed':
                st.error(f"❌ Failed: {task_info['error']}")
            else:
                st.info(f"⏳ {task_info['status']}... {task_info['progress']}%")
        st.divider()

    if st.button("🗑️ Clear Chat", use_container_width=True):
        st.session_state.messages = [
            {"role": "assistant", "content": "Hello! I'm Yukti AI. How can I help you today?", "timestamp": datetime.now().strftime("%H:%M")}
        ]
        st.rerun()

# ----------------------------------------------------------------------
# Custom chat input bar
# ----------------------------------------------------------------------
cols = st.columns([1, 1, 8])
with cols[0]:
    st.markdown('<button class="chat-bar-button" onclick="window.startVoiceRecognition()">🎤</button>', unsafe_allow_html=True)
with cols[1]:
    st.markdown('<button class="chat-bar-button" onclick="window.triggerFileUpload()">📎</button>', unsafe_allow_html=True)
with cols[2]:
    # Handle input clearing correctly
    if st.session_state.clear_input:
        default_value = ""
        st.session_state.clear_input = False
    else:
        default_value = None

    prompt = st.text_input("Message", key="message_input", value=default_value, label_visibility="collapsed", placeholder="Type a message")

# ----------------------------------------------------------------------
# Display all messages
# ----------------------------------------------------------------------
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])
        if "media" in msg:
            for media in msg["media"]:
                if media["type"] == "image":
                    st.image(media["url"], width=300)
                elif media["type"] == "audio":
                    st.audio(media["url"])
                elif media["type"] == "video":
                    st.video(media["url"])
        if "timestamp" in msg:
            st.markdown(f"<div class='message-timestamp'>{msg['timestamp']}</div>", unsafe_allow_html=True)

# ----------------------------------------------------------------------
# Process user message when Enter is pressed
# ----------------------------------------------------------------------
if prompt and prompt.strip():
    current_prompt = prompt
    # Set flag to clear input on next run (instead of modifying widget directly)
    st.session_state.clear_input = True

    uploaded_file_obj = None
    if st.session_state.uploaded_file:
        file_name, file_bytes = st.session_state.uploaded_file
        uploaded_file_obj = BytesIO(file_bytes)
        uploaded_file_obj.name = file_name
        st.session_state.uploaded_file = None

    # Detect language
    try:
        lang_info = detect_language(current_prompt)
        target_lang = lang_info['language']
        explicit = lang_info.get('explicit_instruction')
        logger.info(f"Detected language: {target_lang}")
    except Exception as e:
        logger.warning(f"Language detection failed: {e}")
        target_lang = 'en'
        explicit = None

    if explicit:
        st.session_state.conversation_language = explicit
    elif st.session_state.conversation_language is None:
        st.session_state.conversation_language = target_lang

    # Add user message
    timestamp = datetime.now().strftime("%H:%M")
    st.session_state.messages.append({
        "role": "user",
        "content": current_prompt,
        "timestamp": timestamp
    })
    if DATABASE_AVAILABLE:
        save_message("user", current_prompt, timestamp)

    # Generate assistant response
    with st.chat_message("assistant"):
        response_placeholder = st.empty()
        result = None
        answer = ""
        full_response = ""
        media = []

        try:
            model_key = st.session_state.selected_model
            extra_kwargs = {}
            if uploaded_file_obj is not None:
                with tempfile.NamedTemporaryFile(delete=False, suffix=Path(uploaded_file_obj.name).suffix) as tmp:
                    tmp.write(uploaded_file_obj.getvalue())
                    extra_kwargs["image_url"] = tmp.name

            if model_key in ["Yukti‑Flash", "Yukti‑Quantum"]:
                if not st.session_state.kb_status["ready"]:
                    full_response = "Knowledge base is not ready. Please check the sidebar for details and click 'Update' if needed."
                    response_placeholder.markdown(full_response)
                    result = {"type": "sync", "answer": full_response}
                else:
                    history = [
                        {"role": msg["role"], "content": msg["content"]}
                        for msg in st.session_state.messages[-10:-1]
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
                        st.download_button("📥 Download Audio", data=audio_bytes, file_name=f"yukti_audio_{int(time.time())}.wav")
                        full_response = "Audio generated."
                        result = {"type": "sync", "format": "audio"}
                        media.append({"type": "audio", "url": audio_path})
                    elif model_key == "Yukti‑Image":
                        image_url = model.invoke(current_prompt, **extra_kwargs)
                        st.image(image_url, width=300)
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
                        full_response = f"⏳ Video task started: `{task_id}`"
                        result = {"type": "async", "task_id": task_id}
                    else:
                        history = [
                            {"role": msg["role"], "content": msg["content"]}
                            for msg in st.session_state.messages[-10:-1]
                        ]
                        with st.spinner("Thinking..."):
                            result = think(current_prompt, history, model_key, language=st.session_state.conversation_language)

            if result and result.get("type") == "async":
                st.info(f"Task {result.get('task_id')} submitted. Check sidebar for progress.")
            elif result and result.get("type") == "sync":
                answer = result.get("answer", "")
                if not answer:
                    answer = "(No response generated)"
                if result.get("monologue"):
                    with st.expander("Show thinking process"):
                        st.markdown(result["monologue"])
                # Stream response
                words = answer.split()
                current = ""
                for word in words:
                    current += word + " "
                    response_placeholder.markdown(current + "▌")
                    time.sleep(0.03)
                response_placeholder.markdown(current)
                if result.get("sources"):
                    with st.expander("View source documents"):
                        for i, doc in enumerate(result["sources"][:3]):
                            st.markdown(f"**Source {i+1}:**")
                            st.write(doc.page_content[:500] + "..." if len(doc.page_content) > 500 else doc.page_content)
                st.caption(f"Thought for {result.get('thinking_time', 0):.2f}s")
                full_response = answer

        except Exception as e:
            st.error(f"An unexpected error occurred: {e}")
            logger.exception("Fatal error in message processing")
            full_response = ""

    # Append assistant message
    if result and result.get("type") == "sync" and full_response:
        st.session_state.messages.append({
            "role": "assistant",
            "content": full_response,
            "media": media,
            "timestamp": datetime.now().strftime("%H:%M")
        })
        if DATABASE_AVAILABLE:
            save_message("assistant", full_response, datetime.now().strftime("%H:%M"))
    elif result and result.get("type") == "async":
        st.session_state.messages.append({
            "role": "assistant",
            "content": full_response,
            "timestamp": datetime.now().strftime("%H:%M")
        })
        if DATABASE_AVAILABLE:
            save_message("assistant", full_response, datetime.now().strftime("%H:%M"))

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
            except Exception as e:
                logger.error(f"Error polling task {task_id}: {e}")
    if any_pending:
        time.sleep(2)
        st.rerun()
