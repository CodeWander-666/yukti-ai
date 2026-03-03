"""
Streamlit frontend – WhatsApp/Instagram style chat with voice,
file upload, and real‑time video progress.
"""

import os
import sys
import time
import logging
import tempfile
import base64
from pathlib import Path
from datetime import datetime

# Ensure project root is in path
PROJECT_ROOT = Path(__file__).parent.parent.absolute()
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

try:
    import streamlit as st
except ImportError:
    print("Streamlit not installed. Please install it.")
    sys.exit(1)

import requests
from io import BytesIO

# Local imports
try:
    from langchain_helper import check_kb_status
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
        get_active_tasks,   # still used for polling, but we'll show progress in chat
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
# WhatsApp‑style CSS
# ----------------------------------------------------------------------
st.markdown("""
<style>
    /* Global background */
    .stApp {
        background: #0b141a;
    }
    /* Hide default Streamlit elements */
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    header {visibility: hidden;}
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
    /* Chat input bar */
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
        margin-left: 21rem; /* adjust based on sidebar width */
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
    /* Progress message */
    .progress-message {
        background: #1f2c33;
        border-left: 4px solid #00a884;
        padding: 1rem;
        border-radius: 12px;
        margin: 0.5rem 0;
    }
    /* Video container */
    .video-container {
        margin: 0.5rem 0;
    }
    /* Sidebar styling */
    .css-1d391kg {
        background: #1f2c33;
        border-right: 1px solid #2a3942;
    }
    /* Hide default chat input */
    .stChatInput {
        display: none !important;
    }
</style>
""", unsafe_allow_html=True)

# ----------------------------------------------------------------------
# Custom title (smaller, more subtle)
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
    st.session_state.messages = [
        {"role": "assistant", "content": "Hello! I'm Yukti AI. How can I help you today?", "timestamp": datetime.now().strftime("%H:%M")}
    ]

if "knowledge_base_ready" not in st.session_state:
    try:
        st.session_state.knowledge_base_ready = check_kb_status()
    except Exception as e:
        logger.error(f"Failed to check knowledge base status: {e}")
        st.session_state.knowledge_base_ready = False

if "tasks" not in st.session_state:
    st.session_state.tasks = {}  # task_id -> info

if "conversation_language" not in st.session_state:
    st.session_state.conversation_language = None

if "uploaded_file" not in st.session_state:
    st.session_state.uploaded_file = None

# ----------------------------------------------------------------------
# Sidebar (simplified, no task list)
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
    if st.session_state.knowledge_base_ready:
        st.markdown("✅ **Knowledge Base Active**")
    else:
        st.markdown("⚠️ **Knowledge Base Not Built**")
    st.divider()

    if st.button("🗑️ Clear Chat", use_container_width=True):
        st.session_state.messages = [
            {"role": "assistant", "content": "Hello! I'm Yukti AI. How can I help you today?", "timestamp": datetime.now().strftime("%H:%M")}
        ]
        st.rerun()

# ----------------------------------------------------------------------
# Main chat area – display messages in WhatsApp style
# ----------------------------------------------------------------------
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])
        if "media" in msg:
            for media in msg["media"]:
                if media["type"] == "image":
                    st.image(media["url"], width=300)
                    # Download button inline
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
                elif media["type"] == "audio":
                    st.audio(media["url"])
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
                elif media["type"] == "video":
                    st.video(media["url"])
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
        # Add timestamp
        if "timestamp" in msg:
            st.markdown(f"<div class='message-timestamp'>{msg['timestamp']}</div>", unsafe_allow_html=True)

# ----------------------------------------------------------------------
# JavaScript for voice and file input
# ----------------------------------------------------------------------
st.markdown("""
<script>
let fileInput = null;

document.addEventListener('DOMContentLoaded', function() {
    // Create hidden file input
    fileInput = document.createElement('input');
    fileInput.type = 'file';
    fileInput.id = 'hidden-file-input';
    fileInput.style.display = 'none';
    document.body.appendChild(fileInput);

    fileInput.addEventListener('change', function(e) {
        if (e.target.files.length > 0) {
            const file = e.target.files[0];
            const reader = new FileReader();
            reader.onload = function(ev) {
                const data = ev.target.result; // base64
                // Send to Streamlit via custom event
                const fileDataInput = document.getElementById('file-data-input');
                if (fileDataInput) {
                    fileDataInput.value = data;
                    fileDataInput.dispatchEvent(new Event('input', { bubbles: true }));
                }
                const fileNameInput = document.getElementById('file-name-input');
                if (fileNameInput) {
                    fileNameInput.value = file.name;
                    fileNameInput.dispatchEvent(new Event('input', { bubbles: true }));
                }
            };
            reader.readAsDataURL(file);
        }
    });
});

function startVoiceRecognition() {
    if (!('webkitSpeechRecognition' in window) && !('SpeechRecognition' in window)) {
        alert('Voice recognition not supported in this browser. Please use Chrome or Edge.');
        return;
    }
    const recognition = new (window.SpeechRecognition || window.webkitSpeechRecognition)();
    recognition.lang = 'en-US'; // language will be auto-detected by backend
    recognition.interimResults = false;
    recognition.maxAlternatives = 1;

    recognition.start();

    recognition.onresult = function(event) {
        const transcript = event.results[0][0].transcript;
        const textInput = document.querySelector('input[data-testid="stTextInput"]');
        if (textInput) {
            textInput.value = transcript;
            textInput.dispatchEvent(new Event('input', { bubbles: true }));
        }
    };

    recognition.onerror = function(event) {
        alert('Voice error: ' + event.error);
    };
}

function triggerFileUpload() {
    document.getElementById('hidden-file-input').click();
}
</script>

<!-- Hidden inputs to receive file data -->
<div style="display: none;">
    <input type="text" id="file-data-input" />
    <input type="text" id="file-name-input" />
</div>
""", unsafe_allow_html=True)

# Read hidden inputs via Streamlit's text_input (they will be updated by JS)
file_data = st.text_input("file_data", key="file_data", label_visibility="collapsed", value="")
file_name = st.text_input("file_name", key="file_name", label_visibility="collapsed", value="")

if file_data and file_name and not st.session_state.uploaded_file:
    try:
        header, encoded = file_data.split(",", 1)
        data = base64.b64decode(encoded)
        st.session_state.uploaded_file = (file_name, data)
        # Clear the hidden inputs to avoid re-upload
        st.session_state.file_data = ""
        st.session_state.file_name = ""
    except Exception as e:
        logger.error(f"Failed to decode file: {e}")
        st.session_state.uploaded_file = None

# ----------------------------------------------------------------------
# Custom chat input bar
# ----------------------------------------------------------------------
col1, col2, col3 = st.columns([1, 1, 8])
with col1:
    st.markdown('<button class="chat-bar-button" onclick="startVoiceRecognition()">🎤</button>', unsafe_allow_html=True)
with col2:
    st.markdown('<button class="chat-bar-button" onclick="triggerFileUpload()">📎</button>', unsafe_allow_html=True)
with col3:
    prompt = st.text_input("Message", key="chat_input", label_visibility="collapsed", placeholder="Type a message")

# ----------------------------------------------------------------------
# Progress update for video tasks (runs every 2 seconds)
# ----------------------------------------------------------------------
def update_video_progress():
    """Poll for video task status and update the corresponding message."""
    for task_id, task_info in list(st.session_state.tasks.items()):
        if task_info['status'] in ('submitted', 'pending', 'processing'):
            updated = get_task_status(task_id)
            if updated:
                task_info.update(updated)
                # Find the message that contains this task
                for idx, msg in enumerate(st.session_state.messages):
                    if msg.get('task_id') == task_id:
                        # Update the content
                        if task_info['status'] == 'completed':
                            msg['content'] = "✅ Video generated!"
                            msg['media'] = [{"type": "video", "url": task_info['result_url']}]
                            # Remove from tasks dict
                            del st.session_state.tasks[task_id]
                        elif task_info['status'] == 'failed':
                            msg['content'] = f"❌ Video failed: {task_info['error']}"
                            del st.session_state.tasks[task_id]
                        else:
                            # Update progress
                            progress = task_info.get('progress', 0)
                            msg['content'] = f"⏳ Generating video... {progress}%"
                        break
        elif task_info['status'] == 'completed':
            # Already completed, remove from dict
            del st.session_state.tasks[task_id]

# Add a placeholder for auto-refresh
if st.session_state.get('tasks'):
    update_video_progress()
    time.sleep(2)
    st.rerun()

# ----------------------------------------------------------------------
# Process user message
# ----------------------------------------------------------------------
if prompt:
    # Get uploaded file if any
    uploaded_file_obj = None
    if st.session_state.uploaded_file:
        file_name, file_data_bytes = st.session_state.uploaded_file
        uploaded_file_obj = BytesIO(file_data_bytes)
        uploaded_file_obj.name = file_name
        st.session_state.uploaded_file = None  # clear after use

    # Detect language
    try:
        lang_info = detect_language(prompt)
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
    st.session_state.messages.append({
        "role": "user",
        "content": prompt,
        "timestamp": datetime.now().strftime("%H:%M")
    })

    # Prepare assistant response placeholder
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
                # Save to temp file for model
                with tempfile.NamedTemporaryFile(delete=False, suffix=Path(uploaded_file_obj.name).suffix) as tmp:
                    tmp.write(uploaded_file_obj.getvalue())
                    extra_kwargs["image_url"] = tmp.name

            if model_key in ["Yukti‑Flash", "Yukti‑Quantum"]:
                if not st.session_state.knowledge_base_ready:
                    full_response = "The knowledge base is not ready. Please contact the administrator."
                    response_placeholder.markdown(full_response)
                    result = {"type": "sync", "answer": full_response}
                else:
                    history = [
                        {"role": msg["role"], "content": msg["content"]}
                        for msg in st.session_state.messages[-10:]
                    ]
                    with st.spinner("Thinking..."):
                        result = think(prompt, history, model_key, language=st.session_state.conversation_language)
            else:
                model = load_model(model_key)
                with st.spinner("Generating..."):
                    if model_key == "Yukti‑Audio":
                        voice = "female"
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
                        st.image(image_url, width=300)
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
                        full_response = "⏳ Generating video... 0%"
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
                # We'll update via the polling loop
                pass
            elif result and result.get("type") == "sync":
                answer = result.get("answer", "")
                if not answer:
                    answer = "(No response generated)"
                    st.warning("The model returned an empty response. Please try again.")
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
                            if i < len(result["sources"]) - 1:
                                st.divider()
                st.caption(f"Thought for {result.get('thinking_time', 0):.2f}s")
                full_response = answer

        except Exception as e:
            st.error(f"An unexpected error occurred: {e}")
            logger.exception("Fatal error in message processing")
            full_response = ""

    # Append assistant message to session state
    if result and result.get("type") == "sync" and full_response:
        st.session_state.messages.append({
            "role": "assistant",
            "content": full_response,
            "media": media,
            "timestamp": datetime.now().strftime("%H:%M")
        })
    elif result and result.get("type") == "async":
        # For async, we add a placeholder that will be updated
        st.session_state.messages.append({
            "role": "assistant",
            "content": full_response,
            "task_id": result.get("task_id"),
            "timestamp": datetime.now().strftime("%H:%M")
        })

    # Clear input and rerun
    st.session_state.chat_input = ""
    st.rerun()
