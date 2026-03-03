import streamlit as st
import os

# --- Page Config ---
st.set_page_config(page_title="Yukti AI", page_icon="⚡", layout="wide")

# --- Optimized CSS (Fixed for Overflow & React Stability) ---
st.markdown("""
<style>
    .stApp { background: #0d0b1a; color: #e0e0ff; }
    
    /* Centered Chat Flow */
    .main .block-container {
        max-width: 850px;
        padding-bottom: 150px;
    }

    /* Message Bubbles */
    .stChatMessage { background: transparent !important; border: none !important; }
    .stChatMessage[data-testid="user-message"] {
        background: rgba(255, 0, 255, 0.05) !important;
        border-radius: 24px !important;
        border: 1px solid rgba(255, 0, 255, 0.2) !important;
    }

    /* Floating Gemini Bar */
    .fixed-bottom-bar {
        position: fixed;
        bottom: 30px;
        left: 50%;
        transform: translateX(-50%);
        width: 90%;
        max-width: 800px;
        background: rgba(20, 20, 35, 0.95);
        backdrop-filter: blur(15px);
        border: 1px solid rgba(0, 255, 255, 0.4);
        border-radius: 35px;
        padding: 8px 15px;
        display: flex;
        align-items: center;
        z-index: 9999;
        box-shadow: 0 0 25px rgba(0, 255, 255, 0.2);
    }

    /* Buttons - Removed 'onclick' to prevent React Error #231 */
    .cyber-btn {
        background: transparent; border: none; color: #00ffff;
        font-size: 1.5rem; cursor: pointer; width: 45px; height: 45px;
        display: flex; align-items: center; justify-content: center;
        border-radius: 50%; transition: 0.3s;
    }
    .cyber-btn:hover { background: rgba(0, 255, 255, 0.1); text-shadow: 0 0 10px #00ffff; }
    
    .recording-glow { color: #ff0055 !important; animation: pulse 1s infinite; }
    @keyframes pulse { 50% { opacity: 0.5; } }

    /* Input Styling */
    .stTextInput input { background: transparent !important; border: none !important; color: white !important; }
    div[data-baseweb="input"] { background: transparent !important; border: none !important; }
    
    /* Clean Hide */
    #MainMenu, footer, header { visibility: hidden; }
    .stChatInput { display: none !important; }
</style>
""", unsafe_allow_html=True)

# --- State ---
if "messages" not in st.session_state:
    st.session_state.messages = []
if "file_data" not in st.session_state:
    st.session_state.file_data = None

# --- UI Layout ---
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

# This catches the JS data
file_bridge = st.text_input("FB", key="file_bridge", label_visibility="collapsed")
if file_bridge and not st.session_state.file_data:
    st.session_state.file_data = file_bridge
    st.toast(f"Attached: {file_bridge.split(',')[0]}", icon="📎")

# THE BAR: Note there are NO 'onclick' attributes here. This prevents the React crash.
st.markdown('''
<div class="fixed-bottom-bar">
    <button id="btn-plus" class="cyber-btn">＋</button>
    <div style="flex-grow: 1;"></div>
    <button id="btn-mic" class="cyber-btn">🎙️</button>
</div>
''', unsafe_allow_html=True)

# Overlay the Streamlit input on top of the 'flex-grow' space
col1, col2, col3 = st.columns([1, 8, 2])
with col2:
    user_input = st.text_input("P", key="user_input", placeholder="Ask Yukti AI...", label_visibility="collapsed")
with col3:
    # Removed 'help' parameter to stop Popper.js / preventOverflow warning
    send_btn = st.button("➤", key="send_trigger")

# --- The JavaScript Bridge (Fixed for Stability) ---
st.markdown("""
<script>
    // Create the hidden file input once
    if (!window.fileInputCreated) {
        const fi = window.parent.document.createElement('input');
        fi.type = 'file';
        fi.id = 'hidden-file-sys';
        fi.style.display = 'none';
        window.parent.document.body.appendChild(fi);
        window.fileInputCreated = true;

        fi.onchange = function(e) {
            const file = e.target.files[0];
            const reader = new FileReader();
            reader.onload = function(ev) {
                const bridge = window.parent.document.querySelector('input[aria-label="FB"]');
                bridge.value = file.name + "," + ev.target.result;
                bridge.dispatchEvent(new Event('input', { bubbles: true }));
            };
            reader.readAsDataURL(file);
        };
    }

    // Attach listeners to the custom HTML buttons safely
    const plusBtn = window.parent.document.getElementById('btn-plus');
    const micBtn = window.parent.document.getElementById('btn-mic');

    if (plusBtn) {
        plusBtn.onclick = () => window.parent.document.getElementById('hidden-file-sys').click();
    }

    if (micBtn) {
        micBtn.onclick = () => {
            const rec = new (window.SpeechRecognition || window.webkitSpeechRecognition)();
            rec.onstart = () => micBtn.classList.add('recording-glow');
            rec.onresult = (e) => {
                const val = e.results[0][0].transcript;
                const inp = window.parent.document.querySelector('input[aria-label="P"]');
                inp.value = val;
                inp.dispatchEvent(new Event('input', { bubbles: true }));
            };
            rec.onend = () => micBtn.classList.remove('recording-glow');
            rec.start();
        };
    }
</script>
""", unsafe_allow_html=True)

# --- Logic ---
if send_btn and user_input:
    st.session_state.messages.append({"role": "user", "content": user_input})
    with st.chat_message("assistant"):
        # Put your 'think' or model logic here
        st.write("Neural process initiated...")
        st.session_state.file_data = None # Clear file after send
    st.rerun()
