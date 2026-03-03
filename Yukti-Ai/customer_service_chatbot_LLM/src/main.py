import streamlit as st
import time

# --- 1. PAGE SETUP ---
st.set_page_config(page_title="Yukti AI", page_icon="🤖", layout="wide")

# --- 2. CYBER-GEMINI CSS (The "Pill" Layout) ---
st.markdown("""
<style>
    /* Cyberpunk Theme */
    .stApp { background: #08080b; color: #00ffff; }
    
    /* Centered Chat Flow */
    .main .block-container {
        max-width: 800px;
        padding-bottom: 150px;
    }

    /* THE FLOATING BAR CONTAINER */
    .gemini-bar-container {
        position: fixed;
        bottom: 30px;
        left: 50%;
        transform: translateX(-50%);
        width: 90%;
        max-width: 750px;
        background: rgba(15, 15, 25, 0.95);
        backdrop-filter: blur(10px);
        border: 2px solid #00ffff;
        border-radius: 40px;
        padding: 5px 20px;
        z-index: 9999;
        display: flex;
        align-items: center;
        box-shadow: 0 0 20px rgba(0, 255, 255, 0.3);
    }

    /* Target Streamlit Buttons inside the bar */
    /* We style them to look like Gemini icons */
    div[data-testid="column"] button {
        background: transparent !important;
        border: none !important;
        color: #00ffff !important;
        font-size: 22px !important;
        padding: 0 !important;
        box-shadow: none !important;
    }
    
    div[data-testid="column"] button:hover {
        color: #ff00ff !important;
        transform: scale(1.1);
    }

    /* Hide the default Streamlit chat input if any exists */
    .stChatInput { display: none !important; }
    
    /* Clean Sidebar */
    [data-testid="stSidebar"] { background-color: #0a0a0f !important; border-right: 1px solid #00ffff; }
</style>
""", unsafe_allow_html=True)

# --- 3. SESSION STATE ---
if "messages" not in st.session_state:
    st.session_state.messages = []
if "file_status" not in st.session_state:
    st.session_state.file_status = None

# --- 4. SIDEBAR & SETTINGS ---
with st.sidebar:
    st.markdown("<h2 style='color:#ff00ff;'>YUKTI AI</h2>", unsafe_allow_html=True)
    st.selectbox("Model Core", ["Yukti-Prime", "Yukti-Vision", "Yukti-Audio"])
    if st.button("🗑️ Clear History"):
        st.session_state.messages = []
        st.rerun()

# --- 5. CHAT SCREEN ---
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

# --- 6. THE GEMINI BAR (The "Working" Interface) ---
# This structure uses Streamlit columns INSIDE a styled div
st.markdown('<div class="gemini-bar-container">', unsafe_allow_html=True)

# Column sizes to mimic Gemini: [Plus] [Input] [Mic] [Send]
col_plus, col_input, col_mic, col_send = st.columns([1, 8, 1, 1])

with col_plus:
    # Key 'plus_trigger' allows JS to find this button
    plus_btn = st.button("＋", key="plus_trigger")

with col_input:
    # Use label 'cmd' for JS targeting
    user_query = st.text_input("cmd", placeholder="Ask Yukti AI...", label_visibility="collapsed", key="query_input")

with col_mic:
    mic_btn = st.button("🎙️", key="mic_trigger")

with col_send:
    send_btn = st.button("➤", key="send_trigger")

st.markdown('</div>', unsafe_allow_html=True)

# --- 7. STABLE JAVASCRIPT BRIDGE ---
# This script attaches to the Streamlit buttons WITHOUT crashing React
st.markdown("""
<script>
    // 1. Setup the Bridge
    const mainDoc = window.parent.document;
    
    // Find buttons by their Key (Streamlit adds data-testid or aria-labels)
    const getBtn = (key) => mainDoc.querySelector(`button[kind="secondary"][key="${key}"]`) || 
                           Array.from(mainDoc.querySelectorAll('button')).find(b => b.innerText.includes(key));

    // 2. File Upload Logic
    if (!mainDoc.getElementById('hidden-uploader')) {
        const uploader = mainDoc.createElement('input');
        uploader.type = 'file';
        uploader.id = 'hidden-uploader';
        uploader.style.display = 'none';
        mainDoc.body.appendChild(uploader);
        
        uploader.onchange = (e) => {
            alert("File Linked: " + e.target.files[0].name);
        };
    }

    // 3. Attach Listeners Safely
    // We use a small delay to ensure Streamlit has rendered the buttons
    setTimeout(() => {
        const pBtn = mainDoc.querySelector('button[aria-label="＋"]');
        const mBtn = mainDoc.querySelector('button[aria-label="🎙️"]');

        if (pBtn) pBtn.onclick = () => mainDoc.getElementById('hidden-uploader').click();
        
        if (mBtn) mBtn.onclick = () => {
            const recognition = new (window.SpeechRecognition || window.webkitSpeechRecognition)();
            recognition.onresult = (event) => {
                const text = event.results[0][0].transcript;
                const inputField = mainDoc.querySelector('input[aria-label="cmd"]');
                inputField.value = text;
                inputField.dispatchEvent(new Event('input', { bubbles: true }));
            };
            recognition.start();
        };
    }, 1000);
</script>
""", unsafe_allow_html=True)

# --- 8. LOGIC EXECUTION ---
if send_btn and user_query:
    # Add user message
    st.session_state.messages.append({"role": "user", "content": user_query})
    
    # Generate Response
    with st.chat_message("assistant"):
        with st.spinner("`SYSTEM: Processing Neural Query...`"):
            # Put your actual 'think' or 'load_model' logic here
            time.sleep(1) 
            response = f"Yukti AI Response for: {user_query}"
            st.session_state.messages.append({"role": "assistant", "content": response})
    
    # Clean input and refresh
    st.rerun()
