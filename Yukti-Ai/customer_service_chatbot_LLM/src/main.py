import streamlit as st
import os
import time

# --- 1. PAGE CONFIG & CYBER-GEMINI CSS ---
st.set_page_config(page_title="Yukti AI", page_icon="🤖", layout="wide")

st.markdown("""
<style>
    /* Global Cyberpunk Theme */
    .stApp {
        background: #050505;
        color: #00ffff;
        font-family: 'Courier New', monospace;
    }

    /* Gemini Centered Chat Screen */
    .main .block-container {
        max-width: 850px;
        padding-top: 2rem;
        padding-bottom: 150px; /* Space for the floating bar */
    }

    /* Side Menu & Settings Button Styling */
    [data-testid="stSidebar"] {
        background-color: #0a0a0a !important;
        border-right: 1px solid #ff00ff !important;
        box-shadow: 5px 0 15px rgba(255, 0, 255, 0.3);
    }
    
    /* Message Bubbles */
    .stChatMessage { background: transparent !important; border: none !important; }
    [data-testid="user-message"] {
        background: rgba(255, 0, 255, 0.1) !important;
        border: 1px solid #ff00ff !important;
        border-radius: 20px !important;
        box-shadow: 0 0 10px #ff00ff;
    }

    /* THE FLOATING GEMINI BAR CONTAINER */
    /* This creates the pill shape at the bottom */
    .gemini-bar-outer {
        position: fixed;
        bottom: 30px;
        left: 50%;
        transform: translateX(-50%);
        width: 90%;
        max-width: 800px;
        background: rgba(15, 15, 15, 0.95);
        backdrop-filter: blur(10px);
        border: 2px solid #00ffff;
        border-radius: 40px;
        padding: 5px 15px;
        z-index: 99999;
        display: flex;
        align-items: center;
        box-shadow: 0 0 20px #00ffff;
    }

    /* Styling Native Buttons inside the bar to look like Gemini Icons */
    /* This is the secret to avoiding React Error 231 */
    .gemini-bar-outer button {
        background: transparent !important;
        border: none !important;
        color: #00ffff !important;
        font-size: 1.5rem !important;
        padding: 0 10px !important;
        transition: 0.3s !important;
    }
    .gemini-bar-outer button:hover {
        color: #ff00ff !important;
        text-shadow: 0 0 10px #ff00ff;
        transform: scale(1.2);
    }

    /* Style the Text Input inside the bar */
    .gemini-bar-outer input {
        background: transparent !important;
        border: none !important;
        color: white !important;
        font-size: 1.1rem !important;
    }
    
    /* Clean up default Streamlit UI */
    header, footer, #MainMenu {visibility: hidden;}
    .stChatInput {display: none !important;}
</style>
""", unsafe_allow_html=True)

# --- 2. SESSION STATE ---
if "messages" not in st.session_state:
    st.session_state.messages = []

# --- 3. SIDE MENU (Side Menu & Settings) ---
with st.sidebar:
    st.markdown("<h1 style='color:#ff00ff; text-align:center;'>YUKTI AI</h1>", unsafe_allow_html=True)
    st.markdown("### 🛠️ System Config")
    st.selectbox("Model Core", ["Yukti-Prime", "Yukti-Vision", "Yukti-Audio"])
    st.button("⚙️ Advanced Settings", use_container_width=True)
    st.divider()
    if st.button("🗑️ Purge Neural Memory", use_container_width=True):
        st.session_state.messages = []
        st.rerun()

# --- 4. CHAT SCREEN ---
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

# --- 5. THE GEMINI INTERFACE BAR ---
# We use st.container to create the wrapper, then columns inside it.
# This ensures React controls the buttons, preventing crash #231.
with st.container():
    st.markdown('<div class="gemini-bar-outer">', unsafe_allow_html=True)
    
    col_plus, col_input, col_mic, col_send = st.columns([1, 8, 1, 1])
    
    with col_plus:
        # These are native Streamlit buttons styled by CSS
        plus_btn = st.button("＋", key="gemini_plus")
        
    with col_input:
        # This input is automatically styled by our CSS above
        user_query = st.text_input("cmd", label_visibility="collapsed", placeholder="Enter Command...", key="gemini_input")
        
    with col_mic:
        mic_btn = st.button("🎙️", key="gemini_mic")
        
    with col_send:
        send_btn = st.button("➤", key="gemini_send")
        
    st.markdown('</div>', unsafe_allow_html=True)

# --- 6. ROBUST JAVASCRIPT (The Bridge) ---
# We use addEventListener to attach logic to the NATIVE buttons safely.
st.markdown("""
<script>
    const doc = window.parent.document;
    
    // Find our native buttons by their unique text content
    const findBtn = (text) => Array.from(doc.querySelectorAll('button')).find(b => b.innerText === text);
    
    const plus = findBtn('＋');
    const mic = findBtn('🎙️');

    // Voice Logic Bridge
    if (mic && !mic.dataset.hooked) {
        mic.addEventListener('click', () => {
            const recognition = new (window.SpeechRecognition || window.webkitSpeechRecognition)();
            recognition.onresult = (e) => {
                const transcript = e.results[0][0].transcript;
                const input = doc.querySelector('input[aria-label="cmd"]');
                input.value = transcript;
                input.dispatchEvent(new Event('input', { bubbles: true }));
            };
            recognition.start();
        });
        mic.dataset.hooked = "true";
    }

    // File Upload Bridge
    if (plus && !plus.dataset.hooked) {
        if(!doc.getElementById('yukti-file-sys')){
            const f = doc.createElement('input');
            f.type = 'file'; f.id = 'yukti-file-sys'; f.style.display = 'none';
            doc.body.appendChild(f);
            plus.addEventListener('click', () => f.click());
            f.addEventListener('change', (e) => alert("Attached: " + e.target.files[0].name));
        }
        plus.dataset.hooked = "true";
    }
</script>
""", unsafe_allow_html=True)

# --- 7. MAIN LOGIC EXECUTION ---
if (send_btn or (user_query and user_query != "")) and user_query:
    st.session_state.messages.append({"role": "user", "content": user_query})
    
    # Transition to "Assistant" Screen
    with st.chat_message("assistant"):
        with st.spinner("`Decrypting Neural Request...`"):
            # --- INTEGRATE YOUR MODEL CALL HERE ---
            time.sleep(1)
            response = f"Neural Link Confirmed. Processed: {user_query}"
            st.session_state.messages.append({"role": "assistant", "content": response})
    
    st.rerun()
