"""
Yukti AI – Main Streamlit Application
Integrates all Yukti models with Gemini fallback, letter‑by‑letter thinking display,
and comprehensive error handling.
"""

import os
import sys
import time
from pathlib import Path

# ---------- Path Setup: Add project root to sys.path ----------
PROJECT_ROOT = Path(__file__).parent.parent.parent.parent  # yukti-ai/
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import streamlit as st
import pandas as pd
from langchain_core.documents import Document
from langchain_community.vectorstores import FAISS

# Local imports
from langchain_helper import get_embeddings, VECTORDB_PATH, BASE_DIR, create_vector_db
from think import think
from model_manager import get_available_models, MODELS
from LLM.zhipu.queue_manager import TaskQueue

# ---------- Page Configuration ----------
st.set_page_config(
    page_title="Yukti AI",
    page_icon="✨",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ---------- Custom CSS for Professional Look ----------
st.markdown("""
<style>
    /* Main container */
    .main > div {
        padding: 0 2rem;
    }
    /* Chat message styling */
    .stChatMessage {
        border-radius: 12px;
        padding: 1rem;
        margin-bottom: 0.5rem;
    }
    .stChatMessage[data-testid="user-message"] {
        background-color: #f0f2f6;
    }
    .stChatMessage[data-testid="assistant-message"] {
        background-color: #e3f2fd;
        border-left: 4px solid #1976d2;
    }
    /* Sidebar styling */
    .css-1d391kg {
        background-color: #f8f9fa;
    }
    /* Buttons */
    .stButton > button {
        border-radius: 8px;
        font-weight: 500;
        transition: all 0.2s;
    }
    .stButton > button:hover {
        transform: translateY(-1px);
        box-shadow: 0 4px 8px rgba(0,0,0,0.1);
    }
    /* Input field */
    .stTextInput > div > div > input {
        border-radius: 24px;
        border: 1px solid #ddd;
        padding: 12px 20px;
        font-size: 16px;
    }
    /* Headers */
    h1, h2, h3 {
        font-family: 'Inter', sans-serif;
        font-weight: 500;
    }
    /* Hide Streamlit branding */
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
</style>
""", unsafe_allow_html=True)

# ---------- Title ----------
st.title("Yukti AI")
st.caption("Your Intelligent Assistant – Powered by Zhipu GLM & Gemini")

# ---------- Session State Initialization ----------
if "messages" not in st.session_state:
    st.session_state.messages = [
        {"role": "assistant", "content": "Hello! I'm Yukti AI. How can I help you today?"}
    ]
if "knowledge_base_ready" not in st.session_state:
    st.session_state.knowledge_base_ready = os.path.exists(VECTORDB_PATH)
if "show_thinking" not in st.session_state:
    st.session_state.show_thinking = True  # default to show thinking process

# ---------- Data Sources ----------
SOURCES = [
    {
        "type": "csv",
        "path": os.path.join(BASE_DIR, "dataset", "dataset.csv"),
        "name": "Original Dataset",
        "columns": ["prompt", "response"],
        "content_template": "Q: {prompt}\nA: {response}"
    },
]

# ---------- Helper Functions ----------
def load_all_documents():
    """Load documents from all sources with robust encoding handling."""
    docs = []
    for src in SOURCES:
        if src["type"] == "csv":
            path = src["path"]
            if not os.path.exists(path):
                st.sidebar.warning(f"File not found: {path}")
                continue
            try:
                encodings = ['utf-8', 'cp1252', 'latin-1', 'iso-8859-1']
                df = None
                for enc in encodings:
                    try:
                        df = pd.read_csv(path, encoding=enc, on_bad_lines='skip')
                        break
                    except UnicodeDecodeError:
                        continue
                if df is None:
                    st.sidebar.error(f"Cannot read {path}")
                    return None
                missing = [c for c in src["columns"] if c not in df.columns]
                if missing:
                    st.sidebar.error(f"Missing columns {missing} in {path}")
                    return None
                for idx, row in df.iterrows():
                    content = src["content_template"].format(**{c: row[c] for c in src["columns"]})
                    docs.append(Document(
                        page_content=content,
                        metadata={"source": path, "row": idx}
                    ))
                st.sidebar.success(f"Loaded {len(df)} rows from {src['name']}")
            except Exception as e:
                st.sidebar.error(f"Error reading {path}: {e}")
                return None
    return docs

def rebuild_knowledge_base():
    """Rebuild FAISS index from all sources."""
    with st.spinner("Loading documents..."):
        docs = load_all_documents()
        if docs is None:
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
            return False

def stream_response(placeholder, full_text, delay=0.02):
    """Simulate letter-by-letter streaming."""
    words = full_text.split()
    current = ""
    for word in words:
        current += word + " "
        placeholder.markdown(current + "▌")  # cursor effect
        time.sleep(delay)
    placeholder.markdown(current)  # final without cursor

# ---------- Sidebar ----------
with st.sidebar:
    st.header("Knowledge Base")
    if st.button("Update Knowledge Base", use_container_width=True):
        rebuild_knowledge_base()
    st.divider()

    st.subheader("Status")
    if st.session_state.knowledge_base_ready:
        st.markdown("**Active**")
    else:
        st.markdown("**Not built** – click update above.")
    st.divider()

    # ---------- Model Selector ----------
    st.subheader("Brain")
    model_options = get_available_models()
    selected_model = st.selectbox(
        "Choose thinking style",
        options=model_options,
        format_func=lambda x: f"{x} – {MODELS[x]['description']}",
        key="model_selector"
    )
    st.session_state.selected_model = selected_model
    st.divider()

    # ---------- Thinking Toggle ----------
    st.subheader("Display")
    st.session_state.show_thinking = st.checkbox("Show thinking process", value=True)
    st.divider()

    # ---------- Task Monitor ----------
    st.subheader("📋 Active Tasks")
    task_queue = TaskQueue(db_path=os.path.join(BASE_DIR, "yukti_tasks.db"))
    if st.button("Refresh Tasks"):
        st.rerun()
    active = task_queue.get_active_tasks()
    for task_id, variant, status, progress in active:
        with st.container():
            st.markdown(f"**{variant}** ({task_id[:8]})")
            if status == "processing":
                st.progress(progress/100, text=f"{progress}%")
            elif status in ("submitted", "pending"):
                st.text("⏳ Queued")
            else:
                st.text(f"Status: {status}")
    # Show recent completed tasks
    completed = task_queue.conn.execute(
        "SELECT variant, result_url FROM tasks WHERE status='completed' ORDER BY completed_at DESC LIMIT 5"
    ).fetchall()
    if completed:
        st.markdown("---")
        st.markdown("✅ Recent Results")
        for variant, url in completed:
            st.markdown(f"- {variant}: [View]({url})")
    st.divider()

    st.subheader("Sources")
    for src in SOURCES:
        st.markdown(f"- **{src['name']}**")
    st.divider()

    if st.button("Clear Conversation", use_container_width=True):
        st.session_state.messages = [
            {"role": "assistant", "content": "Hello! I'm Yukti AI. How can I help you today?"}
        ]
        st.rerun()

# ---------- Main Chat Interface ----------
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

if prompt := st.chat_input("Ask me anything..."):
    # Add user message
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    # Generate response
    with st.chat_message("assistant"):
        response_placeholder = st.empty()
        try:
            if not st.session_state.knowledge_base_ready:
                full_response = "The knowledge base is not ready. Please click 'Update Knowledge Base' in the sidebar first."
                response_placeholder.markdown(full_response)
            else:
                # Build conversation history
                history = [
                    {"role": msg["role"], "content": msg["content"]}
                    for msg in st.session_state.messages[-10:]
                ]
                with st.spinner("Thinking..."):
                    result = think(prompt, history, st.session_state.selected_model)

                if result.get("type") == "async":
                    # Async task started (image/video/audio)
                    task_id = result["task_id"]
                    variant = result["variant"]
                    full_response = f"🎨 **{variant} task started!**\n\nTask ID: `{task_id}`\n\nCheck progress in the sidebar."
                    response_placeholder.markdown(full_response)
                else:
                    # Sync response (text/reasoning)
                    # Display thinking process if enabled
                    if st.session_state.show_thinking and result.get("monologue"):
                        with st.expander("Show thinking process"):
                            st.markdown(result["monologue"])
                    # Stream the answer letter by letter
                    answer = result.get("answer", "")
                    stream_response(response_placeholder, answer, delay=0.03)

                    # Display sources if available
                    if result.get("sources"):
                        with st.expander("View source documents"):
                            for i, doc in enumerate(result["sources"][:3]):
                                st.markdown(f"**Source {i+1}:**")
                                st.write(doc.page_content[:500] + "..." if len(doc.page_content) > 500 else doc.page_content)
                                if i < len(result["sources"]) - 1:
                                    st.divider()
                    # Show thinking time
                    st.caption(f"Thought for {result.get('thinking_time', 0):.2f}s")

        except Exception as e:
            st.error(f"An error occurred: {e}")
            full_response = ""

    # Save assistant message (for sync responses, we already have answer; for async, we store the task message)
    if result.get("type") == "sync":
        st.session_state.messages.append({"role": "assistant", "content": result.get("answer", "")})
    else:
        st.session_state.messages.append({"role": "assistant", "content": full_response})
