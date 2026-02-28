import os
import time
import streamlit as st
import pandas as pd
from langchain_core.documents import Document
from langchain_community.vectorstores import FAISS
from langchain_helper import get_embeddings, VECTORDB_PATH, BASE_DIR, create_vector_db
from think import think
from model_manager import get_available_models, MODELS
from LLM.zhipu.queue_manager import TaskQueue  # import directly

# ---------- Page Configuration ----------
st.set_page_config(
    page_title="Yukti AI",
    page_icon="✨",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ---------- Custom CSS (unchanged) ----------
st.markdown("""
<style>
    /* ... your existing CSS ... */
</style>
""", unsafe_allow_html=True)

st.title("Yukti AI")
st.caption("Your Intelligent Assistant – Powered by Zhipu GLM")

# ---------- Session State Initialization ----------
if "messages" not in st.session_state:
    st.session_state.messages = [
        {"role": "assistant", "content": "Hello! I'm Yukti AI. How can I help you today?"}
    ]
if "knowledge_base_ready" not in st.session_state:
    st.session_state.knowledge_base_ready = os.path.exists(VECTORDB_PATH)

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

# ---------- Helper Functions (unchanged) ----------
def load_all_documents():
    # ... same as before ...
    pass

def rebuild_knowledge_base():
    # ... same as before ...
    pass

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
            elif status == "submitted" or status == "pending":
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
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message("assistant"):
        response_placeholder = st.empty()
        try:
            if not st.session_state.knowledge_base_ready:
                full_response = "The knowledge base is not ready. Please click 'Update Knowledge Base' in the sidebar first."
                response_placeholder.markdown(full_response)
            else:
                history = [
                    {"role": msg["role"], "content": msg["content"]}
                    for msg in st.session_state.messages[-10:]
                ]
                with st.spinner("Thinking..."):
                    result = think(prompt, history, st.session_state.selected_model)

                if result.get("type") == "async":
                    # Async task started
                    task_id = result["task_id"]
                    variant = result["variant"]
                    full_response = f"🎨 **{variant} task started!**\n\nTask ID: `{task_id}`\n\nCheck progress in the sidebar."
                    response_placeholder.markdown(full_response)
                else:
                    # Sync response
                    if result.get("monologue"):
                        with st.expander("Show thinking process"):
                            st.markdown(result["monologue"])
                    response_placeholder.markdown(result.get("answer", ""))
                    if result.get("sources"):
                        with st.expander("View source documents"):
                            for i, doc in enumerate(result["sources"][:3]):
                                st.markdown(f"**Source {i+1}:**")
                                st.write(doc.page_content[:500] + "...")
                                if i < len(result["sources"]) - 1:
                                    st.divider()
        except Exception as e:
            st.error(f"An error occurred: {e}")
        st.caption(f"Thought for {result.get('thinking_time', 0):.2f}s" if result.get("type")=="sync" else "")

    st.session_state.messages.append({"role": "assistant", "content": full_response})
