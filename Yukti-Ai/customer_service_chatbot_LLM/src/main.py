import os
import streamlit as st
import pandas as pd
from langchain_core.documents import Document
from langchain_community.vectorstores import FAISS
from langchain_helper import get_qa_chain, get_embeddings, VECTORDB_PATH, BASE_DIR

# ---------- Configuration ----------
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
st.caption("Your Intelligent Assistant – Powered by Gemini")

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

# ---------- Helper Functions ----------
def load_all_documents():
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

# ---------- Sidebar ----------
with st.sidebar:
    st.header("Knowledge Base")
    
    if st.button("🔄 Update Knowledge Base", use_container_width=True):
        rebuild_knowledge_base()
    
    st.divider()
    
    st.subheader("Status")
    if st.session_state.knowledge_base_ready:
        st.markdown("✅ **Active**")
    else:
        st.markdown("❌ **Not built** – click update above.")
    
    st.divider()
    
    st.subheader("Sources")
    for src in SOURCES:
        st.markdown(f"- **{src['name']}**")
    
    st.divider()
    
    if st.button("🗑️ Clear Conversation", use_container_width=True):
        st.session_state.messages = [
            {"role": "assistant", "content": "Hello! I'm Yukti AI. How can I help you today?"}
        ]
        st.rerun()

# ---------- Main Chat Interface ----------
# Display chat history
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])
        # Show source documents if available (for assistant messages)
        if msg["role"] == "assistant" and "sources" in msg:
            with st.expander("View source documents"):
                for i, doc in enumerate(msg["sources"]):
                    st.markdown(f"**Source {i+1}:**")
                    st.write(doc.page_content)
                    if i < len(msg["sources"]) - 1:
                        st.divider()

# Chat input
if prompt := st.chat_input("Ask me anything..."):
    # Add user message
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)
    
    # Generate response
    with st.chat_message("assistant"):
        response_placeholder = st.empty()
        full_response = ""
        sources = []
        
        try:
            if not st.session_state.knowledge_base_ready:
                full_response = "The knowledge base is not ready. Please click 'Update Knowledge Base' in the sidebar first."
            else:
                chain = get_qa_chain()
                result = chain.invoke({"input": prompt})
                full_response = result.get("answer", "I'm sorry, I couldn't generate an answer.")
                sources = result.get("context", [])
            
            response_placeholder.markdown(full_response)
            
            # Show sources if any
            if sources:
                with st.expander("View source documents"):
                    for i, doc in enumerate(sources):
                        st.markdown(f"**Source {i+1}:**")
                        st.write(doc.page_content)
                        if i < len(sources) - 1:
                            st.divider()
            
        except FileNotFoundError as e:
            full_response = "The knowledge base is not ready. Please click 'Update Knowledge Base' in the sidebar first."
            response_placeholder.markdown(full_response)
        except Exception as e:
            full_response = f"An error occurred: {e}"
            response_placeholder.markdown(full_response)
    
    # Save assistant message
    st.session_state.messages.append({
        "role": "assistant",
        "content": full_response,
        "sources": sources
    })
