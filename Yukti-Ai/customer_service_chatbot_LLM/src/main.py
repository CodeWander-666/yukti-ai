import os
import streamlit as st
import pandas as pd
from langchain_core.documents import Document
from langchain_community.vectorstores import FAISS
from langchain_helper import get_qa_chain, get_embeddings, VECTORDB_PATH, BASE_DIR

# ---------- Configuration ----------
SOURCES = [
    {
        "type": "csv",
        "path": os.path.join(BASE_DIR, "dataset", "dataset.csv"),
        "name": "Original Dataset",
        "columns": ["prompt", "response"],
        "content_template": "Q: {prompt}\nA: {response}"
    },
    # Add more CSV files here as needed
    # {
    #     "type": "csv",
    #     "path": os.path.join(BASE_DIR, "dataset", "faq.csv"),
    #     "name": "FAQ",
    #     "columns": ["prompt", "response"],
    #     "content_template": "Q: {prompt}\nA: {response}"
    # },
]

# ---------- Helper Functions ----------
def load_all_documents():
    """Load and combine documents from all sources."""
    docs = []
    for src in SOURCES:
        if src["type"] == "csv":
            path = src["path"]
            if not os.path.exists(path):
                st.warning(f" File not found: {path}")
                continue
            try:
                # Try multiple encodings (for Windows-1252 files)
                encodings = ['utf-8', 'cp1252', 'latin-1', 'iso-8859-1']
                df = None
                for enc in encodings:
                    try:
                        df = pd.read_csv(path, encoding=enc, on_bad_lines='skip')
                        break
                    except UnicodeDecodeError:
                        continue
                if df is None:
                    st.error(f"Could not read {path} with any encoding.")
                    return None
                
                # Check required columns
                missing = [c for c in src["columns"] if c not in df.columns]
                if missing:
                    st.error(f"Missing columns {missing} in {path}")
                    return None
                
                for idx, row in df.iterrows():
                    content = src["content_template"].format(**{c: row[c] for c in src["columns"]})
                    docs.append(Document(
                        page_content=content,
                        metadata={"source": path, "row": idx}
                    ))
                st.success(f" Loaded {len(df)} rows from {src['name']}")
            except Exception as e:
                st.error(f"Error reading {path}: {e}")
                return None
    return docs

def rebuild_knowledge_base():
    """Rebuild FAISS index from all sources."""
    with st.spinner("📥 Loading documents..."):
        docs = load_all_documents()
        if docs is None:
            return False
        if not docs:
            st.warning("No documents found.")
            return False
    
    with st.spinner(" Generating embeddings and building index..."):
        try:
            embeddings = get_embeddings()
            vectordb = FAISS.from_documents(docs, embeddings)
            vectordb.save_local(VECTORDB_PATH)
            st.success(f" Knowledge base rebuilt with {len(docs)} documents!")
            return True
        except Exception as e:
            st.error(f"❌ Build failed: {e}")
            return False

# ---------- Streamlit UI ----------
st.set_page_config(page_title="Customer Service Chatbot", page_icon="AI", layout="wide")
st.title("Yukti AI - Chatbot with Dynamic Knowledge")

# Sidebar
with st.sidebar:
    st.header("⚙️ Admin Panel")
    if st.button("🔄 Update Knowledge Base", use_container_width=True):
        rebuild_knowledge_base()
    
    st.divider()
    st.caption(" Data Sources")
    for src in SOURCES:
        st.write(f"- {src['name']}")
    
    st.divider()
    st.caption(" Info")
    st.write("Click **Update Knowledge Base** after adding new sources.")

# Main chat area
question = st.text_input("💬 Ask a question:", placeholder="Type your question here...")

if question:
    with st.spinner(" Thinking..."):
        try:
            chain = get_qa_chain()
            response = chain.invoke({"input": question})
            st.header("Ans wer")
            st.write(response["answer"])
        except FileNotFoundError as e:
            st.warning(f"{e}\n\nClick **'Update Knowledge Base'** in the sidebar to create it.")
        except Exception as e:
            st.error(f" An error occurred: {e}")
