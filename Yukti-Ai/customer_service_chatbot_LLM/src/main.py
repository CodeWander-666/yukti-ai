import os
import streamlit as st
import pandas as pd
from langchain_core.documents import Document
from langchain_helper import get_qa_chain, get_embeddings, VECTORDB_PATH, BASE_DIR
from langchain_community.vectorstores import FAISS

# ----------------------------------------------------------------------
# Configuration – list all your data sources here
# ----------------------------------------------------------------------
SOURCES = [
    {
        "type": "csv",
        "path": os.path.join(BASE_DIR, "dataset", "dataset.csv"),
        "name": "Original Dataset",
        "columns": ["prompt", "response"],
        "content_template": "Q: {prompt}\nA: {response}"
    },
    # Add more sources as needed, e.g.:
    # {
    #     "type": "csv",
    #     "path": os.path.join(BASE_DIR, "dataset", "faq.csv"),
    #     "name": "FAQ",
    #     "columns": ["prompt", "response"],
    #     "content_template": "Q: {prompt}\nA: {response}"
    # },
]

# ----------------------------------------------------------------------
# Helper to load documents from all sources
# ----------------------------------------------------------------------
def load_all_documents():
    """Fetch documents from all configured sources."""
    docs = []
    for src in SOURCES:
        if src["type"] == "csv":
            path = src["path"]
            if not os.path.exists(path):
                st.warning(f"Source file not found: {path}")
                continue
            try:
                # Try multiple encodings
                encodings = ['utf-8', 'latin-1', 'cp1252', 'iso-8859-1']
                df = None
                for enc in encodings:
                    try:
                        df = pd.read_csv(path, encoding=enc, on_bad_lines='skip')
                        break
                    except UnicodeDecodeError:
                        continue
                if df is None:
                    st.error(f"Could not read {path} with any common encoding.")
                    return None

                # Check required columns
                for col in src["columns"]:
                    if col not in df.columns:
                        st.error(f"Column '{col}' missing in {path}")
                        return None
                for idx, row in df.iterrows():
                    content = src["content_template"].format(**{col: row[col] for col in src["columns"]})
                    docs.append(Document(
                        page_content=content,
                        metadata={"source": path, "row": idx}
                    ))
                st.info(f"Loaded {len(df)} rows from {src['name']}")
            except Exception as e:
                st.error(f"Error reading {path}: {e}")
                return None
        # Add other source types here if needed
    return docs

# ----------------------------------------------------------------------
# Rebuild the vector database from all sources
# ----------------------------------------------------------------------
def rebuild_knowledge_base():
    """Rebuild FAISS index from all sources."""
    with st.spinner("Loading documents from sources..."):
        docs = load_all_documents()
        if docs is None:
            return False
        if not docs:
            st.warning("No documents found. Check your sources.")
            return False

    with st.spinner("Generating embeddings and building index..."):
        try:
            embeddings = get_embeddings()
            vectordb = FAISS.from_documents(docs, embeddings)
            vectordb.save_local(VECTORDB_PATH)
            st.success(f"Knowledge base rebuilt with {len(docs)} documents!")
            return True
        except Exception as e:
            st.error(f"Failed to build index: {e}")
            return False

# ----------------------------------------------------------------------
# Streamlit UI
# ----------------------------------------------------------------------
st.set_page_config(page_title="Customer Service Chatbot", page_icon="🤖")
st.title("🤖 CUSTOMER SERVICE CHATBOT with Dynamic Knowledge")

# Sidebar for admin actions
with st.sidebar:
    st.header("Admin")
    if st.button("🔄 Update Knowledge Base"):
        rebuild_knowledge_base()

    st.markdown("---")
    st.caption("Sources configured:")
    for src in SOURCES:
        st.write(f"- {src['name']}")

# Main chat area
question = st.text_input("Ask a question:")

if question:
    try:
        chain = get_qa_chain()
        response = chain.invoke({"input": question})
        st.header("Answer")
        st.write(response["answer"])
    except FileNotFoundError as e:
        st.warning(f"⚠️ {e}\n\nClick **'Update Knowledge Base'** in the sidebar to create it.")
    except Exception as e:
        st.error(f"An error occurred: {e}")
