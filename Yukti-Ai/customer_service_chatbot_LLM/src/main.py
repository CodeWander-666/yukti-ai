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
    # Add more sources as needed
]

# ----------------------------------------------------------------------
# Helper to load documents from all sources
# ----------------------------------------------------------------------
def load_all_documents():
    """Fetch documents from all configured sources (handles any encoding)."""
    docs = []
    for src in SOURCES:
        if src["type"] == "csv":
            path = src["path"]
            if not os.path.exists(path):
                st.warning(f"Source file not found: {path}")
                continue
            try:
                # Read file in binary mode, decode with replacement
                with open(path, 'rb') as f:
                    raw = f.read()
                # Decode with 'replace' to handle any invalid bytes
                content = raw.decode('utf-8', errors='replace')
                # Split lines and parse CSV manually
                lines = content.splitlines()
                if not lines:
                    st.warning(f"Empty file: {path}")
                    continue
                header = lines[0].split(',')
                # Find indices of required columns
                try:
                    prompt_idx = header.index('prompt')
                    response_idx = header.index('response')
                except ValueError:
                    st.error(f"CSV must have 'prompt' and 'response' columns. Found: {header}")
                    return None

                for i, line in enumerate(lines[1:], start=2):
                    if not line.strip():
                        continue
                    # Simple CSV parsing (handles quoted fields with commas)
                    parts = []
                    in_quote = False
                    current = []
                    for ch in line:
                        if ch == '"' and not in_quote:
                            in_quote = True
                        elif ch == '"' and in_quote:
                            in_quote = False
                        elif ch == ',' and not in_quote:
                            parts.append(''.join(current).strip())
                            current = []
                        else:
                            current.append(ch)
                    parts.append(''.join(current).strip())
                    if len(parts) > max(prompt_idx, response_idx):
                        prompt = parts[prompt_idx].strip('"')
                        response = parts[response_idx].strip('"')
                        content = src["content_template"].format(prompt=prompt, response=response)
                        docs.append(Document(
                            page_content=content,
                            metadata={"source": path, "row": i}
                        ))
                st.info(f"Loaded {len(docs)} rows from {src['name']}")
            except Exception as e:
                st.error(f"Error reading {path}: {e}")
                return None
    return docs
# ----------------------------------------------------------------------
# Rebuild the vector database from all sources
# ----------------------------------------------------------------------
def rebuild_knowledge_base():
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
