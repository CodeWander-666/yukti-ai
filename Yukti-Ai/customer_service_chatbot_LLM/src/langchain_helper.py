"""
Yukti AI – LangChain Helper Module
Optimized for speed: singleton embeddings, connection reuse, concise prompts.
"""

import os
import logging
from functools import lru_cache
from dotenv import load_dotenv

from langchain_community.vectorstores import FAISS
from langchain_community.document_loaders import CSVLoader
from langchain_community.embeddings import HuggingFaceInstructEmbeddings
from langchain_core.prompts import PromptTemplate
from langchain_google_genai import GoogleGenerativeAI
from langchain_classic.chains import create_retrieval_chain
from langchain_classic.chains.combine_documents import create_stuff_documents_chain

load_dotenv()
logger = logging.getLogger(__name__)

# ----------------------------------------------------------------------
# Paths (absolute – works on Streamlit Cloud)
# ----------------------------------------------------------------------
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATASET_PATH = os.path.join(BASE_DIR, "dataset", "dataset.csv")
VECTORDB_PATH = os.path.join(BASE_DIR, "faiss_index")

# ----------------------------------------------------------------------
# Gemini Model Configuration – 2026 ready
# ----------------------------------------------------------------------
GEMINI_MODELS = [
    "gemini-2.5-flash",      # Fast, stable, recommended
    "gemini-2.5-pro",
    "gemini-3-flash",
    "gemini-3.1-pro",
]

# ----------------------------------------------------------------------
# Cached Resources (singletons for speed)
# ----------------------------------------------------------------------
@lru_cache(maxsize=1)
def get_embeddings():
    """Return a cached embedding model (fast, lightweight)."""
    try:
        logger.info("Loading embedding model: all-MiniLM-L6-v2")
        return HuggingFaceInstructEmbeddings(
            model_name="sentence-transformers/all-MiniLM-L6-v2",
            encode_kwargs={'normalize_embeddings': True}
        )
    except Exception as e:
        logger.exception("Failed to load embedding model")
        raise RuntimeError(f"Embedding model unavailable: {e}")

@lru_cache(maxsize=1)
def get_llm():
    """Return a working Gemini LLM with fallback models (cached)."""
    api_key = os.environ.get("GOOGLE_API_KEY")
    if not api_key:
        raise ValueError("GOOGLE_API_KEY missing. Set in .env or Streamlit secrets.")
    
    last_error = None
    for model in GEMINI_MODELS:
        try:
            logger.info(f"Attempting to use model: {model}")
            llm = GoogleGenerativeAI(
                model=model,
                google_api_key=api_key,
                temperature=0.1,
                max_retries=2,
                request_timeout=30
            )
            # Quick test call (optional, remove in production if not needed)
            # llm.invoke("test")
            logger.info(f"Using model: {model}")
            return llm
        except Exception as e:
            last_error = e
            logger.warning(f"Model {model} failed: {e}")
            continue
    raise RuntimeError(f"All Gemini models failed. Last error: {last_error}")

# ----------------------------------------------------------------------
# Vector Database Operations
# ----------------------------------------------------------------------
def create_vector_db():
    """Build FAISS index from CSV (called via UI)."""
    if not os.path.exists(DATASET_PATH):
        raise FileNotFoundError(f"Dataset not found at {DATASET_PATH}")
    
    logger.info(f"Loading CSV from {DATASET_PATH}")
    loader = CSVLoader(
        file_path=DATASET_PATH,
        source_column="prompt",
        encoding="utf-8"
    )
    data = loader.load()
    logger.info(f"Loaded {len(data)} documents")
    
    embeddings = get_embeddings()
    vectordb = FAISS.from_documents(data, embeddings)
    vectordb.save_local(VECTORDB_PATH)
    logger.info(f"Index saved to {VECTORDB_PATH}")

def get_qa_chain():
    """Load index and return retrieval QA chain (fast)."""
    if not os.path.exists(VECTORDB_PATH):
        raise FileNotFoundError("Knowledge base not found. Please build it first.")
    
    embeddings = get_embeddings()
    vectordb = FAISS.load_local(
        VECTORDB_PATH,
        embeddings,
        allow_dangerous_deserialization=True
    )
    
    # Retrieve top 3 documents for speed
    retriever = vectordb.as_retriever(search_kwargs={"k": 3})
    
    # Concise prompt for faster generation
    prompt = PromptTemplate(
        template="""Answer the question using only the context below.
If the answer is not in the context, say "I don't know."

Context:
{context}

Question: {input}""",
        input_variables=["context", "input"]
    )
    
    llm = get_llm()
    combine_docs = create_stuff_documents_chain(llm, prompt)
    chain = create_retrieval_chain(retriever, combine_docs)
    
    return chain

# ----------------------------------------------------------------------
# Retrieval function for thinking engine
# ----------------------------------------------------------------------
def retrieve_documents(query: str, k: int = 5):
    """Retrieve top k relevant documents from the knowledge base."""
    if not os.path.exists(VECTORDB_PATH):
        raise FileNotFoundError("Knowledge base not found.")
    embeddings = get_embeddings()
    vectordb = FAISS.load_local(VECTORDB_PATH, embeddings, allow_dangerous_deserialization=True)
    return vectordb.similarity_search(query, k=k)

# ----------------------------------------------------------------------
# For standalone testing (optional)
# ----------------------------------------------------------------------
if __name__ == "__main__":
    try:
        chain = get_qa_chain()
        response = chain.invoke({"input": "What is this course about?"})
        print("Answer:", response.get("answer"))
    except Exception as e:
        print(f"Error: {e}")
