import os
import logging
from dotenv import load_dotenv
from langchain_community.vectorstores import FAISS
from langchain_community.document_loaders import CSVLoader
from langchain_community.embeddings import HuggingFaceInstructEmbeddings
from langchain_core.prompts import PromptTemplate
from langchain_google_genai import GoogleGenerativeAI
from langchain_classic.chains import create_retrieval_chain
from langchain_classic.chains.combine_documents import create_stuff_documents_chain

load_dotenv()
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATASET_PATH = os.path.join(BASE_DIR, "dataset", "dataset.csv")
VECTORDB_PATH = os.path.join(BASE_DIR, "faiss_index")

# Faster models list (gemini-pro is most reliable)
GEMINI_MODELS = [
    "gemini-1.5-flash",
    "gemini-1.5-pro",
    "gemini-pro",
]

def get_llm():
    api_key = os.environ.get("GOOGLE_API_KEY")
    if not api_key:
        raise ValueError("GOOGLE_API_KEY missing.")
    last_error = None
    for model in GEMINI_MODELS:
        try:
            logger.info(f"Trying model: {model}")
            llm = GoogleGenerativeAI(
                model=model,
                google_api_key=api_key,
                temperature=0.1,
                max_retries=2,
                request_timeout=30
            )
            return llm
        except Exception as e:
            last_error = e
            continue
    raise RuntimeError(f"All Gemini models failed. Last error: {last_error}")

# Faster embedding model
def get_embeddings():
    try:
        return HuggingFaceInstructEmbeddings(
            model_name="sentence-transformers/all-MiniLM-L6-v2",
            encode_kwargs={'normalize_embeddings': True}
        )
    except Exception as e:
        raise RuntimeError(f"Failed to load embedding model: {e}")

def create_vector_db():
    if not os.path.exists(DATASET_PATH):
        raise FileNotFoundError(f"Dataset not found at {DATASET_PATH}")
    logger.info(f"Loading CSV from {DATASET_PATH}")
    loader = CSVLoader(file_path=DATASET_PATH, source_column="prompt", encoding="utf-8")
    data = loader.load()
    logger.info(f"Loaded {len(data)} documents")
    embeddings = get_embeddings()
    vectordb = FAISS.from_documents(data, embeddings)
    vectordb.save_local(VECTORDB_PATH)
    logger.info(f"Index saved to {VECTORDB_PATH}")

def get_qa_chain():
    if not os.path.exists(VECTORDB_PATH):
        raise FileNotFoundError("Knowledge base not found. Click 'Update Knowledge Base' first.")
    embeddings = get_embeddings()
    vectordb = FAISS.load_local(VECTORDB_PATH, embeddings, allow_dangerous_deserialization=True)
    retriever = vectordb.as_retriever(search_kwargs={"k": 3})  # Faster: fewer docs
    prompt = PromptTemplate(
        template="""Given the context below, answer the question.
If the answer is not in the context, say "I don't know."

Context:
{context}

Question: {input}""",
        input_variables=["context", "input"]
    )
    llm = get_llm()
    combine_docs = create_stuff_documents_chain(llm, prompt)
    return create_retrieval_chain(retriever, combine_docs)
