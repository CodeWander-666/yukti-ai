import os
from dotenv import load_dotenv

from langchain_community.vectorstores import FAISS
from langchain_community.document_loaders import CSVLoader
from langchain_community.embeddings import HuggingFaceInstructEmbeddings
from langchain_core.prompts import PromptTemplate
from langchain_google_genai import GoogleGenerativeAI
from langchain_classic.chains import create_retrieval_chain
from langchain_classic.chains.combine_documents import create_stuff_documents_chain

load_dotenv()

# ----------------------------------------------------------------------
# Path configuration (absolute paths for Streamlit Cloud)
# ----------------------------------------------------------------------
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# The CSV file is inside the "dataset" folder
DATASET_PATH = os.path.join(BASE_DIR, "dataset", "dataset.csv")
VECTORDB_PATH = os.path.join(BASE_DIR, "faiss_index")

# ----------------------------------------------------------------------
# Resource initialisation with error handling
# ----------------------------------------------------------------------
def get_llm():
    """Create and return the Google Gemini LLM."""
    api_key = os.environ.get("GOOGLE_API_KEY")
    if not api_key:
        raise ValueError(
            "GOOGLE_API_KEY not found. Please set it in your .env file or Streamlit secrets."
        )
    return GoogleGenerativeAI(
        model="gemini-1.5-flash",
        google_api_key=api_key,
        temperature=0.1
    )

def get_embeddings():
    """Create and return the Hugging Face instructor embeddings."""
    try:
        return HuggingFaceInstructEmbeddings(
            model_name="hkunlp/instructor-large"
        )
    except Exception as e:
        raise RuntimeError(f"Failed to load embedding model: {e}")

# ----------------------------------------------------------------------
# Core functions
# ----------------------------------------------------------------------
def create_vector_db():
    """
    Load CSV data, create a FAISS vector database, and save it locally.
    Raises exceptions with descriptive messages on failure.
    """
    # Check if dataset exists
    if not os.path.exists(DATASET_PATH):
        raise FileNotFoundError(
            f"Dataset file not found at {DATASET_PATH}. "
            "Please ensure 'dataset.csv' is inside the 'dataset/' folder."
        )

    # Load CSV data
    try:
        loader = CSVLoader(file_path=DATASET_PATH, source_column="prompt")
        data = loader.load()
    except Exception as e:
        raise RuntimeError(f"Failed to load CSV data: {e}")

    # Create embeddings and vector store
    embeddings = get_embeddings()
    try:
        vectordb = FAISS.from_documents(documents=data, embedding=embeddings)
    except Exception as e:
        raise RuntimeError(f"Failed to create vector database: {e}")

    # Save the index
    try:
        vectordb.save_local(VECTORDB_PATH)
    except Exception as e:
        raise RuntimeError(f"Failed to save vector database to {VECTORDB_PATH}: {e}")

    print(f"Vector database created successfully at {VECTORDB_PATH}")

def get_qa_chain():
    """
    Load the saved FAISS index and build a retrieval QA chain.
    Returns the chain object. Raises exceptions on failure.
    """
    # Check if the vector database exists
    if not os.path.exists(VECTORDB_PATH):
        raise FileNotFoundError(
            f"Vector database not found at {VECTORDB_PATH}. "
            "Please click 'Create Knowledgebase' first."
        )

    # Load the vector store
    embeddings = get_embeddings()
    try:
        vectordb = FAISS.load_local(
            folder_path=VECTORDB_PATH,
            embeddings=embeddings,
            allow_dangerous_deserialization=True   # required for local pickle files
        )
    except Exception as e:
        raise RuntimeError(f"Failed to load vector database: {e}")

    # Create retriever
    try:
        retriever = vectordb.as_retriever(score_threshold=0.7)
    except Exception as e:
        raise RuntimeError(f"Failed to create retriever: {e}")

    # Define prompt template
    prompt_template = """Given the following context and a question, generate an answer based on this context only.
    In the answer try to provide as much text as possible from the "response" section in the source document context without making much changes.
    If the answer is not found in the context, kindly state "I don't know." Don't try to make up an answer.

    CONTEXT: {context}

    QUESTION: {input}"""

    PROMPT = PromptTemplate(
        template=prompt_template,
        input_variables=["context", "input"]
    )

    # Create LLM
    llm = get_llm()

    # Build chains using classic imports
    try:
        combine_docs_chain = create_stuff_documents_chain(llm, PROMPT)
        retrieval_chain = create_retrieval_chain(retriever, combine_docs_chain)
    except Exception as e:
        raise RuntimeError(f"Failed to build retrieval chain: {e}")

    return retrieval_chain
