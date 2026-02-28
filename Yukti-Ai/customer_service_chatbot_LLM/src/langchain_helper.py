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

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATASET_PATH = os.path.join(BASE_DIR, "dataset", "dataset.csv")
VECTORDB_PATH = os.path.join(BASE_DIR, "faiss_index")

def get_llm():
    api_key = os.environ.get("GOOGLE_API_KEY")
    if not api_key:
        raise ValueError("GOOGLE_API_KEY missing.")
    return GoogleGenerativeAI(model="gemini-1.5-flash", google_api_key=api_key, temperature=0.1)

def get_embeddings():
    return HuggingFaceInstructEmbeddings(model_name="hkunlp/instructor-large")

def create_vector_db():
    if not os.path.exists(DATASET_PATH):
        raise FileNotFoundError(f"Dataset not found at {DATASET_PATH}")
    loader = CSVLoader(file_path=DATASET_PATH, source_column="prompt")
    data = loader.load()
    embeddings = get_embeddings()
    vectordb = FAISS.from_documents(data, embeddings)
    vectordb.save_local(VECTORDB_PATH)

def get_qa_chain():
    if not os.path.exists(VECTORDB_PATH):
        raise FileNotFoundError("Knowledge base not found. Click 'Create Knowledgebase' first.")
    embeddings = get_embeddings()
    vectordb = FAISS.load_local(VECTORDB_PATH, embeddings, allow_dangerous_deserialization=True)
    retriever = vectordb.as_retriever(score_threshold=0.7)
    prompt = PromptTemplate(
        template="""Given context and a question, answer based on context only.
If answer not found, say "I don't know."
CONTEXT: {context}
QUESTION: {input}""",
        input_variables=["context", "input"]
    )
    llm = get_llm()
    combine_docs = create_stuff_documents_chain(llm, prompt)
    return create_retrieval_chain(retriever, combine_docs)
