import os
from dotenv import load_dotenv

from langchain_community.vectorstores import FAISS
from langchain_community.document_loaders import CSVLoader
from langchain_community.embeddings import HuggingFaceInstructEmbeddings
from langchain_core.prompts import PromptTemplate
from langchain_google_genai import GoogleGenerativeAI
from langchain_classic.chains import create_retrieval_chain
from langchain_classic.chains.combine_documents import create_stuff_documents_chain

# Take environment variables from .env (especially GOOGLE_API_KEY)
load_dotenv()  

# 1. Create Google Gemini LLM model (Replaces deprecated PaLM)
llm = GoogleGenerativeAI(
    model="gemini-1.5-flash", 
    google_api_key=os.environ["GOOGLE_API_KEY"], 
    temperature=0.1
)

# 2. Initialize instructor embeddings using the Hugging Face model
instructor_embeddings = HuggingFaceInstructEmbeddings(
    model_name="hkunlp/instructor-large"
)

vectordb_file_path = "faiss_index"

def create_vector_db():
    """Loads CSV data, creates a FAISS vector database, and saves it locally."""
    loader = CSVLoader(file_path="dataset.csv", source_column="prompt")
    data = loader.load()

    # Create a FAISS instance for vector database from 'data'
    vectordb = FAISS.from_documents(documents=data, embedding=instructor_embeddings)

    # Save vector database locally
    vectordb.save_local(vectordb_file_path)
    print("Vector database created and saved successfully!")

def get_qa_chain():
    """Loads the local vector database and builds the modern LCEL retrieval chain."""
    # Load the vector database (requires explicit permission for local pickle files)
    vectordb = FAISS.load_local(
        folder_path=vectordb_file_path, 
        embeddings=instructor_embeddings,
        allow_dangerous_deserialization=True
    )

    # Create a retriever for querying the vector database
    retriever = vectordb.as_retriever(score_threshold=0.7)

    # Modern chains use {input} instead of {question}
    prompt_template = """Given the following context and a question, generate an answer based on this context only.
    In the answer try to provide as much text as possible from the "response" section in the source document context without making much changes.
    If the answer is not found in the context, kindly state "I don't know." Don't try to make up an answer.

    CONTEXT: {context}

    QUESTION: {input}"""

    PROMPT = PromptTemplate(
        template=prompt_template, 
        input_variables=["context", "input"]
    )

    # Create the modern retrieval chain (replaces RetrievalQA)
    combine_docs_chain = create_stuff_documents_chain(llm, PROMPT)
    retrieval_chain = create_retrieval_chain(retriever, combine_docs_chain)

    return retrieval_chain

if __name__ == "__main__":
    # create_vector_db() # Uncomment this if you need to regenerate the database
    
    chain = get_qa_chain()
    
    # Modern chains use .invoke() and return a dictionary where the response is under the "answer" key
    response = chain.invoke({"input": "hello?"})
    print(response["answer"])
