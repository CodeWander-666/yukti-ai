import streamlit as st
from langchain_helper import get_qa_chain, create_vector_db

st.title("CUSTOMER SERVICE CHATBOT 🤖")

# Button to create knowledge base
if st.button("Create Knowledgebase"):
    with st.spinner("Building knowledge base..."):
        try:
            create_vector_db()
            st.success("✅ Knowledge base created successfully!")
        except FileNotFoundError as e:
            st.error(f"❌ File not found: {e}")
        except ValueError as e:
            st.error(f"❌ Configuration error: {e}")
        except Exception as e:
            st.error(f"❌ An unexpected error occurred: {e}")

# Question input
question = st.text_input("Ask a question:")

if question:
    with st.spinner("Searching for answer..."):
        try:
            chain = get_qa_chain()
            response = chain.invoke({"input": question})
            st.header("Answer")
            st.write(response["answer"])
        except FileNotFoundError as e:
            st.warning(f"⚠️ {e}")   # Shows "Please click 'Create Knowledgebase' first"
        except ValueError as e:
            st.error(f"❌ Configuration error: {e}")
        except Exception as e:
            st.error(f"❌ An error occurred: {e}")
