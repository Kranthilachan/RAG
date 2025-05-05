import streamlit as st
from app import (load_data, split_text, get_embeddings, 
                create_vectorstore, create_retriever, create_qa_chain)
from prompt_pem import temp
from harm import chat_model
from langchain_community.vectorstores import Chroma, FAISS

# Initialize session state
if 'qa_chain' not in st.session_state:
    st.session_state.qa_chain = None
if 'initialized' not in st.session_state:
    st.session_state.initialized = False

def initialize_system():
    try:
        # Load and process documents
        documents = load_data()
        texts = split_text(documents)
        embeddings = get_embeddings()
        vectordb = create_vectorstore(texts, embeddings)
        retriever = create_retriever(vectordb, chat_model)
        
        # Get prompt template
        prompt = temp()
        
        # Create QA chain
        st.session_state.qa_chain = create_qa_chain(chat_model, retriever, prompt)
        st.session_state.initialized = True
        return True
    except Exception as e:
        st.error(f"Initialization failed: {str(e)}")
        return False

# Streamlit UI
st.title("Budget Speech Q&A with Gemini")
st.markdown("Ask questions about the budget speech document using RAG and Google Gemini")

# Sidebar for settings
with st.sidebar:
    st.header("Settings")
    if st.button("Initialize System"):
        with st.spinner("Initializing RAG system..."):
            if initialize_system():
                st.success("System initialized successfully!")
            else:
                st.error("Initialization failed")

# Main content area
if not st.session_state.initialized:
    st.warning("Please initialize the system from the sidebar first")
else:
    question = st.text_input("Enter your question about the budget speech:")
    
    if question:
        with st.spinner("Searching for answer..."):
            try:
                # Get answer from QA chain
                result = st.session_state.qa_chain({"query": question})
                answer = result["result"]
                
                # Display answer
                st.subheader("Answer:")
                st.write(answer)
                
            except Exception as e:
                st.error(f"Error getting answer: {str(e)}")