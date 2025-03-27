# Main Streamlit Application for RAG Chatbot

import logging
import os
from pathlib import Path

import streamlit as st
from langchain_core.embeddings import Embeddings

import chroma_integration
import embeddings
import langchain_integration
import pdf_loader
import text_chunking

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Constants
PERSIST_DIRECTORY = "D:\\ml\\projects\\Rag_app\\chroma_db"
OLLAMA_BASE_URL = "http://localhost:11434"
EMBEDDING_MODEL = "nomic-embed-text"
LLM_MODEL = "deepseek-r1:1.5b"
CHUNK_SIZE = 1000
CHUNK_OVERLAP = 200

def initialize_session_state():
    """Initialize session state variables."""
    if "messages" not in st.session_state:
        st.session_state.messages = []
    
    if "rag_chain" not in st.session_state:
        st.session_state.rag_chain = None
    
    if "vector_store" not in st.session_state:
        st.session_state.vector_store = None
        
    if "embedding_model" not in st.session_state:
        st.session_state.embedding_model = None

def load_and_process_document(uploaded_file):
    """Load and process the uploaded document."""
    try:
        # Save the uploaded file
        file_path = pdf_loader.save_uploaded_file(uploaded_file)
        if not file_path:
            st.error("Failed to save the uploaded file.")
            return False
            
        # Load the PDF
        documents = pdf_loader.load_pdf(file_path)
        if not documents:
            st.error("Failed to extract content from the PDF.")
            return False
            
        # Chunk the documents
        chunks = text_chunking.chunk_documents(
            documents, 
            chunk_size=CHUNK_SIZE, 
            chunk_overlap=CHUNK_OVERLAP
        )
        
        # Get embeddings model
        embedding_model = embeddings.get_ollama_embeddings(
            model_name=EMBEDDING_MODEL,
            base_url=OLLAMA_BASE_URL
        )
        st.session_state.embedding_model = embedding_model
        
        # Create vector store
        vector_store = chroma_integration.create_vector_store(
            documents=chunks,
            embeddings=embedding_model,
            persist_directory=PERSIST_DIRECTORY
        )
        st.session_state.vector_store = vector_store
        
        # Get LLM
        llm = langchain_integration.get_ollama_llm(
            model_name=LLM_MODEL,
            base_url=OLLAMA_BASE_URL
        )
        
        # Create retrieval chain
        rag_chain = langchain_integration.create_retrieval_chain(
            vector_store=vector_store,
            llm=llm
        )
        st.session_state.rag_chain = rag_chain
        
        return True
    except Exception as e:
        logger.error(f"Error processing document: {e}")
        st.error(f"An error occurred: {str(e)}")
        return False
    
def main():
    """Main application function."""
    st.set_page_config(
        page_title="Local RAG Assistant Chatbot",
        page_icon="🤖",
        layout="wide",
        initial_sidebar_state="expanded"
    )
    
    # Initialize session state
    initialize_session_state()
    
    # Apply dark theme styling
    st.markdown("""
    <style>
    .stApp {
        background-color: #0e1117;
        color: #fafafa;
    }
    .stTextInput, .stTextArea {
        background-color: #262730;
        color: #fafafa;
    }
    .stButton>button {
        background-color: #4e57d4;
        color: white;
    }
    .chat-message {
        padding: 1rem;
        border-radius: 0.5rem;
        margin-bottom: 1rem;
        display: flex;
    }
    .chat-message.user {
        background-color: #2e2e2e;
    }
    .chat-message.assistant {
        background-color: #004d40;
    }
    .chat-message .avatar {
        width: 40px;
        height: 40px;
        border-radius: 50%;
        object-fit: cover;
        margin-right: 1rem;
    }
    .chat-message .message {
        flex: 1;
    }
    </style>
    """, unsafe_allow_html=True)
    
    # Sidebar for file upload and settings
    with st.sidebar:
        st.title("📄 Document Upload")
        uploaded_file = st.file_uploader("Upload a PDF document", type=["pdf"])
        
        if uploaded_file is not None:
            if st.button("Process Document", key="process"):
                with st.spinner("Processing document..."):
                    success = load_and_process_document(uploaded_file)
                    if success:
                        st.success("Document processed successfully!")
                    else:
                        st.error("Failed to process document.")
        
        st.divider()
        st.subheader("⚙️ Settings")
        
        # Display connection status
        try:
            # Check if Ollama is available
            if st.session_state.embedding_model:
                st.success("✅ Connected to Ollama")
            else:
                # Try to initialize embedding model to check connection
                try:
                    embedding_model = embeddings.get_ollama_embeddings(
                        model_name=EMBEDDING_MODEL,
                        base_url=OLLAMA_BASE_URL
                    )
                    st.session_state.embedding_model = embedding_model
                    st.success("✅ Connected to Ollama")
                except:
                    st.error("❌ Failed to connect to Ollama")
                    st.info("Make sure Ollama is running on http://localhost:11434")
        except:
            st.error("❌ Failed to connect to Ollama")
            st.info("Make sure Ollama is running on http://localhost:11434")
        
        # Check if the correct models are available
        if st.session_state.embedding_model:
            st.info(f"Embedding model: {EMBEDDING_MODEL}")
            st.info(f"LLM model: {LLM_MODEL}")
            
            # Option to clear conversation
            if st.button("Clear Conversation"):
                st.session_state.messages = []
                st.rerun()
    
    # Main chat interface
    st.title("💬 RAG Chatbot")
    
    # Display chat messages
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.write(message["content"])
    
    # Chat input
    if prompt := st.chat_input("Ask a question about your document..."):
        # Add user message to chat history
        st.session_state.messages.append({"role": "user", "content": prompt})
        
        # Display user message
        with st.chat_message("user"):
            st.write(prompt)
            
        # Generate response
        with st.chat_message("assistant"):
            if st.session_state.rag_chain:
                with st.spinner("Thinking..."):
                    response_placeholder = st.empty()
                    response = st.session_state.rag_chain.invoke(prompt)
                    response_placeholder.write(response)
                    
                # Add assistant response to chat history
                st.session_state.messages.append({"role": "assistant", "content": response})
            else:
                st.warning("Please upload and process a document first.")
                # Add default response to chat history
                st.session_state.messages.append({"role": "assistant", "content": "Please upload and process a document first."})
    
    # Display initial instructions if no conversation has started
    if not st.session_state.messages:
        st.info("👈 Please upload a PDF document using the sidebar to get started.")

if __name__ == "__main__":
    main()