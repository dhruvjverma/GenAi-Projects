"""Module for Chroma vector database integration."""

import logging
import os
from typing import List, Optional

from langchain.schema import Document
from langchain_chroma import Chroma
from langchain_core.embeddings import Embeddings

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def create_vector_store(documents: List[Document], 
                        embeddings: Embeddings, 
                        persist_directory: Optional[str] = None) -> Chroma:
    """
    Create a Chroma vector store from documents.
    
    Args:
        documents: List of documents to store.
        embeddings: Embeddings model to use.
        persist_directory: Directory to persist the vector store to.
        
    Returns:
        A Chroma vector store instance.
    """
    try:
        logger.info(f"Creating Chroma vector store with {len(documents)} documents")
        
        # If no persist directory is specified, use a default
        if persist_directory is None:
            persist_directory = "D:\\ml\\projects\\Rag_app\\chroma_db"
            
        # Ensure the directory exists
        os.makedirs(persist_directory, exist_ok=True)
        
        # Create and persist the vector store
        vector_store = Chroma.from_documents(
            documents=documents,
            embedding=embeddings,
            persist_directory=persist_directory
        )
        
        # Add documents to the vector store
        vector_store.add_documents(documents)
        
        logger.info(f"Successfully created and persisted Chroma vector store to {persist_directory}")
        return vector_store
    except Exception as e:
        logger.error(f"Error creating Chroma vector store: {e}")
        raise

def load_vector_store(embeddings: Embeddings, persist_directory: str) -> Optional[Chroma]:
    """
    Load a Chroma vector store from disk.
    
    Args:
        embeddings: Embeddings model to use.
        persist_directory: Directory where the vector store is persisted.
        
    Returns:
        A Chroma vector store instance or None if loading fails.
    """
    try:
        logger.info(f"Loading Chroma vector store from {persist_directory}")
        
        if not os.path.exists(persist_directory):
            logger.warning(f"Persist directory {persist_directory} does not exist")
            return None
            
        vector_store = Chroma(
            embedding_function=embeddings,
            persist_directory=persist_directory
        )
        
        logger.info("Successfully loaded Chroma vector store")
        return vector_store
    except Exception as e:
        logger.error(f"Error loading Chroma vector store: {e}")
        return None