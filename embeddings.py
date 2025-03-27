"""Module for generating embeddings using Ollama."""

import logging
from typing import Optional

from langchain_community.embeddings import OllamaEmbeddings

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def get_ollama_embeddings(model_name: str = "nomic-embed-text", base_url: Optional[str] = None):
    """
    Create an Ollama embeddings model instance.
    
    Args:
        model_name: Name of the Ollama embedding model to use.
        base_url: Base URL for the Ollama API. Defaults to http://localhost:11434.
        
    Returns:
        An OllamaEmbeddings instance.
    """
    try:
        logger.info(f"Creating Ollama embeddings with model: {model_name}")
        
        # Set default base_url if not provided
        if base_url is None:
            base_url = "http://localhost:11434"
            
        embeddings = OllamaEmbeddings(
            model=model_name,
            base_url=base_url
        )
        
        logger.info("Successfully created Ollama embeddings instance")
        return embeddings
    except Exception as e:
        logger.error(f"Error creating Ollama embeddings: {e}")
        raise