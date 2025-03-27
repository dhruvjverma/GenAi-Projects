"""Module for splitting documents into chunks for processing."""

import logging
from typing import List

from langchain.schema import Document
from langchain.text_splitter import RecursiveCharacterTextSplitter

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def chunk_documents(documents: List[Document], chunk_size: int = 1000, chunk_overlap: int = 200) -> List[Document]:
    """
    Split documents into chunks for processing.
    
    Args:
        documents: List of documents to split.
        chunk_size: Maximum size of each chunk.
        chunk_overlap: Overlap between chunks.
        
    Returns:
        List of chunked documents.
    """
    try:
        logger.info(f"Chunking {len(documents)} documents with chunk size {chunk_size} and overlap {chunk_overlap}")
        
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            length_function=len,
            separators=["\n\n", "\n", " ", ""]
        )
        
        chunks = text_splitter.split_documents(documents)
        
        logger.info(f"Created {len(chunks)} chunks from the documents")
        return chunks
    except Exception as e:
        logger.error(f"Error chunking documents: {e}")
        raise