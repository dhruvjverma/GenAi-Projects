"""Module for LangChain integration and RAG chain implementation."""

import logging
from typing import Any, Dict, Optional

from langchain.schema import Document
from langchain_chroma import Chroma
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_ollama import OllamaLLM

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def get_ollama_llm(model_name: str = "deepseek-r1:1.5b", base_url: Optional[str] = None) -> OllamaLLM:
    """
    Create an Ollama LLM instance.
    
    Args:
        model_name: Name of the Ollama model to use.
        base_url: Base URL for the Ollama API.
        
    Returns:
        An OllamaLLM instance.
    """
    try:
        logger.info(f"Creating Ollama LLM with model: {model_name}")
        
        # Set default base_url if not provided
        if base_url is None:
            base_url = "http://localhost:11434"
        
        llm = OllamaLLM(
            model=model_name,
            base_url=base_url,
            temperature=0.1
        )
        
        logger.info("Successfully created Ollama LLM instance")
        return llm
    except Exception as e:
        logger.error(f"Error creating Ollama LLM: {e}")
        raise

def create_retrieval_chain(vector_store: Chroma, llm: OllamaLLM, k: int = 3):
    """
    Create a RAG retrieval chain for question answering.
    
    Args:
        vector_store: Vector store for document retrieval.
        llm: Language model to use for generation.
        k: Number of documents to retrieve.
        
    Returns:
        A runnable chain for question answering.
    """
    try:
        logger.info(f"Creating retrieval chain with k={k}")
        
        # Create a retriever from the vector store
        retriever = vector_store.as_retriever(search_kwargs={"k": k})
        
        # Define the RAG prompt template
        template = """You are a helpful assistant that answers questions based on the provided context.
        
Context:
{context}

Question: {question}

Instructions:
1. Answer the question based solely on the provided context.
2. If the context doesn't contain the answer, say "I don't have enough information to answer this question."
3. Provide detailed and accurate answers.
4. Use bullet points for lists and structured information.

Your answer:"""
        
        prompt = ChatPromptTemplate.from_template(template)
        
        # Format the retrieved documents
        def format_docs(docs):
            return "\n\n".join(doc.page_content for doc in docs)
        
        # Create the RAG chain
        rag_chain = (
            {"context": retriever | format_docs, "question": RunnablePassthrough()}
            | prompt
            | llm
            | StrOutputParser()
        )
        
        logger.info("Successfully created retrieval chain")
        return rag_chain
    except Exception as e:
        logger.error(f"Error creating retrieval chain: {e}")
        raise