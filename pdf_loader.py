# Module For Loading & Processing PDF Doc

import logging
from pathlib import Path
from typing import Optional, Union

from langchain_community.document_loaders import PyPDFLoader

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def load_pdf(file_path: Union[str, Path]) -> list:
    """
    Load a PDF document and extract its content.
    
    Args:
        file_path: Path to the PDF file.
        
    Returns:
        A list of document objects containing the extracted text.
    """
    try:
        logger.info(f"Loading PDF from: {file_path}")
        loader = PyPDFLoader(str(file_path))
        documents = loader.load()
        logger.info(f"Successfully loaded PDF with {len(documents)} pages")
        return documents
    except Exception as e:
        logger.error(f"Error loading PDF: {e}")
        raise

def save_uploaded_file(uploaded_file) -> Optional[str]:
    """
    Save an uploaded file to disk temporarily.
    
    Args:
        uploaded_file: The uploaded file object from Streamlit.
        
    Returns:
        The path to the saved file or None if saving failed.
    """
    try:
        # Create a temporary file path
        temp_dir = Path("./temp")
        temp_dir.mkdir(exist_ok=True)
        
        file_path = temp_dir / uploaded_file.name
        
        # Write the file
        with open(file_path, "wb") as f:
            f.write(uploaded_file.getbuffer())
            
        logger.info(f"Saved uploaded file to {file_path}")
        return str(file_path)
    except Exception as e:
        logger.error(f"Error saving uploaded file: {e}")
        return None