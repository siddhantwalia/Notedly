import nest_asyncio
import os
import requests
import logging
import tempfile
from urllib.parse import urlparse
from pathlib import Path
# from llama_parse import LlamaParse
from dotenv import load_dotenv
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_core.documents import Document as LCDocument
from langchain_community.document_loaders import (
    PyMuPDFLoader,
    Docx2txtLoader,
    UnstructuredEmailLoader
)
from langchain_community.document_loaders.llmsherpa import LLMSherpaFileLoader
nest_asyncio.apply()

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

async def parse_document_from_url(url: str):
    try:
        response = requests.get(url)
        response.raise_for_status()
        
        # Determine file type from Content-Type header
        content_type = response.headers.get('Content-Type', '').lower()
        if 'pdf' in content_type:
            ext = '.pdf'
        elif 'wordprocessingml.document' in content_type:  # For DOCX
            ext = '.docx'
        elif 'eml' in content_type or 'message/rfc822' in content_type:
            ext = '.eml'
        else:
            # Fallback to URL extension if Content-Type is unclear
            parsed_url = urlparse(url)
            ext = Path(parsed_url.path).suffix.lower()
            if ext not in ['.pdf', '.docx', '.eml']:
                raise ValueError(f"Unsupported file type: {ext or content_type}. Only PDF, DOCX, and EML are supported.")
        
        logger.info(f"Detected file type: {ext} for URL: {url}")
        
        with tempfile.NamedTemporaryFile(delete=False, suffix=ext) as tmp_file:
            tmp_file.write(response.content)
            tmp_path = tmp_file.name
        
        try:
            if ext == ".pdf":
                # loader = LLMSherpaFileLoader(
                #     file_path=tmp_path,
                #     new_indent_parser=True,
                #     apply_ocr=True,
                #     strategy="chunks",
                #     llmsherpa_api_url="http://localhost:5010/api/parseDocument?renderFormat=all"
                # )
                loader = PyMuPDFLoader(tmp_path)
            elif ext == ".docx":
                # loader = LLMSherpaFileLoader(
                #     file_path=tmp_path,
                #     new_indent_parser=True,
                #     apply_ocr=True,
                #     strategy="chunks",
                #     llmsherpa_api_url="http://localhost:5010/api/parseDocument?renderFormat=all"
                # )
                loader = Docx2txtLoader(tmp_path)
            elif ext == ".eml":
                loader = UnstructuredEmailLoader(tmp_path)
            else:
                raise ValueError(f"No loader configured for: {ext}")
            
            documents = loader.load()
            return documents
        finally:
            # Clean up temporary file
            if os.path.exists(tmp_path):
                os.unlink(tmp_path)
        
    except requests.RequestException as e:
        raise ValueError(f"Failed to download document: {str(e)}")
    except Exception as e:
        raise ValueError(f"Error parsing document: {str(e)}")

def split_documents(parsed_docs, chunk_size=1000, chunk_overlap=200):
    all_chunks = []

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        separators=["\n\n", "\n", ".", " ", ""]
    )

    try:
        chunks = splitter.split_documents(parsed_docs)
        all_chunks.extend(chunks)
    except Exception as e:
        print(f"Error processing document chunk: {e}")

    return all_chunks
