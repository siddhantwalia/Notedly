import os
import re
import asyncio
import time
import logging
from fastapi import FastAPI, Request, Header, HTTPException
from pydantic import BaseModel
from typing import List, Optional
from model import Prompt, llm, NomicEmbeddings, HuggingFaceEmbed
from utils import parse_document_from_url, split_documents
from langchain_community.vectorstores import FAISS
from langchain_core.runnables import RunnablePassthrough
from dotenv import load_dotenv
import uuid  # For generating session IDs


load_dotenv()
app = FastAPI()


# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


# Global in-memory cache for FAISS retrievers (key: document URL, value: retriever)
faiss_cache = {}  # Add this here for shared access across requests

# Global in-memory cache for chat histories (key: session_id, value: list of (question, answer) tuples)
chat_histories = {}  # For maintaining conversational context across requests


class DocumentRequest(BaseModel):
    documents: str  # URL of the document to process


class ChatRequest(BaseModel):
    document_url: str  # URL of the cached document to query against
    question: str
    session_id: Optional[str] = None  # Optional session ID for chat history continuity


@app.get('/')
async def home():
    return {"home": "The api"}


@app.get("/doc")
async def hehe():
    return {"input_format": "documents : url"}


@app.post("/doc")
async def process_document(req: DocumentRequest):
    start = time.time()
    
    cache_key = req.documents  
    if cache_key in faiss_cache:
        logger.info("Using cached FAISS retriever")
        return {"message": "Document already processed and cached"}
    
    p_time = time.time()
    try:
        parse_doc = await parse_document_from_url(req.documents)
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Error parsing document: {str(e)}")
    logger.info(f"Parsing time: {time.time() - p_time}")
    
    c_time = time.time()
    try:
        chunks = split_documents(parse_doc)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Splitting failed: {str(e)}")
    
    texts = [chunk.page_content for chunk in chunks]
    logger.info(f"Chunking time: {time.time() - c_time}")
    
    e_time = time.time()
    try:
        embedding_model = NomicEmbeddings()
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Embedding failed: {str(e)}")
    logger.info(f"Embedding generation time: {time.time() - e_time}")
    
    s_time = time.time()
    try: 
        db = FAISS.from_documents(chunks, embedding_model)
        retriever = db.as_retriever(search_type="mmr", search_kwargs={"k": 10, "lambda_mult": 0.5})
        

        faiss_cache[cache_key] = retriever
        logger.info("FAISS retriever cached")
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Vector DB error: {str(e)}")
    logger.info(f"Storing VDB time: {time.time() - s_time}")
    
    logger.info(f"Total time: {time.time() - start}")
    return {"message": "Document processed and cached successfully"}


@app.post("/get_response")
async def get_response(req: ChatRequest):
    start = time.time()
    
    cache_key = req.document_url
    if cache_key not in faiss_cache:
        raise HTTPException(status_code=404, detail="Document not found or not processed. Please process via /doc first.")
    
    retriever = faiss_cache[cache_key]
    
    # Handle session ID: generate a new one if not provided
    session_id = req.session_id or str(uuid.uuid4())
    
    # Get or initialize chat history for this session
    chat_history = chat_histories.get(session_id, [])
    
    async def get_answer(question, history): 
        try:
            logger.info(question)
            # Retrieve context based on the current question (optionally incorporate history if needed)
            context_docs = retriever.invoke(question)
            context = "\n".join([doc.page_content for doc in context_docs])
            
            # Format history for the prompt (assuming Prompt can handle a 'history' input)
            formatted_history = "\n".join([f"User: {q}\nAssistant: {a}" for q, a in history])
            
            inputs = {"context": context, "question": question, "history": formatted_history}
            answer = await (Prompt | llm).ainvoke(inputs) 
            return clean_output(answer)
        except Exception as e:
            logger.error(f"Error processing question '{question}': {str(e)}")
            return f"Error: {str(e)}"
    
    def clean_output(answer):
        if hasattr(answer, "content"): 
            content = answer.content
        else:
            content = str(answer)
        content = content.strip()
        content = re.sub(r"<think>.*?</think>", "", content, flags=re.DOTALL)
        content = re.sub(r"\n{3,}", "\n\n", content)
        return content
    
    llm_res = time.time()
    answer = await get_answer(req.question, chat_history)
    # Append to history
    chat_history.append((req.question, answer))
    chat_histories[session_id] = chat_history
    
    logger.info(f"LLM response time: {time.time() - llm_res}")
    
    logger.info(f"Total time: {time.time() - start}")
    return {"answer": answer, "session_id": session_id}
