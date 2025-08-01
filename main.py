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
from langchain_core.prompts import PromptTemplate
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

# Global variable to store the most recent document URL (for single-document assumption)
document_url = ""


class DocumentRequest(BaseModel):
    documents: str  # URL of the document to process


class ChatRequest(BaseModel):
    question: str
    session_id: Optional[str] = None  # Optional session ID for chat history continuity

@app.get('/')
async def home():
    return {"home": "Home Page"}


@app.get("/doc")
async def Doc_input():
    return {"input_format": "documents : url"}


@app.post("/doc")
async def process_document(req: DocumentRequest):
    global document_url  # Declare as global to modify it
    start = time.time()
    document_url = req.documents  # Update the global with the latest document URL
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
    
    if not document_url:
        raise HTTPException(status_code=400, detail="No document has been processed yet. Please call /doc first.")
    
    cache_key = document_url
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



@app.post("/summary")
async def make_summary():  # No request body, as it doesn't need a question
    start = time.time()
    
    if not document_url:
        raise HTTPException(status_code=400, detail="No document has been processed yet. Please call /doc first.")
    
    cache_key = document_url
    if cache_key not in faiss_cache:
        raise HTTPException(status_code=404, detail="Document not found or not processed. Please process via /doc first.")
    
    retriever = faiss_cache[cache_key]
    
    # Generate a session ID for this summary request (optional, can be removed if not needed)
    session_id = str(uuid.uuid4())
    
    # Define the PromptTemplate locally (or import and use globally if preferred)
    summary_prompt = PromptTemplate(
        input_variables=["context"],
        template="""
        You are an expert tutor helping a student understand a topic based on a textbook. Read the following context carefully and give a detailed, well-structured summary. Include definitions, key points, explanations, and examples if relevant.

        Context:
        {context}

        Instructions:
        - Provide a clear and in-depth explanation.
        - Break down complex terms or processes if necessary.
        - Use bullet points or paragraph structure if it helps clarity.
        - Avoid vague answers. Focus strictly on the context.
        - Do not add unrelated information.

        Answer:
        """
    )
    
    async def get_summary(): 
        try:
            # Use a default query to retrieve relevant context for the entire document summary
            default_query = "Summarize the main content and key points of the document"
            logger.info(default_query)
            context_docs = retriever.invoke(default_query)
            context = "\n".join([doc.page_content for doc in context_docs])
            
            inputs = {"context": context}
            answer = await (summary_prompt | llm).ainvoke(inputs) 
            return clean_output(answer)
        except Exception as e:
            logger.error(f"Error generating summary: {str(e)}")
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
    summary = await get_summary()
    
    logger.info(f"LLM response time: {time.time() - llm_res}")
    
    logger.info(f"Total time: {time.time() - start}")
    return {"summary": summary, "session_id": session_id}


# app.post("/make_quiz")
async def make_quiz():
    start = time.time()
    
    if not document_url:
        raise HTTPException(status_code=400, detail="No document has been processed yet. Please call /doc first.")
    
    cache_key = document_url
    if cache_key not in faiss_cache:
        raise HTTPException(status_code=404, detail="Document not found or not processed. Please process via /doc first.")
    
    retriever = faiss_cache[cache_key]
    
    # Generate a session ID for this quiz request (optional, can be removed if not needed)
    session_id = str(uuid.uuid4())
    
    quiz_prompt = PromptTemplate(
        input_variables=["context"],
        template="""
        You are an expert tutor helping a student understand a topic based on a textbook. Read the following context carefully and create exactly 5 quiz questions around it. Include a mix of multiple-choice, true/false, and short-answer questions.

        Context:
        {context}

        Instructions:
        - Generate exactly 5 questions that test key concepts, definitions, and understanding.
        - For each question, provide the correct answer immediately after it, followed by a brief explanation.
        - Use a structured format: number the questions, bold the answers, and italicize explanations.
        - Focus strictly on the context without adding external information.
        - Ensure questions vary in difficulty and cover different parts of the context.

        Quiz:
        """
    )
    
    async def get_quiz(): 
        try:
            # Use a default query to retrieve relevant context for quiz generation
            default_query = "Key concepts and details from the document for creating quiz questions"
            logger.info(default_query)
            context_docs = retriever.invoke(default_query)
            context = "\n".join([doc.page_content for doc in context_docs])
            
            inputs = {"context": context}
            quiz_output = await (quiz_prompt | llm).ainvoke(inputs) 
            return clean_output(quiz_output)
        except Exception as e:
            logger.error(f"Error generating quiz: {str(e)}")
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
    
    def parse_quiz_and_answers(quiz_text):
        questions = []
        answers = []
        # Defensive check on empty input
        if not quiz_text:
            return questions, answers

        # Split by question numbers (assuming format like "1. Question text\n**Answer:** answer\n*Explanation:* exp\n")
        items = re.split(r'(\d+\.\s)', quiz_text)
        if len(items) < 3:
            # Not enough parts for normal parsing, fallback to all text as one question
            questions.append(quiz_text.strip())
            answers.append("")
            return questions, answers

        for i in range(1, len(items), 2):
            if i + 1 < len(items):
                question_part_raw = items[i] + items[i+1]
                # Split question part and answer
                split_answer = question_part_raw.split('**Answer:**')
                question_part = split_answer[0].strip()

                if len(split_answer) > 1:
                    answer_section = split_answer[1]
                    # Try to split explanation if present
                    if '*Explanation:*' in answer_section:
                        answer_part = answer_section.split('*Explanation:*')[0].strip()
                        explanation_part = answer_section.split('*Explanation:*')[1].strip()
                        full_answer = f"{answer_part} ({explanation_part})"
                    else:
                        full_answer = answer_section.strip()
                else:
                    full_answer = ""

                questions.append(question_part)
                answers.append(full_answer)
        return questions, answers
    
    llm_res = time.time()
    quiz_text = await get_quiz()
    quiz_questions, quiz_answers = parse_quiz_and_answers(quiz_text)
    
    logger.info(f"LLM response time: {time.time() - llm_res}")
    
    logger.info(f"Total time: {time.time() - start}")
    return {"quiz": quiz_questions, "answers": quiz_answers, "session_id": session_id}