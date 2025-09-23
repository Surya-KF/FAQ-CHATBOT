"""
FastAPI routes and endpoints for Hospital FAQ Chatbot

This module defines all REST API endpoints including:
- Chat endpoint for conversation with the chatbot
- Document search and retrieval endpoints
- Health check and system status endpoints
- Session management endpoints
"""

import logging
import time
import uuid
from typing import List, Optional
from fastapi import APIRouter, HTTPException, Depends, BackgroundTasks
from fastapi.responses import JSONResponse
from app.models import (
    ChatRequest, ChatResponse, SearchRequest, SearchResponse,
    HealthCheckResponse, SystemStatus, MemoryStats, ErrorResponse,
    IndexingRequest, IndexingResponse
)
from app.retrieval import get_rag_retriever, answer_faq_question
from app.memory import get_memory_manager, get_session_memory
from app.indexing import get_indexer, index_hospital_faqs
from app.db import get_chroma_client
from app.llm import test_gemini_connection
from app.config import settings, validate_configuration

# Configure logging
logger = logging.getLogger(__name__)

# Create API router
router = APIRouter()


@router.post("/chat", response_model=ChatResponse)
async def chat_with_bot(request: ChatRequest) -> ChatResponse:
    """
    Chat with the hospital FAQ bot.
    
    Args:
        request: Chat request containing question and session info
        
    Returns:
        ChatResponse: Bot response with sources and metadata
    """
    start_time = time.time()
    try:
        # Generate session ID if not provided
        session_id = request.session_id or str(uuid.uuid4())
        # Get conversation context from memory
        memory_manager = get_memory_manager()
        conversation_history = memory_manager.get_conversation_context(
            session_id=session_id,
            max_messages=5  # Use last 5 messages for context
        )
        # Get answer using RAG system
        response_text, source_documents = answer_faq_question(
            question=request.question,
            max_context_docs=request.context_limit,
            conversation_history=conversation_history
        )
        # DEBUG: Log the retrieved source documents and their content
        logger.info("[DEBUG] User question: %s", request.question)
        for i, doc in enumerate(source_documents, 1):
            logger.info("[DEBUG] Source %d: %s", i, doc.content)
            logger.info("[DEBUG] Metadata %d: %s", i, doc.metadata)
        # Calculate confidence score (simple heuristic)
        confidence_score = 0.8 if source_documents else 0.3
        if source_documents:
            avg_relevance = sum(doc.relevance_score or 0 for doc in source_documents) / len(source_documents)
            confidence_score = min(0.95, avg_relevance * 1.2)
        # Add conversation turn to memory
        memory_manager.add_conversation_turn(
            session_id=session_id,
            user_message=request.question,
            assistant_response=response_text
        )
        # Calculate response time
        response_time_ms = int((time.time() - start_time) * 1000)
        
        # Prepare response
        chat_response = ChatResponse(
            response=response_text,
            session_id=session_id,
            sources=source_documents if request.include_sources else None,
            confidence_score=confidence_score,
            response_time_ms=response_time_ms
        )
        
        logger.info(f"Chat response generated for session {session_id} in {response_time_ms}ms")
        return chat_response
        
    except Exception as e:
        logger.error(f"Chat endpoint error: {e}")
        raise HTTPException(
            status_code=500,
            detail=f"Failed to generate response: {str(e)}"
        )


@router.post("/search", response_model=SearchResponse)
async def search_documents(request: SearchRequest) -> SearchResponse:
    """
    Search hospital FAQ documents.
    
    Args:
        request: Search request parameters
        
    Returns:
        SearchResponse: Search results with metadata
    """
    try:
        rag_retriever = get_rag_retriever()
        response = rag_retriever.search_documents(request)
        
        logger.info(f"Document search completed: {len(response.results)} results")
        return response
        
    except Exception as e:
        logger.error(f"Search endpoint error: {e}")
        raise HTTPException(
            status_code=500,
            detail=f"Search failed: {str(e)}"
        )


@router.get("/health", response_model=HealthCheckResponse)
async def health_check() -> HealthCheckResponse:
    """
    Health check endpoint.
    
    Returns:
        HealthCheckResponse: System health status
    """
    try:
        # Check various system components
        components = {
            "configuration": validate_configuration(),
            "gemini_api": test_gemini_connection(),
            "database": True,  # Will be set based on actual test
            "memory": True
        }
        
        # Test database connection
        try:
            chroma_client = get_chroma_client()
            chroma_client.get_collection_info()
            components["database"] = True
        except:
            components["database"] = False
        
        # Determine overall status
        overall_status = "healthy" if all(components.values()) else "degraded"
        
        return HealthCheckResponse(
            status=overall_status,
            components=components,
            version="1.0.0"
        )
        
    except Exception as e:
        logger.error(f"Health check error: {e}")
        return HealthCheckResponse(
            status="error",
            components={"error": False},
            version="1.0.0"
        )


@router.get("/status", response_model=SystemStatus)
async def get_system_status() -> SystemStatus:
    """
    Get detailed system status.
    
    Returns:
        SystemStatus: Comprehensive system information
    """
    try:
        # Check component availability
        database_connected = True
        llm_available = test_gemini_connection()
        embedding_service_available = True
        
        # Get database info
        total_documents = 0
        try:
            chroma_client = get_chroma_client()
            collection_info = chroma_client.get_collection_info()
            total_documents = collection_info.get("document_count", 0)
        except:
            database_connected = False
        
        # Get active sessions count
        memory_manager = get_memory_manager()
        active_sessions = memory_manager.get_active_session_count()
        
        return SystemStatus(
            status="operational" if all([database_connected, llm_available, embedding_service_available]) else "degraded",
            database_connected=database_connected,
            llm_available=llm_available,
            embedding_service_available=embedding_service_available,
            total_documents=total_documents,
            active_sessions=active_sessions,
            uptime_seconds=time.time()  # Simplified uptime
        )
        
    except Exception as e:
        logger.error(f"Status endpoint error: {e}")
        raise HTTPException(
            status_code=500,
            detail=f"Failed to get system status: {str(e)}"
        )


@router.get("/sessions", response_model=List[MemoryStats])
async def get_session_stats() -> List[MemoryStats]:
    """
    Get statistics for all active sessions.
    
    Returns:
        List[MemoryStats]: Memory statistics for all sessions
    """
    try:
        memory_manager = get_memory_manager()
        stats = memory_manager.get_all_session_stats()
        
        logger.info(f"Retrieved stats for {len(stats)} sessions")
        return stats
        
    except Exception as e:
        logger.error(f"Session stats endpoint error: {e}")
        raise HTTPException(
            status_code=500,
            detail=f"Failed to get session stats: {str(e)}"
        )


@router.delete("/sessions/{session_id}")
async def clear_session(session_id: str) -> JSONResponse:
    """
    Clear a specific conversation session.
    
    Args:
        session_id: Session ID to clear
        
    Returns:
        JSONResponse: Success or error message
    """
    try:
        memory_manager = get_memory_manager()
        success = memory_manager.clear_session(session_id)
        
        if success:
            return JSONResponse(
                content={"message": f"Session {session_id} cleared successfully"},
                status_code=200
            )
        else:
            return JSONResponse(
                content={"message": f"Session {session_id} not found"},
                status_code=404
            )
            
    except Exception as e:
        logger.error(f"Clear session endpoint error: {e}")
        raise HTTPException(
            status_code=500,
            detail=f"Failed to clear session: {str(e)}"
        )


@router.post("/index", response_model=IndexingResponse)
async def index_documents(
    request: IndexingRequest,
    background_tasks: BackgroundTasks
) -> IndexingResponse:
    """
    Index hospital FAQ documents.
    
    Args:
        request: Indexing request parameters
        background_tasks: Background task manager
        
    Returns:
        IndexingResponse: Indexing results
    """
    try:
        indexer = get_indexer()
        
        # Process files
        response = indexer.process_files(
            file_paths=request.file_paths,
            chunk_size=request.chunk_size,
            chunk_overlap=request.chunk_overlap,
            force_reindex=request.force_reindex
        )
        
        logger.info(f"Document indexing completed: {response.total_documents} documents processed")
        return response
        
    except Exception as e:
        logger.error(f"Index endpoint error: {e}")
        raise HTTPException(
            status_code=500,
            detail=f"Indexing failed: {str(e)}"
        )


@router.post("/index/directory")
async def index_directory(
    directory_path: str,
    background_tasks: BackgroundTasks
) -> IndexingResponse:
    """
    Index all documents in a directory.
    
    Args:
        directory_path: Path to directory to index
        background_tasks: Background task manager
        
    Returns:
        IndexingResponse: Indexing results
    """
    try:
        indexer = get_indexer()
        response = indexer.process_directory(directory_path)
        
        logger.info(f"Directory indexing completed: {directory_path}")
        return response
        
    except Exception as e:
        logger.error(f"Directory index endpoint error: {e}")
        raise HTTPException(
            status_code=500,
            detail=f"Directory indexing failed: {str(e)}"
        )


@router.get("/similar-questions")
async def get_similar_questions(
    question: str,
    limit: int = 5
) -> JSONResponse:
    """
    Get similar questions from the knowledge base.
    
    Args:
        question: Input question
        limit: Maximum number of similar questions
        
    Returns:
        JSONResponse: List of similar questions
    """
    try:
        rag_retriever = get_rag_retriever()
        similar_questions = rag_retriever.get_similar_questions(question, limit)
        
        return JSONResponse(
            content={
                "question": question,
                "similar_questions": similar_questions,
                "count": len(similar_questions)
            }
        )
        
    except Exception as e:
        logger.error(f"Similar questions endpoint error: {e}")
        raise HTTPException(
            status_code=500,
            detail=f"Failed to get similar questions: {str(e)}"
        )


@router.get("/")
async def root() -> JSONResponse:
    """
    Root endpoint with API information.
    
    Returns:
        JSONResponse: API information
    """
    return JSONResponse(
        content={
            "name": "Hospital FAQ Chatbot API",
            "version": "1.0.0",
            "description": "RESTful API for Hospital FAQ Chatbot using LangChain and Gemini 2.5 Flash",
            "endpoints": {
                "chat": "/chat - Chat with the FAQ bot",
                "search": "/search - Search FAQ documents",
                "health": "/health - Health check",
                "status": "/status - System status",
                "sessions": "/sessions - Session management",
                "index": "/index - Document indexing"
            }
        }
    )


# Add router to the app
def get_api_router() -> APIRouter:
    """
    Get the configured API router.
    
    Returns:
        APIRouter: Configured router with all endpoints
    """
    return router