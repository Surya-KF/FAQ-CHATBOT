"""
Pydantic models and schemas for Hospital FAQ Chatbot

This module defines data models for API requests, responses, and internal data structures
using Pydantic for validation and serialization.
"""

from datetime import datetime
from typing import List, Dict, Any, Optional, Union
from enum import Enum
from pydantic import BaseModel, Field, validator
import uuid


class MessageRole(str, Enum):
    """Enumeration for message roles in conversation."""
    USER = "user"
    ASSISTANT = "assistant"
    SYSTEM = "system"


class ConversationMessage(BaseModel):
    """Model for individual conversation messages."""
    role: MessageRole = Field(..., description="Role of the message sender")
    content: str = Field(..., min_length=1, description="Message content")
    timestamp: datetime = Field(default_factory=datetime.utcnow, description="Message timestamp")
    message_id: str = Field(default_factory=lambda: str(uuid.uuid4()), description="Unique message ID")
    
    class Config:
        json_encoders = {
            datetime: lambda v: v.isoformat()
        }


class ChatRequest(BaseModel):
    """Request model for chat endpoint."""
    question: str = Field(..., min_length=1, max_length=1000, description="User's question")
    session_id: Optional[str] = Field(None, description="Session ID for conversation continuity")
    context_limit: Optional[int] = Field(default=5, ge=1, le=20, description="Number of context documents to retrieve")
    include_sources: Optional[bool] = Field(default=True, description="Whether to include source information")
    
    @validator('question')
    def validate_question(cls, v):
        if not v.strip():
            raise ValueError('Question cannot be empty or whitespace only')
        return v.strip()


class SourceDocument(BaseModel):
    """Model for source document information."""
    content: str = Field(..., description="Document content")
    metadata: Dict[str, Any] = Field(default_factory=dict, description="Document metadata")
    relevance_score: Optional[float] = Field(None, ge=0.0, le=1.0, description="Relevance score")
    document_id: str = Field(..., description="Unique document identifier")


class ChatResponse(BaseModel):
    """Response model for chat endpoint."""
    response: str = Field(..., description="Chatbot response")
    session_id: str = Field(..., description="Session ID")
    sources: Optional[List[SourceDocument]] = Field(None, description="Source documents used")
    confidence_score: Optional[float] = Field(None, ge=0.0, le=1.0, description="Response confidence score")
    response_time_ms: Optional[int] = Field(None, description="Response time in milliseconds")
    timestamp: datetime = Field(default_factory=datetime.utcnow, description="Response timestamp")
    
    class Config:
        json_encoders = {
            datetime: lambda v: v.isoformat()
        }


class ConversationHistory(BaseModel):
    """Model for conversation history."""
    session_id: str = Field(..., description="Session identifier")
    messages: List[ConversationMessage] = Field(default_factory=list, description="List of messages")
    created_at: datetime = Field(default_factory=datetime.utcnow, description="Conversation creation time")
    updated_at: datetime = Field(default_factory=datetime.utcnow, description="Last update time")
    summary: Optional[str] = Field(None, description="Conversation summary")
    
    class Config:
        json_encoders = {
            datetime: lambda v: v.isoformat()
        }


class DocumentChunk(BaseModel):
    """Model for document chunks used in indexing."""
    content: str = Field(..., description="Chunk content")
    metadata: Dict[str, Any] = Field(default_factory=dict, description="Chunk metadata")
    chunk_id: str = Field(default_factory=lambda: str(uuid.uuid4()), description="Unique chunk ID")
    source_file: Optional[str] = Field(None, description="Source file path")
    chunk_index: Optional[int] = Field(None, description="Index of chunk in source document")
    embedding: Optional[List[float]] = Field(None, description="Embedding vector")


class IndexingRequest(BaseModel):
    """Request model for document indexing."""
    file_paths: List[str] = Field(..., description="List of file paths to index")
    chunk_size: Optional[int] = Field(default=512, ge=100, le=2000, description="Chunk size for splitting")
    chunk_overlap: Optional[int] = Field(default=60, ge=0, le=500, description="Overlap between chunks")
    force_reindex: Optional[bool] = Field(default=False, description="Force reindexing of existing documents")


class IndexingResponse(BaseModel):
    """Response model for document indexing."""
    success: bool = Field(..., description="Whether indexing was successful")
    total_documents: int = Field(..., description="Total number of documents processed")
    total_chunks: int = Field(..., description="Total number of chunks created")
    processing_time_ms: int = Field(..., description="Processing time in milliseconds")
    errors: List[str] = Field(default_factory=list, description="List of errors encountered")
    indexed_files: List[str] = Field(default_factory=list, description="Successfully indexed files")


class MemoryStats(BaseModel):
    """Model for conversation memory statistics."""
    session_id: str = Field(..., description="Session identifier")
    total_messages: int = Field(..., description="Total number of messages")
    memory_size_kb: float = Field(..., description="Memory size in kilobytes")
    last_activity: datetime = Field(..., description="Last activity timestamp")
    summary_length: Optional[int] = Field(None, description="Length of conversation summary")
    
    class Config:
        json_encoders = {
            datetime: lambda v: v.isoformat()
        }


class SystemStatus(BaseModel):
    """Model for system status information."""
    status: str = Field(..., description="System status")
    database_connected: bool = Field(..., description="Database connection status")
    llm_available: bool = Field(..., description="LLM availability status")
    embedding_service_available: bool = Field(..., description="Embedding service availability")
    total_documents: int = Field(..., description="Total documents in database")
    active_sessions: int = Field(..., description="Number of active sessions")
    uptime_seconds: float = Field(..., description="System uptime in seconds")
    timestamp: datetime = Field(default_factory=datetime.utcnow, description="Status check timestamp")
    
    class Config:
        json_encoders = {
            datetime: lambda v: v.isoformat()
        }


class SearchRequest(BaseModel):
    """Request model for document search."""
    query: str = Field(..., min_length=1, max_length=500, description="Search query")
    limit: Optional[int] = Field(default=10, ge=1, le=50, description="Maximum number of results")
    metadata_filter: Optional[Dict[str, Any]] = Field(None, description="Metadata filter criteria")
    similarity_threshold: Optional[float] = Field(default=0.7, ge=0.0, le=1.0, description="Minimum similarity threshold")


class SearchResponse(BaseModel):
    """Response model for document search."""
    results: List[SourceDocument] = Field(..., description="Search results")
    total_results: int = Field(..., description="Total number of results found")
    query_time_ms: int = Field(..., description="Query execution time in milliseconds")
    query: str = Field(..., description="Original search query")


class HealthCheckResponse(BaseModel):
    """Response model for health check endpoint."""
    status: str = Field(..., description="Health status")
    timestamp: datetime = Field(default_factory=datetime.utcnow, description="Health check timestamp")
    components: Dict[str, bool] = Field(..., description="Component health status")
    version: str = Field(..., description="Application version")
    
    class Config:
        json_encoders = {
            datetime: lambda v: v.isoformat()
        }


class ErrorResponse(BaseModel):
    """Response model for API errors."""
    error: str = Field(..., description="Error message")
    error_code: Optional[str] = Field(None, description="Error code")
    details: Optional[Dict[str, Any]] = Field(None, description="Additional error details")
    timestamp: datetime = Field(default_factory=datetime.utcnow, description="Error timestamp")
    request_id: Optional[str] = Field(None, description="Request identifier for tracking")
    
    class Config:
        json_encoders = {
            datetime: lambda v: v.isoformat()
        }


# Configuration models
class ProcessingConfig(BaseModel):
    """Configuration model for document processing."""
    chunk_size: int = Field(default=512, ge=100, le=2000)
    chunk_overlap: int = Field(default=60, ge=0, le=500)
    max_context_length: int = Field(default=4000, ge=1000, le=8000)
    similarity_threshold: float = Field(default=0.7, ge=0.0, le=1.0)


class LLMConfig(BaseModel):
    """Configuration model for LLM settings."""
    model_name: str = Field(default="gemini-2.0-flash")
    temperature: float = Field(default=0.1, ge=0.0, le=2.0)
    max_tokens: int = Field(default=2048, ge=100, le=4096)
    top_p: float = Field(default=0.8, ge=0.0, le=1.0)
    top_k: int = Field(default=40, ge=1, le=100)


class DatabaseConfig(BaseModel):
    """Configuration model for database settings."""
    collection_name: str = Field(default="hospital_faq")
    persist_directory: str = Field(default="./chroma_db")
    max_connections: int = Field(default=10, ge=1, le=100)


if __name__ == "__main__":
    # Test model validation
    print("Testing Pydantic models...")
    
    # Test ChatRequest
    try:
        request = ChatRequest(question="What are the visiting hours?", session_id="test-123")
        print(f"✅ ChatRequest validation successful: {request.json()}")
    except Exception as e:
        print(f"❌ ChatRequest validation failed: {e}")
    
    # Test ConversationMessage
    try:
        message = ConversationMessage(role=MessageRole.USER, content="Hello")
        print(f"✅ ConversationMessage validation successful")
    except Exception as e:
        print(f"❌ ConversationMessage validation failed: {e}")
    
    print("Model validation tests completed")