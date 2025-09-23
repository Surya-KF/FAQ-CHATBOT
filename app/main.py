"""
FastAPI main application for Hospital FAQ Chatbot

This module sets up the FastAPI application with:
- API configuration and middleware
- CORS handling
- Error handling
- Logging configuration
- Application startup and shutdown events
"""

import logging
import os
import sys
import time
from contextlib import asynccontextmanager
from fastapi import FastAPI, Request, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from fastapi.exceptions import RequestValidationError
import uvicorn
from app.config import settings, validate_configuration
from app.api import get_api_router
from app.models import ErrorResponse

# Configure logging
logging.basicConfig(
    level=getattr(logging, settings.log_level.upper()),
    format=settings.log_format,
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler("hospital_faq_chatbot.log")
    ]
)

logger = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    Application lifespan context manager for startup and shutdown events.
    
    Args:
        app: FastAPI application instance
    """
    # Startup
    logger.info("Starting Hospital FAQ Chatbot API...")
    # Validate configuration
    if not validate_configuration():
        logger.error("Configuration validation failed")
        sys.exit(1)
    # Test connections, run preprocessing, then indexing
    try:
        from app.llm import test_gemini_connection
        from app.db import get_chroma_client
        from app.indexing import index_hospital_faqs
        import subprocess
        # Test Gemini connection
        if not test_gemini_connection():
            logger.warning("Gemini API connection test failed - some features may not work")
        # Test ChromaDB connection
        try:
            chroma_client = get_chroma_client()
            chroma_client.get_collection_info()
            logger.info("ChromaDB connection successful")
        except Exception as e:
            logger.warning(f"ChromaDB connection test failed: {e}")
        # Run preprocessing at startup
        logger.info("Running document preprocessing at server startup...")
        result = subprocess.run([sys.executable, "scripts/preprocess.py"], capture_output=True, text=True)
        if result.returncode == 0:
            logger.info("Preprocessing completed successfully:")
            logger.info(result.stdout)
        else:
            logger.error("Preprocessing failed:")
            logger.error(result.stderr)
        # Run indexing at startup
        logger.info("Running document indexing at server startup...")
        indexing_result = index_hospital_faqs()
        if indexing_result.success:
            logger.info(f"Indexing successful: {indexing_result.total_documents} documents, {indexing_result.total_chunks} chunks.")
        else:
            logger.error(f"Indexing failed at startup: {indexing_result.errors}")
        logger.info("Application startup completed successfully")
    except Exception as e:
        logger.error(f"Application startup failed: {e}")
        sys.exit(1)
    
    yield
    
    # Shutdown
    logger.info("Shutting down Hospital FAQ Chatbot API...")


# Create FastAPI application
app = FastAPI(
    title=settings.api_title,
    description=settings.api_description,
    version=settings.api_version,
    lifespan=lifespan,
    docs_url="/docs",
    redoc_url="/redoc"
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.allowed_origins,
    allow_credentials=True,
    allow_methods=["GET", "POST", "PUT", "DELETE", "OPTIONS"],
    allow_headers=["*"],
)


@app.middleware("http")
async def log_requests(request: Request, call_next):
    """
    Middleware to log all HTTP requests.
    
    Args:
        request: HTTP request
        call_next: Next middleware/endpoint
        
    Returns:
        HTTP response
    """
    start_time = time.time()
    
    # Log request
    logger.info(f"{request.method} {request.url} - {request.client.host if request.client else 'unknown'}")
    
    # Process request
    response = await call_next(request)
    
    # Log response
    process_time = time.time() - start_time
    logger.info(f"{request.method} {request.url} - {response.status_code} - {process_time:.4f}s")
    
    return response


@app.exception_handler(HTTPException)
async def http_exception_handler(request: Request, exc: HTTPException):
    """
    Handle HTTP exceptions.
    
    Args:
        request: HTTP request
        exc: HTTP exception
        
    Returns:
        JSON error response
    """
    logger.error(f"HTTP Exception: {exc.status_code} - {exc.detail}")
    
    error_response = ErrorResponse(
        error=exc.detail,
        error_code=str(exc.status_code)
    )
    
    return JSONResponse(
        status_code=exc.status_code,
        content=error_response.dict()
    )


@app.exception_handler(RequestValidationError)
async def validation_exception_handler(request: Request, exc: RequestValidationError):
    """
    Handle request validation errors.
    
    Args:
        request: HTTP request
        exc: Validation exception
        
    Returns:
        JSON error response
    """
    logger.error(f"Validation Error: {exc.errors()}")
    
    error_response = ErrorResponse(
        error="Request validation failed",
        error_code="VALIDATION_ERROR",
        details={"errors": exc.errors()}
    )
    
    return JSONResponse(
        status_code=422,
        content=error_response.dict()
    )


@app.exception_handler(Exception)
async def general_exception_handler(request: Request, exc: Exception):
    """
    Handle general exceptions.
    
    Args:
        request: HTTP request
        exc: Exception
        
    Returns:
        JSON error response
    """
    logger.error(f"Unhandled Exception: {type(exc).__name__} - {str(exc)}")
    
    error_response = ErrorResponse(
        error="Internal server error",
        error_code="INTERNAL_ERROR",
        details={"exception_type": type(exc).__name__}
    )
    
    return JSONResponse(
        status_code=500,
        content=error_response.dict()
    )


# Include API router
app.include_router(get_api_router(), prefix="/api/v1")


@app.get("/")
async def root():
    """
    Root endpoint with basic API information.
    
    Returns:
        JSON response with API info
    """
    return {
        "name": "Hospital FAQ Chatbot API",
        "version": "1.0.0",
        "status": "operational",
        "docs": "/docs",
        "api_prefix": "/api/v1"
    }


def main():
    """Main entry point for running the application."""
    try:
        logger.info(f"Starting server on {settings.app_host}:{settings.app_port}")
        
        uvicorn.run(
            "app.main:app",
            host=settings.app_host,
            port=settings.app_port,
            reload=settings.debug,
            log_level=settings.log_level.lower(),
            access_log=True
        )
        
    except KeyboardInterrupt:
        logger.info("Server shutdown requested by user")
    except Exception as e:
        logger.error(f"Server startup failed: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()