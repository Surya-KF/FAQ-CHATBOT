"""
Configuration module for Hospital FAQ Chatbot

This module handles all configuration settings including environment variables,
API keys, and application settings using Pydantic Settings.
"""

import os
from typing import Optional
from pydantic import Field
from pydantic_settings import BaseSettings
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()


class Settings(BaseSettings):
    """Application settings and configuration."""
    
    # Embedding Backend Selection
    embedding_backend: str = Field(default="gemini", env="EMBEDDING_BACKEND")  # "gemini" or "sentence-transformers"
    
    # Gemini AI Configuration
    gemini_api_key: Optional[str] = Field(default=None, env="GEMINI_API_KEY")
    gemini_model_name: str = Field(default="gemini-2.5-flash-preview-04-17", env="GEMINI_MODEL_NAME")
    
    # ChromaDB Configuration
    chroma_persist_directory: str = Field(default="./chroma_db", env="CHROMA_PERSIST_DIRECTORY")
    chroma_collection_name: str = Field(default="hospital_faq", env="CHROMA_COLLECTION_NAME")
    
    # Application Configuration
    app_host: str = Field(default="0.0.0.0", env="APP_HOST")
    app_port: int = Field(default=8000, env="APP_PORT")
    debug: bool = Field(default=False, env="DEBUG")
    
    # Streamlit Configuration
    streamlit_server_port: int = Field(default=8501, env="STREAMLIT_SERVER_PORT")
    streamlit_server_address: str = Field(default="0.0.0.0", env="STREAMLIT_SERVER_ADDRESS")
    
    # Document Processing Configuration
    chunk_size: int = Field(default=512, env="CHUNK_SIZE")
    chunk_overlap: int = Field(default=60, env="CHUNK_OVERLAP")
    max_context_length: int = Field(default=4000, env="MAX_CONTEXT_LENGTH")
    
    # Memory Configuration
    max_conversation_history: int = Field(default=10, env="MAX_CONVERSATION_HISTORY")
    memory_type: str = Field(default="buffer", env="MEMORY_TYPE")  # buffer or summary
    
    # Logging Configuration
    log_level: str = Field(default="INFO", env="LOG_LEVEL")
    log_format: str = Field(
        default="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        env="LOG_FORMAT"
    )
    
    # Data Paths
    data_raw_path: str = Field(default="./data/raw", env="DATA_RAW_PATH")
    data_processed_path: str = Field(default="./data/processed", env="DATA_PROCESSED_PATH")
    
    # API Configuration
    api_title: str = Field(default="Hospital FAQ Chatbot API", env="API_TITLE")
    api_description: str = Field(
        default="RESTful API for Hospital FAQ Chatbot using LangChain and Gemini 2.5 Flash",
        env="API_DESCRIPTION"
    )
    api_version: str = Field(default="1.0.0", env="API_VERSION")
    
    # CORS Configuration
    allowed_origins: list[str] = Field(
        default=["http://localhost:8501", "http://localhost:3000"],
        env="ALLOWED_ORIGINS"
    )
    
    model_config = {
        "protected_namespaces": ()
    }


def get_settings() -> Settings:
    """Get application settings instance."""
    return Settings()


def validate_configuration() -> bool:
    """
    Validate that all required configuration is present and valid.
    
    Returns:
        bool: True if configuration is valid, False otherwise
    """
    try:
        settings = get_settings()
        
        # Check required API key
        if not settings.gemini_api_key or settings.gemini_api_key == "your_gemini_api_key_here":
            print("ERROR: GEMINI_API_KEY is not set or is using placeholder value")
            return False
        
        # Check data directories exist
        os.makedirs(settings.data_raw_path, exist_ok=True)
        os.makedirs(settings.data_processed_path, exist_ok=True)
        os.makedirs(settings.chroma_persist_directory, exist_ok=True)
        
        return True
        
    except Exception as e:
        print(f"Configuration validation failed: {e}")
        return False


# Global settings instance
settings = get_settings()


if __name__ == "__main__":
    # Test configuration
    if validate_configuration():
        print("✅ Configuration is valid")
        print(f"Gemini Model: {settings.gemini_model_name}")
        print(f"ChromaDB Collection: {settings.chroma_collection_name}")
        print(f"API Host: {settings.app_host}:{settings.app_port}")
    else:
        print("❌ Configuration validation failed")
