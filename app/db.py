"""
ChromaDB integration module for Hospital FAQ Chatbot

This module handles vector database operations including:
- ChromaDB client setup and configuration
- Vector storage and retrieval
- Collection management
- Query operations for semantic search
"""

import logging
from typing import List, Dict, Any, Optional, Tuple
import os
import sys

# Handle ONNX Runtime import with fallback for Windows DLL issues
try:
    import onnxruntime
    print("Real ONNX Runtime loaded successfully")
except ImportError as e:
    print(f"ONNX Runtime import failed: {e}")
    print("Using dummy ONNX Runtime - install Visual C++ Redistributable for full functionality")
    
    # Create a dummy onnxruntime module
    class DummyONNXRuntime:
        """Dummy ONNX Runtime module for Windows compatibility."""
        
        class InferenceSession:
            def __init__(self, *args, **kwargs):
                pass
            
            def run(self, *args, **kwargs):
                return [[0.0] * 384]  # Return dummy embedding vector
        
        @staticmethod
        def get_available_providers():
            return ['CPUExecutionProvider']
    
    # Install the dummy module
    sys.modules['onnxruntime'] = DummyONNXRuntime()

import chromadb
from chromadb.config import Settings as ChromaSettings
from langchain_community.vectorstores import Chroma
from langchain.schema import Document
from app.config import settings

# Configure logging
logger = logging.getLogger(__name__)


class ChromaDBClient:
    """ChromaDB client for vector storage and retrieval operations."""
    
    def __init__(self):
        """Initialize ChromaDB client with configuration."""
        self.persist_directory = settings.chroma_persist_directory
        self.collection_name = settings.chroma_collection_name
        self.client = None
        self.collection = None
        self.vector_store = None
        self._initialize_client()
    
    def _initialize_client(self) -> None:
        """Initialize ChromaDB client and collection."""
        try:
            # Create ChromaDB client with persistence
            chroma_settings = ChromaSettings(
                persist_directory=self.persist_directory,
                anonymized_telemetry=False
            )
            
            self.client = chromadb.PersistentClient(
                path=self.persist_directory,
                settings=chroma_settings
            )
            
            # Get or create collection with custom embedding function
            # We'll use None here and let LangChain handle the embeddings
            try:
                self.collection = self.client.get_collection(name=self.collection_name)
                logger.info(f"Retrieved existing collection: {self.collection_name}")
            except Exception:
                # Collection doesn't exist, create it without default embedding function
                self.collection = self.client.create_collection(
                    name=self.collection_name,
                    metadata={"description": "Hospital FAQ embeddings"},
                    embedding_function=None  # Let LangChain handle embeddings
                )
                logger.info(f"Created new collection: {self.collection_name}")
            
            logger.info(f"ChromaDB client initialized with collection: {self.collection_name}")
            
        except Exception as e:
            logger.error(f"Failed to initialize ChromaDB client: {e}")
            raise
    
    def create_langchain_vector_store(self, embedding_function) -> Chroma:
        """
        Create LangChain Chroma vector store instance.
        
        Args:
            embedding_function: Embedding function for generating vectors
            
        Returns:
            Chroma: LangChain Chroma vector store instance
        """
        try:
            self.vector_store = Chroma(
                client=self.client,
                collection_name=self.collection_name,
                embedding_function=embedding_function,
                persist_directory=self.persist_directory
            )
            
            logger.info("LangChain Chroma vector store created successfully")
            return self.vector_store
            
        except Exception as e:
            logger.error(f"Failed to create LangChain vector store: {e}")
            raise
    
    def add_documents(
        self, 
        documents: List[str], 
        metadatas: Optional[List[Dict[str, Any]]] = None,
        ids: Optional[List[str]] = None
    ) -> None:
        """
        Add documents to the ChromaDB collection.
        
        Args:
            documents: List of document texts to add
            metadatas: Optional list of metadata dictionaries
            ids: Optional list of document IDs
        """
        try:
            if not ids:
                ids = [f"doc_{i}" for i in range(len(documents))]
            
            if not metadatas:
                metadatas = [{"source": "hospital_faq"} for _ in documents]
            
            self.collection.add(
                documents=documents,
                metadatas=metadatas,
                ids=ids
            )
            
            logger.info(f"Added {len(documents)} documents to collection")
            
        except Exception as e:
            logger.error(f"Failed to add documents: {e}")
            raise
    
    def query_similar_documents(
        self, 
        query_text: str, 
        n_results: int = 5,
        where_filter: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """
        Query similar documents from the collection.
        
        Args:
            query_text: Query text to search for
            n_results: Number of results to return
            where_filter: Optional metadata filter
            
        Returns:
            Dict containing query results
        """
        try:
            results = self.collection.query(
                query_texts=[query_text],
                n_results=n_results,
                where=where_filter
            )
            
            logger.info(f"Query returned {len(results['documents'][0])} results")
            return results
            
        except Exception as e:
            logger.error(f"Failed to query documents: {e}")
            raise
    
    def get_collection_info(self) -> Dict[str, Any]:
        """
        Get information about the current collection.
        
        Returns:
            Dict containing collection information
        """
        try:
            count = self.collection.count()
            peek_results = self.collection.peek()
            
            return {
                "collection_name": self.collection_name,
                "document_count": count,
                "sample_documents": peek_results.get("documents", [])[:3]
            }
            
        except Exception as e:
            logger.error(f"Failed to get collection info: {e}")
            return {"error": str(e)}
    
    def delete_collection(self) -> bool:
        """
        Delete the current collection.
        
        Returns:
            bool: True if successful, False otherwise
        """
        try:
            self.client.delete_collection(name=self.collection_name)
            logger.info(f"Deleted collection: {self.collection_name}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to delete collection: {e}")
            return False
    
    def reset_collection(self) -> None:
        """Reset the collection by deleting and recreating it."""
        try:
            self.delete_collection()
            self.collection = self.client.get_or_create_collection(
                name=self.collection_name,
                metadata={"description": "Hospital FAQ embeddings"}
            )
            logger.info(f"Reset collection: {self.collection_name}")
            
        except Exception as e:
            logger.error(f"Failed to reset collection: {e}")
            raise
    
    def search_by_metadata(
        self, 
        metadata_filter: Dict[str, Any], 
        n_results: int = 10
    ) -> List[Dict[str, Any]]:
        """
        Search documents by metadata criteria.
        
        Args:
            metadata_filter: Metadata filter criteria
            n_results: Maximum number of results to return
            
        Returns:
            List of matching documents with metadata
        """
        try:
            results = self.collection.get(
                where=metadata_filter,
                limit=n_results
            )
            
            documents = []
            for i, doc in enumerate(results.get("documents", [])):
                documents.append({
                    "id": results["ids"][i],
                    "document": doc,
                    "metadata": results["metadatas"][i]
                })
            
            logger.info(f"Found {len(documents)} documents matching metadata filter")
            return documents
            
        except Exception as e:
            logger.error(f"Failed to search by metadata: {e}")
            return []


def get_chroma_client() -> ChromaDBClient:
    """
    Get a ChromaDB client instance.
    
    Returns:
        ChromaDBClient: Initialized ChromaDB client
    """
    return ChromaDBClient()


def create_retriever(embedding_function, search_kwargs: Optional[Dict[str, Any]] = None):
    """
    Create a retriever for document search.
    
    Args:
        embedding_function: Function to generate embeddings
        search_kwargs: Additional search parameters
        
    Returns:
        VectorStoreRetriever: Configured retriever instance
    """
    try:
        client = get_chroma_client()
        vector_store = client.create_langchain_vector_store(embedding_function)
        
        if search_kwargs is None:
            search_kwargs = {"k": 5}
        
        retriever = vector_store.as_retriever(search_kwargs=search_kwargs)
        logger.info("Created retriever with ChromaDB vector store")
        return retriever
        
    except Exception as e:
        logger.error(f"Failed to create retriever: {e}")
        raise


if __name__ == "__main__":
    # Test ChromaDB client
    try:
        client = get_chroma_client()
        info = client.get_collection_info()
        print(f"✅ ChromaDB client test successful")
        print(f"Collection info: {info}")
        
    except Exception as e:
        print(f"❌ ChromaDB client test failed: {e}")