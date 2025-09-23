"""
Retrieval and RAG (Retrieval-Augmented Generation) system for Hospital FAQ Chatbot

This module handles:
- Semantic search and document retrieval
- Context preparation and ranking
- RAG pipeline integration
- Response generation with retrieved context
"""

import logging
import time
from typing import List, Dict, Any, Optional, Tuple
from langchain.schema import Document
from langchain.vectorstores.base import VectorStoreRetriever
from langchain.chains import RetrievalQA
from langchain.prompts import PromptTemplate
from app.config import settings
from app.db import get_chroma_client, create_retriever
from app.llm import get_llm_client, get_embedding_client
from app.models import SourceDocument, SearchRequest, SearchResponse

# Configure logging
logger = logging.getLogger(__name__)


class RAGRetriever:
    """Retrieval-Augmented Generation system for hospital FAQ queries."""
    
    def __init__(self):
        """Initialize the RAG retriever with required components."""
        self.chroma_client = get_chroma_client()
        self.llm_client = get_llm_client()
        self.embedding_client = get_embedding_client()
        
        # Create retriever with custom search parameters
        self.retriever = create_retriever(
            embedding_function=self.embedding_client.get_embedding_function(),
            search_kwargs={
                "k": 10,  # Retrieve more documents initially
                "fetch_k": 20,  # Fetch more candidates for reranking
                "include_metadata": True
            }
        )
        
        self.vector_store = self.chroma_client.create_langchain_vector_store(
            embedding_function=self.embedding_client.get_embedding_function()
        )
        
        logger.info("RAG retriever initialized")
    
    def retrieve_context(
        self, 
        query: str, 
        max_results: int = 5,
        similarity_threshold: float = 0.2,  # Lowered threshold for more recall
        metadata_filter: Optional[Dict[str, Any]] = None
    ) -> List[SourceDocument]:
        """
        Retrieve relevant context documents for a query.
        
        Args:
            query: User query to search for
            max_results: Maximum number of results to return
            similarity_threshold: Minimum similarity threshold
            metadata_filter: Optional metadata filter
            
        Returns:
            List of relevant source documents
        """
        try:
            start_time = time.time()
            # Perform similarity search with scores
            search_results = self.vector_store.similarity_search_with_score(
                query=query,
                k=max_results * 4,  # Get even more results for filtering
                filter=metadata_filter
            )
            # Debug: Log all candidate document scores
            logger.info("[DEBUG] Candidate document scores for query: '%s'", query)
            for i, (doc, score) in enumerate(search_results, 1):
                similarity_score = 1 / (1 + score) if score > 0 else 1.0
                logger.info("[DEBUG] Candidate %d: score=%.4f, similarity=%.4f, content=%.80s", i, score, similarity_score, doc.page_content.replace('\n', ' ')[:80])
            # Convert to SourceDocument objects with filtering
            source_documents = []
            for doc, score in search_results:
                similarity_score = 1 / (1 + score) if score > 0 else 1.0
                if similarity_score >= similarity_threshold:
                    source_doc = SourceDocument(
                        content=doc.page_content,
                        metadata=doc.metadata,
                        relevance_score=similarity_score,
                        document_id=doc.metadata.get("chunk_id", "unknown")
                    )
                    source_documents.append(source_doc)
            # Sort by relevance score and limit results
            source_documents.sort(key=lambda x: x.relevance_score or 0, reverse=True)
            source_documents = source_documents[:max_results]
            retrieval_time = int((time.time() - start_time) * 1000)
            logger.info(f"Retrieved {len(source_documents)} relevant documents in {retrieval_time}ms")
            return source_documents
        except Exception as e:
            logger.error(f"Failed to retrieve context: {e}")
            return []
    
    def prepare_context(self, source_documents: List[SourceDocument]) -> str:
        """
        Prepare context string from source documents.
        
        Args:
            source_documents: List of source documents
            
        Returns:
            Formatted context string
        """
        try:
            if not source_documents:
                return "No relevant information found in the hospital FAQ database."
            
            context_parts = []
            for i, doc in enumerate(source_documents, 1):
                # Add document number and content
                doc_context = f"Document {i}:\n{doc.content.strip()}"
                
                # Add source information if available
                source_file = doc.metadata.get("source", "")
                if source_file:
                    doc_context += f"\n[Source: {source_file}]"
                
                # Add relevance score if available
                if doc.relevance_score:
                    doc_context += f"\n[Relevance: {doc.relevance_score:.2f}]"
                
                context_parts.append(doc_context)
            
            # Join all context parts
            full_context = "\n\n".join(context_parts)
            
            # Truncate if too long
            if len(full_context) > settings.max_context_length:
                full_context = full_context[:settings.max_context_length] + "\n\n[Context truncated...]"
            
            return full_context
            
        except Exception as e:
            logger.error(f"Failed to prepare context: {e}")
            return "Error preparing context from retrieved documents."
    
    def generate_rag_response(
        self, 
        question: str, 
        context: str,
        conversation_history: Optional[List[Dict[str, str]]] = None
    ) -> str:
        """
        Generate a response using retrieved context and LLM.
        
        Args:
            question: User's question
            context: Retrieved context from documents
            conversation_history: Previous conversation messages
            
        Returns:
            Generated response
        """
        try:
            logger.info("\n---\n[DEBUG] Context passed to LLM for question: '%s'\n%s\n---\n", question, context)
            return self.llm_client.generate_contextual_response(
                question=question,
                context=context,
                conversation_history=conversation_history
            )
        except Exception as e:
            logger.error(f"Failed to generate RAG response: {e}")
            return "I apologize, but I'm having trouble generating a response. Please try rephrasing your question or contact our support staff."
    
    def answer_question(
        self, 
        question: str, 
        max_context_docs: int = 10,  # Increase to 10 for more recall
        include_sources: bool = True,
        conversation_history: Optional[List[Dict[str, str]]] = None
    ) -> Tuple[str, List[SourceDocument]]:
        """
        Answer a question using the RAG pipeline.
        
        Args:
            question: User's question
            max_context_docs: Maximum number of context documents to use
            include_sources: Whether to return source documents
            conversation_history: Previous conversation messages
            
        Returns:
            Tuple of (response, source_documents)
        """
        try:
            # Retrieve relevant context
            source_documents = self.retrieve_context(
                query=question,
                max_results=max_context_docs,
                similarity_threshold=0.3  # Lower threshold for aggressive recall
            )
            # DEBUG: Print out the retrieved source documents
            logger.info("[DEBUG] Retrieved source documents for question: '%s'", question)
            for i, doc in enumerate(source_documents, 1):
                logger.info("[DEBUG] Source %d: %s", i, doc.content)
                logger.info("[DEBUG] Metadata %d: %s", i, doc.metadata)
            # Prepare context string
            context = self.prepare_context(source_documents)
            # DEBUG: Print out the context string
            logger.info("[DEBUG] Context string for question '%s':\n%s", question, context)
            # Generate response using RAG
            response = self.generate_rag_response(
                question=question,
                context=context,
                conversation_history=conversation_history
            )
            # Return sources if requested
            sources = source_documents if include_sources else []
            logger.info(f"Generated RAG response for question: {question[:50]}...")
            return response, sources
        except Exception as e:
            logger.error(f"Failed to answer question: {e}")
            error_response = "I apologize, but I'm experiencing technical difficulties. Please contact our hospital staff for immediate assistance."
            return error_response, []
    
    def search_documents(self, request: SearchRequest) -> SearchResponse:
        """
        Search documents based on a search request.
        
        Args:
            request: Search request parameters
            
        Returns:
            Search response with results
        """
        try:
            start_time = time.time()
            
            # Retrieve documents
            source_documents = self.retrieve_context(
                query=request.query,
                max_results=request.limit,
                similarity_threshold=request.similarity_threshold,
                metadata_filter=request.metadata_filter
            )
            
            query_time = int((time.time() - start_time) * 1000)
            
            return SearchResponse(
                results=source_documents,
                total_results=len(source_documents),
                query_time_ms=query_time,
                query=request.query
            )
            
        except Exception as e:
            logger.error(f"Document search failed: {e}")
            return SearchResponse(
                results=[],
                total_results=0,
                query_time_ms=0,
                query=request.query
            )
    
    def get_similar_questions(self, question: str, limit: int = 5) -> List[str]:
        """
        Find similar questions from the knowledge base.
        
        Args:
            question: Input question
            limit: Maximum number of similar questions to return
            
        Returns:
            List of similar questions
        """
        try:
            # Search for similar content
            results = self.vector_store.similarity_search(
                query=question,
                k=limit * 2,  # Get more results to find diverse questions
                filter={"document_type": "hospital_faq"}
            )
            
            # Extract unique questions/content
            similar_questions = []
            seen_content = set()
            
            for doc in results:
                content = doc.page_content.strip()
                
                # Skip if we've seen similar content
                if content not in seen_content and len(content) > 10:
                    similar_questions.append(content)
                    seen_content.add(content)
                    
                    if len(similar_questions) >= limit:
                        break
            
            return similar_questions
            
        except Exception as e:
            logger.error(f"Failed to find similar questions: {e}")
            return []
    
    def rerank_documents(
        self, 
        query: str, 
        documents: List[SourceDocument]
    ) -> List[SourceDocument]:
        """
        Rerank documents based on relevance to query.
        
        Args:
            query: Search query
            documents: List of documents to rerank
            
        Returns:
            Reranked list of documents
        """
        try:
            # Simple reranking based on keyword matching and metadata
            scored_docs = []
            
            query_lower = query.lower()
            query_words = set(query_lower.split())
            
            for doc in documents:
                score = doc.relevance_score or 0
                
                # Boost score based on exact keyword matches
                content_lower = doc.content.lower()
                word_matches = sum(1 for word in query_words if word in content_lower)
                keyword_boost = word_matches * 0.1
                
                # Boost based on metadata (e.g., document type, recency)
                metadata_boost = 0
                if doc.metadata.get("document_type") == "hospital_faq":
                    metadata_boost += 0.05
                
                final_score = score + keyword_boost + metadata_boost
                scored_docs.append((final_score, doc))
            
            # Sort by final score and return documents
            scored_docs.sort(key=lambda x: x[0], reverse=True)
            reranked_docs = [doc for _, doc in scored_docs]
            
            # Update relevance scores
            for i, doc in enumerate(reranked_docs):
                doc.relevance_score = scored_docs[i][0]
            
            return reranked_docs
            
        except Exception as e:
            logger.error(f"Failed to rerank documents: {e}")
            return documents


def get_rag_retriever() -> RAGRetriever:
    """
    Get a RAG retriever instance.
    
    Returns:
        RAGRetriever: Initialized RAG retriever
    """
    return RAGRetriever()


def answer_faq_question(
    question: str, 
    max_context_docs: int = 5,
    conversation_history: Optional[List[Dict[str, str]]] = None
) -> Tuple[str, List[SourceDocument]]:
    """
    Answer a hospital FAQ question using RAG.
    
    Args:
        question: User's question
        max_context_docs: Maximum context documents to use
        conversation_history: Previous conversation
        
    Returns:
        Tuple of (response, source_documents)
    """
    try:
        retriever = get_rag_retriever()
        return retriever.answer_question(
            question=question,
            max_context_docs=max_context_docs,
            conversation_history=conversation_history
        )
        
    except Exception as e:
        logger.error(f"Failed to answer FAQ question: {e}")
        error_response = "I'm experiencing technical difficulties. Please contact hospital staff for assistance."
        return error_response, []


if __name__ == "__main__":
    """Test the RAG retrieval system."""
    print("Testing RAG retrieval system...")
    
    try:
        # Test retrieval
        retriever = get_rag_retriever()
        
        # Test question
        test_question = "What are the visiting hours?"
        response, sources = retriever.answer_question(test_question)
        
        print(f"✅ RAG system test successful")
        print(f"Question: {test_question}")
        print(f"Response: {response[:200]}...")
        print(f"Sources found: {len(sources)}")
        
    except Exception as e:
        print(f"❌ RAG system test failed: {e}")