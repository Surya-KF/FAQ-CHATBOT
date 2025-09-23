"""
Document indexing pipeline for Hospital FAQ Chatbot

This module handles:
- Document loading and preprocessing
- Text chunking using LangChain's RecursiveCharacterTextSplitter
- Embedding generation using Gemini embedding model
- Vector storage in ChromaDB
- Index management and updates
"""

import os
import logging
import time
from pathlib import Path
from typing import List, Dict, Any, Optional, Tuple
import asyncio
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import (
    TextLoader, 
    PyPDFLoader, 
    DirectoryLoader,
    UnstructuredMarkdownLoader
)
from langchain.schema import Document
from app.config import settings
from app.db import get_chroma_client
from app.llm import get_embedding_client
from app.models import DocumentChunk, IndexingRequest, IndexingResponse

# Configure logging
logger = logging.getLogger(__name__)


class DocumentIndexer:
    """Document indexing pipeline for hospital FAQ documents."""
    
    def __init__(self):
        """Initialize the document indexer with required clients."""
        self.chroma_client = get_chroma_client()
        self.embedding_client = get_embedding_client()
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=settings.chunk_size,
            chunk_overlap=settings.chunk_overlap,
            length_function=len,
            separators=["\n\n", "\n", ".", "!", "?", ",", " ", ""]
        )
        
        # Initialize vector store
        self.vector_store = self.chroma_client.create_langchain_vector_store(
            embedding_function=self.embedding_client.get_embedding_function()
        )
        
        logger.info("Document indexer initialized")
    
    def load_documents_from_directory(
        self,
        directory_path: str,
        file_extensions: Optional[List[str]] = None
    ) -> List[Document]:
        """
        Load documents from a directory, supporting PDF, TXT, MD, JSON (Q&A), CSV, and TSV files.
        Each Q&A or row is loaded as a separate document for structured retrieval.
        """
        import json
        import csv
        documents = []
        try:
            if file_extensions is None:
                file_extensions = ['.txt', '.md', '.pdf', '.json', '.csv', '.tsv']
            directory = Path(directory_path)
            if not directory.exists():
                logger.warning(f"Directory does not exist: {directory_path}")
                return documents
            for ext in file_extensions:
                for file_path in directory.rglob(f'*{ext}'):
                    file_path_str = str(file_path)
                    if ext == '.pdf':
                        loader = PyPDFLoader(file_path_str)
                        docs = loader.load()
                        documents.extend(docs)
                    elif ext == '.md':
                        loader = UnstructuredMarkdownLoader(file_path_str)
                        docs = loader.load()
                        documents.extend(docs)
                    elif ext == '.txt':
                        loader = TextLoader(file_path_str)
                        docs = loader.load()
                        documents.extend(docs)
                    elif ext == '.json':
                        # Expecting list of Q&A dicts or dict with 'questions'/'answers'
                        try:
                            with open(file_path_str, 'r', encoding='utf-8') as f:
                                data = json.load(f)
                            if isinstance(data, list):
                                for idx, item in enumerate(data):
                                    if isinstance(item, dict):
                                        content = item.get('question', '') + '\n' + item.get('answer', '')
                                        meta = {"source": file_path_str, "row": idx, **item}
                                        documents.append(Document(page_content=content, metadata=meta))
                            elif isinstance(data, dict):
                                # Try to extract Q&A pairs from dict
                                if 'questions' in data and 'answers' in data:
                                    for idx, (q, a) in enumerate(zip(data['questions'], data['answers'])):
                                        content = q + '\n' + a
                                        meta = {"source": file_path_str, "row": idx}
                                        documents.append(Document(page_content=content, metadata=meta))
                        except Exception as e:
                            logger.error(f"Failed to load JSON {file_path_str}: {e}")
                    elif ext in ['.csv', '.tsv']:
                        delimiter = ',' if ext == '.csv' else '\t'
                        try:
                            with open(file_path_str, 'r', encoding='utf-8') as f:
                                reader = csv.DictReader(f, delimiter=delimiter)
                                for idx, row in enumerate(reader):
                                    # Try to combine Q/A columns if present
                                    if 'question' in row and 'answer' in row:
                                        content = row['question'] + '\n' + row['answer']
                                    else:
                                        content = '\n'.join(str(v) for v in row.values())
                                    meta = {"source": file_path_str, "row": idx, **row}
                                    documents.append(Document(page_content=content, metadata=meta))
                        except Exception as e:
                            logger.error(f"Failed to load {ext.upper()} {file_path_str}: {e}")
            logger.info(f"Loaded {len(documents)} documents from {directory_path}")
            return documents
        except Exception as e:
            logger.error(f"Failed to load documents from directory: {e}")
            return []
    
    def load_single_document(self, file_path: str) -> Optional[Document]:
        """
        Load a single document, supporting PDF, TXT, MD, JSON (Q&A), CSV, and TSV files.
        For JSON/CSV/TSV, returns the first Q&A or row as a document.
        """
        import json
        import csv
        try:
            file_ext = Path(file_path).suffix.lower()
            if file_ext == '.pdf':
                loader = PyPDFLoader(file_path)
                documents = loader.load()
                if documents:
                    logger.info(f"Loaded document: {file_path}")
                    return documents[0]
            elif file_ext == '.md':
                loader = UnstructuredMarkdownLoader(file_path)
                documents = loader.load()
                if documents:
                    logger.info(f"Loaded document: {file_path}")
                    return documents[0]
            elif file_ext == '.txt':
                loader = TextLoader(file_path)
                documents = loader.load()
                if documents:
                    logger.info(f"Loaded document: {file_path}")
                    return documents[0]
            elif file_ext == '.json':
                with open(file_path, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                if isinstance(data, list) and data:
                    item = data[0]
                    if isinstance(item, dict):
                        content = item.get('question', '') + '\n' + item.get('answer', '')
                        meta = {"source": file_path, "row": 0, **item}
                        return Document(page_content=content, metadata=meta)
                elif isinstance(data, dict):
                    if 'questions' in data and 'answers' in data and data['questions'] and data['answers']:
                        content = data['questions'][0] + '\n' + data['answers'][0]
                        meta = {"source": file_path, "row": 0}
                        return Document(page_content=content, metadata=meta)
            elif file_ext in ['.csv', '.tsv']:
                delimiter = ',' if file_ext == '.csv' else '\t'
                with open(file_path, 'r', encoding='utf-8') as f:
                    reader = csv.DictReader(f, delimiter=delimiter)
                    for idx, row in enumerate(reader):
                        if idx == 0:
                            if 'question' in row and 'answer' in row:
                                content = row['question'] + '\n' + row['answer']
                            else:
                                content = '\n'.join(str(v) for v in row.values())
                            meta = {"source": file_path, "row": idx, **row}
                            return Document(page_content=content, metadata=meta)
            else:
                loader = TextLoader(file_path)
                documents = loader.load()
                if documents:
                    logger.info(f"Loaded document: {file_path}")
                    return documents[0]
        except Exception as e:
            logger.error(f"Failed to load document {file_path}: {e}")
        return None
    
    def create_chunks(
        self, 
        documents: List[Document],
        chunk_size: Optional[int] = None,
        chunk_overlap: Optional[int] = None
    ) -> List[DocumentChunk]:
        """
        Split documents into chunks.
        
        Args:
            documents: List of documents to chunk
            chunk_size: Override default chunk size
            chunk_overlap: Override default chunk overlap
            
        Returns:
            List of document chunks
        """
        try:
            # Create text splitter with custom parameters if provided
            if chunk_size or chunk_overlap:
                splitter = RecursiveCharacterTextSplitter(
                    chunk_size=chunk_size or settings.chunk_size,
                    chunk_overlap=chunk_overlap or settings.chunk_overlap,
                    length_function=len,
                    separators=["\n\n", "\n", ".", "!", "?", ",", " ", ""]
                )
            else:
                splitter = self.text_splitter
            
            chunks = []
            total_chunks = 0
            
            for doc_idx, document in enumerate(documents):
                # Split document into chunks
                doc_chunks = splitter.split_documents([document])
                
                for chunk_idx, chunk in enumerate(doc_chunks):
                    # Create DocumentChunk with metadata
                    chunk_metadata = {
                        **chunk.metadata,
                        "source_document_index": doc_idx,
                        "chunk_index": chunk_idx,
                        "total_chunks": len(doc_chunks),
                        "chunk_size": len(chunk.page_content),
                        "document_type": "hospital_faq"
                    }
                    
                    document_chunk = DocumentChunk(
                        content=chunk.page_content,
                        metadata=chunk_metadata,
                        source_file=chunk.metadata.get("source", "unknown"),
                        chunk_index=chunk_idx
                    )
                    
                    chunks.append(document_chunk)
                    total_chunks += 1
            
            logger.info(f"Created {total_chunks} chunks from {len(documents)} documents")
            return chunks
            
        except Exception as e:
            logger.error(f"Failed to create chunks: {e}")
            return []
    
    def index_chunks(self, chunks: List[DocumentChunk]) -> bool:
        """
        Index document chunks in ChromaDB.
        
        Args:
            chunks: List of document chunks to index
            
        Returns:
            bool: True if successful, False otherwise
        """
        try:
            if not chunks:
                logger.warning("No chunks provided for indexing")
                return True
            
            # Prepare data for vector store
            texts = [chunk.content for chunk in chunks]
            metadatas = [chunk.metadata for chunk in chunks]
            ids = [chunk.chunk_id for chunk in chunks]
            
            # Add to vector store
            self.vector_store.add_texts(
                texts=texts,
                metadatas=metadatas,
                ids=ids
            )
            
            logger.info(f"Successfully indexed {len(chunks)} chunks")
            return True
            
        except Exception as e:
            logger.error(f"Failed to index chunks: {e}")
            return False
    
    def process_files(
        self, 
        file_paths: List[str],
        chunk_size: Optional[int] = None,
        chunk_overlap: Optional[int] = None,
        force_reindex: bool = False
    ) -> IndexingResponse:
        """
        Process and index a list of files.
        
        Args:
            file_paths: List of file paths to process
            chunk_size: Override default chunk size
            chunk_overlap: Override default chunk overlap
            force_reindex: Whether to force reindexing
            
        Returns:
            IndexingResponse with processing results
        """
        start_time = time.time()
        indexed_files = []
        errors = []
        total_chunks = 0
        
        try:
            for file_path in file_paths:
                try:
                    # Check if file exists
                    if not os.path.exists(file_path):
                        error_msg = f"File not found: {file_path}"
                        errors.append(error_msg)
                        logger.warning(error_msg)
                        continue
                    
                    # Load document
                    document = self.load_single_document(file_path)
                    if not document:
                        error_msg = f"Failed to load document: {file_path}"
                        errors.append(error_msg)
                        continue
                    
                    # Create chunks
                    chunks = self.create_chunks([document], chunk_size, chunk_overlap)
                    if not chunks:
                        error_msg = f"Failed to create chunks for: {file_path}"
                        errors.append(error_msg)
                        continue
                    
                    # Index chunks
                    if self.index_chunks(chunks):
                        indexed_files.append(file_path)
                        total_chunks += len(chunks)
                        logger.info(f"Successfully processed {file_path} ({len(chunks)} chunks)")
                    else:
                        error_msg = f"Failed to index chunks for: {file_path}"
                        errors.append(error_msg)
                
                except Exception as e:
                    error_msg = f"Error processing {file_path}: {str(e)}"
                    errors.append(error_msg)
                    logger.error(error_msg)
            
            processing_time = int((time.time() - start_time) * 1000)
            success = len(indexed_files) > 0
            
            response = IndexingResponse(
                success=success,
                total_documents=len(indexed_files),
                total_chunks=total_chunks,
                processing_time_ms=processing_time,
                errors=errors,
                indexed_files=indexed_files
            )
            
            logger.info(f"Indexing completed: {len(indexed_files)} files, {total_chunks} chunks, {processing_time}ms")
            return response
            
        except Exception as e:
            processing_time = int((time.time() - start_time) * 1000)
            error_msg = f"Indexing pipeline failed: {str(e)}"
            logger.error(error_msg)
            
            return IndexingResponse(
                success=False,
                total_documents=0,
                total_chunks=0,
                processing_time_ms=processing_time,
                errors=[error_msg],
                indexed_files=[]
            )
    
    def process_directory(
        self, 
        directory_path: str,
        file_extensions: Optional[List[str]] = None,
        chunk_size: Optional[int] = None,
        chunk_overlap: Optional[int] = None
    ) -> IndexingResponse:
        """
        Process all documents in a directory.
        
        Args:
            directory_path: Path to directory to process
            file_extensions: File extensions to include
            chunk_size: Override default chunk size
            chunk_overlap: Override default chunk overlap
            
        Returns:
            IndexingResponse with processing results
        """
        try:
            # Load all documents from directory
            documents = self.load_documents_from_directory(directory_path, file_extensions)
            
            if not documents:
                return IndexingResponse(
                    success=False,
                    total_documents=0,
                    total_chunks=0,
                    processing_time_ms=0,
                    errors=[f"No documents found in directory: {directory_path}"],
                    indexed_files=[]
                )
            
            # Extract file paths
            file_paths = [doc.metadata.get("source", "unknown") for doc in documents]
            
            # Process files
            return self.process_files(file_paths, chunk_size, chunk_overlap)
            
        except Exception as e:
            error_msg = f"Failed to process directory {directory_path}: {str(e)}"
            logger.error(error_msg)
            
            return IndexingResponse(
                success=False,
                total_documents=0,
                total_chunks=0,
                processing_time_ms=0,
                errors=[error_msg],
                indexed_files=[]
            )
    
    def get_index_stats(self) -> Dict[str, Any]:
        """
        Get statistics about the current index.
        
        Returns:
            Dictionary with index statistics
        """
        try:
            collection_info = self.chroma_client.get_collection_info()
            return {
                "collection_name": collection_info.get("collection_name"),
                "total_documents": collection_info.get("document_count", 0),
                "sample_documents": collection_info.get("sample_documents", [])
            }
            
        except Exception as e:
            logger.error(f"Failed to get index stats: {e}")
            return {"error": str(e)}


def get_indexer() -> DocumentIndexer:
    """
    Get a DocumentIndexer instance.
    
    Returns:
        DocumentIndexer: Initialized document indexer
    """
    return DocumentIndexer()


def index_hospital_faqs() -> IndexingResponse:
    """
    Index hospital FAQ documents from the configured data directory.
    
    Returns:
        IndexingResponse: Results of the indexing operation
    """
    try:
        indexer = get_indexer()
        # Process processed data directory (Q&A pairs)
        processed_data_path = settings.data_processed_path
        if not os.path.exists(processed_data_path):
            os.makedirs(processed_data_path, exist_ok=True)
            logger.warning(f"Created empty processed data directory: {processed_data_path}")

        response = indexer.process_directory(
            directory_path=processed_data_path,
            file_extensions=['.txt', '.json']
        )
        return response
    except Exception as e:
        logger.error(f"Failed to index hospital FAQs: {e}")
        return IndexingResponse(
            success=False,
            total_documents=0,
            total_chunks=0,
            processing_time_ms=0,
            errors=[str(e)],
            indexed_files=[]
        )


if __name__ == "__main__":
    """Main entry point for document indexing."""
    print("Starting Hospital FAQ document indexing...")
    
    # Index hospital FAQs
    result = index_hospital_faqs()
    
    if result.success:
        print(f"✅ Indexing successful!")
        print(f"   Documents processed: {result.total_documents}")
        print(f"   Chunks created: {result.total_chunks}")
        print(f"   Processing time: {result.processing_time_ms}ms")
    else:
        print(f"❌ Indexing failed!")
        if result.errors:
            print("Errors:")
            for error in result.errors:
                print(f"   - {error}")
    
    # Get index statistics
    try:
        indexer = get_indexer()
        stats = indexer.get_index_stats()
        print(f"\nIndex Statistics:")
        print(f"   Collection: {stats.get('collection_name')}")
        print(f"   Total documents: {stats.get('total_documents', 0)}")
        
    except Exception as e:
        print(f"Failed to get index stats: {e}")