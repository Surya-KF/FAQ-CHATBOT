"""
Conversation memory management for Hospital FAQ Chatbot

This module handles:
- Session management and conversation tracking
- Message history storage and retrieval
- Conversation summarization for long-term memory
- Context management for multi-turn conversations
"""

import logging
import time
import json
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any
from langchain.memory import (
    ConversationBufferMemory, 
    ConversationSummaryMemory,
    ConversationBufferWindowMemory
)
from langchain.schema import BaseMessage, HumanMessage, AIMessage
from app.config import settings
from app.llm import get_llm_client
from app.models import ConversationMessage, ConversationHistory, MemoryStats, MessageRole

# Configure logging
logger = logging.getLogger(__name__)


class SessionMemoryManager:
    """Manages conversation memory for individual chat sessions."""
    
    def __init__(self, session_id: str):
        """
        Initialize session memory manager.
        
        Args:
            session_id: Unique session identifier
        """
        self.session_id = session_id
        self.llm_client = get_llm_client()
        
        # Initialize conversation history
        self.conversation_history = ConversationHistory(session_id=session_id)
        
        # Choose memory type based on configuration
        if settings.memory_type.lower() == "summary":
            self.memory = ConversationSummaryMemory(
                llm=self.llm_client.chat_model,
                max_token_limit=1000,
                return_messages=True
            )
        elif settings.memory_type.lower() == "window":
            self.memory = ConversationBufferWindowMemory(
                k=settings.max_conversation_history,
                return_messages=True
            )
        else:
            self.memory = ConversationBufferMemory(
                return_messages=True
            )
        
        logger.info(f"Session memory manager initialized for session: {session_id}")
    
    def add_message(self, role: MessageRole, content: str) -> ConversationMessage:
        """
        Add a message to the conversation history.
        
        Args:
            role: Role of the message sender
            content: Message content
            
        Returns:
            ConversationMessage: The added message
        """
        try:
            # Create conversation message
            message = ConversationMessage(
                role=role,
                content=content
            )
            
            # Add to conversation history
            self.conversation_history.messages.append(message)
            self.conversation_history.updated_at = datetime.utcnow()
            
            # Add to LangChain memory
            if role == MessageRole.USER:
                self.memory.chat_memory.add_user_message(content)
            elif role == MessageRole.ASSISTANT:
                self.memory.chat_memory.add_ai_message(content)
            
            # Manage memory size
            self._manage_memory_size()
            
            logger.info(f"Added {role.value} message to session {self.session_id}")
            return message
            
        except Exception as e:
            logger.error(f"Failed to add message to session {self.session_id}: {e}")
            raise
    
    def get_conversation_context(self, max_messages: Optional[int] = None) -> List[Dict[str, str]]:
        """
        Get conversation context for RAG processing.
        
        Args:
            max_messages: Maximum number of recent messages to include
            
        Returns:
            List of conversation messages as dictionaries
        """
        try:
            messages = self.conversation_history.messages
            
            # Limit to recent messages if specified
            if max_messages:
                messages = messages[-max_messages:]
            
            # Convert to simple format for RAG
            context = []
            for msg in messages:
                context.append({
                    "role": msg.role.value,
                    "content": msg.content,
                    "timestamp": msg.timestamp.isoformat()
                })
            
            return context
            
        except Exception as e:
            logger.error(f"Failed to get conversation context for session {self.session_id}: {e}")
            return []
    
    def get_memory_variables(self) -> Dict[str, Any]:
        """
        Get memory variables for prompt templates.
        
        Returns:
            Dictionary with memory variables
        """
        try:
            return self.memory.load_memory_variables({})
        except Exception as e:
            logger.error(f"Failed to get memory variables for session {self.session_id}: {e}")
            return {}
    
    def summarize_conversation(self) -> str:
        """
        Generate a summary of the conversation.
        
        Returns:
            Conversation summary
        """
        try:
            if not self.conversation_history.messages:
                return ""
            
            # Prepare conversation for summarization
            conversation_text = []
            for msg in self.conversation_history.messages:
                conversation_text.append(f"{msg.role.value}: {msg.content}")
            
            # Generate summary using LLM
            summary = self.llm_client.summarize_conversation(
                conversation_history=self.get_conversation_context()
            )
            
            # Update conversation history with summary
            self.conversation_history.summary = summary
            
            logger.info(f"Generated conversation summary for session {self.session_id}")
            return summary
            
        except Exception as e:
            logger.error(f"Failed to summarize conversation for session {self.session_id}: {e}")
            return "Summary unavailable"
    
    def _manage_memory_size(self) -> None:
        """Manage memory size by removing old messages or creating summaries."""
        try:
            max_messages = settings.max_conversation_history
            
            if len(self.conversation_history.messages) > max_messages:
                if settings.memory_type.lower() == "summary":
                    # Create summary and keep recent messages
                    self.summarize_conversation()
                    
                    # Keep only recent messages
                    recent_messages = self.conversation_history.messages[-max_messages//2:]
                    self.conversation_history.messages = recent_messages
                    
                else:
                    # Simply remove oldest messages
                    self.conversation_history.messages = self.conversation_history.messages[-max_messages:]
                
                logger.info(f"Managed memory size for session {self.session_id}")
                
        except Exception as e:
            logger.error(f"Failed to manage memory size for session {self.session_id}: {e}")
    
    def get_memory_stats(self) -> MemoryStats:
        """
        Get memory statistics for the session.
        
        Returns:
            MemoryStats: Memory usage statistics
        """
        try:
            # Calculate memory size (rough estimate)
            total_content = "".join([msg.content for msg in self.conversation_history.messages])
            memory_size_kb = len(total_content.encode('utf-8')) / 1024
            
            summary_length = len(self.conversation_history.summary) if self.conversation_history.summary else 0
            
            return MemoryStats(
                session_id=self.session_id,
                total_messages=len(self.conversation_history.messages),
                memory_size_kb=memory_size_kb,
                last_activity=self.conversation_history.updated_at,
                summary_length=summary_length
            )
            
        except Exception as e:
            logger.error(f"Failed to get memory stats for session {self.session_id}: {e}")
            return MemoryStats(
                session_id=self.session_id,
                total_messages=0,
                memory_size_kb=0.0,
                last_activity=datetime.utcnow()
            )
    
    def clear_memory(self) -> None:
        """Clear all conversation memory for the session."""
        try:
            self.conversation_history.messages = []
            self.conversation_history.summary = None
            self.conversation_history.updated_at = datetime.utcnow()
            self.memory.clear()
            
            logger.info(f"Cleared memory for session {self.session_id}")
            
        except Exception as e:
            logger.error(f"Failed to clear memory for session {self.session_id}: {e}")


class GlobalMemoryManager:
    """Manages memory across all chat sessions."""
    
    def __init__(self):
        """Initialize global memory manager."""
        self.sessions: Dict[str, SessionMemoryManager] = {}
        self.session_timeouts: Dict[str, datetime] = {}
        self.session_timeout_minutes = 60  # Default session timeout
        
        logger.info("Global memory manager initialized")
    
    def get_session(self, session_id: str) -> SessionMemoryManager:
        """
        Get or create a session memory manager.
        
        Args:
            session_id: Session identifier
            
        Returns:
            SessionMemoryManager: Session memory manager
        """
        try:
            # Clean up expired sessions
            self._cleanup_expired_sessions()
            
            # Get existing session or create new one
            if session_id not in self.sessions:
                self.sessions[session_id] = SessionMemoryManager(session_id)
                logger.info(f"Created new session: {session_id}")
            
            # Update session timeout
            self.session_timeouts[session_id] = datetime.utcnow() + timedelta(minutes=self.session_timeout_minutes)
            
            return self.sessions[session_id]
            
        except Exception as e:
            logger.error(f"Failed to get session {session_id}: {e}")
            raise
    
    def add_conversation_turn(
        self, 
        session_id: str, 
        user_message: str, 
        assistant_response: str
    ) -> None:
        """
        Add a complete conversation turn (user message + assistant response).
        
        Args:
            session_id: Session identifier
            user_message: User's message
            assistant_response: Assistant's response
        """
        try:
            session = self.get_session(session_id)
            
            # Add user message
            session.add_message(MessageRole.USER, user_message)
            
            # Add assistant response
            session.add_message(MessageRole.ASSISTANT, assistant_response)
            
            logger.info(f"Added conversation turn to session {session_id}")
            
        except Exception as e:
            logger.error(f"Failed to add conversation turn to session {session_id}: {e}")
    
    def get_conversation_context(
        self, 
        session_id: str, 
        max_messages: Optional[int] = None
    ) -> List[Dict[str, str]]:
        """
        Get conversation context for a session.
        
        Args:
            session_id: Session identifier
            max_messages: Maximum number of recent messages
            
        Returns:
            List of conversation messages
        """
        try:
            if session_id in self.sessions:
                return self.sessions[session_id].get_conversation_context(max_messages)
            else:
                return []
                
        except Exception as e:
            logger.error(f"Failed to get conversation context for session {session_id}: {e}")
            return []
    
    def get_all_session_stats(self) -> List[MemoryStats]:
        """
        Get memory statistics for all active sessions.
        
        Returns:
            List of memory statistics
        """
        try:
            stats = []
            for session_id, session in self.sessions.items():
                stats.append(session.get_memory_stats())
            
            return stats
            
        except Exception as e:
            logger.error(f"Failed to get session stats: {e}")
            return []
    
    def clear_session(self, session_id: str) -> bool:
        """
        Clear a specific session.
        
        Args:
            session_id: Session to clear
            
        Returns:
            bool: True if successful, False otherwise
        """
        try:
            if session_id in self.sessions:
                self.sessions[session_id].clear_memory()
                del self.sessions[session_id]
                if session_id in self.session_timeouts:
                    del self.session_timeouts[session_id]
                
                logger.info(f"Cleared session {session_id}")
                return True
            
            return False
            
        except Exception as e:
            logger.error(f"Failed to clear session {session_id}: {e}")
            return False
    
    def _cleanup_expired_sessions(self) -> None:
        """Clean up expired sessions to free memory."""
        try:
            current_time = datetime.utcnow()
            expired_sessions = []
            
            for session_id, timeout_time in self.session_timeouts.items():
                if current_time > timeout_time:
                    expired_sessions.append(session_id)
            
            for session_id in expired_sessions:
                self.clear_session(session_id)
                logger.info(f"Cleaned up expired session: {session_id}")
                
        except Exception as e:
            logger.error(f"Failed to cleanup expired sessions: {e}")
    
    def get_active_session_count(self) -> int:
        """
        Get the number of active sessions.
        
        Returns:
            int: Number of active sessions
        """
        self._cleanup_expired_sessions()
        return len(self.sessions)


# Global memory manager instance
global_memory_manager = GlobalMemoryManager()


def get_memory_manager() -> GlobalMemoryManager:
    """
    Get the global memory manager instance.
    
    Returns:
        GlobalMemoryManager: The global memory manager
    """
    return global_memory_manager


def get_session_memory(session_id: str) -> SessionMemoryManager:
    """
    Get session memory manager for a specific session.
    
    Args:
        session_id: Session identifier
        
    Returns:
        SessionMemoryManager: Session memory manager
    """
    return global_memory_manager.get_session(session_id)


if __name__ == "__main__":
    """Test conversation memory system."""
    print("Testing conversation memory system...")
    
    try:
        # Test session creation
        test_session_id = "test-session-123"
        session = get_session_memory(test_session_id)
        
        # Test adding messages
        session.add_message(MessageRole.USER, "What are the visiting hours?")
        session.add_message(MessageRole.ASSISTANT, "Visiting hours are from 9 AM to 8 PM daily.")
        
        # Test getting context
        context = session.get_conversation_context()
        print(f"✅ Memory system test successful")
        print(f"Session ID: {test_session_id}")
        print(f"Messages: {len(context)}")
        
        # Test memory stats
        stats = session.get_memory_stats()
        print(f"Memory size: {stats.memory_size_kb:.2f} KB")
        
    except Exception as e:
        print(f"❌ Memory system test failed: {e}")