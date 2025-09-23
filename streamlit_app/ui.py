"""
Streamlit UI components and logic for Hospital FAQ Chatbot

This module provides:
- Chat interface for user interaction
- Session management and conversation history
- Source document display
- System status and configuration
"""

import streamlit as st
import requests
import json
import time
from typing import List, Dict, Any, Optional
from datetime import datetime
import uuid

# API Configuration
API_BASE_URL = "http://localhost:8000/api/v1"

def init_session_state():
    """Initialize Streamlit session state variables."""
    if 'session_id' not in st.session_state:
        st.session_state.session_id = str(uuid.uuid4())
    
    if 'messages' not in st.session_state:
        st.session_state.messages = []
    
    if 'chat_history' not in st.session_state:
        st.session_state.chat_history = []


def make_api_request(endpoint: str, method: str = "GET", data: Dict = None) -> Optional[Dict]:
    """
    Make API request to the backend.
    
    Args:
        endpoint: API endpoint
        method: HTTP method
        data: Request data for POST requests
        
    Returns:
        API response data or None if failed
    """
    try:
        url = f"{API_BASE_URL}{endpoint}"
        
        if method.upper() == "POST":
            response = requests.post(url, json=data, timeout=30)
        elif method.upper() == "GET":
            response = requests.get(url, timeout=30)
        else:
            st.error(f"Unsupported HTTP method: {method}")
            return None
        
        if response.status_code == 200:
            return response.json()
        else:
            st.error(f"API Error {response.status_code}: {response.text}")
            return None
            
    except requests.exceptions.ConnectionError:
        st.error("❌ Cannot connect to the API server. Please ensure the backend is running on localhost:8000")
        return None
    except requests.exceptions.Timeout:
        st.error("⏱️ Request timed out. Please try again.")
        return None
    except Exception as e:
        st.error(f"Request failed: {str(e)}")
        return None


def send_message(question: str, include_sources: bool = True) -> Optional[Dict]:
    """
    Send a message to the chatbot.
    
    Args:
        question: User question
        include_sources: Whether to include source documents
        
    Returns:
        Chat response or None if failed
    """
    request_data = {
        "question": question,
        "session_id": st.session_state.session_id,
        "context_limit": 5,
        "include_sources": include_sources
    }
    
    return make_api_request("/chat", "POST", request_data)


def get_system_status() -> Optional[Dict]:
    """
    Get system status from the API.
    
    Returns:
        System status data or None if failed
    """
    return make_api_request("/status")


def search_documents(query: str, limit: int = 10) -> Optional[Dict]:
    """
    Search FAQ documents.
    
    Args:
        query: Search query
        limit: Maximum results
        
    Returns:
        Search results or None if failed
    """
    request_data = {
        "query": query,
        "limit": limit,
        "similarity_threshold": 0.6
    }
    
    return make_api_request("/search", "POST", request_data)


def clear_session() -> bool:
    """
    Clear the current session.
    
    Returns:
        True if successful, False otherwise
    """
    response = make_api_request(f"/sessions/{st.session_state.session_id}", "DELETE")
    return response is not None


def display_message(role: str, content: str, timestamp: Optional[str] = None):
    """
    Display a chat message.
    
    Args:
        role: Message role (user/assistant)
        content: Message content
        timestamp: Optional timestamp
    """
    if role == "user":
        with st.chat_message("user"):
            st.write(content)
            if timestamp:
                st.caption(f"🕒 {timestamp}")
    else:
        with st.chat_message("assistant"):
            st.write(content)
            if timestamp:
                st.caption(f"🕒 {timestamp}")


def display_sources(sources: List[Dict]) -> None:
    """
    Display source documents in an expandable section.
    
    Args:
        sources: List of source documents
    """
    if not sources:
        return
    
    with st.expander(f"📚 Sources ({len(sources)} documents)", expanded=False):
        for i, source in enumerate(sources, 1):
            st.markdown(f"**Source {i}** (Relevance: {source.get('relevance_score', 0):.2f})")
            # Display content in a code block for better formatting
            st.code(source.get('content', ''), language='text')
            if i < len(sources):
                st.divider()

    # Display metadata outside the expander
    for i, source in enumerate(sources, 1):
        metadata = source.get('metadata', {})
        if metadata:
            with st.expander(f"Metadata for Source {i}", expanded=False):
                st.json(metadata)


def chat_interface():
    """Main chat interface."""
    st.header("💬 Hospital FAQ Chatbot")
    
    # Display session info
    col1, col2, col3 = st.columns([2, 1, 1])
    
    with col1:
        st.caption(f"Session ID: {st.session_state.session_id[:8]}...")
    
    with col2:
        if st.button("🗑️ Clear Chat", help="Clear conversation history"):
            if clear_session():
                st.session_state.messages = []
                st.session_state.chat_history = []
                st.success("Chat cleared!")
                st.rerun()
            else:
                st.error("Failed to clear chat")
    
    with col3:
        if st.button("🔄 New Session", help="Start a new conversation"):
            st.session_state.session_id = str(uuid.uuid4())
            st.session_state.messages = []
            st.session_state.chat_history = []
            st.success("New session started!")
            st.rerun()
    
    # Display chat history
    for message in st.session_state.messages:
        display_message(
            role=message["role"],
            content=message["content"],
            timestamp=message.get("timestamp")
        )
        
        # Display sources if available
        if message["role"] == "assistant" and "sources" in message and message["sources"]:
            display_sources(message["sources"])
    
    # Chat input
    if prompt := st.chat_input("Ask me about hospital services, policies, or procedures..."):
        # Add user message to chat
        user_message = {
            "role": "user",
            "content": prompt,
            "timestamp": datetime.now().strftime("%H:%M:%S")
        }
        st.session_state.messages.append(user_message)
        
        # Display user message
        display_message("user", prompt, user_message["timestamp"])
        
        # Show typing indicator
        with st.chat_message("assistant"):
            with st.spinner("Thinking..."):
                # Send message to API
                response = send_message(prompt)
        
        if response:
            # Add assistant response to chat
            assistant_message = {
                "role": "assistant",
                "content": response["response"],
                "timestamp": datetime.now().strftime("%H:%M:%S"),
                "sources": response.get("sources", []),
                "confidence_score": response.get("confidence_score"),
                "response_time_ms": response.get("response_time_ms")
            }
            st.session_state.messages.append(assistant_message)
            
            # Display assistant response
            display_message("assistant", response["response"], assistant_message["timestamp"])
            
            # Display sources
            if response.get("sources"):
                display_sources(response["sources"])
            
            # Display response metadata
            col1, col2 = st.columns(2)
            with col1:
                if response.get("confidence_score"):
                    confidence = response["confidence_score"]
                    st.caption(f"🎯 Confidence: {confidence:.1%}")
            
            with col2:
                if response.get("response_time_ms"):
                    response_time = response["response_time_ms"]
                    st.caption(f"⚡ Response time: {response_time}ms")
            
            st.rerun()
        else:
            st.error("Failed to get response from the chatbot. Please try again.")


def search_interface():
    """Document search interface."""
    st.header("🔍 Search FAQ Documents")
    
    # Search input
    search_query = st.text_input(
        "Search hospital FAQ documents",
        placeholder="Enter your search query here...",
        help="Search for specific topics, keywords, or questions"
    )
    
    col1, col2 = st.columns([1, 3])
    
    with col1:
        max_results = st.selectbox("Max Results", [5, 10, 15, 20], index=1)
    
    if search_query:
        if st.button("🔍 Search", type="primary"):
            with st.spinner("Searching documents..."):
                results = search_documents(search_query, max_results)
            
            if results and results.get("results"):
                st.success(f"Found {len(results['results'])} results in {results.get('query_time_ms', 0)}ms")
                
                # Display results
                for i, result in enumerate(results["results"], 1):
                    with st.expander(f"Result {i} - Relevance: {result.get('relevance_score', 0):.2f}", expanded=False):
                        st.write(result.get("content", ""))
                        
                        # Show metadata
                        metadata = result.get("metadata", {})
                        if metadata:
                            st.caption("**Metadata:**")
                            st.json(metadata)
            else:
                st.warning("No results found for your search query.")


def system_status_interface():
    """System status and monitoring interface."""
    st.header("📊 System Status")
    
    if st.button("🔄 Refresh Status"):
        st.rerun()
    
    # Get system status
    status = get_system_status()
    
    if status:
        # Overall status
        status_text = status.get("status", "unknown")
        if status_text == "operational":
            st.success(f"✅ System Status: {status_text.title()}")
        else:
            st.warning(f"⚠️ System Status: {status_text.title()}")
        
        # Component status
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("🔧 Components")
            components = {
                "Database": status.get("database_connected", False),
                "LLM Service": status.get("llm_available", False),
                "Embedding Service": status.get("embedding_service_available", False)
            }
            
            for component, is_healthy in components.items():
                if is_healthy:
                    st.success(f"✅ {component}")
                else:
                    st.error(f"❌ {component}")
        
        with col2:
            st.subheader("📈 Statistics")
            st.metric("Total Documents", status.get("total_documents", 0))
            st.metric("Active Sessions", status.get("active_sessions", 0))
            
            uptime = status.get("uptime_seconds", 0)
            if uptime > 0:
                uptime_hours = uptime / 3600
                st.metric("Uptime (hours)", f"{uptime_hours:.1f}")
        
        # Timestamp
        st.caption(f"Last updated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    
    else:
        st.error("❌ Failed to retrieve system status")


def main():
    """Main Streamlit application."""
    # Page configuration
    st.set_page_config(
        page_title="Hospital FAQ Chatbot",
        page_icon="🏥",
        layout="wide",
        initial_sidebar_state="expanded"
    )
    
    # Initialize session state
    init_session_state()
    
    # Sidebar navigation
    st.sidebar.title("🏥 Hospital FAQ Chatbot")
    st.sidebar.markdown("---")
    
    page = st.sidebar.selectbox(
        "Navigate",
        ["💬 Chat", "🔍 Search", "📊 System Status"],
        index=0
    )
    
    # Display selected page
    if page == "💬 Chat":
        chat_interface()
    elif page == "🔍 Search":
        search_interface()
    elif page == "📊 System Status":
        system_status_interface()
    
    # Sidebar information
    st.sidebar.markdown("---")
    st.sidebar.markdown("""
    ### ℹ️ About
    This chatbot helps answer questions about:
    - Hospital services and departments
                        st.caption(f"Metadata for Source {i}:")
                        st.json(metadata)
    - Contact information
    - COVID-19 guidelines
    
    ### 🆘 Need Help?
    If you can't find the information you need, please contact:
    - Main desk: (555) 123-4567
    - Emergency: 911
    """)
    
    # Footer
    st.sidebar.markdown("---")
    st.sidebar.caption("Powered by LangChain & Gemini 2.5 Flash")


if __name__ == "__main__":
    main()