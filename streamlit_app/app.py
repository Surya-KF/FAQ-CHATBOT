"""
Streamlit application entry point for Hospital FAQ Chatbot

This module serves as the main entry point for the Streamlit web application.
"""

import sys
import os

# Add the parent directory to the Python path to import from streamlit_app
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from streamlit_app.ui import main

if __name__ == "__main__":
    main()