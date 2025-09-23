# Hospital FAQ Chatbot - Multi-stage Docker build with UV package manager
# This Dockerfile creates an optimized container for the Hospital FAQ Chatbot

# Stage 1: Build stage with UV
FROM python:3.11-slim as builder

# Set environment variables
ENV PYTHONUNBUFFERED=1
ENV PYTHONDONTWRITEBYTECODE=1

# Install system dependencies
RUN apt-get update && apt-get install -y \
    curl \
    gcc \
    g++ \
    && rm -rf /var/lib/apt/lists/*

# Install UV (modern Python package manager)
COPY --from=ghcr.io/astral-sh/uv:latest /uv /bin/uv

# Set work directory
WORKDIR /app

# Copy dependency files
COPY pyproject.toml ./
COPY requirements.txt ./

# Create virtual environment and install dependencies using UV
RUN uv venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"

# Install Python dependencies
RUN uv pip install -r requirements.txt

# Stage 2: Runtime stage
FROM python:3.11-slim as runtime

# Set environment variables
ENV PYTHONUNBUFFERED=1
ENV PYTHONDONTWRITEBYTECODE=1
ENV PATH="/opt/venv/bin:$PATH"

# Install runtime system dependencies
RUN apt-get update && apt-get install -y \
    curl \
    && rm -rf /var/lib/apt/lists/* \
    && apt-get clean

# Create non-root user for security
RUN groupadd -r appuser && useradd -r -g appuser appuser

# Set work directory
WORKDIR /app

# Copy virtual environment from builder stage
COPY --from=builder /opt/venv /opt/venv

# Copy application code
COPY app/ ./app/
COPY streamlit_app/ ./streamlit_app/
COPY scripts/ ./scripts/
COPY .env.example ./.env.example

# Create necessary directories
RUN mkdir -p data/raw data/processed chroma_db logs

# Set proper permissions
RUN chown -R appuser:appuser /app

# Switch to non-root user
USER appuser

# Expose ports
EXPOSE 8000 8501

# Health check
HEALTHCHECK --interval=30s --timeout=30s --start-period=5s --retries=3 \
    CMD curl -f http://localhost:8000/health || exit 1

# Default command (can be overridden)
CMD ["python", "app/main.py"]

# Multi-service variant with docker-compose
# To run both API and Streamlit, use docker-compose.yml

# Labels for better container management
LABEL maintainer="Hospital FAQ Chatbot Team"
LABEL version="1.0.0"
LABEL description="Hospital FAQ Chatbot with LangChain, Gemini 2.5 Flash, and ChromaDB"