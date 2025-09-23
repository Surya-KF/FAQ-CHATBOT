.
├── .github/
│   └── INSTRUCTIONS.md        # Guidance/instructions for Copilot and team
│   └── filestructure.md 
├── app/
│   ├── __init__.py
│   ├── config.py              # App/global configurations, secrets via env vars
│   ├── db.py                  # ChromaDB setup and query logic
│   ├── indexing.py            # Chunking, embedding, indexing pipeline
│   ├── llm.py                 # LangChain Gemini 2.5 Flash integration
│   ├── memory.py              # Session/contextuality logic
│   ├── retrieval.py           # Retrieval and search logic (hybrid RAG pipeline)
│   ├── api.py                 # FastAPI route and handler logic
│   ├── models.py              # Data and Pydantic schemas
│   └── main.py                # FastAPI entrypoint
├── streamlit_app/
│   ├── __init__.py
│   ├── ui.py                  # Streamlit UI logic
│   └── app.py                 # Streamlit entrypoint
├── scripts/
│   └── preprocess.py          # Clean/prepare hospital FAQs for indexing
├── data/
│   ├── raw/                   # Source data (hospital faq, policies)
│   └── processed/             # Cleaned, chunked docs for embedding
├── requirements.txt
├── pyproject.toml             # UV package management
├── Dockerfile                 # Docker setup, uses UV for pip/install
├── .env.example               # Sample environment variables (API keys etc.)
└── README.md                  # Full setup and usage guide
