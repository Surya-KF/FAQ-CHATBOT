```markdown
# Hospital FAQ Chatbot: LangChain + Gemini 2.5 Flash + ChromaDB

---

## Overview

This project builds a conversational FAQ chatbot tailored for hospital use. Key technologies:

- **LangChain** for orchestration and integration
- **Gemini 2.5 Flash** (Google Generative AI) for text generation
- **gemini-embedding-001** for lightweight, high-quality embeddings
- **ChromaDB** as vector database for semantic search
- **FastAPI** for backend API
- **Streamlit** for frontend UI
- **UV** for package & environment management
- **Docker** for containerized deployment

---

## Project File Structure

```
.
├── .github/
│   └── INSTRUCTIONS.md        # This instructions file
├── app/
│   ├── config.py              # Environment settings and secrets
│   ├── db.py                  # ChromaDB client setup & query functions
│   ├── indexing.py            # Document chunking, embedding and indexing pipeline
│   ├── llm.py                 # Gemini 2.5 Flash model integration
│   ├── memory.py              # Session/context management for conversation
│   ├── retrieval.py           # Retrieval-augmented generation (RAG) search logic
│   ├── api.py                 # FastAPI endpoints and routing
│   ├── models.py              # Pydantic schemas and data models
│   └── main.py                # FastAPI startup & app configuration
├── streamlit_app/
│   ├── ui.py                  # Streamlit frontend logic and interaction
│   └── app.py                 # Streamlit app entrypoint
├── scripts/
│   └── preprocess.py          # Cleaning and preparing raw hospital FAQs
├── data/
│   ├── raw/                   # Original hospital FAQ documents
│   └── processed/             # Chunked and cleaned documents for indexing
├── requirements.txt           # Minimal python package dependencies
├── pyproject.toml             # UV compatible package manager config
├── Dockerfile                 # Container setup with UV install and app startup
├── .env.example               # Example environment variables (API keys etc.)
└── README.md                  # Setup and usage documentation
```

---

## Gemini 2.5 Flash & Embeddings Setup

### Environment

- Acquire Gemini API key from [Google AI Studio](https://aistudio.google.com/)
- Store in `.env` as `GEMINI_API_KEY`
- Load securely in `config.py`

### Text Generation Sample

```
import google.generativeai as genai
import os

genai.configure(api_key=os.getenv("GEMINI_API_KEY"))
model = genai.GenerativeModel(model_name="gemini-2.5-flash-preview-04-17")

response = model.generate_text(prompt="Explain hospital visiting hours.")
print(response.text)
```

### Using `gemini-embedding-001`

```
texts = [
    "What are the visiting hours at the hospital?",
    "COVID-19 safety procedures."
]

client = genai.Client()

response = client.models.embed_content(
    model="gemini-embedding-001",
    contents=texts
)

embeddings = response.embeddings  # List of 768-dimensional vectors
```

- Embeddings power efficient semantic search and document retrieval.

---

## Chunking & Indexing Best Practices

- Use LangChain’s `RecursiveCharacterTextSplitter`:
  - `chunk_size = 512`
  - `chunk_overlap = 60`

This balances context retention and fit for Gemini models while improving retrieval fidelity.

- Preprocess raw FAQs in `scripts/preprocess.py`
- Perform chunking, embedding, and indexing in `app/indexing.py`

---

## ChromaDB Integration

- Set up ChromaDB vector storage in `app/db.py`
- Use LangChain's `Chroma` wrapper or native Chroma client
- Persist collections locally or via remote service

```
import chromadb
from langchain.vectorstores import Chroma

client = chromadb.Client()
collection_name = "hospital_faq"

embedding_function = YOUR_EMBEDDING_CALLABLE

vector_store = Chroma(
    client=client,
    collection_name=collection_name,
    embedding_function=embedding_function,
    persist_directory="./chroma_db"
)
```

---

## Retrieval & Conversational Memory

- Implement retrieval with RAG pipeline (`app/retrieval.py`)
- Use Chroma retriever: `vector_store.as_retriever()`
- Manage user/session conversation history context in `app/memory.py` with LangChain Memory classes (`ConversationBufferMemory` or `ConversationSummaryMemory`)
- Tie memory to unique user/session IDs for continuity

---

## API & Frontend

- `app/api.py` for FastAPI REST endpoints (clean separation from logic)
- `streamlit_app/ui.py` provides Streamlit user interface; fetches chatbot answers via API

---

## Package Management & Deployment

- Use `pyproject.toml` and `uv` for lightweight package management:
  
  ```
  uv pip install .
  ```

- Dockerfile supports multi-stage build with UV setup
- Use environment variables for secrets; `.env.example` provided

---

## Running the Project

- Start backend API server:

  ```
  python app/main.py
  ```

- Run Streamlit frontend web app:

  ```
  streamlit run streamlit_app/app.py
  ```

- Index or re-index documents after preprocessing:

  ```
  python app/indexing.py
  ```

---

## Developer Guidelines for Copilot & Contributors

- Enforce modular code practices; one core functionality per file
- Use type annotations and PEP8 style consistently
- Document all major functions and classes with docstrings
- Never hardcode API keys or secrets; always use environment configuration
- Leverage Gemini 2.5 Flash and `gemini-embedding-001` for synergy and optimal performance
- Regularly update ChromaDB indices when documents change
- All changes should pass testing before merging

---

## References & Resources

- [Google Gemini API Embeddings Documentation](https://ai.google.dev/gemini-api/docs/embeddings)
- [LangChain Chroma Integration](https://python.langchain.com/docs/integrations/vectorstores/chroma/)
- [LangChain Chatbot Tutorial](https://python.langchain.com/docs/tutorials/chatbot/)
- [Google Gemini Models Overview](https://cloud.google.com/vertex-ai/generative-ai/docs/models)

---

*This instruction file ensures a clean, modern, and scalable architecture for a hospital FAQ chatbot built with LangChain and Gemini 2.5 Flash, streamlined for GitHub Copilot and collaborative development.*
```

[1](https://docs.github.com/copilot/customizing-copilot/adding-custom-instructions-for-github-copilot)
[2](https://www.freecodecamp.org/news/github-flavored-markdown-syntax-examples/)
[3](https://docs.github.com/en/get-started/writing-on-github/getting-started-with-writing-and-formatting-on-github/quickstart-for-writing-on-github)
[4](https://gist.github.com/allysonsilva/85fff14a22bbdf55485be947566cc09e)
[5](https://docs.github.com/github/writing-on-github/getting-started-with-writing-and-formatting-on-github/basic-writing-and-formatting-syntax)
[6](https://www.markdownguide.org/basic-syntax/)
[7](https://www.reddit.com/r/GithubCopilot/comments/1llss4p/this_is_my_generalinstructionsmd_file_to_use_with/)
[8](https://github.com/adam-p/markdown-here/wiki/markdown-cheatsheet)
[9](https://google.github.io/styleguide/docguide/style.html)