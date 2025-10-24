# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Development Commands

### Running the Application
```bash
# Quick start (requires ANTHROPIC_API_KEY in .env)
./run.sh

# Manual start
cd backend && uv run uvicorn app:app --reload --port 8000
```

### Setup Requirements
- Python 3.13+ with `uv` package manager
- `.env` file in root with `ANTHROPIC_API_KEY=your_key_here`
- Application runs on http://localhost:8000
- Dependencies in root `pyproject.toml` (NOT backend/pyproject.toml)

### Testing
```bash
# Run all tests
uv run pytest

# Run specific test file
uv run pytest backend/tests/test_rag_system.py

# Run single test
uv run pytest backend/tests/test_rag_system.py::TestRAGSystem::test_query_basic_flow

# Run by marker
uv run pytest -m unit        # Unit tests only
uv run pytest -m integration # Integration tests only
uv run pytest -m api         # API endpoint tests only

# Verbose output with print statements
uv run pytest -v -s
```

### Dependencies
- Dependencies managed via root `pyproject.toml` and `uv.lock`
- Install: `uv sync`
- Key dependencies: FastAPI, ChromaDB, Anthropic SDK, Sentence Transformers
- Test dependencies: pytest, pytest-mock, pytest-asyncio, httpx

## Architecture Overview

### Project Structure
```
├── pyproject.toml           # Root dependencies (use this, not backend/pyproject.toml)
├── backend/                 # Python backend (all .py files here)
│   ├── app.py              # FastAPI application entry point
│   ├── rag_system.py       # Main RAG orchestrator
│   ├── vector_store.py     # ChromaDB dual-collection storage
│   ├── search_tools.py     # Claude tool definitions
│   ├── ai_generator.py     # Claude API integration
│   ├── document_processor.py
│   ├── session_manager.py
│   ├── models.py           # Pydantic data models
│   ├── config.py           # Configuration settings
│   └── tests/              # Test suite (70+ tests)
├── frontend/               # Static HTML/CSS/JS served by FastAPI
│   ├── index.html
│   ├── script.js
│   └── style.css
└── docs/                   # Course documents (auto-loaded on startup)
```

### Core RAG System Flow
The system follows a 5-component RAG architecture orchestrated by `rag_system.py`:

1. **Document Processing** (`document_processor.py`): Extracts structured course metadata and chunks content
2. **Vector Storage** (`vector_store.py`): Dual ChromaDB collections for course catalog + content chunks
3. **Search Tools** (`search_tools.py`): Tool-based semantic search with course name resolution
4. **AI Generation** (`ai_generator.py`): Claude API integration with tool calling
5. **Session Management** (`session_manager.py`): Conversation history tracking

### Key Data Flow
```
Course Documents → DocumentProcessor → VectorStore (2 collections)
                                   ↓
User Query → RAGSystem → SearchTool → VectorStore.search() → AI Generation → Response
             ↓
        FastAPI app.py → Frontend (index.html)
```

### Critical Architecture Details

**Vector Store Design (`vector_store.py`)**:
- Uses **dual collections**: `course_catalog` (metadata) + `course_content` (chunks)
- Course name resolution via semantic search on catalog before content search
- Supports filtering by course title and lesson number
- Course titles serve as unique identifiers throughout system

**Tool-Based Search (`search_tools.py`)**:
- Claude uses `search_course_content` tool for all content queries
- Handles partial course name matching (e.g., "MCP" finds "MCP Server Development")  
- Formats results with course/lesson context headers
- Tracks sources for frontend display

**Document Structure (`document_processor.py`)**:
Expected course document format:
```
Course Title: [title]
Course Link: [url]
Course Instructor: [instructor]

Lesson X: [lesson title]
Lesson Link: [url]
[content...]

Lesson Y: [lesson title]
[content...]
```

**Configuration (`config.py`)**:
- Chunk size: 800 chars, overlap: 100 chars
- Uses Claude Sonnet 4 model
- ChromaDB path: `./chroma_db`
- Max conversation history: 2 exchanges

### Component Interactions

**RAGSystem orchestrates** (`rag_system.py`):
- Document ingestion via `add_course_folder()` - processes `/docs` on startup
- Query processing via `query()` - coordinates search tools + AI generation
- Prevents duplicate course loading by checking existing titles
- Returns (answer, sources) tuple for API responses

**VectorStore provides** (`vector_store.py`):
- `search()` - unified interface with course name resolution
- `add_course_metadata()` / `add_course_content()` - dual collection storage
- Course analytics and metadata retrieval
- Handles fuzzy course name matching via semantic search

**AI Generator handles** (`ai_generator.py`):
- Tool execution workflow with Claude API
- System prompt emphasizes search tool usage for course queries
- Response generation with conversation history context
- Iterative tool-use loop until final text response

**FastAPI Application** (`app.py`):
- POST `/api/query` - Main query endpoint, returns answer + sources
- GET `/api/courses` - Course statistics endpoint
- Serves static frontend from `/frontend` directory at root `/`
- CORS enabled for development

## Important Implementation Notes

### Core Conventions
- **Course titles are unique IDs** across all components (metadata, chunks, search)
- **Chunk content includes prefixes** with course/lesson context for better retrieval
- **Always use `uv`** for all Python operations (never `pip` or direct `python`)
  - Run server: `uv run uvicorn app:app`
  - Run tests: `uv run pytest`
  - Execute scripts: `uv run python script.py`

### Data Flow Details
- `ToolManager` (`search_tools.py`) tracks search sources for frontend display
- `SessionManager` limits history to prevent context overflow (configurable in `config.py`)
- Document processor expects specific format in `/docs` folder (see Document Structure above)
- ChromaDB persists to `./backend/chroma_db/` directory
- Embedding model downloads on first run (~400MB for sentence-transformers)

### Testing Architecture
- `conftest.py` provides comprehensive fixtures for all components
- Mocks available for: VectorStore, Anthropic client, RAG system, tool responses
- Tests organized by markers: `@pytest.mark.unit`, `@pytest.mark.integration`, `@pytest.mark.api`
- API tests use FastAPI TestClient (no server startup needed)
- always use uv to run the server.  do not use pip directly.
- make sure to use uv to manage all dependencies