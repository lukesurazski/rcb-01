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

### Dependencies
- Dependencies managed via `pyproject.toml` and `uv.lock`
- Install: `uv sync`
- Key dependencies: FastAPI, ChromaDB, Anthropic SDK, Sentence Transformers

### Code Quality Tools
```bash
# Format code with Black and isort
./scripts/format.sh

# Run linting checks with flake8
./scripts/lint.sh

# Run complete quality checks (format + lint + tests)
./scripts/quality.sh
```

**Formatting Standards**:
- Black for code formatting (88 character line length)
- isort for import sorting (Black-compatible profile)
- flake8 for linting with Black-compatible settings

## Architecture Overview

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

**RAGSystem orchestrates**:
- Document ingestion via `add_course_folder()` - processes `/docs` on startup
- Query processing via `query()` - coordinates search tools + AI generation
- Prevents duplicate course loading by checking existing titles

**VectorStore provides**:
- `search()` - unified interface with course name resolution
- `add_course_metadata()` / `add_course_content()` - dual collection storage
- Course analytics and metadata retrieval

**AI Generator handles**:
- Tool execution workflow with Claude
- System prompt emphasizes search tool usage for course queries
- Response generation with conversation history context

## Important Implementation Notes

- Course titles are used as unique IDs across all components
- Chunk content includes course/lesson context prefixes for better retrieval
- Tool manager tracks search sources for frontend source display
- Session manager limits history to prevent context overflow
- FastAPI serves both API endpoints and static frontend files
- CORS configured for development with wildcard origins
- always use uv to run the server do not use pip directly
- make sure to use uv to manage all dependencies
- use uv to run python files