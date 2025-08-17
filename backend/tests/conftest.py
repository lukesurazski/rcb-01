import pytest
import sys
import os
from unittest.mock import Mock, MagicMock, patch
from typing import List, Dict, Any
from fastapi import FastAPI
from fastapi.testclient import TestClient
import tempfile
import shutil

# Add the backend directory to the Python path
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from models import Course, Lesson, CourseChunk
from vector_store import SearchResults
from search_tools import CourseSearchTool, ToolManager
from ai_generator import AIGenerator
from rag_system import RAGSystem
from config import config

@pytest.fixture
def sample_course():
    """Create a sample course for testing"""
    return Course(
        title="Building Towards Computer Use with Anthropic",
        course_link="https://www.deeplearning.ai/short-courses/building-toward-computer-use-with-anthropic/",
        instructor="Colt Steele",
        lessons=[
            Lesson(
                lesson_number=0,
                title="Introduction",
                lesson_link="https://learn.deeplearning.ai/courses/building-toward-computer-use-with-anthropic/lesson/a6k0z/introduction"
            ),
            Lesson(
                lesson_number=1,
                title="Anthropic Background",
                lesson_link="https://learn.deeplearning.ai/courses/building-toward-computer-use-with-anthropic/lesson/b7l1m/background"
            )
        ]
    )

@pytest.fixture
def sample_course_chunks():
    """Create sample course chunks for testing"""
    return [
        CourseChunk(
            content="Lesson 0 content: Welcome to Building Toward Computer Use with Anthropic. This course teaches about computer use capabilities.",
            course_title="Building Towards Computer Use with Anthropic",
            lesson_number=0,
            chunk_index=0
        ),
        CourseChunk(
            content="More content about computer use and its applications in AI systems.",
            course_title="Building Towards Computer Use with Anthropic", 
            lesson_number=0,
            chunk_index=1
        ),
        CourseChunk(
            content="Lesson 1 content: Anthropic is an AI safety company focused on developing safe AI systems.",
            course_title="Building Towards Computer Use with Anthropic",
            lesson_number=1,
            chunk_index=2
        )
    ]

@pytest.fixture
def mock_vector_store():
    """Create a mock vector store for testing"""
    mock = Mock()
    
    # Configure default search behavior
    mock.search.return_value = SearchResults(
        documents=["Sample search result content"],
        metadata=[{"course_title": "Test Course", "lesson_number": 1}],
        distances=[0.1]
    )
    
    # Configure link methods
    mock.get_course_link.return_value = "https://test-course.com"
    mock.get_lesson_link.return_value = "https://test-course.com/lesson1"
    
    return mock

@pytest.fixture
def mock_anthropic_client():
    """Create a mock Anthropic client for testing"""
    mock_client = Mock()
    
    # Create mock response object
    mock_response = Mock()
    mock_response.stop_reason = "end_turn"
    mock_response.content = [Mock(text="Test AI response", type="text")]
    
    mock_client.messages.create.return_value = mock_response
    return mock_client

@pytest.fixture
def mock_tool_use_response():
    """Create a mock response that includes tool use"""
    mock_response = Mock()
    mock_response.stop_reason = "tool_use"
    
    # Mock tool use content block
    mock_tool_block = Mock()
    mock_tool_block.type = "tool_use"
    mock_tool_block.name = "search_course_content"
    mock_tool_block.id = "tool_123"
    mock_tool_block.input = {"query": "test query", "course_name": "Test Course"}
    
    mock_response.content = [mock_tool_block]
    return mock_response

@pytest.fixture
def sample_search_results():
    """Create sample search results for testing"""
    return SearchResults(
        documents=[
            "[Building Towards Computer Use with Anthropic - Lesson 0]\nWelcome to the course about computer use capabilities.",
            "[Building Towards Computer Use with Anthropic - Lesson 1]\nAnthropic focuses on AI safety and alignment."
        ],
        metadata=[
            {"course_title": "Building Towards Computer Use with Anthropic", "lesson_number": 0},
            {"course_title": "Building Towards Computer Use with Anthropic", "lesson_number": 1}
        ],
        distances=[0.1, 0.2]
    )

@pytest.fixture 
def empty_search_results():
    """Create empty search results for testing"""
    return SearchResults(
        documents=[],
        metadata=[],
        distances=[]
    )

@pytest.fixture
def error_search_results():
    """Create error search results for testing"""
    return SearchResults.empty("Test error message")

@pytest.fixture
def mock_config():
    """Create a mock config for testing"""
    config = Mock()
    config.CHUNK_SIZE = 800
    config.CHUNK_OVERLAP = 100
    config.CHROMA_PATH = "./test_chroma_db"
    config.EMBEDDING_MODEL = "test-model"
    config.MAX_RESULTS = 5
    config.ANTHROPIC_API_KEY = "test-key"
    config.ANTHROPIC_MODEL = "claude-sonnet-4-20250514"
    config.MAX_HISTORY = 2
    return config

@pytest.fixture
def mock_rag_system(mock_config):
    """Create a RAG system with mocked components"""
    with patch('rag_system.DocumentProcessor'), \
         patch('rag_system.VectorStore') as mock_vector_store_class, \
         patch('rag_system.AIGenerator') as mock_ai_gen_class, \
         patch('rag_system.SessionManager') as mock_session_class, \
         patch('rag_system.ToolManager') as mock_tool_manager_class:
        
        # Configure mocked classes
        mock_vector_store = Mock()
        mock_vector_store_class.return_value = mock_vector_store
        
        mock_ai_generator = Mock()
        mock_ai_gen_class.return_value = mock_ai_generator
        
        mock_session_manager = Mock()
        mock_session_class.return_value = mock_session_manager
        
        mock_tool_manager = Mock()
        mock_tool_manager_class.return_value = mock_tool_manager
        
        # Create RAG system
        rag = RAGSystem(mock_config)
        
        # Store mocks for test access
        rag._mock_vector_store = mock_vector_store
        rag._mock_ai_generator = mock_ai_generator
        rag._mock_session_manager = mock_session_manager
        rag._mock_tool_manager = mock_tool_manager
        
        return rag

@pytest.fixture
def test_app():
    """Create a test FastAPI app without static file mounting issues"""
    from fastapi import FastAPI, HTTPException
    from fastapi.middleware.cors import CORSMiddleware
    from pydantic import BaseModel
    from typing import List, Optional, Union, Dict, Any
    from unittest.mock import Mock
    
    # Create test app
    app = FastAPI(title="Test Course Materials RAG System")
    
    # Add CORS middleware
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )
    
    # Create mock RAG system for testing
    mock_rag_system = Mock()
    
    # Pydantic models
    class QueryRequest(BaseModel):
        query: str
        session_id: Optional[str] = None

    class QueryResponse(BaseModel):
        answer: str
        sources: List[Union[str, Dict[str, Any]]]
        session_id: str

    class CourseStats(BaseModel):
        total_courses: int
        course_titles: List[str]
    
    @app.post("/api/query", response_model=QueryResponse)
    async def query_documents(request: QueryRequest):
        """Test endpoint for query processing"""
        try:
            # Mock response from RAG system
            session_id = request.session_id or "test-session-123"
            answer, sources = mock_rag_system.query(request.query, session_id)
            
            return QueryResponse(
                answer=answer,
                sources=sources,
                session_id=session_id
            )
        except Exception as e:
            raise HTTPException(status_code=500, detail=str(e))

    @app.get("/api/courses", response_model=CourseStats)
    async def get_course_stats():
        """Test endpoint for course statistics"""
        try:
            analytics = mock_rag_system.get_course_analytics()
            return CourseStats(
                total_courses=analytics["total_courses"],
                course_titles=analytics["course_titles"]
            )
        except Exception as e:
            raise HTTPException(status_code=500, detail=str(e))
    
    @app.get("/")
    async def root():
        """Test root endpoint"""
        return {"message": "RAG System API Test"}
    
    # Store mock for test access
    app.state.mock_rag_system = mock_rag_system
    
    return app

@pytest.fixture
def test_client(test_app):
    """Create a test client for the FastAPI app"""
    return TestClient(test_app)

@pytest.fixture
def temp_chroma_db():
    """Create a temporary ChromaDB directory for testing"""
    temp_dir = tempfile.mkdtemp(prefix="test_chroma_")
    yield temp_dir
    shutil.rmtree(temp_dir, ignore_errors=True)

@pytest.fixture
def mock_anthropic_response():
    """Create a complete mock response for Anthropic API"""
    mock_response = Mock()
    mock_response.stop_reason = "end_turn"
    mock_response.content = [Mock(text="This is a test response from Claude.", type="text")]
    mock_response.usage = Mock(input_tokens=100, output_tokens=50)
    return mock_response

@pytest.fixture
def api_test_data():
    """Common test data for API endpoints"""
    return {
        "valid_query": {
            "query": "What is computer use in AI?",
            "session_id": "test-session-123"
        },
        "valid_query_no_session": {
            "query": "Tell me about Anthropic's AI safety work"
        },
        "empty_query": {
            "query": ""
        },
        "mock_query_response": (
            "Computer use refers to AI systems' ability to interact with computer interfaces.",
            [
                {"course_title": "Building Towards Computer Use", "lesson_number": 1},
                {"course_title": "AI Safety Course", "lesson_number": 2}
            ]
        ),
        "mock_analytics": {
            "total_courses": 3,
            "course_titles": [
                "Building Towards Computer Use with Anthropic",
                "AI Safety and Alignment",
                "Advanced Machine Learning"
            ]
        }
    }