import pytest
import sys
import os
from unittest.mock import Mock, MagicMock, patch
from typing import List, Dict, Any

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