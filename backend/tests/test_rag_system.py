import pytest
from unittest.mock import Mock, patch, MagicMock
from rag_system import RAGSystem
from vector_store import SearchResults
from search_tools import CourseSearchTool, ToolManager
from ai_generator import AIGenerator
from session_manager import SessionManager

class TestRAGSystemIntegration:
    """Test suite for RAG system end-to-end functionality"""
    
    def test_query_course_content_question(self, mock_rag_system):
        """Test RAG system handling course content questions"""
        # Configure AI generator to return response with sources
        mock_rag_system._mock_ai_generator.generate_response.return_value = "Course content response"
        
        # Configure tool manager to return sources
        mock_rag_system._mock_tool_manager.get_last_sources.return_value = [
            {"text": "Test Course - Lesson 1", "url": "https://test.com/lesson1"}
        ]
        
        response, sources = mock_rag_system.query("What does the course cover about computer use?")
        
        assert response == "Course content response"
        assert len(sources) == 1
        assert sources[0]["text"] == "Test Course - Lesson 1"
        
        # Verify AI generator was called with tools
        mock_rag_system._mock_ai_generator.generate_response.assert_called_once()
        call_args = mock_rag_system._mock_ai_generator.generate_response.call_args
        
        assert "tools" in call_args.kwargs
        assert "tool_manager" in call_args.kwargs
    
    def test_query_with_session_history(self, mock_rag_system):
        """Test RAG system with conversation history"""
        session_id = "test_session"
        mock_rag_system._mock_session_manager.get_conversation_history.return_value = "Previous conversation context"
        mock_rag_system._mock_ai_generator.generate_response.return_value = "Response with context"
        mock_rag_system._mock_tool_manager.get_last_sources.return_value = []
        
        response, sources = mock_rag_system.query("Follow-up question", session_id=session_id)
        
        assert response == "Response with context"
        
        # Verify conversation history was retrieved and used
        mock_rag_system._mock_session_manager.get_conversation_history.assert_called_once_with(session_id)
        
        # Verify AI generator received history
        call_args = mock_rag_system._mock_ai_generator.generate_response.call_args
        assert call_args.kwargs["conversation_history"] == "Previous conversation context"
        
        # Verify exchange was added to session
        mock_rag_system._mock_session_manager.add_exchange.assert_called_once_with(
            session_id, "Follow-up question", "Response with context"
        )
    
    def test_query_without_session(self, mock_rag_system):
        """Test RAG system query without session tracking"""
        mock_rag_system._mock_ai_generator.generate_response.return_value = "No session response"
        mock_rag_system._mock_tool_manager.get_last_sources.return_value = []
        
        response, sources = mock_rag_system.query("Standalone question")
        
        assert response == "No session response"
        
        # Verify no session manager calls were made
        mock_rag_system._mock_session_manager.get_conversation_history.assert_not_called()
        mock_rag_system._mock_session_manager.add_exchange.assert_not_called()
        
        # Verify AI generator was called without history
        call_args = mock_rag_system._mock_ai_generator.generate_response.call_args
        assert call_args.kwargs["conversation_history"] is None
    
    def test_query_prompt_formatting(self, mock_rag_system):
        """Test that user queries are properly formatted for the AI"""
        mock_rag_system._mock_ai_generator.generate_response.return_value = "Test response"
        mock_rag_system._mock_tool_manager.get_last_sources.return_value = []
        
        user_query = "How do I implement computer use?"
        mock_rag_system.query(user_query)
        
        # Verify the prompt was formatted correctly
        call_args = mock_rag_system._mock_ai_generator.generate_response.call_args
        formatted_query = call_args.kwargs["query"]
        
        assert f"Answer this question about course materials: {user_query}" == formatted_query
    
    def test_sources_reset_after_query(self, mock_rag_system):
        """Test that sources are reset after each query"""
        mock_rag_system._mock_ai_generator.generate_response.return_value = "Test response"
        mock_rag_system._mock_tool_manager.get_last_sources.return_value = [{"text": "test", "url": "test.com"}]
        
        mock_rag_system.query("Test question")
        
        # Verify sources were retrieved and then reset
        mock_rag_system._mock_tool_manager.get_last_sources.assert_called_once()
        mock_rag_system._mock_tool_manager.reset_sources.assert_called_once()
    
    def test_tool_manager_setup(self, mock_rag_system):
        """Test that tool manager is properly configured with search tools"""
        # Verify tool manager exists and tools are set up
        assert hasattr(mock_rag_system, 'tool_manager')
        assert hasattr(mock_rag_system, 'search_tool')
        assert hasattr(mock_rag_system, 'outline_tool')

class TestRAGSystemContentQueryHandling:
    """Simplified tests for content query handling"""
    
    def test_query_triggers_ai_generator_with_tools(self, mock_rag_system):
        """Test that queries trigger AI generator with proper tool setup"""
        mock_rag_system._mock_ai_generator.generate_response.return_value = "AI response"
        mock_rag_system._mock_tool_manager.get_last_sources.return_value = []
        
        response, sources = mock_rag_system.query("Test course question")
        
        # Verify AI generator was called with tools
        call_args = mock_rag_system._mock_ai_generator.generate_response.call_args
        assert "tools" in call_args.kwargs
        assert "tool_manager" in call_args.kwargs
        assert response == "AI response"

class TestRAGSystemAnalytics:
    """Test RAG system analytics and metadata functionality"""
    
    def test_get_course_analytics(self, mock_rag_system):
        """Test course analytics retrieval"""
        mock_rag_system._mock_vector_store.get_course_count.return_value = 5
        mock_rag_system._mock_vector_store.get_existing_course_titles.return_value = [
            "Course 1", "Course 2", "Course 3", "Course 4", "Course 5"
        ]
        
        analytics = mock_rag_system.get_course_analytics()
        
        assert analytics["total_courses"] == 5
        assert len(analytics["course_titles"]) == 5
        assert "Course 1" in analytics["course_titles"]