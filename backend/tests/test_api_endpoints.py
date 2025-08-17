"""
API endpoint tests for the RAG system FastAPI application.

This module tests the FastAPI endpoints for proper request/response handling,
error conditions, and integration with the underlying RAG system.
"""

import pytest
from fastapi.testclient import TestClient
from unittest.mock import Mock, patch
import json


@pytest.mark.api
class TestQueryEndpoint:
    """Test the /api/query POST endpoint"""
    
    def test_query_with_session_id(self, test_client, api_test_data):
        """Test successful query with provided session ID"""
        # Setup mock response
        mock_rag = test_client.app.state.mock_rag_system
        mock_rag.query.return_value = api_test_data["mock_query_response"]
        
        # Make request
        response = test_client.post(
            "/api/query",
            json=api_test_data["valid_query"]
        )
        
        # Assertions
        assert response.status_code == 200
        data = response.json()
        
        assert "answer" in data
        assert "sources" in data
        assert "session_id" in data
        assert data["session_id"] == api_test_data["valid_query"]["session_id"]
        assert data["answer"] == api_test_data["mock_query_response"][0]
        assert data["sources"] == api_test_data["mock_query_response"][1]
        
        # Verify RAG system was called correctly
        mock_rag.query.assert_called_once_with(
            api_test_data["valid_query"]["query"],
            api_test_data["valid_query"]["session_id"]
        )
    
    def test_query_without_session_id(self, test_client, api_test_data):
        """Test query without session ID generates one automatically"""
        # Setup mock response
        mock_rag = test_client.app.state.mock_rag_system
        mock_rag.query.return_value = api_test_data["mock_query_response"]
        
        # Make request
        response = test_client.post(
            "/api/query",
            json=api_test_data["valid_query_no_session"]
        )
        
        # Assertions
        assert response.status_code == 200
        data = response.json()
        
        assert "answer" in data
        assert "sources" in data
        assert "session_id" in data
        assert data["session_id"] == "test-session-123"  # Default from test fixture
        
        # Verify RAG system was called with generated session ID
        mock_rag.query.assert_called_once_with(
            api_test_data["valid_query_no_session"]["query"],
            "test-session-123"
        )
    
    def test_query_with_empty_string(self, test_client, api_test_data):
        """Test query with empty string"""
        # Setup mock response
        mock_rag = test_client.app.state.mock_rag_system
        mock_rag.query.return_value = ("Please provide a valid query.", [])
        
        # Make request
        response = test_client.post(
            "/api/query",
            json=api_test_data["empty_query"]
        )
        
        # Assertions
        assert response.status_code == 200
        data = response.json()
        
        assert "answer" in data
        assert "sources" in data
        assert "session_id" in data
    
    def test_query_with_rag_system_error(self, test_client, api_test_data):
        """Test query when RAG system raises an exception"""
        # Setup mock to raise exception
        mock_rag = test_client.app.state.mock_rag_system
        mock_rag.query.side_effect = Exception("RAG system error")
        
        # Make request
        response = test_client.post(
            "/api/query",
            json=api_test_data["valid_query"]
        )
        
        # Assertions
        assert response.status_code == 500
        data = response.json()
        assert "detail" in data
        assert "RAG system error" in data["detail"]
    
    def test_query_with_invalid_json(self, test_client):
        """Test query with malformed JSON"""
        response = test_client.post(
            "/api/query",
            data="invalid json"
        )
        
        assert response.status_code == 422
    
    def test_query_missing_required_field(self, test_client):
        """Test query without required 'query' field"""
        response = test_client.post(
            "/api/query",
            json={"session_id": "test-123"}
        )
        
        assert response.status_code == 422
        data = response.json()
        assert "detail" in data


@pytest.mark.api
class TestCoursesEndpoint:
    """Test the /api/courses GET endpoint"""
    
    def test_get_course_stats_success(self, test_client, api_test_data):
        """Test successful retrieval of course statistics"""
        # Setup mock response
        mock_rag = test_client.app.state.mock_rag_system
        mock_rag.get_course_analytics.return_value = api_test_data["mock_analytics"]
        
        # Make request
        response = test_client.get("/api/courses")
        
        # Assertions
        assert response.status_code == 200
        data = response.json()
        
        assert "total_courses" in data
        assert "course_titles" in data
        assert data["total_courses"] == api_test_data["mock_analytics"]["total_courses"]
        assert data["course_titles"] == api_test_data["mock_analytics"]["course_titles"]
        
        # Verify RAG system was called
        mock_rag.get_course_analytics.assert_called_once()
    
    def test_get_course_stats_with_rag_system_error(self, test_client):
        """Test course stats when RAG system raises an exception"""
        # Setup mock to raise exception
        mock_rag = test_client.app.state.mock_rag_system
        mock_rag.get_course_analytics.side_effect = Exception("Database connection failed")
        
        # Make request
        response = test_client.get("/api/courses")
        
        # Assertions
        assert response.status_code == 500
        data = response.json()
        assert "detail" in data
        assert "Database connection failed" in data["detail"]
    
    def test_get_course_stats_empty_response(self, test_client):
        """Test course stats with empty analytics response"""
        # Setup mock response
        mock_rag = test_client.app.state.mock_rag_system
        mock_rag.get_course_analytics.return_value = {
            "total_courses": 0,
            "course_titles": []
        }
        
        # Make request
        response = test_client.get("/api/courses")
        
        # Assertions
        assert response.status_code == 200
        data = response.json()
        
        assert data["total_courses"] == 0
        assert data["course_titles"] == []


@pytest.mark.api
class TestRootEndpoint:
    """Test the root / endpoint"""
    
    def test_root_endpoint(self, test_client):
        """Test the root endpoint returns correct message"""
        response = test_client.get("/")
        
        assert response.status_code == 200
        data = response.json()
        assert data == {"message": "RAG System API Test"}


@pytest.mark.api
class TestAPIResponseFormat:
    """Test API response format and headers"""
    
    def test_query_response_headers(self, test_client, api_test_data):
        """Test that query endpoint returns proper headers"""
        # Setup mock response
        mock_rag = test_client.app.state.mock_rag_system
        mock_rag.query.return_value = api_test_data["mock_query_response"]
        
        response = test_client.post(
            "/api/query",
            json=api_test_data["valid_query"]
        )
        
        assert response.status_code == 200
        assert "application/json" in response.headers.get("content-type", "")
    
    def test_courses_response_headers(self, test_client, api_test_data):
        """Test that courses endpoint returns proper headers"""
        # Setup mock response
        mock_rag = test_client.app.state.mock_rag_system
        mock_rag.get_course_analytics.return_value = api_test_data["mock_analytics"]
        
        response = test_client.get("/api/courses")
        
        assert response.status_code == 200
        assert "application/json" in response.headers.get("content-type", "")
    
    def test_cors_headers_present(self, test_client):
        """Test that CORS headers are properly set"""
        response = test_client.get("/")
        
        # CORS headers should be present due to middleware
        assert response.status_code == 200


@pytest.mark.api
class TestAPIEdgeCases:
    """Test edge cases and boundary conditions"""
    
    def test_query_with_very_long_string(self, test_client):
        """Test query with very long input string"""
        # Setup mock response
        mock_rag = test_client.app.state.mock_rag_system
        mock_rag.query.return_value = ("Response to long query", [])
        
        long_query = "What is AI? " * 1000  # Very long query
        
        response = test_client.post(
            "/api/query",
            json={"query": long_query}
        )
        
        assert response.status_code == 200
        data = response.json()
        assert "answer" in data
    
    def test_query_with_special_characters(self, test_client):
        """Test query with special characters and unicode"""
        # Setup mock response
        mock_rag = test_client.app.state.mock_rag_system
        mock_rag.query.return_value = ("Response with unicode", [])
        
        special_query = "What about AI safety? 🤖 特殊字符 test"
        
        response = test_client.post(
            "/api/query",
            json={"query": special_query}
        )
        
        assert response.status_code == 200
        data = response.json()
        assert "answer" in data
    
    def test_concurrent_requests(self, test_client, api_test_data):
        """Test handling multiple concurrent requests"""
        import concurrent.futures
        import threading
        
        # Setup mock response
        mock_rag = test_client.app.state.mock_rag_system
        mock_rag.query.return_value = api_test_data["mock_query_response"]
        
        def make_request():
            return test_client.post(
                "/api/query",
                json={"query": "Concurrent test query"}
            )
        
        # Make 5 concurrent requests
        with concurrent.futures.ThreadPoolExecutor(max_workers=5) as executor:
            futures = [executor.submit(make_request) for _ in range(5)]
            responses = [future.result() for future in concurrent.futures.as_completed(futures)]
        
        # All requests should succeed
        for response in responses:
            assert response.status_code == 200
            data = response.json()
            assert "answer" in data
            assert "sources" in data
            assert "session_id" in data


@pytest.mark.integration
@pytest.mark.api
class TestAPIIntegration:
    """Integration tests for API endpoints with realistic scenarios"""
    
    def test_query_flow_with_multiple_sources(self, test_client):
        """Test realistic query with multiple course sources"""
        # Setup realistic mock response
        mock_rag = test_client.app.state.mock_rag_system
        mock_rag.query.return_value = (
            "Computer use in AI involves systems that can interact with computer interfaces to perform tasks autonomously. This capability is being developed by companies like Anthropic to create more useful AI assistants.",
            [
                {"course_title": "Building Towards Computer Use with Anthropic", "lesson_number": 1, "source": "Introduction to Computer Use"},
                {"course_title": "Building Towards Computer Use with Anthropic", "lesson_number": 2, "source": "Technical Implementation"},
                {"course_title": "AI Safety and Alignment", "lesson_number": 3, "source": "Safety Considerations"}
            ]
        )
        
        response = test_client.post(
            "/api/query",
            json={"query": "What is computer use in AI and how does Anthropic approach it?"}
        )
        
        assert response.status_code == 200
        data = response.json()
        
        # Verify realistic response structure
        assert len(data["sources"]) == 3
        assert "Computer use" in data["answer"]
        assert "Anthropic" in data["answer"]
        
        # Verify sources contain expected fields
        for source in data["sources"]:
            assert "course_title" in source
            assert "lesson_number" in source
    
    def test_course_analytics_realistic_data(self, test_client):
        """Test course analytics with realistic course data"""
        mock_rag = test_client.app.state.mock_rag_system
        mock_rag.get_course_analytics.return_value = {
            "total_courses": 5,
            "course_titles": [
                "Building Towards Computer Use with Anthropic",
                "AI Safety and Alignment",
                "Advanced Machine Learning Techniques",
                "Natural Language Processing Fundamentals",
                "Ethics in AI Development"
            ]
        }
        
        response = test_client.get("/api/courses")
        
        assert response.status_code == 200
        data = response.json()
        
        assert data["total_courses"] == 5
        assert len(data["course_titles"]) == 5
        
        # Verify realistic course titles
        expected_courses = [
            "Building Towards Computer Use with Anthropic",
            "AI Safety and Alignment",
            "Advanced Machine Learning Techniques",
            "Natural Language Processing Fundamentals", 
            "Ethics in AI Development"
        ]
        
        for course in expected_courses:
            assert course in data["course_titles"]


# Test configuration validation
@pytest.mark.api
def test_api_test_markers():
    """Verify that API test markers are properly configured"""
    import inspect
    
    # Get all test functions in this module
    current_module = inspect.getmodule(inspect.currentframe())
    
    api_test_count = 0
    for name, obj in inspect.getmembers(current_module):
        if inspect.isclass(obj) and name.startswith('Test'):
            # Check if class has api marker
            if hasattr(obj, 'pytestmark'):
                markers = [mark.name for mark in obj.pytestmark if hasattr(mark, 'name')]
                if 'api' in markers:
                    api_test_count += 1
    
    # Ensure we have API tests
    assert api_test_count > 0, "No API test classes found with @pytest.mark.api"