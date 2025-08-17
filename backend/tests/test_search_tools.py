import pytest
from unittest.mock import Mock, patch
from search_tools import CourseSearchTool, ToolManager
from vector_store import SearchResults

class TestCourseSearchTool:
    """Test suite for CourseSearchTool.execute method"""
    
    def test_execute_basic_query_success(self, mock_vector_store, sample_search_results):
        """Test execute with basic query returning results"""
        mock_vector_store.search.return_value = sample_search_results
        
        tool = CourseSearchTool(mock_vector_store)
        result = tool.execute("test query")
        
        mock_vector_store.search.assert_called_once_with(
            query="test query",
            course_name=None,
            lesson_number=None
        )
        
        assert "[Building Towards Computer Use with Anthropic - Lesson 0]" in result
        assert "Welcome to the course about computer use capabilities." in result
        assert "[Building Towards Computer Use with Anthropic - Lesson 1]" in result
        assert "Anthropic focuses on AI safety and alignment." in result
    
    def test_execute_with_course_filter(self, mock_vector_store, sample_search_results):
        """Test execute with course name filter"""
        mock_vector_store.search.return_value = sample_search_results
        
        tool = CourseSearchTool(mock_vector_store)
        result = tool.execute("test query", course_name="Computer Use")
        
        mock_vector_store.search.assert_called_once_with(
            query="test query",
            course_name="Computer Use",
            lesson_number=None
        )
        
        assert "Computer Use" in result
    
    def test_execute_with_lesson_filter(self, mock_vector_store, sample_search_results):
        """Test execute with lesson number filter"""
        mock_vector_store.search.return_value = sample_search_results
        
        tool = CourseSearchTool(mock_vector_store)
        result = tool.execute("test query", lesson_number=1)
        
        mock_vector_store.search.assert_called_once_with(
            query="test query",
            course_name=None,
            lesson_number=1
        )
        
        assert "Lesson 1" in result
    
    def test_execute_with_both_filters(self, mock_vector_store, sample_search_results):
        """Test execute with both course name and lesson number filters"""
        mock_vector_store.search.return_value = sample_search_results
        
        tool = CourseSearchTool(mock_vector_store)
        result = tool.execute("test query", course_name="Computer Use", lesson_number=1)
        
        mock_vector_store.search.assert_called_once_with(
            query="test query",
            course_name="Computer Use", 
            lesson_number=1
        )
        
        assert "Computer Use" in result and "Lesson 1" in result
    
    def test_execute_empty_results(self, mock_vector_store, empty_search_results):
        """Test execute when no results are found"""
        mock_vector_store.search.return_value = empty_search_results
        
        tool = CourseSearchTool(mock_vector_store)
        result = tool.execute("nonexistent query")
        
        assert result == "No relevant content found."
    
    def test_execute_empty_results_with_course_filter(self, mock_vector_store, empty_search_results):
        """Test execute with course filter when no results found"""
        mock_vector_store.search.return_value = empty_search_results
        
        tool = CourseSearchTool(mock_vector_store)
        result = tool.execute("test query", course_name="Nonexistent Course")
        
        assert result == "No relevant content found in course 'Nonexistent Course'."
    
    def test_execute_empty_results_with_lesson_filter(self, mock_vector_store, empty_search_results):
        """Test execute with lesson filter when no results found"""
        mock_vector_store.search.return_value = empty_search_results
        
        tool = CourseSearchTool(mock_vector_store)
        result = tool.execute("test query", lesson_number=99)
        
        assert result == "No relevant content found in lesson 99."
    
    def test_execute_empty_results_with_both_filters(self, mock_vector_store, empty_search_results):
        """Test execute with both filters when no results found"""
        mock_vector_store.search.return_value = empty_search_results
        
        tool = CourseSearchTool(mock_vector_store)
        result = tool.execute("test query", course_name="Test Course", lesson_number=5)
        
        assert result == "No relevant content found in course 'Test Course' in lesson 5."
    
    def test_execute_error_handling(self, mock_vector_store, error_search_results):
        """Test execute when vector store returns error"""
        mock_vector_store.search.return_value = error_search_results
        
        tool = CourseSearchTool(mock_vector_store)
        result = tool.execute("test query")
        
        assert result == "Test error message"
    
    def test_last_sources_tracking_with_links(self, mock_vector_store, sample_search_results):
        """Test that last_sources properly tracks sources with links"""
        mock_vector_store.search.return_value = sample_search_results
        mock_vector_store.get_lesson_link.side_effect = [
            "https://test.com/lesson0",
            "https://test.com/lesson1"
        ]
        
        tool = CourseSearchTool(mock_vector_store)
        tool.execute("test query")
        
        assert len(tool.last_sources) == 2
        assert tool.last_sources[0]["text"] == "Building Towards Computer Use with Anthropic - Lesson 0"
        assert tool.last_sources[0]["url"] == "https://test.com/lesson0"
        assert tool.last_sources[1]["text"] == "Building Towards Computer Use with Anthropic - Lesson 1"
        assert tool.last_sources[1]["url"] == "https://test.com/lesson1"
    
    def test_last_sources_tracking_without_links(self, mock_vector_store, sample_search_results):
        """Test last_sources when no links are available"""
        mock_vector_store.search.return_value = sample_search_results
        mock_vector_store.get_lesson_link.return_value = None
        
        tool = CourseSearchTool(mock_vector_store)
        tool.execute("test query")
        
        assert len(tool.last_sources) == 2
        assert tool.last_sources[0]["url"] is None
        assert tool.last_sources[1]["url"] is None
    
    def test_format_results_structure(self, mock_vector_store):
        """Test the _format_results method output structure"""
        results = SearchResults(
            documents=["Test content 1", "Test content 2"],
            metadata=[
                {"course_title": "Test Course", "lesson_number": 1},
                {"course_title": "Test Course", "lesson_number": 2}
            ],
            distances=[0.1, 0.2]
        )
        
        tool = CourseSearchTool(mock_vector_store)
        formatted = tool._format_results(results)
        
        assert "[Test Course - Lesson 1]" in formatted
        assert "[Test Course - Lesson 2]" in formatted
        assert "Test content 1" in formatted
        assert "Test content 2" in formatted
        
        # Results should be separated by double newlines
        assert "\n\n" in formatted
    
    def test_format_results_without_lesson_number(self, mock_vector_store):
        """Test formatting when lesson_number is missing"""
        results = SearchResults(
            documents=["Course overview content"],
            metadata=[{"course_title": "Test Course"}],
            distances=[0.1]
        )
        
        tool = CourseSearchTool(mock_vector_store)
        formatted = tool._format_results(results)
        
        assert "[Test Course]" in formatted
        assert "Course overview content" in formatted
    
    def test_get_tool_definition(self, mock_vector_store):
        """Test tool definition structure"""
        tool = CourseSearchTool(mock_vector_store)
        definition = tool.get_tool_definition()
        
        assert definition["name"] == "search_course_content"
        assert "description" in definition
        assert "input_schema" in definition
        
        schema = definition["input_schema"]
        assert schema["type"] == "object"
        assert "query" in schema["properties"]
        assert "course_name" in schema["properties"]
        assert "lesson_number" in schema["properties"]
        assert schema["required"] == ["query"]

class TestToolManager:
    """Test suite for ToolManager functionality"""
    
    def test_register_tool(self, mock_vector_store):
        """Test tool registration"""
        manager = ToolManager()
        tool = CourseSearchTool(mock_vector_store)
        
        manager.register_tool(tool)
        
        assert "search_course_content" in manager.tools
        assert manager.tools["search_course_content"] == tool
    
    def test_get_tool_definitions(self, mock_vector_store):
        """Test getting all tool definitions"""
        manager = ToolManager()
        tool = CourseSearchTool(mock_vector_store)
        manager.register_tool(tool)
        
        definitions = manager.get_tool_definitions()
        
        assert len(definitions) == 1
        assert definitions[0]["name"] == "search_course_content"
    
    def test_execute_tool_success(self, mock_vector_store, sample_search_results):
        """Test successful tool execution"""
        mock_vector_store.search.return_value = sample_search_results
        
        manager = ToolManager()
        tool = CourseSearchTool(mock_vector_store)
        manager.register_tool(tool)
        
        result = manager.execute_tool("search_course_content", query="test")
        
        assert "Computer Use" in result
        mock_vector_store.search.assert_called_once()
    
    def test_execute_tool_not_found(self):
        """Test execution of non-existent tool"""
        manager = ToolManager()
        
        result = manager.execute_tool("nonexistent_tool", query="test")
        
        assert result == "Tool 'nonexistent_tool' not found"
    
    def test_get_last_sources(self, mock_vector_store, sample_search_results):
        """Test retrieving sources from last search"""
        mock_vector_store.search.return_value = sample_search_results
        mock_vector_store.get_lesson_link.return_value = "https://test.com/lesson"
        
        manager = ToolManager()
        tool = CourseSearchTool(mock_vector_store)
        manager.register_tool(tool)
        
        # Execute search to populate sources
        manager.execute_tool("search_course_content", query="test")
        
        sources = manager.get_last_sources()
        assert len(sources) > 0
    
    def test_reset_sources(self, mock_vector_store):
        """Test resetting sources across all tools"""
        manager = ToolManager()
        tool = CourseSearchTool(mock_vector_store)
        tool.last_sources = [{"text": "test", "url": "test.com"}]
        manager.register_tool(tool)
        
        manager.reset_sources()
        
        assert tool.last_sources == []