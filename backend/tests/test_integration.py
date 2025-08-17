import pytest
import tempfile
import shutil
from unittest.mock import patch, Mock
from rag_system import RAGSystem
from config import Config
from search_tools import CourseSearchTool
from vector_store import VectorStore

class TestRealSystemIntegration:
    """Integration tests against real components to identify actual issues"""
    
    @pytest.fixture
    def temp_config(self):
        """Create a temporary config for real testing"""
        # Create temporary directory for test database
        temp_dir = tempfile.mkdtemp()
        
        config = Config()
        config.CHROMA_PATH = f"{temp_dir}/test_chroma"
        config.ANTHROPIC_API_KEY = "test-key-fake"  # Will be mocked
        config.MAX_RESULTS = 3
        config.COURSE_NAME_SIMILARITY_THRESHOLD = 1.6
        config.CONTENT_RELEVANCE_THRESHOLD = 1.8
        
        yield config
        
        # Cleanup
        shutil.rmtree(temp_dir, ignore_errors=True)
    
    @pytest.fixture
    def real_vector_store(self, temp_config):
        """Create a real vector store for testing"""
        return VectorStore(
            chroma_path=temp_config.CHROMA_PATH,
            embedding_model=temp_config.EMBEDDING_MODEL,
            max_results=temp_config.MAX_RESULTS,
            course_threshold=temp_config.COURSE_NAME_SIMILARITY_THRESHOLD,
            content_threshold=temp_config.CONTENT_RELEVANCE_THRESHOLD
        )
    
    @pytest.fixture
    def populated_vector_store(self, real_vector_store, sample_course, sample_course_chunks):
        """Create a vector store with test data"""
        real_vector_store.add_course_metadata(sample_course)
        real_vector_store.add_course_content(sample_course_chunks)
        return real_vector_store
    
    def test_course_search_tool_with_real_vector_store(self, populated_vector_store):
        """Test CourseSearchTool against real vector store"""
        tool = CourseSearchTool(populated_vector_store)
        
        # Test basic search
        result = tool.execute("computer use")
        
        # Should find content related to computer use
        assert "Building Towards Computer Use" in result
        assert len(tool.last_sources) > 0
        
        # Test course-specific search
        result = tool.execute("computer use", course_name="Building Towards")
        assert "Building Towards Computer Use" in result
        
        # Test lesson-specific search
        result = tool.execute("content", lesson_number=0)
        assert "Lesson 0" in result
    
    def test_course_search_tool_no_results(self, populated_vector_store):
        """Test CourseSearchTool when no results are found due to relevance threshold"""
        tool = CourseSearchTool(populated_vector_store)
        
        result = tool.execute("completely unrelated quantum physics")
        
        # Should now properly filter out irrelevant results
        assert "No relevant content found" in result
        assert len(tool.last_sources) == 0
    
    def test_course_search_tool_invalid_course(self, populated_vector_store):
        """Test CourseSearchTool with invalid course name"""
        tool = CourseSearchTool(populated_vector_store)
        
        result = tool.execute("test", course_name="Nonexistent Course")
        
        # Should now properly reject invalid course names due to threshold
        assert "No course found matching" in result
    
    @patch('anthropic.Anthropic')
    def test_rag_system_with_mocked_ai(self, mock_anthropic_class, temp_config, sample_course, sample_course_chunks):
        """Test RAG system with real vector store but mocked AI"""
        # Configure mock AI client
        mock_client = Mock()
        mock_response = Mock()
        mock_response.stop_reason = "end_turn"
        mock_response.content = [Mock(text="Mocked AI response about computer use")]
        mock_client.messages.create.return_value = mock_response
        mock_anthropic_class.return_value = mock_client
        
        # Create RAG system
        rag = RAGSystem(temp_config)
        
        # Add test data
        rag.vector_store.add_course_metadata(sample_course)
        rag.vector_store.add_course_content(sample_course_chunks)
        
        # Test query
        response, sources = rag.query("What does the course teach about computer use?")
        
        assert response == "Mocked AI response about computer use"
        
        # Verify AI was called with tools
        mock_client.messages.create.assert_called()
        call_args = mock_client.messages.create.call_args
        assert "tools" in call_args.kwargs
    
    @patch('anthropic.Anthropic')
    def test_rag_system_tool_execution_flow(self, mock_anthropic_class, temp_config, sample_course, sample_course_chunks):
        """Test RAG system when AI actually tries to use tools"""
        # Configure mock AI client to return tool use first, then final response
        mock_client = Mock()
        
        # First response: tool use
        mock_tool_response = Mock()
        mock_tool_response.stop_reason = "tool_use"
        mock_tool_block = Mock()
        mock_tool_block.type = "tool_use"
        mock_tool_block.name = "search_course_content"
        mock_tool_block.id = "tool_123"
        mock_tool_block.input = {"query": "computer use"}
        mock_tool_response.content = [mock_tool_block]
        
        # Second response: final answer
        mock_final_response = Mock()
        mock_final_response.content = [Mock(text="Based on the search, computer use involves...")]
        
        mock_client.messages.create.side_effect = [mock_tool_response, mock_final_response]
        mock_anthropic_class.return_value = mock_client
        
        # Create RAG system with real data
        rag = RAGSystem(temp_config)
        rag.vector_store.add_course_metadata(sample_course)
        rag.vector_store.add_course_content(sample_course_chunks)
        
        # Test query that should trigger tool use
        response, sources = rag.query("What does the course teach about computer use?")
        
        assert response == "Based on the search, computer use involves..."
        
        # Verify two API calls were made (tool use + final response)
        assert mock_client.messages.create.call_count == 2
        
        # Should have sources from the search
        assert len(sources) > 0
    
    def test_vector_store_search_functionality(self, populated_vector_store):
        """Test vector store search capabilities directly"""
        # Test basic search
        results = populated_vector_store.search("computer use")
        assert not results.is_empty()
        assert len(results.documents) > 0
        
        # Test course name resolution
        results = populated_vector_store.search("content", course_name="Building Towards")
        assert not results.is_empty()
        
        # Test lesson filtering
        results = populated_vector_store.search("content", lesson_number=0)
        assert not results.is_empty()
        
        # Test combined filtering
        results = populated_vector_store.search("content", course_name="Building Towards", lesson_number=0)
        assert not results.is_empty()
    
    def test_vector_store_course_name_resolution(self, populated_vector_store):
        """Test vector store's course name resolution capability"""
        # Test partial course name matching (should work - high similarity)
        resolved = populated_vector_store._resolve_course_name("Building")
        assert resolved == "Building Towards Computer Use with Anthropic"
        
        # Test exact match
        resolved = populated_vector_store._resolve_course_name("Building Towards Computer Use with Anthropic")
        assert resolved == "Building Towards Computer Use with Anthropic"
        
        # Test non-existent course (should now be rejected due to threshold)
        resolved = populated_vector_store._resolve_course_name("Nonexistent Course")
        assert resolved is None
    
    def test_search_tool_real_vector_store_error_cases(self, real_vector_store):
        """Test search tool error handling with empty vector store"""
        tool = CourseSearchTool(real_vector_store)
        
        # Search in empty database
        result = tool.execute("any query")
        assert "No relevant content found" in result
        
        # Search with course filter in empty database
        result = tool.execute("query", course_name="Nonexistent")
        assert "No course found matching" in result or "No relevant content found" in result
    
    def test_document_processor_integration(self, temp_config):
        """Test document processor with real course document"""
        from document_processor import DocumentProcessor
        
        # Create a test document
        test_doc_content = """Course Title: Test Integration Course
Course Link: https://test.com/course
Course Instructor: Test Instructor

Lesson 1: Introduction to Testing
Lesson Link: https://test.com/lesson1
This is the content of lesson 1 about testing fundamentals.

Lesson 2: Advanced Testing
This is lesson 2 content about advanced testing concepts."""
        
        temp_file = tempfile.NamedTemporaryFile(mode='w', suffix='.txt', delete=False)
        temp_file.write(test_doc_content)
        temp_file.close()
        
        try:
            processor = DocumentProcessor(temp_config.CHUNK_SIZE, temp_config.CHUNK_OVERLAP)
            course, chunks = processor.process_course_document(temp_file.name)
            
            # Verify course metadata
            assert course.title == "Test Integration Course"
            assert course.course_link == "https://test.com/course"
            assert course.instructor == "Test Instructor"
            assert len(course.lessons) == 2
            
            # Verify chunks were created
            assert len(chunks) > 0
            assert all(chunk.course_title == "Test Integration Course" for chunk in chunks)
            
        finally:
            import os
            os.unlink(temp_file.name)