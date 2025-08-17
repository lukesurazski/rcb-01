import pytest
from unittest.mock import Mock, patch, MagicMock
from ai_generator import AIGenerator

class TestAIGenerator:
    """Test suite for AIGenerator tool calling functionality"""
    
    def test_generate_response_without_tools(self, mock_anthropic_client):
        """Test basic response generation without tools"""
        with patch('anthropic.Anthropic', return_value=mock_anthropic_client):
            generator = AIGenerator("test-key", "claude-sonnet-4-20250514")
            
            result = generator.generate_response("What is 2+2?")
            
            assert result == "Test AI response"
            mock_anthropic_client.messages.create.assert_called_once()
            
            # Verify no tools were passed
            call_args = mock_anthropic_client.messages.create.call_args
            assert "tools" not in call_args.kwargs
    
    def test_generate_response_with_tools_no_tool_use(self, mock_anthropic_client):
        """Test response generation with tools available but not used"""
        mock_tools = [{"name": "search_course_content", "description": "Search courses"}]
        mock_tool_manager = Mock()
        
        with patch('anthropic.Anthropic', return_value=mock_anthropic_client):
            generator = AIGenerator("test-key", "claude-sonnet-4-20250514")
            
            result = generator.generate_response(
                "What is 2+2?",
                tools=mock_tools,
                tool_manager=mock_tool_manager
            )
            
            assert result == "Test AI response"
            
            # Verify tools were passed to API
            call_args = mock_anthropic_client.messages.create.call_args
            assert call_args.kwargs["tools"] == mock_tools
            assert call_args.kwargs["tool_choice"] == {"type": "auto"}
    
    def test_generate_response_with_tool_use(self, mock_anthropic_client, mock_tool_use_response):
        """Test response generation when Claude decides to use tools"""
        # First call returns tool use, second call returns final response
        final_response = Mock()
        final_response.content = [Mock(text="Based on search results, here's the answer")]
        mock_anthropic_client.messages.create.side_effect = [mock_tool_use_response, final_response]
        
        mock_tool_manager = Mock()
        mock_tool_manager.execute_tool.return_value = "Mock search results content"
        mock_tools = [{"name": "search_course_content", "description": "Search courses"}]
        
        with patch('anthropic.Anthropic', return_value=mock_anthropic_client):
            generator = AIGenerator("test-key", "claude-sonnet-4-20250514")
            
            result = generator.generate_response(
                "What does the course cover about computer use?",
                tools=mock_tools,
                tool_manager=mock_tool_manager,
                enable_sequential_tools=False  # Test single-round behavior
            )
            
            assert result == "Based on search results, here's the answer"
            
            # Verify tool was executed
            mock_tool_manager.execute_tool.assert_called_once_with(
                "search_course_content",
                query="test query",
                course_name="Test Course"
            )
            
            # Verify two API calls were made
            assert mock_anthropic_client.messages.create.call_count == 2
    
    def test_handle_tool_execution_single_tool(self, mock_anthropic_client, mock_tool_use_response):
        """Test _handle_tool_execution with single tool call"""
        final_response = Mock()
        final_response.content = [Mock(text="Tool execution completed")]
        mock_anthropic_client.messages.create.return_value = final_response
        
        mock_tool_manager = Mock()
        mock_tool_manager.execute_tool.return_value = "Search results for tool"
        
        base_params = {
            "messages": [{"role": "user", "content": "test query"}],
            "system": "Test system prompt"
        }
        
        with patch('anthropic.Anthropic', return_value=mock_anthropic_client):
            generator = AIGenerator("test-key", "claude-sonnet-4-20250514")
            
            result = generator._handle_tool_execution(
                mock_tool_use_response, 
                base_params, 
                mock_tool_manager
            )
            
            assert result == "Tool execution completed"
            
            # Verify tool was executed with correct parameters
            mock_tool_manager.execute_tool.assert_called_once_with(
                "search_course_content",
                query="test query",
                course_name="Test Course"
            )
    
    def test_handle_tool_execution_message_flow(self, mock_anthropic_client, mock_tool_use_response):
        """Test that messages are properly constructed during tool execution"""
        final_response = Mock()
        final_response.content = [Mock(text="Final response")]
        mock_anthropic_client.messages.create.return_value = final_response
        
        mock_tool_manager = Mock()
        mock_tool_manager.execute_tool.return_value = "Tool result content"
        
        base_params = {
            "messages": [{"role": "user", "content": "original query"}],
            "system": "System prompt"
        }
        
        with patch('anthropic.Anthropic', return_value=mock_anthropic_client):
            generator = AIGenerator("test-key", "claude-sonnet-4-20250514")
            
            generator._handle_tool_execution(
                mock_tool_use_response,
                base_params,
                mock_tool_manager
            )
            
            # Check the final API call parameters
            final_call_args = mock_anthropic_client.messages.create.call_args
            messages = final_call_args.kwargs["messages"]
            
            # Should have: original user message, assistant tool use, user tool result
            assert len(messages) == 3
            assert messages[0]["role"] == "user"
            assert messages[0]["content"] == "original query"
            assert messages[1]["role"] == "assistant"
            assert messages[2]["role"] == "user"
            assert messages[2]["content"][0]["type"] == "tool_result"
            assert messages[2]["content"][0]["content"] == "Tool result content"
    
    def test_conversation_history_integration(self, mock_anthropic_client):
        """Test that conversation history is properly included in system prompt"""
        conversation_history = "User: Previous question\nAssistant: Previous answer"
        
        with patch('anthropic.Anthropic', return_value=mock_anthropic_client):
            generator = AIGenerator("test-key", "claude-sonnet-4-20250514")
            
            generator.generate_response(
                "New question",
                conversation_history=conversation_history
            )
            
            call_args = mock_anthropic_client.messages.create.call_args
            system_content = call_args.kwargs["system"]
            
            assert conversation_history in system_content
            assert "Previous conversation:" in system_content
    
    def test_api_parameters_structure(self, mock_anthropic_client):
        """Test that API parameters are structured correctly"""
        with patch('anthropic.Anthropic', return_value=mock_anthropic_client):
            generator = AIGenerator("test-key", "claude-sonnet-4-20250514")
            
            generator.generate_response("Test query")
            
            call_args = mock_anthropic_client.messages.create.call_args
            kwargs = call_args.kwargs
            
            # Verify required parameters
            assert kwargs["model"] == "claude-sonnet-4-20250514"
            assert kwargs["temperature"] == 0
            assert kwargs["max_tokens"] == 800
            assert kwargs["messages"] == [{"role": "user", "content": "Test query"}]
            assert "system" in kwargs
    
    def test_tool_execution_error_handling(self, mock_anthropic_client, mock_tool_use_response):
        """Test error handling when tool execution fails in sequential mode"""
        mock_tool_manager = Mock()
        mock_tool_manager.execute_tool.side_effect = Exception("Tool execution failed")
        
        # Should continue with final response despite tool error
        final_response = Mock()
        final_response.stop_reason = "end_turn"
        final_response.content = [Mock(text="Error handled response")]
        mock_anthropic_client.messages.create.side_effect = [mock_tool_use_response, final_response]
        
        with patch('anthropic.Anthropic', return_value=mock_anthropic_client):
            generator = AIGenerator("test-key", "claude-sonnet-4-20250514")
            
            # Should handle tool error gracefully without raising exception
            result = generator.generate_response(
                "Test query",
                tools=[{"name": "test_tool"}],
                tool_manager=mock_tool_manager
            )
            
            assert result == "Error handled response"
            assert mock_anthropic_client.messages.create.call_count == 2
    
    def test_sequential_tool_execution_two_rounds(self, mock_anthropic_client):
        """Test sequential tool execution with two complete rounds"""
        # First response: tool use
        first_tool_response = Mock()
        first_tool_response.stop_reason = "tool_use"
        first_tool_block = Mock()
        first_tool_block.type = "tool_use"
        first_tool_block.name = "get_course_outline"
        first_tool_block.id = "tool_1"
        first_tool_block.input = {"course_name": "Test Course"}
        first_tool_response.content = [first_tool_block]
        
        # Second response: another tool use
        second_tool_response = Mock()
        second_tool_response.stop_reason = "tool_use"
        second_tool_block = Mock()
        second_tool_block.type = "tool_use"
        second_tool_block.name = "search_course_content"
        second_tool_block.id = "tool_2"
        second_tool_block.input = {"query": "lesson 4", "course_name": "Test Course"}
        second_tool_response.content = [second_tool_block]
        
        # Final response: no tool use
        final_response = Mock()
        final_response.stop_reason = "end_turn"
        final_response.content = [Mock(text="Final answer after two searches")]
        
        mock_anthropic_client.messages.create.side_effect = [
            first_tool_response, second_tool_response, final_response
        ]
        
        mock_tool_manager = Mock()
        mock_tool_manager.execute_tool.side_effect = [
            "Course outline results",
            "Lesson 4 content results"
        ]
        
        with patch('anthropic.Anthropic', return_value=mock_anthropic_client):
            generator = AIGenerator("test-key", "claude-sonnet-4-20250514")
            
            result = generator.generate_response(
                "What topic is covered in lesson 4 of Test Course?",
                tools=[{"name": "get_course_outline"}, {"name": "search_course_content"}],
                tool_manager=mock_tool_manager
            )
            
            assert result == "Final answer after two searches"
            assert mock_anthropic_client.messages.create.call_count == 3
            assert mock_tool_manager.execute_tool.call_count == 2
    
    def test_sequential_tool_execution_max_rounds_termination(self, mock_anthropic_client):
        """Test that sequential execution terminates after 2 rounds"""
        # Both responses trigger tool use
        tool_response = Mock()
        tool_response.stop_reason = "tool_use"
        tool_block = Mock()
        tool_block.type = "tool_use"
        tool_block.name = "search_course_content"
        tool_block.id = "tool_123"
        tool_block.input = {"query": "test"}
        tool_response.content = [tool_block]
        
        # Final response after max rounds
        final_response = Mock()
        final_response.stop_reason = "tool_use"  # Still wants tools but max reached
        final_response.content = [Mock(text="Response after max rounds")]
        
        mock_anthropic_client.messages.create.side_effect = [
            tool_response, tool_response, final_response
        ]
        
        mock_tool_manager = Mock()
        mock_tool_manager.execute_tool.return_value = "Tool result"
        
        with patch('anthropic.Anthropic', return_value=mock_anthropic_client):
            generator = AIGenerator("test-key", "claude-sonnet-4-20250514")
            
            result = generator.generate_response(
                "Complex query requiring multiple searches",
                tools=[{"name": "search_course_content"}],
                tool_manager=mock_tool_manager
            )
            
            assert result == "Response after max rounds"
            # Should stop after 2 rounds (initial + 2 sequential)
            assert mock_anthropic_client.messages.create.call_count == 3
            assert mock_tool_manager.execute_tool.call_count == 2
    
    def test_sequential_tool_execution_single_round_completion(self, mock_anthropic_client):
        """Test sequential execution that completes in first round"""
        # First response: tool use
        tool_response = Mock()
        tool_response.stop_reason = "tool_use"
        tool_block = Mock()
        tool_block.type = "tool_use"
        tool_block.name = "search_course_content"
        tool_block.id = "tool_123"
        tool_block.input = {"query": "test"}
        tool_response.content = [tool_block]
        
        # Second response: end turn (no more tools)
        final_response = Mock()
        final_response.stop_reason = "end_turn"
        final_response.content = [Mock(text="Complete answer from first search")]
        
        mock_anthropic_client.messages.create.side_effect = [tool_response, final_response]
        
        mock_tool_manager = Mock()
        mock_tool_manager.execute_tool.return_value = "Comprehensive search results"
        
        with patch('anthropic.Anthropic', return_value=mock_anthropic_client):
            generator = AIGenerator("test-key", "claude-sonnet-4-20250514")
            
            result = generator.generate_response(
                "What is covered in the course?",
                tools=[{"name": "search_course_content"}],
                tool_manager=mock_tool_manager
            )
            
            assert result == "Complete answer from first search"
            assert mock_anthropic_client.messages.create.call_count == 2
            assert mock_tool_manager.execute_tool.call_count == 1
    
    def test_sequential_tool_execution_with_feature_toggle_disabled(self, mock_anthropic_client, mock_tool_use_response):
        """Test that sequential tools can be disabled via feature toggle"""
        final_response = Mock()
        final_response.content = [Mock(text="Single round response")]
        mock_anthropic_client.messages.create.side_effect = [mock_tool_use_response, final_response]
        
        mock_tool_manager = Mock()
        mock_tool_manager.execute_tool.return_value = "Tool result"
        
        with patch('anthropic.Anthropic', return_value=mock_anthropic_client):
            generator = AIGenerator("test-key", "claude-sonnet-4-20250514")
            
            result = generator.generate_response(
                "Test query",
                tools=[{"name": "search_course_content"}],
                tool_manager=mock_tool_manager,
                enable_sequential_tools=False
            )
            
            assert result == "Single round response"
            # Should only make 2 calls (initial + single tool round)
            assert mock_anthropic_client.messages.create.call_count == 2
    
    def test_sequential_tool_execution_error_fallback(self, mock_anthropic_client, mock_tool_use_response):
        """Test that sequential execution falls back to single-round on error"""
        mock_tool_manager = Mock()
        
        # Make sure initial response triggers tool use
        mock_anthropic_client.messages.create.return_value = mock_tool_use_response
        
        with patch('anthropic.Anthropic', return_value=mock_anthropic_client):
            generator = AIGenerator("test-key", "claude-sonnet-4-20250514")
            
            # Mock methods to test fallback behavior
            with patch.object(generator, '_handle_sequential_tool_execution', side_effect=Exception("Sequential failed")) as mock_sequential:
                with patch.object(generator, '_handle_tool_execution', return_value="Fallback response") as mock_single:
                    
                    result = generator.generate_response(
                        "Test query",
                        tools=[{"name": "search_course_content"}],
                        tool_manager=mock_tool_manager
                    )
                    
                    assert result == "Fallback response"
                    mock_sequential.assert_called_once()
                    mock_single.assert_called_once()
    
    def test_system_prompt_content(self):
        """Test that system prompt contains expected tool usage guidelines"""
        generator = AIGenerator("test-key", "claude-sonnet-4-20250514")
        
        assert "search_course_content" in generator.SYSTEM_PROMPT
        assert "get_course_outline" in generator.SYSTEM_PROMPT
        assert "Course content questions" in generator.SYSTEM_PROMPT
        assert "Sequential Tool Usage" in generator.SYSTEM_PROMPT
        assert "up to 2 rounds" in generator.SYSTEM_PROMPT