import anthropic
from typing import List, Optional, Dict, Any

class AIGenerator:
    """Handles interactions with Anthropic's Claude API for generating responses"""
    
    # Static system prompt to avoid rebuilding on each call
    SYSTEM_PROMPT = """ You are an AI assistant specialized in course materials and educational content with access to comprehensive search tools for course information.

Available Tools:
- **search_course_content**: For searching specific course content and materials
- **get_course_outline**: For getting complete course outlines including title, course link, and all lessons

Tool Usage Guidelines:
- Use **get_course_outline** for questions about:
  - Course structure, outline, or lesson list
  - "What lessons are in..." or "What's covered in..." queries
  - Overview of course organization
- Use **search_course_content** for questions about:
  - Specific content within lessons
  - Detailed educational materials
  - Implementation details or explanations

**Sequential Tool Usage** (up to 2 rounds):
- **Round 1**: Get initial information (e.g., course outline or broad search)
- **Round 2**: Refine with specific searches based on Round 1 results
- **Examples**:
  - First get course outline, then search specific lessons mentioned
  - First broad search, then focused search on specific topics found
  - Search one course, then search related courses for comparison

**Termination Conditions**:
- Provide final answer when you have sufficient information
- Stop tool usage after 2 rounds maximum
- Stop if no additional tools would improve the response

- Synthesize tool results into accurate, fact-based responses
- If tools yield no results, state this clearly without offering alternatives

Response Protocol:
- **General knowledge questions**: Answer using existing knowledge without tools
- **Course outline questions**: Use get_course_outline tool first, optionally follow up with content search
- **Course content questions**: Use search_course_content tool first, optionally follow up with related searches
- **Complex queries**: Use multiple rounds to gather comprehensive information
- **No meta-commentary**:
 - Provide direct answers only — no reasoning process, tool explanations, or question-type analysis
 - Do not mention "based on the search results" or "using the tool"

All responses must be:
1. **Brief, Concise and focused** - Get to the point quickly
2. **Educational** - Maintain instructional value
3. **Clear** - Use accessible language
4. **Example-supported** - Include relevant examples when they aid understanding
Provide only the direct answer to what was asked, synthesized from all tool results if multiple searches were performed.
"""
    
    def __init__(self, api_key: str, model: str):
        self.client = anthropic.Anthropic(api_key=api_key)
        self.model = model
        
        # Pre-build base API parameters
        self.base_params = {
            "model": self.model,
            "temperature": 0,
            "max_tokens": 800
        }
    
    def generate_response(self, query: str,
                         conversation_history: Optional[str] = None,
                         tools: Optional[List] = None,
                         tool_manager=None,
                         enable_sequential_tools: bool = True) -> str:
        """
        Generate AI response with optional sequential tool usage and conversation context.
        
        Args:
            query: The user's question or request
            conversation_history: Previous messages for context
            tools: Available tools the AI can use
            tool_manager: Manager to execute tools
            enable_sequential_tools: Whether to enable sequential tool calling (default: True)
            
        Returns:
            Generated response as string
        """
        
        # Build system content efficiently - avoid string ops when possible
        system_content = (
            f"{self.SYSTEM_PROMPT}\n\nPrevious conversation:\n{conversation_history}"
            if conversation_history 
            else self.SYSTEM_PROMPT
        )
        
        # Prepare API call parameters efficiently
        api_params = {
            **self.base_params,
            "messages": [{"role": "user", "content": query}],
            "system": system_content
        }
        
        # Add tools if available
        if tools:
            api_params["tools"] = tools
            api_params["tool_choice"] = {"type": "auto"}
        
        # Get response from Claude
        try:
            response = self.client.messages.create(**api_params)
        except Exception as e:
            # Re-raise API errors with more context
            raise Exception(f"Anthropic API error: {type(e).__name__}: {str(e)}")

        # Handle tool execution if needed
        if response.stop_reason == "tool_use" and tool_manager:
            if enable_sequential_tools:
                try:
                    return self._handle_sequential_tool_execution(response, api_params, tool_manager)
                except Exception:
                    # Fallback to single-round execution on sequential error
                    return self._handle_tool_execution(response, api_params, tool_manager)
            else:
                return self._handle_tool_execution(response, api_params, tool_manager)

        # Return direct response - handle empty content gracefully
        if not response.content or len(response.content) == 0:
            raise Exception(f"Empty response from Claude API (stop_reason: {response.stop_reason})")

        if not hasattr(response.content[0], 'text'):
            raise Exception(f"Unexpected content type in response: {type(response.content[0])}")

        return response.content[0].text
    
    def _handle_sequential_tool_execution(self, initial_response, base_params: Dict[str, Any], tool_manager, max_rounds: int = 2):
        """
        Handle sequential tool execution across multiple API calls.
        
        Args:
            initial_response: The response containing initial tool use requests
            base_params: Base API parameters (including tools)
            tool_manager: Manager to execute tools
            max_rounds: Maximum number of tool execution rounds (default: 2)
            
        Returns:
            Final response text after sequential tool execution
        """
        try:
            messages = base_params["messages"].copy()
            current_response = initial_response
            round_count = 0
            
            while (current_response.stop_reason == "tool_use" and 
                   round_count < max_rounds and 
                   tool_manager):
                
                round_count += 1
                
                # Add AI's tool use response to conversation
                messages.append({"role": "assistant", "content": current_response.content})
                
                # Execute tools and collect results
                tool_results = self._execute_tools_for_round(current_response, tool_manager)
                
                if tool_results:
                    messages.append({"role": "user", "content": tool_results})
                else:
                    break  # No tools executed, exit loop
                
                # Prepare next API call with tools still available
                next_params = {
                    **self.base_params,
                    "messages": messages,
                    "system": base_params["system"],
                    "tools": base_params.get("tools", []),  # Keep tools available
                    "tool_choice": {"type": "auto"}
                }
                
                # Get next response
                try:
                    current_response = self.client.messages.create(**next_params)
                except Exception as e:
                    return f"API error during sequential execution: {str(e)}"
            
            # Return final response content with safety checks
            if not current_response.content or len(current_response.content) == 0:
                return "No response generated from Claude API"

            if not hasattr(current_response.content[0], 'text'):
                return f"Unexpected response format from Claude API"

            return current_response.content[0].text

        except Exception as e:
            # Re-raise to trigger fallback at generate_response level
            raise e
    
    def _execute_tools_for_round(self, response, tool_manager):
        """Execute all tool calls for a single round and return results"""
        tool_results = []
        
        for content_block in response.content:
            if content_block.type == "tool_use":
                try:
                    tool_result = tool_manager.execute_tool(
                        content_block.name, 
                        **content_block.input
                    )
                    
                    tool_results.append({
                        "type": "tool_result",
                        "tool_use_id": content_block.id,
                        "content": tool_result
                    })
                except Exception as e:
                    # Handle individual tool execution errors
                    tool_results.append({
                        "type": "tool_result",
                        "tool_use_id": content_block.id,
                        "content": f"Tool execution error: {str(e)}",
                        "is_error": True
                    })
        
        return tool_results

    def _handle_tool_execution(self, initial_response, base_params: Dict[str, Any], tool_manager):
        """
        Handle execution of tool calls and get follow-up response (single round).
        
        Args:
            initial_response: The response containing tool use requests
            base_params: Base API parameters
            tool_manager: Manager to execute tools
            
        Returns:
            Final response text after tool execution
        """
        # Start with existing messages
        messages = base_params["messages"].copy()
        
        # Add AI's tool use response
        messages.append({"role": "assistant", "content": initial_response.content})
        
        # Execute all tool calls and collect results
        tool_results = self._execute_tools_for_round(initial_response, tool_manager)
        
        # Add tool results as single message
        if tool_results:
            messages.append({"role": "user", "content": tool_results})
        
        # Prepare final API call without tools
        final_params = {
            **self.base_params,
            "messages": messages,
            "system": base_params["system"]
        }
        
        # Get final response
        try:
            final_response = self.client.messages.create(**final_params)
        except Exception as e:
            raise Exception(f"Anthropic API error in tool execution: {type(e).__name__}: {str(e)}")

        # Handle response with safety checks
        if not final_response.content or len(final_response.content) == 0:
            return "No response generated after tool execution"

        if not hasattr(final_response.content[0], 'text'):
            return f"Unexpected response format after tool execution"

        return final_response.content[0].text