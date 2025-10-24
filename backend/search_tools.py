from typing import Dict, Any, Optional
from abc import ABC, abstractmethod
from vector_store import VectorStore, SearchResults


class Tool(ABC):
    """Abstract base class for all tools"""
    
    @abstractmethod
    def get_tool_definition(self) -> Dict[str, Any]:
        """Return Anthropic tool definition for this tool"""
        pass
    
    @abstractmethod
    def execute(self, **kwargs) -> str:
        """Execute the tool with given parameters"""
        pass


class CourseSearchTool(Tool):
    """Tool for searching course content with semantic course name matching"""
    
    def __init__(self, vector_store: VectorStore):
        self.store = vector_store
        self.last_sources = []  # Track sources from last search
    
    def get_tool_definition(self) -> Dict[str, Any]:
        """Return Anthropic tool definition for this tool"""
        return {
            "name": "search_course_content",
            "description": "Search course materials with smart course name matching and lesson filtering",
            "input_schema": {
                "type": "object",
                "properties": {
                    "query": {
                        "type": "string", 
                        "description": "What to search for in the course content"
                    },
                    "course_name": {
                        "type": "string",
                        "description": "Course title (partial matches work, e.g. 'MCP', 'Introduction')"
                    },
                    "lesson_number": {
                        "type": "integer",
                        "description": "Specific lesson number to search within (e.g. 1, 2, 3)"
                    }
                },
                "required": ["query"]
            }
        }
    
    def execute(self, query: str, course_name: Optional[str] = None, lesson_number: Optional[int] = None) -> str:
        """
        Execute the search tool with given parameters.
        
        Args:
            query: What to search for
            course_name: Optional course filter
            lesson_number: Optional lesson filter
            
        Returns:
            Formatted search results or error message
        """
        
        # Use the vector store's unified search interface
        results = self.store.search(
            query=query,
            course_name=course_name,
            lesson_number=lesson_number
        )
        
        # Handle errors
        if results.error:
            return results.error
        
        # Handle empty results
        if results.is_empty():
            filter_info = ""
            if course_name:
                filter_info += f" in course '{course_name}'"
            if lesson_number:
                filter_info += f" in lesson {lesson_number}"
            return f"No relevant content found{filter_info}."
        
        # Format and return results
        return self._format_results(results)
    
    def _format_results(self, results: SearchResults) -> str:
        """Format search results with course and lesson context"""
        formatted = []
        sources = []  # Track sources for the UI
        
        for doc, meta in zip(results.documents, results.metadata):
            course_title = meta.get('course_title', 'unknown')
            lesson_num = meta.get('lesson_number')
            
            # Build context header
            header = f"[{course_title}"
            if lesson_num is not None:
                header += f" - Lesson {lesson_num}"
            header += "]"
            
            # Track source for the UI with link if available
            source_text = course_title
            if lesson_num is not None:
                source_text += f" - Lesson {lesson_num}"
                # Try to get lesson link
                lesson_link = self.store.get_lesson_link(course_title, lesson_num)
                if lesson_link:
                    sources.append({"text": source_text, "url": lesson_link})
                else:
                    sources.append({"text": source_text, "url": None})
            else:
                # For course-only results, try to get course link
                course_link = self.store.get_course_link(course_title)
                if course_link:
                    sources.append({"text": source_text, "url": course_link})
                else:
                    sources.append({"text": source_text, "url": None})
            
            formatted.append(f"{header}\n{doc}")
        
        # Store sources for retrieval
        self.last_sources = sources
        
        return "\n\n".join(formatted)


class CourseOutlineTool(Tool):
    """Tool for getting complete course outlines with lesson structure"""
    
    def __init__(self, vector_store: VectorStore):
        self.store = vector_store
        self.last_sources = []  # Track sources from last search
    
    def get_tool_definition(self) -> Dict[str, Any]:
        """Return Anthropic tool definition for this tool"""
        return {
            "name": "get_course_outline",
            "description": "Get complete course outline including title, link, and all lessons",
            "input_schema": {
                "type": "object",
                "properties": {
                    "course_name": {
                        "type": "string",
                        "description": "Course title (partial matches work, e.g. 'MCP', 'Introduction')"
                    }
                },
                "required": ["course_name"]
            }
        }
    
    def execute(self, course_name: str) -> str:
        """
        Execute the outline tool to get course structure.
        
        Args:
            course_name: Course title or partial name
            
        Returns:
            Formatted course outline or error message
        """
        # Get course outline from vector store
        outline_data = self.store.get_course_outline(course_name)
        
        if not outline_data:
            return f"No course found matching '{course_name}'"
        
        # Format the outline response
        return self._format_outline(outline_data)
    
    def _format_outline(self, outline_data: Dict[str, Any]) -> str:
        """Format course outline data into readable text"""
        course_title = outline_data.get('course_title', 'Unknown Course')
        course_link = outline_data.get('course_link', '')
        instructor = outline_data.get('instructor', '')
        lessons = outline_data.get('lessons', [])
        
        # Track source for the UI
        source_text = f"{course_title} - Course Outline"
        self.last_sources = [{"text": source_text, "url": course_link}]
        
        # Build formatted outline
        result = f"**{course_title}**\n"
        if instructor:
            result += f"Instructor: {instructor}\n"
        if course_link:
            result += f"Course Link: {course_link}\n"
        
        result += "\n**Course Outline:**\n"
        
        if lessons:
            for lesson in lessons:
                lesson_num = lesson.get('lesson_number', '?')
                lesson_title = lesson.get('lesson_title', 'Untitled Lesson')
                result += f"Lesson {lesson_num}: {lesson_title}\n"
        else:
            result += "No lessons available\n"
        
        return result


class ToolManager:
    """Manages available tools for the AI"""
    
    def __init__(self):
        self.tools = {}
    
    def register_tool(self, tool: Tool):
        """Register any tool that implements the Tool interface"""
        tool_def = tool.get_tool_definition()
        tool_name = tool_def.get("name")
        if not tool_name:
            raise ValueError("Tool must have a 'name' in its definition")
        self.tools[tool_name] = tool

    
    def get_tool_definitions(self) -> list:
        """Get all tool definitions for Anthropic tool calling"""
        return [tool.get_tool_definition() for tool in self.tools.values()]
    
    def execute_tool(self, tool_name: str, **kwargs) -> str:
        """Execute a tool by name with given parameters"""
        if tool_name not in self.tools:
            return f"Tool '{tool_name}' not found"
        
        return self.tools[tool_name].execute(**kwargs)
    
    def get_last_sources(self) -> list:
        """Get sources from the last search operation"""
        # Check all tools for last_sources attribute
        for tool in self.tools.values():
            if hasattr(tool, 'last_sources') and tool.last_sources:
                return tool.last_sources
        return []

    def reset_sources(self):
        """Reset sources from all tools that track sources"""
        for tool in self.tools.values():
            if hasattr(tool, 'last_sources'):
                tool.last_sources = []