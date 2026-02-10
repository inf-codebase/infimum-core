"""
Tool registry for managing and discovering available tools.
"""

from typing import Dict, List, Optional, Type, Any
from langchain_core.tools import BaseTool

from .search_tools import web_search, google_search, set_serper_api_key
from .calculation_tools import calculator, compound_interest
from .weather_tools import weather, set_weather_api_key
from .time_tools import current_time, world_clock
from .web_tools import web_scrape


class ToolRegistry:
    """
    Registry for managing and discovering available tools.
    """
    
    def __init__(self):
        self._tools: Dict[str, BaseTool] = {}
        self._tool_classes: Dict[str, Type[BaseTool]] = {}
        self._categories: Dict[str, List[str]] = {
            "search": [],
            "calculation": [],
            "weather": [],
            "time": [],
            "web": [],
            "utility": [],
        }
    
    def register_tool(self, tool: BaseTool, category: str = "utility") -> None:
        """Register a tool instance."""
        self._tools[tool.name] = tool
        self._tool_classes[tool.name] = type(tool)
        
        if category in self._categories:
            if tool.name not in self._categories[category]:
                self._categories[category].append(tool.name)
        else:
            self._categories[category] = [tool.name]
    
    def get_tool(self, name: str) -> Optional[BaseTool]:
        """Get a tool by name."""
        return self._tools.get(name)
    
    def get_tools(self, category: Optional[str] = None) -> List[BaseTool]:
        """Get all tools or tools in a specific category."""
        if category is None:
            return list(self._tools.values())
        
        if category not in self._categories:
            return []
        
        return [self._tools[name] for name in self._categories[category] if name in self._tools]
    
    def get_tool_names(self, category: Optional[str] = None) -> List[str]:
        """Get tool names, optionally filtered by category."""
        if category is None:
            return list(self._tools.keys())
        
        return self._categories.get(category, [])
    
    def list_categories(self) -> List[str]:
        """List all available categories."""
        return list(self._categories.keys())
    
    def get_tool_info(self, name: str) -> Optional[Dict[str, Any]]:
        """Get information about a specific tool."""
        tool = self._tools.get(name)
        if not tool:
            return None
        
        info = {
            "name": tool.name,
            "description": tool.description,
            "category": self._get_tool_category(name),
        }
        
        # Add additional info if available
        if hasattr(tool, 'tool_id'):
            info["tool_id"] = tool.tool_id
        if hasattr(tool, 'get_tool_info'):
            info.update(tool.get_tool_info())
        
        return info
    
    def _get_tool_category(self, tool_name: str) -> str:
        """Find the category for a given tool name."""
        for category, tools in self._categories.items():
            if tool_name in tools:
                return category
        return "utility"
    
    def search_tools(self, query: str) -> List[str]:
        """Search for tools by name or description."""
        query_lower = query.lower()
        matching_tools = []
        
        for name, tool in self._tools.items():
            if (query_lower in name.lower() or 
                query_lower in tool.description.lower()):
                matching_tools.append(name)
        
        return matching_tools


def get_default_tools(
    include_search: bool = True,
    include_calculation: bool = True,
    include_weather: bool = True,
    include_time: bool = True,
    include_web: bool = True,
    serp_api_key: Optional[str] = None,
    weather_api_key: Optional[str] = None,
) -> ToolRegistry:
    """
    Create a tool registry with default tools.
    
    Args:
        include_search: Include search tools
        include_calculation: Include calculation tools
        include_weather: Include weather tools
        include_time: Include time tools
        include_web: Include web scraping tools
        serp_api_key: Serper API key for Google search
        weather_api_key: OpenWeather API key for weather data
    
    Returns:
        ToolRegistry with registered default tools
    """
    registry = ToolRegistry()
    
    # Search tools
    if include_search:
        registry.register_tool(web_search, "search")
        if serp_api_key:
            set_serper_api_key(serp_api_key)
            registry.register_tool(google_search, "search")
    
    # Calculation tools
    if include_calculation:
        registry.register_tool(calculator, "calculation")
        registry.register_tool(compound_interest, "calculation")
    
    # Weather tools
    if include_weather:
        if weather_api_key:
            set_weather_api_key(weather_api_key)
        registry.register_tool(weather, "weather")
    
    # Time tools
    if include_time:
        registry.register_tool(current_time, "time")
        registry.register_tool(world_clock, "time")
    
    # Web tools
    if include_web:
        registry.register_tool(web_scrape, "web")
    
    return registry


def create_langchain_tools(registry: ToolRegistry) -> List[BaseTool]:
    """
    Convert registry tools to LangChain format.
    
    Args:
        registry: Tool registry
    
    Returns:
        List of LangChain tools
    """
    return registry.get_tools()