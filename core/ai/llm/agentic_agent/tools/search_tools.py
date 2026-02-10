"""
Search tools for web search functionality using decorator approach.
"""

import os
from typing import Any, Dict, Optional
from langchain_core.tools import tool
from langchain_community.utilities import GoogleSerperAPIWrapper


@tool
def web_search(query: str, num_results: int = 5) -> str:
    """Search the web for information using DuckDuckGo.
    
    Args:
        query: The search query
        num_results: Number of results to return (1-20, default: 5)
    
    Returns:
        str: Formatted search results with titles, URLs and snippets
    """
    try:
        # Validate num_results
        if num_results < 1 or num_results > 20:
            return "Error: num_results must be between 1 and 20"
            
        from duckduckgo_search import DDGS
        
        with DDGS() as ddgs:
            results = list(ddgs.text(query, max_results=num_results))
            
            if not results:
                return f"No search results found for query: {query}"
            
            formatted_results = []
            for i, result in enumerate(results, 1):
                title = result.get("title", "No title")
                url = result.get("href", "No URL")
                snippet = result.get("body", "No snippet")
                
                formatted_results.append(f"{i}. {title}\n   URL: {url}\n   {snippet}\n")
            
            return f"Search results for '{query}':\n\n" + "\n".join(formatted_results)
                
    except ImportError:
        return "Error: duckduckgo-search package not installed. Install with: pip install duckduckgo-search"
    except Exception as e:
        return f"Search failed: {str(e)}"


# Global variable to store the API key for google_search
_serper_api_key: Optional[str] = None

def set_serper_api_key(api_key: str):
    """Set the Serper API key for Google search."""
    global _serper_api_key
    _serper_api_key = api_key

@tool
def google_search(query: str, num_results: int = 5) -> str:
    """Search Google using Serper API.
    
    Args:
        query: The search query
        num_results: Number of results to return (1-20, default: 5)
    
    Returns:
        str: Formatted Google search results
    """
    try:
        # Validate num_results
        if num_results < 1 or num_results > 20:
            return "Error: num_results must be between 1 and 20"
            
        # Get API key from global variable or environment
        api_key = _serper_api_key or os.getenv("SERP_API_KEY")
        if not api_key:
            return "Error: SERP_API_KEY is required for Google search. Set environment variable or call set_serper_api_key()"
        
        search_wrapper = GoogleSerperAPIWrapper(
            serper_api_key=api_key,
            k=num_results
        )
        
        results = search_wrapper.results(query)
        
        if not results or "organic" not in results:
            return f"No Google search results found for query: {query}"
        
        # Format results consistently
        formatted_results = []
        for i, result in enumerate(results["organic"], 1):
            title = result.get("title", "No title")
            url = result.get("link", "No URL") 
            snippet = result.get("snippet", "No snippet")
            
            formatted_results.append(f"{i}. {title}\n   URL: {url}\n   {snippet}\n")
        
        return f"Google search results for '{query}':\n\n" + "\n".join(formatted_results)
        
    except Exception as e:
        return f"Google search failed: {str(e)}"


# Legacy classes for backward compatibility
class WebSearchTool:
    def __init__(self):
        self.tool = web_search
        self.name = web_search.name
        self.description = web_search.description

class SerperSearchTool:
    def __init__(self, api_key: Optional[str] = None):
        if api_key:
            set_serper_api_key(api_key)
        self.tool = google_search
        self.name = google_search.name
        self.description = google_search.description