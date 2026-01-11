import requests
import json
import math
import os
import re

from src.core.utils import auto_config

class ToolManager:
    def __init__(self, tools=None):
        self.tools = {}
        
        # Register provided tools
        if tools:
            for tool in tools:
                self.register_tool(tool)
    
    def register_tool(self, tool):
        """Register a tool with the manager"""
        self.tools[tool.name] = tool
    
    def execute_tool(self, tool_name, tool_input):
        """Execute a specific tool"""
        if tool_name not in self.tools:
            return f"Error: Tool '{tool_name}' not found."
        
        try:
            return self.tools[tool_name].execute(tool_input)
        except Exception as e:
            return f"Error executing tool '{tool_name}': {str(e)}"
    
    def get_available_tools(self):
        """Get list of available tools"""
        return list(self.tools.keys())


class Tool:
    """Base class for all tools"""
    def __init__(self, name, description):
        self.name = name
        self.description = description
    
    def execute(self, input_text):
        """Execute the tool functionality"""
        raise NotImplementedError("Tool subclasses must implement execute method")


class WebSearchTool(Tool):
    def __init__(self):
        super().__init__("search", "Search the web for information")
        self.api_key = auto_config.SERP_API_KEY
    
    def execute(self, query):
        """Execute a web search, trying SerpAPI first and falling back to DuckDuckGo if needed"""
        try:
            # Try SerpAPI first
            serpapi_result = self._serpapi_search(query)
            
            # If SerpAPI returns an error or no results, try DuckDuckGo
            if "No search results found" in serpapi_result or "Search API error" in serpapi_result:
                return self._duckduckgo_search(query)
            
            return serpapi_result
                
        except Exception as e:
            # If any error occurs, try DuckDuckGo as fallback
            try:
                return self._duckduckgo_search(query)
            except Exception as e2:
                return f"Search error: Both SerpAPI and DuckDuckGo failed. Last error: {str(e2)}"
    
    def _serpapi_search(self, query):
        """Execute a web search using SerpAPI (Google)"""
        url = "https://serpapi.com/search"
        params = {
            'api_key': self.api_key,
            'q': query,
            'engine': 'google',
            'num': 5,  # Number of results to return
            'gl': 'us'  # Country to use for the search (United States)
        }
        
        response = requests.get(url, params=params)
        
        if response.status_code != 200:
            return f"Search API error: {response.status_code} - {response.text}"
            
        results = response.json()
        formatted_results = []
        
        if 'organic_results' in results:
            for item in results['organic_results']:
                formatted_results.append({
                    'title': item.get('title', ''),
                    'link': item.get('link', ''),
                    'snippet': item.get('snippet', '')
                })
            
            return json.dumps(formatted_results, indent=2)
        else:
            return "No search results found"
    
    def _duckduckgo_search(self, query):
        """Execute a web search using DuckDuckGo"""
        try:
            from duckduckgo_search import DDGS
            
            # Initialize DDGS and perform the search
            with DDGS() as ddgs:
                # Get search results with a reasonable timeout
                results = list(ddgs.text(query, max_results=5))
                
                # Format results to match SerpAPI format
                formatted_results = []
                for item in results:
                    formatted_results.append({
                        'title': item.get('title', ''),
                        'link': item.get('href', ''),
                        'snippet': item.get('body', '')
                    })
                
                if formatted_results:
                    return json.dumps(formatted_results, indent=2)
                else:
                    return "No search results found"
                    
        except ImportError:
            return "Error: duckduckgo_search package not installed. Please install it using 'pip install duckduckgo-search'"
        except Exception as e:
            return f"DuckDuckGo search error: {str(e)}"


class CalculatorTool(Tool):
    def __init__(self):
        super().__init__("calculator", "Perform mathematical calculations")
    
    def execute(self, expression):
        """Execute a mathematical calculation"""
        try:
            # Extract mathematical expression from the prompt
            if "compound interest" in expression.lower():
                # Handle compound interest calculation
                # Extract values using regex
                principal = float(re.search(r'\$(\d+(?:,\d+)?(?:\.\d+)?)', expression).group(1).replace(',', ''))
                years = int(re.search(r'(\d+)\s+years', expression).group(1))
                rate = float(re.search(r'(\d+(?:\.\d+)?)\s*%', expression).group(1)) / 100
                
                # Default to annual compounding if not specified
                n = 1  # Number of compounding periods per year
                
                # Calculate compound interest using the correct formula: A = P(1 + r/n)^(nt)
                amount = principal * (1 + rate/n) ** (n * years)
                interest = amount - principal
                
                return f"Compound Interest Calculation:\nPrincipal: ${principal:,.2f}\nAnnual Rate: {rate*100}%\nYears: {years}\nCompounding Periods per Year: {n}\nFinal Amount: ${amount:,.2f}\nTotal Interest: ${interest:,.2f}"
            
            # For other mathematical expressions
            # Make this safer by limiting allowed operations
            if any(op in expression for op in ["import", "exec", "eval", "os.", "sys."]):
                return "Error: Potentially unsafe operation"
            
            # Create a safer environment for eval
            safe_dict = {
                'abs': abs, 'round': round, 'min': min, 'max': max,
                'int': int, 'float': float, 'sum': sum,
                'pow': pow, 'sqrt': math.sqrt, 'sin': math.sin, 'cos': math.cos,
                'tan': math.tan, 'pi': math.pi, 'e': math.e
            }
            
            # Try to extract just the mathematical expression
            # Replace ^ with ** for Python exponentiation
            expression = expression.replace('^', '**')
            math_expr = re.search(r'([\d+\-*/().\s]+)', expression)
            if math_expr:
                # Use eval with a safer environment
                result = eval(math_expr.group(1), {"__builtins__": {}}, safe_dict)
                return f"Result: {result}"
            else:
                return "Error: No valid mathematical expression found"
                
        except Exception as e:
            return f"Calculation error: {str(e)}"


class WeatherTool(Tool):
    def __init__(self):
        super().__init__("weather", "Get weather information for a location")
        self.api_key = os.getenv("WEATHER_API_KEY")  # You would need a weather API key
    
    def execute(self, location):
        """Get weather for a location"""
        # In a real implementation, you would call a weather API
        # This is a simplified mock implementation
        try:
            # # Simulate API call
            # weather_info = {
            #     "location": location,
            #     "temperature": "72°F",
            #     "condition": "Partly Cloudy",
            #     "humidity": "45%",
            #     "forecast": "Clear skies expected later"
            # }
            
            location_list = [
                "21.033333,105.849998",
                "minneapolis,mn",
                "43.67,-70.26",
                "80452",
                "berkley,uk"
            ]
            
            if location not in location_list: # for testing only
                location = location_list[0]
                        
            weather_info = self.__get_weather_of(location)
            return json.dumps(weather_info, indent=2)
        except Exception as e:
            return f"Weather error: {str(e)}"
        
    def __get_weather_of(self, location):
        print(f"retrieving data for {location}...")
        request_fields = [
            'place.name',
            'place.country',
            'periods.tempF',
            'periods.feelslikeF',
            'periods.humidity',
        ]

        formatted_fields = []
        formatted_fields = ','.join(request_fields)

        res = requests.request(
            method="GET",
            url=f"https://api.aerisapi.com/conditions/{location}",
            params={
                "client_id": "k6RBjSk0ADFQF22aa9FYf",
                "client_secret": "PwPEzNV8d8qqXz8ARLoBdWREMqkMbJ4dISqQPw3R",
                "fields": formatted_fields,
            }
        )
        
        if res.status_code != 200:
            raise Exception(f"status code was not 200: {res.status_code}")
        
        api_response_body = json.loads(res.text)
        
        return api_response_body


class TimeTool(Tool):
    def __init__(self):
        super().__init__("time", "Get current time information")
    
    def execute(self, input_text):
        """Get current time information"""
        try:
            from datetime import datetime
            import pytz
            
            # Get current time in UTC
            utc_now = datetime.now(pytz.UTC)
            
            # Format the time information
            time_info = {
                "utc_time": utc_now.strftime("%Y-%m-%d %H:%M:%S %Z"),
                "unix_timestamp": int(utc_now.timestamp()),
                "timezone": "UTC"
            }
            
            return json.dumps(time_info, indent=2)
        except Exception as e:
            return f"Time error: {str(e)}"
        
        
