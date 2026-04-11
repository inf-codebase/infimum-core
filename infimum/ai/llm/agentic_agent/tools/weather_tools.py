"""
Weather information tools using decorator approach.
"""

import os
import random
from typing import Any, Dict, Optional
from langchain_core.tools import tool
import requests


# Global variable to store the API key
_weather_api_key: Optional[str] = None

def set_weather_api_key(api_key: str):
    """Set the OpenWeather API key."""
    global _weather_api_key
    _weather_api_key = api_key


def _get_mock_weather(location: str, units: str = "metric") -> str:
    """Generate mock weather data when API key is not available."""
    # Mock weather conditions
    conditions = [
        "Clear", "Partly Cloudy", "Cloudy", "Light Rain", 
        "Heavy Rain", "Thunderstorm", "Snow", "Fog"
    ]
    
    temp = round(random.uniform(10, 30), 1)
    feels_like = round(random.uniform(8, 32), 1)
    description = random.choice(conditions)
    humidity = random.randint(30, 90)
    wind_speed = round(random.uniform(0, 15), 1)
    units_symbol = "°C" if units == "metric" else "°F"
    
    return f"""Weather for {location} (Mock Data):
🌡️ Temperature: {temp}{units_symbol} (feels like {feels_like}{units_symbol})
🌤️ Condition: {description}
💧 Humidity: {humidity}%
💨 Wind Speed: {wind_speed} m/s

Note: Mock data - set OPENWEATHER_API_KEY for real weather data"""


@tool
def weather(location: str, units: str = "metric") -> str:
    """Get current weather information for a location.
    
    Args:
        location: Location to get weather for (city, coordinates, etc.)
        units: Units for temperature - "metric" for Celsius or "imperial" for Fahrenheit
    
    Returns:
        str: Formatted weather information including temperature, conditions, humidity, etc.
    """
    try:
        # Validate units
        if units not in ["metric", "imperial"]:
            return "Error: units must be 'metric' or 'imperial'"
            
        # Get API key from global variable or environment
        api_key = _weather_api_key or os.getenv("OPENWEATHER_API_KEY")
        if not api_key:
            return _get_mock_weather(location, units)
        
        # OpenWeatherMap API endpoint
        base_url = "http://api.openweathermap.org/data/2.5/weather"
        params = {
            "q": location,
            "appid": api_key,
            "units": units,
        }
        
        response = requests.get(base_url, params=params, timeout=10)
        response.raise_for_status()
        
        data = response.json()
        
        # Extract relevant information
        location_name = f"{data['name']}, {data['sys']['country']}"
        temp = data['main']['temp']
        feels_like = data['main']['feels_like']
        description = data['weather'][0]['description'].title()
        humidity = data['main']['humidity']
        pressure = data['main']['pressure']
        wind_speed = data.get('wind', {}).get('speed', 0)
        wind_direction = data.get('wind', {}).get('deg', 0)
        cloudiness = data['clouds']['all']
        visibility = data.get('visibility', 0) / 1000  # Convert to km
        
        units_symbol = "°C" if units == "metric" else "°F"
        wind_unit = "m/s" if units == "metric" else "mph"
        
        return f"""Weather for {location_name}:
🌡️ Temperature: {temp}{units_symbol} (feels like {feels_like}{units_symbol})
🌤️ Condition: {description}
💧 Humidity: {humidity}%
📊 Pressure: {pressure} hPa
💨 Wind: {wind_speed} {wind_unit} at {wind_direction}°
☁️ Cloudiness: {cloudiness}%
👁️ Visibility: {visibility} km"""
        
    except requests.RequestException as e:
        return f"Weather API request failed: {str(e)}"
    except KeyError as e:
        return f"Unexpected weather API response format: {str(e)}"
    except Exception as e:
        return f"Weather lookup failed: {str(e)}"


# Legacy class for backward compatibility
class WeatherTool:
    def __init__(self, api_key: Optional[str] = None):
        if api_key:
            set_weather_api_key(api_key)
        self.tool = weather
        self.name = weather.name
        self.description = weather.description