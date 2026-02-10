"""
Time and date utilities using decorator approach.
"""

from datetime import datetime, timezone
from typing import List, Optional
from langchain_core.tools import tool
import pytz


@tool
def current_time(timezone_name: Optional[str] = None, format_type: str = "iso") -> str:
    """Get current time and date information.
    
    Args:
        timezone_name: Timezone (e.g., 'US/Eastern', 'UTC'). Defaults to UTC.
        format_type: Format for time display - "iso", "human", or "timestamp"
    
    Returns:
        str: Formatted current time information
    """
    try:
        # Validate format_type
        if format_type not in ["iso", "human", "timestamp"]:
            return "Error: format_type must be 'iso', 'human', or 'timestamp'"
            
        # Get current UTC time
        utc_now = datetime.now(timezone.utc)
        
        # Convert to requested timezone if specified
        if timezone_name:
            try:
                target_tz = pytz.timezone(timezone_name)
                local_time = utc_now.astimezone(target_tz)
                tz_display = timezone_name
            except pytz.exceptions.UnknownTimeZoneError:
                return f"Error: Unknown timezone '{timezone_name}'. Using UTC instead."
        else:
            local_time = utc_now
            tz_display = "UTC"
        
        # Format based on requested format
        if format_type == "human":
            formatted_time = local_time.strftime("%A, %B %d, %Y at %I:%M:%S %p %Z")
        elif format_type == "timestamp":
            formatted_time = str(int(local_time.timestamp()))
        else:  # iso format (default)
            formatted_time = local_time.isoformat()
        
        return f"""Current Time Information:
🕐 Time: {formatted_time}
🌍 Timezone: {tz_display}
📅 Date: {local_time.strftime("%Y-%m-%d")}
📆 Day: {local_time.strftime("%A")}
⏰ UTC Time: {utc_now.isoformat()}
#️⃣ Timestamp: {int(local_time.timestamp())}"""
        
    except Exception as e:
        return f"Time lookup failed: {str(e)}"


@tool
def world_clock(timezones: List[str] = None) -> str:
    """Get current time in multiple timezones.
    
    Args:
        timezones: List of timezones to display (defaults to ["UTC", "US/Eastern", "US/Pacific"])
    
    Returns:
        str: Formatted time information for multiple timezones
    """
    try:
        if timezones is None:
            timezones = ["UTC", "US/Eastern", "US/Pacific"]
            
        utc_now = datetime.now(timezone.utc)
        results = []
        results.append(f"World Clock - {utc_now.strftime('%Y-%m-%d')}")
        results.append("=" * 40)
        
        for tz_name in timezones:
            try:
                if tz_name == "UTC":
                    local_time = utc_now
                else:
                    tz = pytz.timezone(tz_name)
                    local_time = utc_now.astimezone(tz)
                
                formatted_time = local_time.strftime("%I:%M:%S %p")
                date_str = local_time.strftime("%A, %B %d")
                results.append(f"🌍 {tz_name}: {formatted_time} ({date_str})")
                
            except pytz.exceptions.UnknownTimeZoneError:
                results.append(f"❌ {tz_name}: Unknown timezone")
        
        return "\n".join(results)
        
    except Exception as e:
        return f"World clock lookup failed: {str(e)}"


# Legacy classes for backward compatibility
class TimeTool:
    def __init__(self):
        self.tool = current_time
        self.name = current_time.name
        self.description = current_time.description

class WorldClockTool:
    def __init__(self):
        self.tool = world_clock
        self.name = world_clock.name
        self.description = world_clock.description