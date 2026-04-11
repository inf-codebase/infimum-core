import json
import requests
from loguru import logger
from typing import Optional

from urllib.parse import urlparse



def remove_trailing_slash(url: str):
    """Remove trailing slash from a URL."""
    return url.rstrip("/")


def normalize_url_data(url_data):
    """Normalize URL data by trimming and removing trailing slash from the URL."""
    normalized_url_data = url_data.copy()
    normalized_url_data["url"] = remove_trailing_slash(url_data.get("url", "").strip())

    return normalized_url_data


def parse_proxy_url(proxy_url: str):
    parsed_url = urlparse(proxy_url)
    proxy_dict = {
        "server": f"{parsed_url.scheme}://{parsed_url.hostname}:{parsed_url.port}",
    }
    if parsed_url.username and parsed_url.password:
        proxy_dict["username"] = parsed_url.username
        proxy_dict["password"] = parsed_url.password
    return proxy_dict


def jina_read_content(url: str, retry_limit: int = 3) -> Optional[str]:
    """
    Get content from a URL using Jina Reader API.
    
    Args:
        url (str): The URL to read content from
        retry_limit (int): Number of retry attempts
        
    Returns:
        Optional[str]: The extracted content or None if failed
    """
    reader_url = f"https://r.jina.ai/{url}"
    headers = {
        "Accept": "text/event-stream",
        "X-No-Cache": "true",
        "X-Timeout": "40",
        "X-Remove-Selector": "nav, header, footer",
    }

    logger.info(f"Jina is reading URL: {url}")
    
    for attempt in range(retry_limit):
        try:
            response = requests.get(reader_url, headers=headers, timeout=45)
            response.raise_for_status()

            last_stream = None
            for stream in response.iter_lines():
                if stream:
                    last_stream = stream.decode("utf-8")

            if last_stream and last_stream.startswith("data: "):
                json_data = last_stream[6:]
                parsed_data = json.loads(json_data)
                content: str = parsed_data.get("content", "")
                
                if content:
                    return content
                    
            logger.warning(f"Attempt {attempt + 1}/{retry_limit} failed to get content")
            
        except Exception as e:
            logger.error(f"Error reading content: {str(e)}")
            if attempt == retry_limit - 1:
                raise

    return None 