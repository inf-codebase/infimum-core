"""
Web scraping and content extraction tools using decorator approach.
"""

import re
from urllib.parse import urljoin, urlparse
from langchain_core.tools import tool
import requests
from bs4 import BeautifulSoup


def _extract_main_content(soup: BeautifulSoup) -> str:
    """Extract main text content from the page."""
    # Try to find main content areas first
    main_selectors = [
        'main', 
        'article', 
        '[role="main"]',
        '.content',
        '.main-content',
        '#content',
        '#main'
    ]
    
    main_content = None
    for selector in main_selectors:
        main_content = soup.select_one(selector)
        if main_content:
            break
    
    # Fallback to body if no main content found
    if not main_content:
        main_content = soup.find('body')
    
    if not main_content:
        return soup.get_text()
    
    # Extract text and clean up
    text = main_content.get_text(separator=' ', strip=True)
    
    # Clean up whitespace
    text = re.sub(r'\s+', ' ', text)
    text = re.sub(r'\n\s*\n', '\n', text)
    
    return text.strip()


def _extract_links(soup: BeautifulSoup, base_url: str) -> list:
    """Extract links from the page."""
    links = []
    
    for link in soup.find_all('a', href=True):
        href = link['href']
        text = link.get_text(strip=True)
        
        # Convert relative URLs to absolute
        full_url = urljoin(base_url, href)
        
        # Skip empty links and fragments
        if not text or href.startswith('#'):
            continue
        
        # Skip javascript and mailto links
        if href.startswith(('javascript:', 'mailto:')):
            continue
        
        links.append({
            "text": text[:100],  # Limit text length
            "url": full_url,
            "is_external": urlparse(full_url).netloc != urlparse(base_url).netloc
        })
    
    # Remove duplicates based on URL
    seen_urls = set()
    unique_links = []
    for link in links:
        if link["url"] not in seen_urls:
            seen_urls.add(link["url"])
            unique_links.append(link)
    
    return unique_links[:20]  # Limit to 20 links


@tool
def web_scrape(url: str, extract_links: bool = False, max_length: int = 5000) -> str:
    """Extract text content and optionally links from web pages.
    
    Args:
        url: URL to scrape (must include http:// or https://)
        extract_links: Whether to extract links from the page (default: False)
        max_length: Maximum content length (100-50000, default: 5000)
    
    Returns:
        str: Formatted web page content including title, content, and optionally links
    """
    try:
        # Validate URL
        parsed = urlparse(url)
        if not parsed.scheme or not parsed.netloc:
            return "Error: Invalid URL format. URL must include http:// or https://"
        
        # Validate max_length
        if max_length < 100 or max_length > 50000:
            return "Error: max_length must be between 100 and 50000"
        
        # Set up headers to mimic a real browser
        headers = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36',
            'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,*/*;q=0.8',
            'Accept-Language': 'en-US,en;q=0.5',
            'Accept-Encoding': 'gzip, deflate',
            'Connection': 'keep-alive',
        }
        
        # Make the request
        response = requests.get(url, headers=headers, timeout=30, allow_redirects=True)
        response.raise_for_status()
        
        # Parse with BeautifulSoup
        soup = BeautifulSoup(response.content, 'html.parser')
        
        # Remove script and style elements
        for script in soup(["script", "style", "nav", "footer", "aside"]):
            script.decompose()
        
        # Extract title
        title = soup.find('title')
        title_text = title.get_text().strip() if title else "No title"
        
        # Extract main content
        content_text = _extract_main_content(soup)
        
        # Limit content length
        original_length = len(content_text)
        if len(content_text) > max_length:
            content_text = content_text[:max_length] + "... [truncated]"
        
        # Build result
        result_parts = [
            f"📄 Web Page Content",
            f"🌐 URL: {url}",
            f"📋 Title: {title_text}",
            f"📊 Status: {response.status_code}",
            f"📏 Content Length: {len(content_text)} chars" + (f" (truncated from {original_length})" if original_length > max_length else ""),
            "",
            "📖 Content:",
            content_text
        ]
        
        # Extract links if requested
        if extract_links:
            links = _extract_links(soup, url)
            if links:
                result_parts.append("")
                result_parts.append(f"🔗 Links found ({len(links)}):")
                for i, link in enumerate(links, 1):
                    external_indicator = " (external)" if link["is_external"] else ""
                    result_parts.append(f"{i}. {link['text']}{external_indicator}")
                    result_parts.append(f"   {link['url']}")
            else:
                result_parts.append("")
                result_parts.append("🔗 No links found")
        
        return "\n".join(result_parts)
        
    except requests.RequestException as e:
        return f"Failed to fetch URL: {str(e)}"
    except Exception as e:
        return f"Web scraping failed: {str(e)}"


# Legacy class for backward compatibility
class WebScrapeTool:
    def __init__(self):
        self.tool = web_scrape
        self.name = web_scrape.name
        self.description = web_scrape.description