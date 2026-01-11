from abc import ABC, abstractmethod
import asyncio
import random

from bs4 import BeautifulSoup

from src.core.ai.preprocessing.crawling_utils import jina_read_content, parse_proxy_url

from src.core.engine import ParameterizedInjection
from src.core.utils.constants import CrawlingType
from src.core.utils import auto_config
import requests
import json
import re
from pydantic import BaseModel, Field
from typing import List, Optional, Any
from playwright.sync_api import sync_playwright
import html2text
from firecrawl import FirecrawlApp
from loguru import logger
from playwright.async_api import async_playwright
from langchain_core.prompts import ChatPromptTemplate            
from typing import List


class Crawler(ABC):
    @abstractmethod
    def crawl(self, *args, **kwargs):
        pass
    
class PDFCrawler(Crawler):
    def crawl(self, pdf_url, **kwargs):
        """Extract text from PDF file on the intenet

        Args:
            pdf_url (str): url of pdf. 

        Raises:
            Exception: _description_

        Returns:
            _type_: _description_
        """
        
        reader_url = (
                "https://r.jina.ai/" + pdf_url
        ) 
        headers = {
            "Accept": "text/event-stream",
            "X-No-Cache": "true",
            "X-Timeout": "40",
            "X-Remove-Selector": "nav, header, footer",
        }

        retry_limit = 2
        for _ in range(retry_limit):
            response = requests.get(reader_url, headers=headers)

            if not (200 <= response.status_code < 400):
                raise Exception("Cannot scrape")

            last_stream = None
            for stream in response.iter_lines():
                if stream:
                    last_stream = stream.decode("utf-8")

            if last_stream is not None and last_stream.startswith("data: "):
                # Remove the 'data: ' prefix and parse the JSON
                json_data = last_stream[6:]
                parsed_data = json.loads(json_data)
                url_content: str = parsed_data.get("content", "")
                
                return url_content
        
class WebCrawler(Crawler):
    def __init__(self):
        self.h2t = html2text.HTML2Text()
        self.h2t.ignore_links = False
        self.h2t.ignore_images = False
        self.h2t.ignore_tables = False
    
    def crawl(self, url, dto_class, **kwargs):
        """Crawl content of a web link. 

        Args:
            url (str): Web url to be extracted. 

        Returns:
            _type_: _description_
        """
        self.url = url
        
        # Check if we're already in an event loop
        try:
            loop = asyncio.get_event_loop()
            if loop.is_running():
                # Use a different approach when already in a running loop
                # Create a new thread with a new event loop
                import threading
                import concurrent.futures
                
                def run_in_new_loop():
                    new_loop = asyncio.new_event_loop()
                    asyncio.set_event_loop(new_loop)
                    try:
                        return new_loop.run_until_complete(self.scrape())
                    finally:
                        new_loop.close()
                
                with concurrent.futures.ThreadPoolExecutor() as executor:
                    future = executor.submit(run_in_new_loop)
                    content = future.result()
            else:
                # If we're not in a running loop, we can use the existing one
                content = loop.run_until_complete(self.scrape())
        except RuntimeError:
            # If getting the event loop fails, fall back to asyncio.run
            content = asyncio.run(self.scrape())
            
        return content
    
    async def scrape(self) -> Optional[str]:
        """
        Scrape content using multiple strategies with fallbacks.
        Returns markdown formatted content.
        
        Strategy order:
        1. Simple requests (free, fast)
        2. Jina Reader (free API)
        3. Firecrawl (paid but reliable)
        4. Playwright (paid, complex JS rendering)
        """
        strategies = [
            self._try_requests_scrape,    # Try simple request first (free)
            self._try_jina_scrape,        # Then Jina Reader (free API)
            self._try_firecrawl_scrape,   # Then Firecrawl (paid service - but limited to 5000 requests)
            self._try_playwright_scrape    # Last resort for complex JS sites
        ]
        
        for strategy in strategies:
            try:
                content = await strategy()
                if content and len(content.strip()) > 1000:  # Basic content validation
                    logger.info(f"Successfully scraped content using {strategy.__name__}")
                    return self._clean_markdown(content)
            except Exception as e:
                logger.warning(f"Strategy {strategy.__name__} failed: {str(e)}")
                continue
                
        raise Exception(f"All scraping strategies failed for URL: {self.url}")

    async def _try_requests_scrape(self) -> Optional[str]:
        """Attempt to scrape using simple requests with BeautifulSoup (Free)"""
        headers = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36',
            'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8',
            'Accept-Language': 'en-US,en;q=0.5',
            'Connection': 'keep-alive',
        }
        
        response = requests.get(self.url, headers=headers, timeout=30)
        response.raise_for_status()
        
        soup = BeautifulSoup(response.text, 'html.parser')
        
        # Remove unwanted elements
        for element in soup.find_all(['script', 'style', 'nav', 'header', 'footer', 'iframe']):
            element.decompose()
            
        # Try to find main content with expanded selector list
        content = None
        selectors = [
            'main', 'article', '#content', '.content', 
            '[role="main"]', '.main-content', '#main-content',
            '.post-content', '.entry-content', '.article-content'
        ]
        
        for selector in selectors:
            content = soup.select_one(selector)
            if content and len(str(content).strip()) > 100:
                break
                
        html = str(content) if content else str(soup.body)
        return self.h2t.handle(html)

    async def _try_jina_scrape(self) -> Optional[str]:
        """Attempt to scrape using Jina Reader API (Free)"""
        return jina_read_content(self.url)
    
    async def _try_firecrawl_scrape(self) -> Optional[str]:
        """Attempt to scrape using Firecrawl (Paid service)"""
        app = FirecrawlApp(api_key=auto_config.FIRECRAWL_API_KEY)
        data = app.scrape_url(self.url, {
            'formats': ['markdown'],
            'markdownOptions': {
                'removeScripts': True,
                'removeStyles': True,
                'removeNavigation': True,
                'removeAds': True,
                'removeComments': True
            }
        })
        return data.get("markdown")

    async def _try_playwright_scrape(self) -> Optional[str]:
        """Attempt to scrape using Playwright with advanced features"""
        USER_AGENTS = [
            "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36",
            "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/605.1.15 (KHTML, like Gecko) Version/14.0.3 Safari/605.1.15",
            "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Edge/91.0.864.59 Safari/537.36"
        ]
        
        proxy = auto_config.PROXY_LIST
        user_agent = random.choice(USER_AGENTS)
        
        async with async_playwright() as p:
            browser_args = [
                '--disable-dev-shm-usage',
                '--no-sandbox',
                '--disable-setuid-sandbox',
                '--disable-blink-features=AutomationControlled',
                '--disable-infobars',
                '--window-size=1920,1080',
                '--start-maximized'
            ]
            
            try:
                browser = await p.chromium.launch(
                    headless=True,
                    proxy={"server": proxy},
                    args=browser_args
                )
                
                context = await browser.new_context(
                    viewport={'width': 1920, 'height': 1080},
                    user_agent=user_agent,
                    ignore_https_errors=True,
                    extra_http_headers={
                        'Accept-Language': 'en-US,en;q=0.9',
                        'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8',
                        'DNT': '1',
                        'Connection': 'keep-alive'
                    }
                )
                
                # Enable stealth mode
                await context.add_init_script("""
                    Object.defineProperty(navigator, 'webdriver', {
                        get: () => undefined
                    });
                """)
                
                page = await context.new_page()
                await page.set_default_timeout(30000)
                
                # Intercept and abort unnecessary requests
                await page.route("**/*.{png,jpg,jpeg,gif,svg,css,woff,woff2,ttf}", 
                    lambda route: route.abort())
                
                try:
                    response = await page.goto(self.url, 
                        wait_until="networkidle",
                        timeout=30000
                    )
                    
                    if not response.ok:
                        raise Exception(f"Failed to load page: {response.status}")
                    
                    # Wait for content to load
                    await page.wait_for_selector('body')
                    await asyncio.sleep(2)
                    
                    # Try to find and click "Accept Cookies" buttons
                    cookie_selectors = [
                        'button[id*="cookie"][id*="accept" i]',
                        'button[class*="cookie"][class*="accept" i]',
                        '[id*="cookie-banner"] button',
                        '[class*="cookie-banner"] button'
                    ]
                    
                    for selector in cookie_selectors:
                        try:
                            await page.click(selector, timeout=2000)
                            await asyncio.sleep(1)
                        except:
                            continue
                    
                    # Get main content with expanded selectors
                    content_selectors = [
                        'main',
                        'article',
                        '#content',
                        '.content',
                        '[role="main"]',
                        '.main-content',
                        '#main-content',
                        '.article-content',
                        'body'
                    ]
                    
                    for selector in content_selectors:
                        try:
                            element = await page.query_selector(selector)
                            if element:
                                # Remove unwanted elements
                                await page.evaluate("""(element) => {
                                    const removeSelectors = ['nav', 'header', 'footer', 'script', 'style', 'iframe', '.ad'];
                                    removeSelectors.forEach(selector => {
                                        element.querySelectorAll(selector).forEach(el => el.remove());
                                    });
                                }""", element)
                                
                                html = await element.inner_html()
                                if len(html.strip()) > 100:
                                    return self.h2t.handle(html)
                        except Exception as e:
                            logger.debug(f"Failed with selector {selector}: {str(e)}")
                            continue
                    
                    # Fallback to full body
                    html = await page.content()
                    return self.h2t.handle(html)
                    
                except Exception as e:
                    logger.error(f"Playwright scraping failed: {str(e)}")
                    raise
                
                finally:
                    await page.close()
                    
            except Exception as e:
                logger.error(f"Playwright browser failed: {str(e)}")
                raise
            
            finally:
                if 'browser' in locals():
                    await browser.close()
    
    def _clean_markdown(self, content: str) -> str:
            """Clean and normalize markdown content"""
            lines = content.split('\n')
            cleaned_lines = []
            
            for line in lines:
                line = line.strip()
                if line and not line.isspace():
                    # Remove excessive newlines and spaces
                    line = ' '.join(line.split())
                    cleaned_lines.append(line)
                    
            return '\n\n'.join(cleaned_lines)
        
    
class InterestRateItem(BaseModel):
    tenor: str = Field(..., description="The tenor period like 1M, 3M, 6M, etc.")
    rate: str = Field(..., description="The interest rate as a string representation of a percentage")
    
class InterestRateList(BaseModel):
    rates: List[InterestRateItem] = Field(..., description="List of interest rates with tenor and rate")
    
class TextCrawler(Crawler):
    MODELS = [
        auto_config.OPENAI_MODEL,  # OpenAI
        auto_config.GEMINI_MODEL,  # Gemini
        auto_config.MISTRAL_MODEL,  # Mistral
        auto_config.XAI_MODEL,  # Grok
        auto_config.DEEPSEEK_MODEL  # DeepSeek
    ]
    
    def crawl(self, text_prompt, dto_class, model=None, retry_count=0) -> List[BaseModel]:
        """Extract data from text using AI models
        
        Args:
            text_prompt: Text prompt to send to AI model
            dto_class: Data transfer object class to convert response to
            model: Specific model to use, defaults to first available model
            retry_count: Number of retries attempted
            
        Returns:
            List of data objects converted from model response
        """
        if not model:
            model = self.MODELS[0]
            
        try:
            # Call the appropriate model API
            if "grok" in model.lower():
                response = self._call_xai_api(text_prompt, model)
            elif "gpt" in model.lower():
                response = self._call_openai_api(text_prompt, model)
            elif "gemini" in model.lower():
                response = self._call_gemini_api(text_prompt, model)
            elif "mistral" in model.lower():
                response = self._call_mistral_api(text_prompt, model)
            elif "deepseek" in model.lower():
                response = self._call_deepseek_api(text_prompt, model)
            else:
                raise ValueError(f"Unsupported model: {model}")
                
            # Extract JSON from response and convert to dto_class
            return self.extract_json_from_output(response, dto_class, model, retry_count)
            
        except Exception as e:
            logger.error(f"Error using model {model}: {str(e)}")
            return self._try_next_model(text_prompt, dto_class, model, retry_count)
    
    def extract_json_from_output(self, raw_output, dto_class, model=None, retry_count=0):
        """Extract JSON from model output and convert to DTO objects"""
        
        # Use convert_response_to_schema if raw_output is an AI message object
        if hasattr(raw_output, 'tool_calls') or hasattr(raw_output, 'additional_kwargs'):
            content = self.convert_response_to_schema(raw_output, dto_class)
        else:
            content = raw_output
            
        # If content is not a string (e.g., it's already a parsed object), return it
        if not isinstance(content, str):
            return content
                    
        try:
            # Try to extract JSON array from the response
            json_str = re.search(r'\[.*\]', content, re.DOTALL)
            if json_str:
                json_str = json_str.group()
            else:
                # If no JSON array found, check for JSON object
                json_str = re.search(r'\{.*\}', content, re.DOTALL)
                if json_str:
                    json_str = json_str.group()
                else:
                    json_str = content
                    
            # Clean up the JSON string
            json_str = json_str.replace("'", '"')
            json_str = re.sub(r'(\w+):', r'"\1":', json_str)  # Convert keys to quoted strings
            
            # Parse the JSON
            import json
            parsed_data = json.loads(json_str)
            
            # Convert to list if it's a dict
            if isinstance(parsed_data, dict):
                parsed_data = [parsed_data]
                
            # Convert each item to the DTO class
            return [dto_class(**item) for item in parsed_data]
            
        except Exception as e:
            logger.error(f"Error parsing output from {model}: {str(e)}")
            logger.debug(f"Raw output: {content}")
            
            if retry_count < len(self.MODELS) - 1:
                # Try with next model
                return self._try_next_model(content, dto_class, model, retry_count)
            return []
    
    def convert_response_to_schema(self, response: str, schema_type: Any) -> Any:
        """Extract data from AIMessage response format and convert to a list of schema objects"""
        try:
            # Case 1: Response has direct tool_calls attribute
            if hasattr(response, 'tool_calls') and response.tool_calls:
                tool_call = response.tool_calls[0]
                if 'args' in tool_call:
                    args = tool_call['args']
                    # Check if args is a list or a single object
                    if isinstance(args, list):
                        return [schema_type(**item) for item in args]
                    elif isinstance(args, dict) and "rates" in args and isinstance(args["rates"], list):
                        return [schema_type(**item) for item in args["rates"]]
                    elif isinstance(args, InterestRateList):
                        return [schema_type(**item) for item in args.rates]
                    else:
                        return [schema_type(**args)]
            
            # Case 2: Response has tool_calls in additional_kwargs
            elif hasattr(response, 'additional_kwargs') and 'tool_calls' in response.additional_kwargs:
                tool_call = response.additional_kwargs['tool_calls'][0]
                
                # Extract arguments
                if 'args' in tool_call:
                    args = tool_call['args']
                else:
                    # Parse from function arguments JSON string
                    import json
                    args = json.loads(tool_call['function']['arguments'])
                
                # Convert to list of schema objects
                if isinstance(args, list):
                    return [schema_type(**item) for item in args]
                elif isinstance(args, dict) and "rates" in args and isinstance(args["rates"], list):
                    return [schema_type(**item) for item in args["rates"]]  
                elif isinstance(args, InterestRateList):
                    return [schema_type(**item) for item in args.rates]
                else:
                    return [schema_type(**args)]
            
            # Case 3: Response has content attribute for text responses
            elif hasattr(response, 'content'):
                return response.content
            
            # Default case: Return the response itself
            return response
            
        except Exception as e:
            logger.error(f"Error converting response to schema: {str(e)}")
            # Return the raw response content as fallback
            if hasattr(response, 'content'):
                return response.content
            return response
            
    def _try_next_model(self, text_prompt, dto_class, current_model, retry_count):
        """Try the next available model if the current one fails"""
        retry_count += 1
        if retry_count >= len(self.MODELS):
            logger.error("All models failed to extract data")
            return []
            
        # Find index of current model and get the next one
        try:
            current_idx = self.MODELS.index(current_model)
            next_model = self.MODELS[(current_idx + 1) % len(self.MODELS)]
        except ValueError:
            next_model = self.MODELS[0]
            
        logger.info(f"Trying next model: {next_model}")
        return self.crawl(text_prompt, dto_class, next_model, retry_count)

    def _call_api_with_schema(self, text_prompt, model, provider="openai"):
        """Generic method to call any LLM API with schema binding for structured output
        
        Args:
            text_prompt: The prompt to send to the LLM
            model: The specific model to use
            provider: The provider (openai, gemini, mistral, etc.)
            
        Returns:
            The structured response from the LLM
        """
        try:
            # Set up system message
            system_message = (
                "You are a financial data extraction expert. "
                "Extract interest rate information from the provided text. "
                "Return a structured list of tenor and rate pairs."
            )
            
            # Create prompt template
            prompt = ChatPromptTemplate.from_messages([
                ("system", system_message),
                ("human", "{text_prompt}")
            ])
            
            # Get the appropriate LLM based on provider
            if provider == "openai":
                from langchain_openai import ChatOpenAI
                llm = ChatOpenAI(model=model, api_key=auto_config.get_random_config(auto_config.OPENAI_API_KEY))
            elif provider == "gemini":
                from langchain_google_genai import ChatGoogleGenerativeAI
                llm = ChatGoogleGenerativeAI(model=model, api_key=auto_config.get_random_config(auto_config.GOOGLE_API_KEY))
            elif provider == "mistral":
                from langchain_mistralai import ChatMistralAI
                llm = ChatMistralAI(model=model, api_key=auto_config.get_random_config(auto_config.MISTRAL_API_KEY))
            elif provider == "xai":
                from langchain_xai import ChatXai
                llm = ChatXai(model=model, api_key=auto_config.get_random_config(auto_config.XAI_API_KEY))
            elif provider == "deepseek":
                from langchain_deepseek import ChatDeepseek
                llm = ChatDeepseek(model=model, api_key=auto_config.get_random_config(auto_config.DEEPSEEK_API_KEY))
            elif provider == "local":
                from langchain_community.llms import LlamaCpp
                llm = LlamaCpp(model_path=auto_config.get_random_config(auto_config.LOCAL_MODEL_PATH))
            else:
                raise ValueError(f"Unsupported provider: {provider}")
            
            # Bind schema to LLM
            llm_with_schema = llm.bind_tools([InterestRateList])
            
            # Create chain
            chain = prompt | llm_with_schema
            
            # Invoke chain
            response = chain.invoke({"text_prompt": text_prompt})
            
            logger.debug(f"Response from {provider} ({model}): {response}")
            return response
            
        except Exception as e:
            logger.error(f"Error calling {provider} API with {model}: {str(e)}")
            # Return a raw string as fallback
            return f"Error extracting interest rates. Please check the following text: {text_prompt}"
    
    def _call_openai_api(self, text_prompt, model):
        """Call OpenAI API with schema binding"""
        return self._call_api_with_schema(text_prompt, model, provider="openai")
    
    def _call_gemini_api(self, text_prompt, model):
        """Call Gemini API with schema binding"""
        return self._call_api_with_schema(text_prompt, model, provider="gemini")
    
    def _call_mistral_api(self, text_prompt, model):
        """Call Mistral API with schema binding"""
        return self._call_api_with_schema(text_prompt, model, provider="mistral")
    
    def _call_xai_api(self, text_prompt, model):
        """Call XAI (Grok) API with schema binding"""
        return self._call_api_with_schema(text_prompt, model, provider="xai")
    
    def _call_deepseek_api(self, text_prompt, model):
        """Call Deepseek API with schema binding"""
        return self._call_api_with_schema(text_prompt, model, provider="deepseek")
    
    def _call_local_api(self, text_prompt, model):
        """Call local model API with schema binding"""
        return self._call_api_with_schema(text_prompt, model, provider="local")

# Factory class to create crawlers
class CrawlerFactory:
    @staticmethod
    def create_crawler(crawling_type: CrawlingType) -> Crawler:
        """
        Factory method to create specific crawler based on crawling type
        
        Args:
            crawling_type (CrawlingType): Type of crawler to create
        
        Returns:
            Crawler: Specific crawler instance
        
        Raises:
            ValueError: If an unsupported crawling type is provided
        """
        crawlers = {
            CrawlingType.PDF: PDFCrawler,
            CrawlingType.WEB: WebCrawler,
            CrawlingType.TEXT: TextCrawler
        }
        
        crawler_class = crawlers.get(crawling_type)
        if not crawler_class:
            raise ValueError(f"Unsupported crawling type: {crawling_type}")
        
        return crawler_class()
    

class CrawlerFactoryInjection(ParameterizedInjection):
    def __init__(self, crawling_type: CrawlingType):
        super().__init__()
        self.crawling_type = crawling_type
        
    def on_call_function_action_and_return_params(self):
        return [CrawlerFactory.create_crawler(self.crawling_type)]
    
    def on_call_params_action(self):
        return super().on_call_params_action()