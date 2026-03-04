# knowledge_updater/connectors.py (enhanced version)

import time
import random
import logging
import requests
from urllib.parse import urlparse, urljoin
from typing import List, Dict, Any, Optional
from pathlib import Path

from langchain_core.documents import Document
from bs4 import BeautifulSoup

# For JavaScript-heavy sites
try:
    from playwright.sync_api import sync_playwright
    PLAYWRIGHT_AVAILABLE = True
except ImportError:
    PLAYWRIGHT_AVAILABLE = False

# For full website crawling
try:
    import crawlfish
    CRAWLFISH_AVAILABLE = True
except ImportError:
    CRAWLFISH_AVAILABLE = False

logger = logging.getLogger(__name__)

class AdvancedWebCrawler:
    """
    Intelligent web crawler with anti-detection features and support for both
    static and dynamic sites.
    """
    
    def __init__(self, respect_robots=True, rate_delay=(2, 5), 
                 use_proxy_rotation=False, proxy_list=None):
        self.respect_robots = respect_robots
        self.rate_delay = rate_delay  # (min, max) seconds between requests
        self.use_proxy_rotation = use_proxy_rotation
        self.proxy_list = proxy_list or []
        self.session = requests.Session()
        self._setup_session()
        
    def _setup_session(self):
        """Configure session with realistic browser headers."""
        self.session.headers.update({
            'User-Agent': self._get_random_user_agent(),
            'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,*/*;q=0.8',
            'Accept-Language': 'en-US,en;q=0.5',
            'Accept-Encoding': 'gzip, deflate',
            'Connection': 'keep-alive',
            'Upgrade-Insecure-Requests': '1',
        })
    
    def _get_random_user_agent(self):
        """Rotate through realistic user agents [citation:3]."""
        agents = [
            'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36',
            'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/605.1.15 (KHTML, like Gecko) Version/17.1 Safari/605.1.15',
            'Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/119.0.0.0 Safari/537.36'
        ]
        return random.choice(agents)
    
    def _respectful_delay(self):
        """Add random delay between requests to avoid rate limiting [citation:7]."""
        delay = random.uniform(self.rate_delay[0], self.rate_delay[1])
        time.sleep(delay)
    
    def check_robots_txt(self, url):
        """Check and respect robots.txt rules [citation:5]."""
        if not self.respect_robots:
            return True
            
        try:
            from urllib.robotparser import RobotFileParser
            parsed = urlparse(url)
            robots_url = f"{parsed.scheme}://{parsed.netloc}/robots.txt"
            
            rp = RobotFileParser()
            rp.set_url(robots_url)
            rp.read()
            
            return rp.can_fetch(self.session.headers['User-Agent'], url)
        except Exception as e:
            logger.warning(f"Could not check robots.txt for {url}: {e}")
            return True  # Assume allowed if check fails
    
    def scrape_static(self, url: str) -> Optional[str]:
        """
        Scrape static sites using requests + BeautifulSoup.
        Best for simple pages where content is in initial HTML [citation:1].
        """
        try:
            self._respectful_delay()
            response = self.session.get(url, timeout=30)
            response.raise_for_status()
            
            soup = BeautifulSoup(response.text, 'html.parser')
            
            # Remove script/style elements for cleaner text
            for script in soup(["script", "style"]):
                script.decompose()
                
            return soup.get_text()
        except Exception as e:
            logger.error(f"Static scrape failed for {url}: {e}")
            return None
    
    def scrape_dynamic(self, url: str, wait_for_selector: str = None) -> Optional[str]:
        """
        Scrape JavaScript-heavy sites using Playwright.
        Essential for SPAs and dynamic content [citation:1][citation:8].
        """
        if not PLAYWRIGHT_AVAILABLE:
            logger.error("Playwright not installed. Install with: pip install playwright")
            return None
            
        try:
            with sync_playwright() as p:
                # Launch browser in stealth mode
                browser = p.chromium.launch(headless=True)
                context = browser.new_context(
                    viewport={'width': 1920, 'height': 1080},
                    user_agent=self._get_random_user_agent()
                )
                page = context.new_page()
                
                # Navigate and wait for content
                page.goto(url, wait_until='networkidle')
                
                if wait_for_selector:
                    page.wait_for_selector(wait_for_selector, timeout=10000)
                
                # Optional: simulate human behavior [citation:3]
                page.mouse.move(random.randint(100, 500), random.randint(100, 500))
                
                content = page.content()
                browser.close()
                
                # Parse with BeautifulSoup for clean text
                soup = BeautifulSoup(content, 'html.parser')
                for script in soup(["script", "style"]):
                    script.decompose()
                    
                return soup.get_text()
        except Exception as e:
            logger.error(f"Dynamic scrape failed for {url}: {e}")
            return None
    
    def crawl_website(self, start_url: str, max_pages: int = 50) -> List[Document]:
        """
        Crawl entire website using crawlfish library.
        Discovers and processes all pages within the domain [citation:2].
        """
        if not CRAWLFISH_AVAILABLE:
            logger.error("crawlfish not installed. Install with: pip install crawlfish")
            return []
            
        try:
            logger.info(f"Starting website crawl of {start_url} (max {max_pages} pages)")
            website_report = crawlfish.explore_website(start_url, crawl_limit=max_pages)
            
            documents = []
            domain = urlparse(start_url).netloc
            
            for page_data in website_report.pages_data:
                # Extract clean text content
                text = page_data.get('text', '')
                if not text and page_data.get('html'):
                    soup = BeautifulSoup(page_data['html'], 'html.parser')
                    for script in soup(["script", "style"]):
                        script.decompose()
                    text = soup.get_text()
                
                doc = Document(
                    page_content=text,
                    metadata={
                        'source': page_data.get('url'),
                        'domain': domain,
                        'title': page_data.get('title', ''),
                        'crawl_depth': page_data.get('depth', 0)
                    }
                )
                documents.append(doc)
                
            logger.info(f"Crawled {len(documents)} pages from {start_url}")
            return documents
            
        except Exception as e:
            logger.error(f"Website crawl failed for {start_url}: {e}")
            return []
    
    def scrape_with_ai(self, url: str, extraction_prompt: str = None) -> Optional[str]:
        """
        Advanced: Use AI to intelligently extract and structure content.
        Converts web content to clean markdown for RAG systems [citation:4].
        """
        try:
            # Option 1: Use crawl4ai (modern AI-powered scraper)
            from crawl4ai import WebCrawler
            
            crawler = WebCrawler(verbose=True)
            result = crawler.crawl(url)
            
            # Returns clean markdown, ideal for LLM consumption
            return result.markdown if result else None
            
        except ImportError:
            # Option 2: Fallback to firecrawl or similar
            try:
                import requests
                response = requests.post(
                    'https://api.firecrawl.dev/v1/scrape',
                    json={'url': url, 'formats': ['markdown']}
                )
                if response.status_code == 200:
                    return response.json().get('markdown')
            except:
                pass
                
            logger.warning("AI scraping libraries not available")
            return None
    
    def extract_specific_elements(self, url: str, css_selectors: Dict[str, str]) -> Dict[str, Any]:
        """
        Extract specific data fields using CSS selectors [citation:8].
        Useful for structured data like product prices, reviews, etc.
        
        Example:
            selectors = {
                'title': 'h1.product-title',
                'price': '.price',
                'description': 'div.description'
            }
        """
        result = {}
        try:
            html = self.scrape_static(url) or self.scrape_dynamic(url)
            if not html:
                return result
                
            soup = BeautifulSoup(html, 'html.parser')
            
            for field, selector in css_selectors.items():
                elements = soup.select(selector)
                if elements:
                    # Handle multiple elements (lists)
                    if len(elements) > 1:
                        result[field] = [el.get_text(strip=True) for el in elements]
                    else:
                        result[field] = elements[0].get_text(strip=True)
                        
            return result
        except Exception as e:
            logger.error(f"Element extraction failed for {url}: {e}")
            return result
