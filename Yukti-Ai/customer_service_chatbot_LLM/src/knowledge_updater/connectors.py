"""
Data source connectors for knowledge updater.
Supports CSV files, RSS feeds, REST APIs, local uploads, and website crawling.
Now includes a function for on‑demand scraping of any URL.
"""

import logging
import time
import requests
import feedparser
from urllib.parse import urlparse
from pathlib import Path
from typing import List, Dict, Any, Optional

from langchain_core.documents import Document
from langchain_community.document_loaders import CSVLoader, TextLoader, PyPDFLoader
from bs4 import BeautifulSoup

# For JavaScript‑heavy sites
try:
    from playwright.sync_api import sync_playwright
    PLAYWRIGHT_AVAILABLE = True
except ImportError:
    PLAYWRIGHT_AVAILABLE = False

from .config import DATASET_PATH, UPLOADS_PATH, SOURCES

logger = logging.getLogger(__name__)

# ----------------------------------------------------------------------
# Static sources (for periodic rebuild)
# ----------------------------------------------------------------------

def fetch_csv(file_path: Path) -> List[Document]:
    """Read a CSV file with multiple encoding fallbacks."""
    if not file_path.exists():
        logger.warning(f"CSV file not found: {file_path}")
        return []
    encodings = ['utf-8', 'cp1252', 'latin-1', 'iso-8859-1']
    for enc in encodings:
        try:
            loader = CSVLoader(str(file_path), encoding=enc)
            docs = loader.load()
            logger.info(f"Loaded {len(docs)} documents from {file_path.name} (encoding {enc})")
            return docs
        except UnicodeDecodeError:
            continue
        except Exception as e:
            logger.warning(f"Failed to load {file_path} with {enc}: {e}")
    logger.error(f"Could not read {file_path} with any encoding.")
    return []

def fetch_uploads() -> List[Document]:
    """Scan UPLOADS_PATH for supported files and load them."""
    supported_extensions = {'.csv': CSVLoader, '.txt': TextLoader, '.pdf': PyPDFLoader}
    docs = []
    for file_path in UPLOADS_PATH.glob("*"):
        if file_path.is_file() and file_path.suffix.lower() in supported_extensions:
            loader_class = supported_extensions[file_path.suffix.lower()]
            try:
                loader = loader_class(str(file_path))
                file_docs = loader.load()
                docs.extend(file_docs)
                logger.info(f"Loaded {len(file_docs)} docs from {file_path.name}")
            except Exception as e:
                logger.error(f"Failed to load {file_path.name}: {e}")
    return docs

def fetch_rss(feed_config: Dict[str, Any]) -> List[Document]:
    """Fetch entries from an RSS feed."""
    if not feed_config.get("enabled", False):
        return []
    url = feed_config["url"]
    try:
        feed = feedparser.parse(url)
        docs = []
        for entry in feed.entries:
            content = entry.get("summary", entry.get("description", ""))
            if content:
                doc = Document(
                    page_content=content,
                    metadata={"source": url, "title": entry.get("title"), "published": entry.get("published")}
                )
                docs.append(doc)
        logger.info(f"Fetched {len(docs)} items from RSS {url}")
        return docs
    except Exception as e:
        logger.error(f"RSS fetch failed for {url}: {e}")
        return []

def fetch_api(api_config: Dict[str, Any]) -> List[Document]:
    """Fetch data from a REST API (expects JSON array)."""
    if not api_config.get("enabled", False):
        return []
    url = api_config["url"]
    try:
        resp = requests.get(url, timeout=10)
        resp.raise_for_status()
        data = resp.json()
        docs = []
        if isinstance(data, list):
            for item in data:
                content = str(item)
                doc = Document(page_content=content, metadata={"source": url})
                docs.append(doc)
        logger.info(f"Fetched {len(docs)} items from API {url}")
        return docs
    except Exception as e:
        logger.error(f"API fetch failed for {url}: {e}")
        return []

def fetch_websites() -> List[Document]:
    """Crawl websites configured in web_sources.json."""
    docs = []
    for site in SOURCES.get("websites", []):
        if not site.get("enabled", False):
            continue
        url = site["url"]
        max_pages = site.get("max_pages", 50)
        use_js = site.get("use_javascript", False)
        logger.info(f"Crawling {url} (max {max_pages} pages, JS={use_js})")
        try:
            # We'll use a simple crawler – for production, consider using a dedicated library like `crawlfish`
            from .crawler import crawl_website
            site_docs = crawl_website(url, max_pages, use_js)
            docs.extend(site_docs)
            logger.info(f"Crawled {len(site_docs)} pages from {url}")
        except Exception as e:
            logger.error(f"Website crawl failed for {url}: {e}")
    return docs

def fetch_all_sources() -> List[Document]:
    """Fetch documents from all enabled sources."""
    all_docs = []
    all_docs.extend(fetch_csv(DATASET_PATH))
    all_docs.extend(fetch_uploads())
    for feed in SOURCES.get("rss", []):
        all_docs.extend(fetch_rss(feed))
    for api in SOURCES.get("api", []):
        all_docs.extend(fetch_api(api))
    all_docs.extend(fetch_websites())
    return all_docs

# ----------------------------------------------------------------------
# On‑demand web scraper (for real‑time user queries)
# ----------------------------------------------------------------------

class WebScraper:
    """
    Scrapes any URL on demand, with support for both static and JavaScript‑rendered pages.
    Respects robots.txt and includes polite delays.
    """
    def __init__(self, respect_robots=True, delay=1.0):
        self.respect_robots = respect_robots
        self.delay = delay
        self.session = requests.Session()
        self.session.headers.update({
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36'
        })

    def _check_robots(self, url):
        if not self.respect_robots:
            return True
        from urllib.robotparser import RobotFileParser
        parsed = urlparse(url)
        robots_url = f"{parsed.scheme}://{parsed.netloc}/robots.txt"
        rp = RobotFileParser()
        rp.set_url(robots_url)
        try:
            rp.read()
            return rp.can_fetch(self.session.headers['User-Agent'], url)
        except Exception as e:
            logger.warning(f"Could not check robots.txt for {url}: {e}")
            return True

    def scrape_static(self, url: str) -> Optional[str]:
        """Scrape a static page using requests + BeautifulSoup."""
        if not self._check_robots(url):
            logger.warning(f"robots.txt disallows scraping {url}")
            return None
        time.sleep(self.delay)
        try:
            resp = self.session.get(url, timeout=30)
            resp.raise_for_status()
            soup = BeautifulSoup(resp.text, 'html.parser')
            # Remove script/style elements
            for script in soup(["script", "style"]):
                script.decompose()
            text = soup.get_text()
            lines = (line.strip() for line in text.splitlines())
            chunks = (phrase.strip() for line in lines for phrase in line.split("  "))
            text = ' '.join(chunk for chunk in chunks if chunk)
            return text
        except Exception as e:
            logger.error(f"Static scrape failed for {url}: {e}")
            return None

    def scrape_dynamic(self, url: str, wait_for_selector: str = None) -> Optional[str]:
        """Scrape a JavaScript‑heavy page using Playwright."""
        if not PLAYWRIGHT_AVAILABLE:
            logger.error("Playwright not installed. Cannot scrape dynamic pages.")
            return None
        if not self._check_robots(url):
            logger.warning(f"robots.txt disallows scraping {url}")
            return None
        time.sleep(self.delay)
        try:
            with sync_playwright() as p:
                browser = p.chromium.launch(headless=True)
                context = browser.new_context(
                    viewport={'width': 1920, 'height': 1080},
                    user_agent=self.session.headers['User-Agent']
                )
                page = context.new_page()
                page.goto(url, wait_until='networkidle')
                if wait_for_selector:
                    page.wait_for_selector(wait_for_selector, timeout=10000)
                content = page.content()
                browser.close()
                soup = BeautifulSoup(content, 'html.parser')
                for script in soup(["script", "style"]):
                    script.decompose()
                text = soup.get_text()
                lines = (line.strip() for line in text.splitlines())
                chunks = (phrase.strip() for line in lines for phrase in line.split("  "))
                text = ' '.join(chunk for chunk in chunks if chunk)
                return text
        except Exception as e:
            logger.error(f"Dynamic scrape failed for {url}: {e}")
            return None

    def scrape(self, url: str, use_js: bool = False, wait_for: str = None) -> Optional[str]:
        """Unified scrape method."""
        if use_js:
            return self.scrape_dynamic(url, wait_for)
        else:
            return self.scrape_static(url)

# Global scraper instance
scraper = WebScraper()

def scrape_url(url: str, use_js: bool = False) -> Optional[str]:
    """Convenience function to scrape a URL and return clean text."""
    return scraper.scrape(url, use_js)
