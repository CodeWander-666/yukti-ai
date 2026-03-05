"""
Enhanced website crawler for indexing entire sites.
Uses requests + BeautifulSoup for static sites, and optionally Playwright for JS.
Includes anti‑block strategies: rotating user agents, smart delays, proxy support.
"""

import time
import logging
from urllib.parse import urljoin, urlparse
from collections import deque
from typing import List, Optional

import requests
from bs4 import BeautifulSoup
from langchain_core.documents import Document

# Import the anti‑block manager
from .anti_block import AntiBlockManager

logger = logging.getLogger(__name__)

try:
    from playwright.sync_api import sync_playwright
    PLAYWRIGHT_AVAILABLE = True
except ImportError:
    PLAYWRIGHT_AVAILABLE = False

# Global anti‑block instance
_anti_block = AntiBlockManager()

def crawl_website(start_url: str, max_pages: int = 50, use_js: bool = False) -> List[Document]:
    """
    Crawl a website starting from start_url, up to max_pages.
    Returns a list of Documents with page text and metadata.
    Uses anti‑block strategies to avoid detection.
    """
    visited = set()
    queue = deque([start_url])
    docs = []
    domain = urlparse(start_url).netloc

    # For JS crawling, we might need a persistent browser
    browser = None
    playwright = None
    if use_js and PLAYWRIGHT_AVAILABLE:
        playwright = sync_playwright().start()
        browser = playwright.chromium.launch(headless=True)

    try:
        while queue and len(visited) < max_pages:
            url = queue.popleft()
            if url in visited:
                continue
            logger.debug(f"Crawling {url}")

            # Introduce smart delay to mimic human behavior
            _anti_block.smart_delay(base_delay=1.0)

            try:
                if use_js and browser:
                    # For JS pages, we use Playwright (anti‑block not as critical,
                    # but we can still set viewport and user agent)
                    context = browser.new_context(
                        viewport={'width': 1920, 'height': 1080},
                        user_agent=_anti_block.get_headers()['User-Agent']
                    )
                    page = context.new_page()
                    page.goto(url, wait_until='networkidle')
                    content = page.content()
                    page.close()
                    context.close()
                else:
                    # Static scraping with anti‑block headers and optional proxy
                    headers = _anti_block.get_headers()
                    proxies = _anti_block.get_proxy()  # None if no proxies configured
                    resp = requests.get(
                        url,
                        headers=headers,
                        proxies=proxies,
                        timeout=30
                    )
                    resp.raise_for_status()
                    content = resp.text

                soup = BeautifulSoup(content, 'html.parser')
                # Extract text
                for script in soup(["script", "style"]):
                    script.decompose()
                text = soup.get_text()
                lines = (line.strip() for line in text.splitlines())
                chunks = (phrase.strip() for line in lines for phrase in line.split("  "))
                text = ' '.join(chunk for chunk in chunks if chunk)

                doc = Document(
                    page_content=text,
                    metadata={'source': url, 'domain': domain}
                )
                docs.append(doc)
                visited.add(url)

                # Find new links
                for link in soup.find_all('a', href=True):
                    full_url = urljoin(url, link['href'])
                    if urlparse(full_url).netloc == domain and full_url not in visited:
                        queue.append(full_url)

            except requests.exceptions.RequestException as e:
                logger.warning(f"Failed to crawl {url}: {e}")
                # If it's a 429 (rate limit), the anti_block might have handled it,
                # but we don't have the response object here. For now, just log and continue.
                continue
            except Exception as e:
                logger.warning(f"Unexpected error on {url}: {e}")
                continue

    finally:
        if browser:
            browser.close()
        if playwright:
            playwright.stop()

    logger.info(f"Crawled {len(docs)} pages from {start_url}")
    return docs
