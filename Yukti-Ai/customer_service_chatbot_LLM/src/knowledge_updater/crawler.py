"""
Simple website crawler for indexing entire sites.
Uses requests + BeautifulSoup for static sites, and optionally Playwright for JS.
"""

import time
import logging
from urllib.parse import urljoin, urlparse
from collections import deque
from typing import List, Optional

import requests
from bs4 import BeautifulSoup
from langchain_core.documents import Document

logger = logging.getLogger(__name__)

try:
    from playwright.sync_api import sync_playwright
    PLAYWRIGHT_AVAILABLE = True
except ImportError:
    PLAYWRIGHT_AVAILABLE = False

def crawl_website(start_url: str, max_pages: int = 50, use_js: bool = False) -> List[Document]:
    """
    Crawl a website starting from start_url, up to max_pages.
    Returns a list of Documents with page text and metadata.
    """
    visited = set()
    queue = deque([start_url])
    docs = []
    domain = urlparse(start_url).netloc

    headers = {
        'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36'
    }

    # For JS crawling, we might need a persistent browser
    browser = None
    if use_js and PLAYWRIGHT_AVAILABLE:
        playwright = sync_playwright().start()
        browser = playwright.chromium.launch(headless=True)

    try:
        while queue and len(visited) < max_pages:
            url = queue.popleft()
            if url in visited:
                continue
            logger.debug(f"Crawling {url}")
            time.sleep(1)  # polite delay

            try:
                if use_js and browser:
                    page = browser.new_page()
                    page.goto(url, wait_until='networkidle')
                    content = page.content()
                    page.close()
                else:
                    resp = requests.get(url, headers=headers, timeout=30)
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

            except Exception as e:
                logger.warning(f"Failed to crawl {url}: {e}")

    finally:
        if browser:
            browser.close()
            playwright.stop()

    logger.info(f"Crawled {len(docs)} pages from {start_url}")
    return docs
