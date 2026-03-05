"""
Anti‑blocking strategies for web scraping: proxy rotation, user‑agent rotation, smart delays.
"""

import random
import time
import logging
from typing import Optional

import requests

logger = logging.getLogger(__name__)

class AntiBlockManager:
    """
    Manages anti‑detection strategies to avoid being blocked.
    """
    def __init__(self):
        self.user_agents = [
            'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36',
            'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/605.1.15 (KHTML, like Gecko) Version/17.1 Safari/605.1.15',
            'Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/119.0.0.0 Safari/537.36',
            # Add more realistic UAs
        ]
        self.proxies = []          # Populate with your proxy list
        self.request_count = 0
        self.last_request_time = 0

    def get_headers(self) -> dict:
        """Generate realistic browser headers with random User‑Agent."""
        return {
            'User-Agent': random.choice(self.user_agents),
            'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8',
            'Accept-Language': 'en-US,en;q=0.5',
            'Accept-Encoding': 'gzip, deflate',
            'DNT': '1',
            'Connection': 'keep-alive',
            'Upgrade-Insecure-Requests': '1',
        }

    def get_proxy(self) -> Optional[dict]:
        """Return a random proxy dict (e.g., {'http': 'http://proxy:port'}) or None."""
        if not self.proxies:
            return None
        return random.choice(self.proxies)

    def smart_delay(self, base_delay: float = 2.0) -> None:
        """
        Introduce a random delay with jitter to mimic human behavior.
        """
        jitter = random.uniform(0.5, 2.0)
        delay = base_delay * jitter
        time.sleep(delay)
        # Occasionally add longer pauses
        if random.random() < 0.1:  # 10% chance
            time.sleep(random.uniform(5, 15))

    def handle_rate_limit(self, response: requests.Response) -> bool:
        """
        Handle 429 Too Many Requests. Returns True if we should retry.
        """
        if response.status_code == 429:
            retry_after = int(response.headers.get('Retry-After', 60))
            logger.warning(f"Rate limited. Waiting {retry_after} seconds...")
            time.sleep(retry_after)
            return True
        return False
