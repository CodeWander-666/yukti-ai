# knowledge_updater/anti_block.py

import random
import time
from typing import List
import requests

class AntiBlockManager:
    """
    Manages anti-detection strategies to avoid being blocked [citation:3].
    """
    
    def __init__(self):
        self.proxies = []
        self.user_agents = [
            'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36...',
            'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/605.1.15...',
            # Add more realistic UAs
        ]
        self.request_count = 0
        self.last_request_time = 0
    
    def rotate_proxy(self):
        """Rotate IP addresses using proxy pool [citation:3]."""
        if not self.proxies:
            return None
        return random.choice(self.proxies)
    
    def get_headers(self):
        """Generate realistic browser headers."""
        return {
            'User-Agent': random.choice(self.user_agents),
            'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8',
            'Accept-Language': 'en-US,en;q=0.5',
            'Accept-Encoding': 'gzip, deflate',
            'DNT': '1',
            'Connection': 'keep-alive',
            'Upgrade-Insecure-Requests': '1',
        }
    
    def smart_delay(self, base_delay=2):
        """
        Implement intelligent delays with jitter [citation:7].
        Mimics human browsing patterns.
        """
        # Add random delay
        jitter = random.uniform(0.5, 2.0)
        time.sleep(base_delay * jitter)
        
        # Occasionally add longer pauses
        if random.random() < 0.1:  # 10% chance
            time.sleep(random.uniform(5, 15))
    
    def handle_rate_limit(self, response):
        """Handle 429 Too Many Requests responses [citation:5]."""
        if response.status_code == 429:
            retry_after = int(response.headers.get('Retry-After', 60))
            logger.warning(f"Rate limited. Waiting {retry_after} seconds...")
            time.sleep(retry_after)
            return True
        return False
