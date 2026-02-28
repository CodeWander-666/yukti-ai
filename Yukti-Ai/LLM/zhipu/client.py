"""
Synchronous client for Zhipu chat completions (sync models only).
Includes retry logic and error handling.
"""

import requests
import time
import logging
import streamlit as st

logger = logging.getLogger(__name__)

ZHIPU_API_BASE = "https://open.bigmodel.cn/api/paas/v4"
MAX_RETRIES = 3
RETRY_DELAY = 2  # seconds

class ZhipuSyncClient:
    def __init__(self, api_key=None):
        self.api_key = api_key or st.secrets.get("ZHIPU_API_KEY")
        if not self.api_key:
            raise ValueError("ZHIPU_API_KEY not found in secrets.")
        self.session = requests.Session()
        self.session.headers.update({
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json"
        })

    def chat(self, model: str, prompt: str, temperature: float = 0.1, max_tokens: int = 1024) -> str:
        """
        Send a synchronous chat request with retries.
        Returns the generated text.
        """
        data = {
            "model": model,
            "messages": [{"role": "user", "content": prompt}],
            "temperature": temperature,
            "max_tokens": max_tokens
        }
        for attempt in range(MAX_RETRIES):
            try:
                resp = self.session.post(f"{ZHIPU_API_BASE}/chat/completions", json=data)
                resp.raise_for_status()
                result = resp.json()
                return result["choices"][0]["message"]["content"]
            except requests.exceptions.RequestException as e:
                logger.warning(f"Attempt {attempt+1} failed: {e}")
                if attempt < MAX_RETRIES - 1:
                    time.sleep(RETRY_DELAY * (2 ** attempt))  # exponential backoff
                else:
                    raise Exception(f"Zhipu API error after {MAX_RETRIES} attempts: {e}")
        raise Exception("Unexpected error in ZhipuSyncClient.chat")
