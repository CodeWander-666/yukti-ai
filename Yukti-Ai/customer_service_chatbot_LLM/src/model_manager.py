# Add imports at top
import threading
import os
import time
import re
from datetime import datetime
from pathlib import Path
from typing import Dict, Any, Optional, List, Tuple

# ... (existing imports)

# Add a semaphore per model for async tasks
_async_semaphores = {}
_async_lock = threading.Lock()

def _get_async_semaphore(model_key: str) -> threading.Semaphore:
    with _async_lock:
        if model_key not in _async_semaphores:
            limit = _MODEL_REGISTRY[model_key].get("concurrency_limit", 5)
            _async_semaphores[model_key] = threading.Semaphore(limit)
        return _async_semaphores[model_key]

# Modify ZhipuTaskQueue.submit_async to acquire semaphore
def submit_async(self, variant: str, model: str, prompt: str, **kwargs) -> str:
    sem = _get_async_semaphore(model)
    acquired = sem.acquire(blocking=True, timeout=60)  # wait up to 60 sec
    if not acquired:
        raise RuntimeError(f"Concurrency limit for {model} reached; try again later.")
    try:
        # ... existing submission code
        return local_id
    finally:
        sem.release()

# In ZhipuClient.audio(), validate response structure
def audio(self, model: str, prompt: str, voice: str = None, language: str = None) -> str:
    # ... (existing code)
    resp = self._request_with_retry("POST", url, json=payload)
    data = resp.json()
    try:
        audio_b64 = data["choices"][0]["message"]["audio"]["data"]
    except (KeyError, IndexError, TypeError) as e:
        logger.error(f"Unexpected audio response: {data}")
        raise RuntimeError("Audio generation failed: unexpected response format") from e
    # ... (rest unchanged)

# In ZhipuClient.video_submit(), accept language and prepend instruction
def video_submit(self, model: str, prompt: str, language: str = None, **kwargs) -> str:
    if language and language != 'en':
        lang_name = get_language_name(language)
        prompt = f"[Generate video with narration in {lang_name}]\n{prompt}"
    # ... (existing submission code)

# Update Gemini model selection to be dynamic
class GeminiClient:
    def __init__(self, api_key: str):
        self.client = genai.Client(api_key=api_key)
        self._available_models = None
        self._selected_model = None
        self._last_refresh = 0
        self._refresh_interval = 3600  # refresh model list every hour

    def _refresh_models(self):
        now = time.time()
        if self._available_models is None or (now - self._last_refresh) > self._refresh_interval:
            try:
                models = self.client.models.list()
                self._available_models = [
                    m.name for m in models 
                    if 'generateContent' in (getattr(m, 'supported_actions', []) or [])
                ]
                self._last_refresh = now
                logger.info(f"Refreshed Gemini models: {self._available_models}")
            except Exception as e:
                logger.warning(f"Failed to refresh Gemini models: {e}")
                if self._available_models is None:
                    self._available_models = []

    def _select_model(self) -> str:
        self._refresh_models()
        preferred = [
            "gemini-2.0-flash-exp",
            "gemini-1.5-pro",
            "gemini-1.5-flash",
            "gemini-1.0-pro",
        ]
        available = self._available_models
        if not available:
            raise RuntimeError("No Gemini models available")
        for pref in preferred:
            for av in available:
                if pref in av or av.endswith(pref):
                    logger.info(f"Selected Gemini model: {av}")
                    return av
        logger.warning(f"None of {preferred} found, using first available: {available[0]}")
        return available[0]
