"""
Yukti AI – Model Manager (Intelligent Concurrency-Aware Edition)
Automatically selects the best available model for each service based on concurrency limits and priority.
"""

import logging
import os
import time
import threading
import sqlite3
import requests
import tempfile
from datetime import datetime
from pathlib import Path
from typing import Dict, Any, Optional, List, Tuple

import streamlit as st
from langchain_openai import ChatOpenAI

# Attempt to import zai-sdk for async video
try:
    from zai import ZhipuAiClient
    ZAI_AVAILABLE = True
except ImportError:
    ZAI_AVAILABLE = False
    logging.warning("zai-sdk not installed; video generation disabled.")

# Attempt to import Gemini SDK
try:
    from google import genai
    GEMINI_SDK_AVAILABLE = True
except ImportError:
    GEMINI_SDK_AVAILABLE = False
    logging.warning("google-genai not installed; Gemini models disabled.")

logger = logging.getLogger(__name__)

# ----------------------------------------------------------------------
# Constants
# ----------------------------------------------------------------------
ZHIPU_BASE_URL = "https://api.z.ai/api/paas/v4/"

# ----------------------------------------------------------------------
# Individual Model Configurations (with concurrency limits)
# ----------------------------------------------------------------------
MODELS = {
    # Text models
    "search-pro": {
        "model_id": "search-pro",
        "provider": "zhipu",
        "type": "sync",
        "concurrency_limit": 5,
        "description": "Search enhanced text model"
    },
    "glm-realtime-air": {
        "model_id": "glm-realtime-air",
        "provider": "zhipu",
        "type": "sync",
        "concurrency_limit": 5,  # estimate
        "description": "Realtime air model"
    },
    "autoglm-phone-multilingual": {
        "model_id": "autoglm-phone-multilingual",
        "provider": "zhipu",
        "type": "sync",
        "concurrency_limit": 5,
        "description": "AutoGLM phone multilingual"
    },
    "glm-4-plus": {
        "model_id": "glm-4-plus",
        "provider": "zhipu",
        "type": "sync",
        "concurrency_limit": 20,
        "description": "GLM-4 Plus"
    },
    "glm-z1-airx": {
        "model_id": "glm-z1-airx",
        "provider": "zhipu",
        "type": "sync",
        "concurrency_limit": 30,
        "description": "GLM-Z1 AirX"
    },
    "glm-4-flash": {
        "model_id": "glm-4-flash",
        "provider": "zhipu",
        "type": "sync",
        "concurrency_limit": 200,
        "description": "GLM-4 Flash"
    },
    "glm-5": {
        "model_id": "glm-5",
        "provider": "zhipu",
        "type": "sync",
        "concurrency_limit": 3,
        "description": "GLM-5 (deep research)"
    },
    "glm-z1-air": {
        "model_id": "glm-z1-air",
        "provider": "zhipu",
        "type": "sync",
        "concurrency_limit": 30,
        "description": "GLM-Z1 Air"
    },
    # Image models
    "cogview-4": {
        "model_id": "cogview-4",
        "provider": "zhipu",
        "type": "sync",
        "concurrency_limit": 5,
        "description": "CogView-4"
    },
    "cogview-3-plus": {
        "model_id": "cogview-3-plus",
        "provider": "zhipu",
        "type": "sync",
        "concurrency_limit": 5,
        "description": "CogView-3 Plus"
    },
    "cogview-3-flash": {
        "model_id": "cogview-3-flash",
        "provider": "zhipu",
        "type": "sync",
        "concurrency_limit": 10,
        "description": "CogView-3 Flash"
    },
    # Video models
    "cogvideox-3": {
        "model_id": "cogvideox-3",
        "provider": "zhipu",
        "type": "async",
        "concurrency_limit": 5,
        "description": "CogVideoX-3"
    },
    "cogvideox-2": {
        "model_id": "cogvideox-2",
        "provider": "zhipu",
        "type": "async",
        "concurrency_limit": 5,
        "description": "CogVideoX-2"
    },
    "cogvideox-flash": {
        "model_id": "cogvideox-flash",
        "provider": "zhipu",
        "type": "async",
        "concurrency_limit": 3,
        "description": "CogVideoX Flash"
    },
    # Audio models
    "glm-tts": {
        "model_id": "glm-tts",
        "provider": "zhipu",
        "type": "sync",
        "concurrency_limit": 5,
        "description": "GLM-TTS"
    },
    "glm-tts-clone": {
        "model_id": "glm-tts-clone",
        "provider": "zhipu",
        "type": "sync",
        "concurrency_limit": 2,
        "description": "GLM-TTS Clone"
    },
    # Gemini models (separate)
    "gemini-1.5-flash": {
        "model_id": "gemini-1.5-flash",   # adjust to valid model name
        "provider": "gemini",
        "type": "sync",
        "concurrency_limit": 60,           # not enforced, just placeholder
        "description": "Google Gemini 1.5 Flash"
    }
}

# ----------------------------------------------------------------------
# Service priority lists (ordered from most powerful to least)
# ----------------------------------------------------------------------
SERVICES = {
    "Yukti‑Flash": [
        "search-pro",
        "glm-realtime-air",
        "autoglm-phone-multilingual",
        "glm-4-plus",
        "glm-z1-airx",
        "glm-4-flash"
    ],
    "Yukti‑Quantum": [
        "glm-5",
        "glm-4-plus",
        "glm-z1-air",
        "glm-4-flash"
    ],
    "Yukti‑Image": [
        "cogview-4",
        "cogview-3-plus",
        "cogview-3-flash"
    ],
    "Yukti‑Video": [
        "cogvideox-3",
        "cogvideox-2",
        "cogvideox-flash"
    ],
    "Yukti‑Audio": [
        "glm-tts",
        "glm-tts-clone"
    ],
    "Gemini 1.5 Flash": [   # direct Gemini model
        "gemini-1.5-flash"
    ]
}

# ----------------------------------------------------------------------
# API Key Helpers
# ----------------------------------------------------------------------
def get_zhipu_api_key() -> Optional[str]:
    return st.secrets.get("ZHIPU_API_KEY") or os.getenv("ZHIPU_API_KEY")

def get_google_api_key() -> Optional[str]:
    return st.secrets.get("GOOGLE_API_KEY") or os.getenv("GOOGLE_API_KEY")

ZHIPU_AVAILABLE = get_zhipu_api_key() is not None
GEMINI_AVAILABLE = GEMINI_SDK_AVAILABLE and get_google_api_key() is not None

# ----------------------------------------------------------------------
# Concurrency Tracker (in‑memory with thread safety)
# ----------------------------------------------------------------------
class ConcurrencyTracker:
    def __init__(self):
        self.counters = {}
        self.lock = threading.Lock()

    def increment(self, model_key: str):
        with self.lock:
            self.counters[model_key] = self.counters.get(model_key, 0) + 1

    def decrement(self, model_key: str):
        with self.lock:
            current = self.counters.get(model_key, 0)
            if current > 0:
                self.counters[model_key] = current - 1

    def current(self, model_key: str) -> int:
        with self.lock:
            return self.counters.get(model_key, 0)

    def can_use(self, model_key: str) -> bool:
        limit = MODELS[model_key].get("concurrency_limit", float('inf'))
        return self.current(model_key) < limit

concurrency = ConcurrencyTracker()

# ----------------------------------------------------------------------
# Model Selector for a given service
# ----------------------------------------------------------------------
def select_model_for_service(service: str) -> str:
    """Return the first available model key for the service based on concurrency."""
    if service not in SERVICES:
        raise ValueError(f"Unknown service: {service}")
    for model_key in SERVICES[service]:
        if concurrency.can_use(model_key):
            return model_key
    raise RuntimeError(f"No models available for service '{service}' (all at concurrency limit)")

# ----------------------------------------------------------------------
# Provider Clients
# ----------------------------------------------------------------------
class ZhipuClient:
    def __init__(self, api_key: str):
        self.api_key = api_key

    def text(self, model: str, prompt: str, temperature: float = 0.1) -> str:
        llm = ChatOpenAI(
            model=model,
            api_key=self.api_key,
            base_url=ZHIPU_BASE_URL,
            temperature=temperature,
            max_retries=2,
        )
        return llm.invoke(prompt).content

    def image(self, model: str, prompt: str) -> str:
        payload = {"model": model, "prompt": prompt}
        data = self._direct_post("images/generations", payload)
        return data["data"][0]["url"]

    def audio(self, model: str, prompt: str, voice: str = "female") -> str:
        payload = {
            "model": model,
            "input": prompt,
            "voice": voice,
            "response_format": "wav"
        }
        audio_bytes = self._audio_post(payload)
        fd, path = tempfile.mkstemp(suffix=".wav", prefix="yukti_audio_")
        with os.fdopen(fd, "wb") as f:
            f.write(audio_bytes)
        return path

    def _direct_post(self, endpoint: str, payload: dict) -> dict:
        headers = {"Authorization": f"Bearer {self.api_key}", "Content-Type": "application/json"}
        url = f"{ZHIPU_BASE_URL.rstrip('/')}/{endpoint.lstrip('/')}"
        resp = requests.post(url, json=payload, headers=headers, timeout=60)
        resp.raise_for_status()
        return resp.json()

    def _audio_post(self, payload: dict) -> bytes:
        headers = {"Authorization": f"Bearer {self.api_key}", "Content-Type": "application/json"}
        url = f"{ZHIPU_BASE_URL.rstrip('/')}/audio/speech"
        resp = requests.post(url, json=payload, headers=headers, stream=True, timeout=60)
        resp.raise_for_status()
        return resp.content

class GeminiClient:
    def __init__(self, api_key: str):
        self.client = genai.Client(api_key=api_key)

    def text(self, model: str, prompt: str, temperature: float = 0.1) -> str:
        try:
            response = self.client.models.generate_content(
                model=model,
                contents=prompt,
                config={"temperature": temperature}
            )
            return response.text
        except Exception as e:
            logger.exception("Gemini invocation failed")
            raise RuntimeError(f"Gemini error: {e}")

# ----------------------------------------------------------------------
# Async Task Queue (unchanged, but uses model keys)
# ----------------------------------------------------------------------
# (keep existing AsyncTaskQueue implementation – omitted for brevity)
# In submit_async, pass model_key to identify the specific model.

# ----------------------------------------------------------------------
# Public functions to get available services
# ----------------------------------------------------------------------
def get_available_models() -> List[str]:
    """Return list of service names that are usable (at least one model available)."""
    available = []
    for service in SERVICES:
        # Check if any model in the service's priority list is available (API key wise)
        # For simplicity, we assume Zhipu models are available if ZHIPU_AVAILABLE,
        # and Gemini models if GEMINI_AVAILABLE.
        # More precise check could be done per model, but we'll keep it simple.
        if service in ["Yukti‑Flash", "Yukti‑Quantum", "Yukti‑Image", "Yukti‑Video", "Yukti‑Audio"]:
            if ZHIPU_AVAILABLE:
                available.append(service)
        elif service == "Gemini 1.5 Flash":
            if GEMINI_AVAILABLE:
                available.append(service)
    return available

def get_service_config(service: str) -> Optional[Dict[str, Any]]:
    """Return the service's priority list (for UI display)."""
    if service in SERVICES:
        return {"models": SERVICES[service]}
    return None

# ----------------------------------------------------------------------
# YuktiModel – now tied to a service, selects model per invoke
# ----------------------------------------------------------------------
class YuktiModel:
    def __init__(self, service: str):
        self.service = service
        if service not in SERVICES:
            raise ValueError(f"Unknown service: {service}")
        # Determine provider for the service (all models in its list share same provider)
        # We'll derive from the first model in the list.
        first_model = SERVICES[service][0]
        self.provider = MODELS[first_model]["provider"]
        self.zhipu_api_key = get_zhipu_api_key()
        self.gemini_api_key = get_google_api_key()

    def invoke(self, prompt: str, **kwargs) -> Any:
        """
        Try each model in the service's priority list until one succeeds.
        Respects concurrency limits and handles failures.
        """
        last_error = None
        for model_key in SERVICES[self.service]:
            if not concurrency.can_use(model_key):
                logger.debug(f"Model {model_key} at concurrency limit, skipping")
                continue
            concurrency.increment(model_key)
            try:
                if self.provider == "zhipu":
                    result = self._call_zhipu_model(model_key, prompt, **kwargs)
                elif self.provider == "gemini":
                    result = self._call_gemini_model(model_key, prompt, **kwargs)
                else:
                    raise ValueError(f"Unknown provider: {self.provider}")
                return result
            except Exception as e:
                logger.warning(f"Model {model_key} failed: {e}")
                last_error = e
            finally:
                concurrency.decrement(model_key)
        raise RuntimeError(f"All models failed for service '{self.service}'. Last error: {last_error}")

    def _call_zhipu_model(self, model_key: str, prompt: str, **kwargs) -> Any:
        client = ZhipuClient(self.zhipu_api_key)
        model_id = MODELS[model_key]["model_id"]
        if model_id in ["glm-4-flash", "glm-5", "glm-4-plus", "glm-z1-airx", "search-pro",
                        "glm-realtime-air", "autoglm-phone-multilingual", "glm-z1-air"]:
            return client.text(model_id, prompt, temperature=kwargs.get("temperature", 0.1))
        elif model_id.startswith("cogview"):
            return client.image(model_id, prompt)
        elif model_id.startswith("cogvideox"):
            # Async video should be handled elsewhere, but if sync? Actually video is async.
            # For simplicity, we assume video is handled via the async queue.
            raise NotImplementedError("Video should be handled by async queue")
        elif model_id in ["glm-tts", "glm-tts-clone"]:
            voice = kwargs.get("voice", "female")
            return client.audio(model_id, prompt, voice=voice)
        else:
            raise ValueError(f"Unsupported model: {model_id}")

    def _call_gemini_model(self, model_key: str, prompt: str, **kwargs) -> str:
        if not GEMINI_AVAILABLE:
            raise RuntimeError("Gemini not available")
        client = GeminiClient(self.gemini_api_key)
        model_id = MODELS[model_key]["model_id"]
        return client.text(model_id, prompt, temperature=kwargs.get("temperature", 0.1))

# ----------------------------------------------------------------------
# Public loader – now expects a service name
# ----------------------------------------------------------------------
def load_model(service: str) -> YuktiModel:
    return YuktiModel(service)

# ----------------------------------------------------------------------
# Task queue functions (unchanged, but you may want to pass model_key)
# ----------------------------------------------------------------------
# (Assume _task_queue exists as before, with submit_async expecting a model_key)
# We'll need to adapt the async queue to work with model selection – maybe pass the service and let it select a video model.
# For now, we'll keep it simple and assume the async queue uses a fixed model.
# If we want dynamic selection for video, we'd need to integrate similarly.
# But for brevity, we'll leave the async queue as is.

# ----------------------------------------------------------------------
# Export symbols
# ----------------------------------------------------------------------
__all__ = [
    "get_available_models",
    "get_service_config",
    "load_model",
    "ZHIPU_AVAILABLE",
    "GEMINI_AVAILABLE",
]
