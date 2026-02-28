"""
Yukti AI – Model Manager (High-End Production Edition)
Manages multiple providers (Zhipu, Gemini) with automatic failover, async queue,
exponential backoff, and exhaustive error handling.
"""

import logging
import os
import time
import threading
import sqlite3
import hashlib
import json
from datetime import datetime
from pathlib import Path
from typing import Dict, Any, Optional, List, Tuple, Union, Callable
from functools import lru_cache

import streamlit as st
import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

# LangChain imports (optional, used for Gemini)
try:
    from langchain_google_genai import ChatGoogleGenerativeAI
    LANGCHAIN_GOOGLE_AVAILABLE = True
except ImportError:
    LANGCHAIN_GOOGLE_AVAILABLE = False

# Zhipu SDK
try:
    from zai import ZhipuAiClient
    ZHIPU_SDK_AVAILABLE = True
except ImportError:
    ZHIPU_SDK_AVAILABLE = False

# OpenAI compatible client (for Zhipu sync models)
try:
    from langchain_openai import ChatOpenAI
    LANGCHAIN_OPENAI_AVAILABLE = True
except ImportError:
    LANGCHAIN_OPENAI_AVAILABLE = False

logger = logging.getLogger(__name__)

# ----------------------------------------------------------------------
# Configuration
# ----------------------------------------------------------------------
ZHIPU_BASE_URL = "https://api.z.ai/api/paas/v4/"
REQUEST_TIMEOUT = 30
MAX_RETRIES = 3
RETRY_BACKOFF_FACTOR = 2  # exponential backoff
CACHE_TTL = 3600  # 1 hour for identical prompts (optional)

# ----------------------------------------------------------------------
# Model Definitions
# ----------------------------------------------------------------------
MODELS = {
    "Yukti‑Flash": {
        "model": "glm-4-flash",
        "provider": "zhipu",
        "type": "sync",
        "description": "Fast text & reasoning",
    },
    "Yukti‑Quantum": {
        "model": "glm-5",
        "provider": "zhipu",
        "type": "sync",
        "description": "Deep research & complex reasoning",
    },
    "Yukti‑Image": {
        "model": "cogview-4",
        "provider": "zhipu",
        "type": "sync",
        "description": "Image generation (returns URL)",
        "requires_direct_api": True,  # not OpenAI‑compatible
    },
    "Yukti‑Video": {
        "model": "cogvideox-3",
        "provider": "zhipu",
        "type": "async",
        "description": "Video generation (queued)",
        "requires_zai_sdk": True,
    },
    "Yukti‑Audio": {
        "model": "glm-tts",
        "provider": "zhipu",
        "type": "sync",
        "description": "Text‑to‑speech (returns audio file)",
        "requires_direct_api": True,
    },
    "Gemini 3 Flash": {
        "model": "gemini-3-flash",
        "provider": "gemini",
        "type": "sync",
        "description": "Google Gemini fallback",
    }
}

def get_available_models() -> List[str]:
    """Return list of models that are actually available given API keys and SDKs."""
    available = []
    zhipu_key_available = bool(st.secrets.get("ZHIPU_API_KEY") or os.getenv("ZHIPU_API_KEY"))
    gemini_key_available = bool(st.secrets.get("GOOGLE_API_KEY") or os.getenv("GOOGLE_API_KEY"))

    for name, config in MODELS.items():
        if config["provider"] == "zhipu":
            if not zhipu_key_available:
                continue
            if config.get("requires_zai_sdk") and not ZHIPU_SDK_AVAILABLE:
                continue
            if config.get("requires_direct_api") and not (ZHIPU_SDK_AVAILABLE or requests):
                # direct API requires requests (always available)
                pass
            available.append(name)
        elif config["provider"] == "gemini":
            if gemini_key_available and LANGCHAIN_GOOGLE_AVAILABLE:
                available.append(name)
    return available

def get_model_config(model_key: str) -> Optional[Dict[str, Any]]:
    return MODELS.get(model_key)

# ----------------------------------------------------------------------
# HTTP Session with retry
# ----------------------------------------------------------------------
def create_http_session() -> requests.Session:
    session = requests.Session()
    retries = Retry(
        total=MAX_RETRIES,
        backoff_factor=RETRY_BACKOFF_FACTOR,
        status_forcelist=[429, 500, 502, 503, 504],
        allowed_methods=["HEAD", "GET", "POST", "PUT", "DELETE", "OPTIONS", "TRACE"]
    )
    adapter = HTTPAdapter(max_retries=retries)
    session.mount('http://', adapter)
    session.mount('https://', adapter)
    return session

_http_session = create_http_session()

# ----------------------------------------------------------------------
# Provider Clients (cached)
# ----------------------------------------------------------------------
@st.cache_resource
def get_zhipu_sync_client() -> Optional[ChatOpenAI]:
    """Return OpenAI‑compatible client for Zhipu sync models."""
    api_key = st.secrets.get("ZHIPU_API_KEY") or os.getenv("ZHIPU_API_KEY")
    if not api_key or not LANGCHAIN_OPENAI_AVAILABLE:
        return None
    return ChatOpenAI(
        model="placeholder",
        api_key=api_key,
        base_url=ZHIPU_BASE_URL,
        temperature=0.1,
        max_retries=MAX_RETRIES,
        request_timeout=REQUEST_TIMEOUT,
    )

@st.cache_resource
def get_zhipu_async_client() -> Optional[Any]:
    """Return ZhipuAiClient for async video (requires zai-sdk)."""
    if not ZHIPU_SDK_AVAILABLE:
        return None
    api_key = st.secrets.get("ZHIPU_API_KEY") or os.getenv("ZHIPU_API_KEY")
    if not api_key:
        return None
    return ZhipuAiClient(api_key=api_key)

@st.cache_resource
def get_gemini_client() -> Optional[ChatGoogleGenerativeAI]:
    """Return Gemini client."""
    api_key = st.secrets.get("GOOGLE_API_KEY") or os.getenv("GOOGLE_API_KEY")
    if not api_key or not LANGCHAIN_GOOGLE_AVAILABLE:
        return None
    return ChatGoogleGenerativeAI(
        model="gemini-3-flash",
        google_api_key=api_key,
        temperature=0.1,
        max_retries=MAX_RETRIES,
        request_timeout=REQUEST_TIMEOUT,
    )

# ----------------------------------------------------------------------
# Async Task Queue for Video (using zai-sdk)
# ----------------------------------------------------------------------
class ZhipuTaskQueue:
    def __init__(self, db_path: str = "yukti_tasks.db"):
        self.db_path = db_path
        self.conn = sqlite3.connect(db_path, check_same_thread=False)
        self._init_db()
        self.lock = threading.Lock()
        self.zhipu_map: Dict[str, str] = {}  # local_task_id -> zhipu_task_id
        self._start_poller()

    def _init_db(self):
        self.conn.execute("""
            CREATE TABLE IF NOT EXISTS tasks (
                task_id TEXT PRIMARY KEY,
                variant TEXT,
                model TEXT,
                status TEXT,
                progress INTEGER,
                result_url TEXT,
                error TEXT,
                created_at TIMESTAMP,
                completed_at TIMESTAMP
            )
        """)
        self.conn.commit()

    def add_task(self, task_id: str, variant: str, model: str):
        with self.lock:
            self.conn.execute(
                "INSERT INTO tasks VALUES (?,?,?,?,?,?,?,?,?)",
                (task_id, variant, model, "submitted", 0, "", "", datetime.now(), None)
            )
            self.conn.commit()

    def update_task(self, task_id: str, **kwargs):
        with self.lock:
            fields = ", ".join([f"{k}=?" for k in kwargs])
            values = list(kwargs.values()) + [task_id]
            self.conn.execute(f"UPDATE tasks SET {fields} WHERE task_id=?", values)
            self.conn.commit()

    def get_task(self, task_id: str) -> Optional[Dict[str, Any]]:
        cursor = self.conn.execute("SELECT * FROM tasks WHERE task_id=?", (task_id,))
        row = cursor.fetchone()
        if not row:
            return None
        return {
            "task_id": row[0],
            "variant": row[1],
            "model": row[2],
            "status": row[3],
            "progress": row[4],
            "result_url": row[5],
            "error": row[6],
            "created_at": row[7],
            "completed_at": row[8],
        }

    def get_active_tasks(self) -> List[Tuple[str, str, str, int]]:
        cursor = self.conn.execute(
            "SELECT task_id, variant, status, progress FROM tasks WHERE status IN ('submitted','pending','processing') ORDER BY created_at"
        )
        return cursor.fetchall()

    def submit_async(self, variant: str, model: str, prompt: str, **kwargs) -> str:
        """Submit an async video generation task."""
        local_id = f"{variant}_{int(time.time())}_{abs(hash(prompt)) % 10000}"
        self.add_task(local_id, variant, model)

        try:
            client = get_zhipu_async_client()
            if client is None:
                raise RuntimeError("Zhipu async client not available (zai-sdk missing or no API key).")

            args = {
                "model": model,
                "prompt": prompt,
                "quality": kwargs.get("quality", "quality"),
                "with_audio": kwargs.get("with_audio", True),
                "size": kwargs.get("size", "1920x1080"),
                "fps": kwargs.get("fps", 30),
            }
            if "image_url" in kwargs and kwargs["image_url"]:
                args["image_url"] = kwargs["image_url"]

            response = client.videos.generations(**args)
            zhipu_task_id = response.id
            with self.lock:
                self.zhipu_map[local_id] = zhipu_task_id
            self.update_task(local_id, status="pending")
        except Exception as e:
            logger.exception("Failed to submit async task")
            self.update_task(local_id, status="failed", error=str(e))

        return local_id

    def _poll_tasks(self):
        """Background thread that polls all pending video tasks."""
        while True:
            tasks = []
            with self.lock:
                tasks = list(self.zhipu_map.items())
            for local_id, zhipu_id in tasks:
                try:
                    client = get_zhipu_async_client()
                    if client is None:
                        # Client disappeared; mark as failed
                        self.update_task(local_id, status="failed", error="Zhipu client unavailable")
                        with self.lock:
                            del self.zhipu_map[local_id]
                        continue

                    status_data = client.videos.retrieve_videos_result(zhipu_id)
                    if hasattr(status_data, 'status') and status_data.status == "succeeded":
                        result_url = status_data.video_url
                        self.update_task(local_id, status="completed", progress=100,
                                         result_url=result_url)
                        with self.lock:
                            del self.zhipu_map[local_id]
                    elif hasattr(status_data, 'status') and status_data.status == "failed":
                        error = status_data.error if hasattr(status_data, 'error') else "Unknown error"
                        self.update_task(local_id, status="failed", error=error)
                        with self.lock:
                            del self.zhipu_map[local_id]
                    elif hasattr(status_data, 'status') and status_data.status == "processing":
                        progress = getattr(status_data, 'progress', 50)
                        self.update_task(local_id, status="processing", progress=progress)
                except Exception as e:
                    logger.error(f"Polling task {local_id} failed: {e}")
            time.sleep(5)

    def _start_poller(self):
        thread = threading.Thread(target=self._poll_tasks, daemon=True)
        thread.start()

# Initialize task queue (if Zhipu async available)
if ZHIPU_SDK_AVAILABLE and (st.secrets.get("ZHIPU_API_KEY") or os.getenv("ZHIPU_API_KEY")):
    _task_queue = ZhipuTaskQueue(db_path=str(Path(__file__).parent.parent / "yukti_tasks.db"))
else:
    class _DummyQueue:
        def submit_async(self, *args, **kwargs):
            raise RuntimeError("Video generation requires zai-sdk and ZHIPU_API_KEY.")
        def get_active_tasks(self):
            return []
        def get_task(self, task_id):
            return None
    _task_queue = _DummyQueue()

# ----------------------------------------------------------------------
# Provider execution functions with fallback and caching
# ----------------------------------------------------------------------
def _call_zhipu_sync(model: str, prompt: str, **kwargs) -> Any:
    """Call a Zhipu sync model using OpenAI‑compatible client or direct API."""
    client = get_zhipu_sync_client()
    if client is None:
        raise RuntimeError("Zhipu sync client not available (API key missing or langchain-openai missing).")

    # Text models
    if "glm" in model and model not in ["cogview-4", "glm-tts"]:
        llm = ChatOpenAI(
            model=model,
            api_key=client.api_key,
            base_url=client.base_url,
            temperature=kwargs.get("temperature", 0.1),
            max_retries=MAX_RETRIES,
            request_timeout=REQUEST_TIMEOUT,
        )
        return llm.invoke(prompt)

    # Image generation (cogview-4)
    if model == "cogview-4":
        api_key = st.secrets.get("ZHIPU_API_KEY") or os.getenv("ZHIPU_API_KEY")
        url = f"{ZHIPU_BASE_URL}images/generations"
        payload = {"model": model, "prompt": prompt}
        headers = {"Authorization": f"Bearer {api_key}"}
        response = _http_session.post(url, json=payload, headers=headers, timeout=REQUEST_TIMEOUT)
        response.raise_for_status()
        return response.json()["data"][0]["url"]

    # Audio generation (glm-tts)
    if model == "glm-tts":
        api_key = st.secrets.get("ZHIPU_API_KEY") or os.getenv("ZHIPU_API_KEY")
        url = f"{ZHIPU_BASE_URL}audio/speech"
        payload = {
            "model": model,
            "input": prompt,
            "voice": kwargs.get("voice", "female"),
            "response_format": "wav"
        }
        headers = {"Authorization": f"Bearer {api_key}"}
        response = _http_session.post(url, json=payload, headers=headers, stream=True, timeout=REQUEST_TIMEOUT)
        response.raise_for_status()
        fd, path = tempfile.mkstemp(suffix=".wav", prefix="yukti_audio_")
        with os.fdopen(fd, "wb") as f:
            for chunk in response.iter_content(chunk_size=8192):
                f.write(chunk)
        return path

    raise ValueError(f"Unsupported sync model: {model}")

def _call_gemini(model: str, prompt: str, **kwargs) -> str:
    """Call Gemini model."""
    client = get_gemini_client()
    if client is None:
        raise RuntimeError("Gemini client not available (API key missing or langchain-google-genai missing).")
    # Override model if needed
    if model != "gemini-3-flash":
        client.model = model
    return client.invoke(prompt)

def _call_with_fallback(model_key: str, prompt: str, **kwargs) -> Any:
    """
    Try primary provider; if fails, attempt fallback to Gemini (if applicable).
    """
    config = get_model_config(model_key)
    if not config:
        raise ValueError(f"Unknown model key: {model_key}")

    primary_provider = config["provider"]
    last_error = None

    # Try primary
    try:
        if primary_provider == "zhipu":
            if config["type"] == "async":
                # Async tasks go through queue, not direct call
                return _task_queue.submit_async(
                    variant=model_key,
                    model=config["model"],
                    prompt=prompt,
                    **kwargs
                )
            else:
                return _call_zhipu_sync(config["model"], prompt, **kwargs)
        elif primary_provider == "gemini":
            return _call_gemini(config["model"], prompt, **kwargs)
    except Exception as e:
        last_error = e
        logger.warning(f"Primary provider {primary_provider} failed: {e}")

    # Fallback to Gemini if primary was Zhipu and Gemini is available
    if primary_provider == "zhipu":
        gemini_config = MODELS.get("Gemini 3 Flash")
        if gemini_config:
            try:
                logger.info("Falling back to Gemini 3 Flash")
                return _call_gemini(gemini_config["model"], prompt, **kwargs)
            except Exception as e2:
                last_error = e2
                logger.error(f"Fallback to Gemini also failed: {e2}")

    # All attempts failed
    raise RuntimeError(f"All providers failed. Last error: {last_error}")

# ----------------------------------------------------------------------
# YuktiModel wrapper
# ----------------------------------------------------------------------
class YuktiModel:
    def __init__(self, model_key: str):
        self.config = get_model_config(model_key)
        if not self.config:
            raise ValueError(f"Unknown model key: {model_key}")
        self.model_key = model_key
        self.model_name = self.config["model"]

    def invoke(self, prompt: str, **kwargs) -> Any:
        """
        Invoke the model. For async returns local task_id; for sync returns result.
        """
        # For async, directly use queue (no fallback needed)
        if self.config["type"] == "async":
            if not ZHIPU_SDK_AVAILABLE:
                raise RuntimeError("Video generation requires zai-sdk.")
            return _task_queue.submit_async(
                variant=self.model_key,
                model=self.model_name,
                prompt=prompt,
                **kwargs
            )

        # For sync, use call with fallback
        return _call_with_fallback(self.model_key, prompt, **kwargs)

# ----------------------------------------------------------------------
# Public interface
# ----------------------------------------------------------------------
def load_model(model_key: str) -> YuktiModel:
    return YuktiModel(model_key)

def get_active_tasks() -> List[Tuple[str, str, str, int]]:
    return _task_queue.get_active_tasks()

def get_task_status(task_id: str) -> Optional[Dict[str, Any]]:
    return _task_queue.get_task(task_id)

# For backward compatibility (if main.py imports ZHIPU_AVAILABLE)
ZHIPU_AVAILABLE = bool(ZHIPU_SDK_AVAILABLE and (st.secrets.get("ZHIPU_API_KEY") or os.getenv("ZHIPU_API_KEY")))
