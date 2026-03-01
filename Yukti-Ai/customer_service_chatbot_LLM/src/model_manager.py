"""
Yukti AI – Model Manager (Final Production Edition)
Handles all Yukti services with concurrency‑aware model selection, Gemini as a separate service,
async video queue, and full compatibility with existing files.
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
_MODEL_REGISTRY = {
    # Text models (Zhipu)
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
        "concurrency_limit": 30,
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
        "model_id": "gemini-1.5-flash",   # adjust to valid model name if needed
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
    "Gemini 1.5 Flash": [   # direct Gemini service
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
        limit = _MODEL_REGISTRY[model_key].get("concurrency_limit", float('inf'))
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
# Async Task Queue for Video
# ----------------------------------------------------------------------
if ZAI_AVAILABLE and ZHIPU_AVAILABLE:
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
            """Submit an async video generation task using zai-sdk."""
            local_id = f"{variant}_{int(time.time())}_{abs(hash(prompt)) % 10000}"
            self.add_task(local_id, variant, model)

            try:
                api_key = get_zhipu_api_key()
                if not api_key:
                    raise RuntimeError("Zhipu API key not configured.")
                client = ZhipuAiClient(api_key=api_key)
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
            """Background thread polling all pending video tasks."""
            while True:
                tasks = []
                with self.lock:
                    tasks = list(self.zhipu_map.items())
                for local_id, zhipu_id in tasks:
                    try:
                        api_key = get_zhipu_api_key()
                        if not api_key:
                            raise RuntimeError("Zhipu API key missing.")
                        client = ZhipuAiClient(api_key=api_key)
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

    _task_queue = ZhipuTaskQueue(db_path=str(Path(__file__).parent.parent / "yukti_tasks.db"))
else:
    class _DummyQueue:
        def submit_async(self, *args, **kwargs):
            raise RuntimeError("Video generation requires zai-sdk and Zhipu API key.")
        def get_active_tasks(self):
            return []
        def get_task(self, task_id):
            return None
    _task_queue = _DummyQueue()

# ----------------------------------------------------------------------
# Public task functions
# ----------------------------------------------------------------------
def get_active_tasks() -> List[Tuple[str, str, str, int]]:
    return _task_queue.get_active_tasks()

def get_task_status(task_id: str) -> Optional[Dict[str, Any]]:
    return _task_queue.get_task(task_id)

# ----------------------------------------------------------------------
# Build backward‑compatible MODELS dictionary for main.py
# ----------------------------------------------------------------------
MODELS = {}
for service, models in SERVICES.items():
    first_model = models[0]
    desc = _MODEL_REGISTRY[first_model]["description"]
    MODELS[service] = {
        "description": desc,
        "models": models   # optional, not used by main.py but could be useful
    }

# ----------------------------------------------------------------------
# YuktiModel – now tied to a service, selects model per invoke
# ----------------------------------------------------------------------
class YuktiModel:
    def __init__(self, service: str):
        self.service = service
        if service not in SERVICES:
            raise ValueError(f"Unknown service: {service}")
        # Determine provider for the service (all models in its list share same provider)
        first_model = SERVICES[service][0]
        self.provider = _MODEL_REGISTRY[first_model]["provider"]
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
        model_id = _MODEL_REGISTRY[model_key]["model_id"]
        if model_id in ["glm-4-flash", "glm-5", "glm-4-plus", "glm-z1-airx", "search-pro",
                        "glm-realtime-air", "autoglm-phone-multilingual", "glm-z1-air"]:
            return client.text(model_id, prompt, temperature=kwargs.get("temperature", 0.1))
        elif model_id.startswith("cogview"):
            return client.image(model_id, prompt)
        elif model_id.startswith("cogvideox"):
            # Video is async; if we reach here, it's a sync call error
            raise NotImplementedError("Video should be handled via async queue")
        elif model_id in ["glm-tts", "glm-tts-clone"]:
            voice = kwargs.get("voice", "female")
            return client.audio(model_id, prompt, voice=voice)
        else:
            raise ValueError(f"Unsupported model: {model_id}")

    def _call_gemini_model(self, model_key: str, prompt: str, **kwargs) -> str:
        if not GEMINI_AVAILABLE:
            raise RuntimeError("Gemini not available")
        client = GeminiClient(self.gemini_api_key)
        model_id = _MODEL_REGISTRY[model_key]["model_id"]
        return client.text(model_id, prompt, temperature=kwargs.get("temperature", 0.1))

# ----------------------------------------------------------------------
# Public loader – now expects a service name
# ----------------------------------------------------------------------
def load_model(service: str) -> YuktiModel:
    return YuktiModel(service)

# ----------------------------------------------------------------------
# Public configuration functions
# ----------------------------------------------------------------------
def get_available_models() -> List[str]:
    """Return list of service names that are usable (at least one model available)."""
    available = []
    for service in SERVICES:
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

def get_model_config(service: str) -> Optional[Dict[str, Any]]:
    """
    Return the configuration of the first model in the service's priority list.
    Used by think.py to determine model type (sync/async) and other metadata.
    """
    if service not in SERVICES:
        return None
    first_model = SERVICES[service][0]
    config = _MODEL_REGISTRY[first_model].copy()
    config['service'] = service
    config['model_key'] = first_model
    return config

# ----------------------------------------------------------------------
# Export symbols
# ----------------------------------------------------------------------
__all__ = [
    "get_available_models",
    "get_service_config",
    "get_model_config",
    "load_model",
    "get_active_tasks",
    "get_task_status",
    "ZHIPU_AVAILABLE",
    "GEMINI_AVAILABLE",
    "MODELS",
]
