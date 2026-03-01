"""
Yukti AI – Model Manager (Ultimate Production Edition)
Handles all Yukti models via Zhipu with Gemini fallback. Includes async video queue,
comprehensive error handling, and is designed for scalability.
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
    logging.warning("google-genai not installed; Gemini fallback disabled.")

logger = logging.getLogger(__name__)

# ----------------------------------------------------------------------
# Constants
# ----------------------------------------------------------------------
ZHIPU_BASE_URL = "https://api.z.ai/api/paas/v4/"
DEFAULT_GEMINI_MODEL = "gemini-1.5-flash"   # adjust as needed

# ----------------------------------------------------------------------
# Model Registry
# ----------------------------------------------------------------------
MODELS = {
    "Yukti‑Flash": {
        "model": "glm-4-flash",
        "type": "sync",
        "provider": "zhipu",
        "description": "Fast text & reasoning",
        "requires_zai": False,
    },
    "Yukti‑Quantum": {
        "model": "glm-5",
        "type": "sync",
        "provider": "zhipu",
        "description": "Deep research & complex reasoning",
        "requires_zai": False,
    },
    "Yukti‑Image": {
        "model": "cogview-4",
        "type": "sync",
        "provider": "zhipu",
        "description": "Image generation (returns URL)",
        "requires_zai": False,
    },
    "Yukti‑Video": {
        "model": "cogvideox-3",
        "type": "async",
        "provider": "zhipu",
        "description": "Video generation (queued)",
        "requires_zai": True,
    },
    "Yukti‑Audio": {
        "model": "glm-tts",
        "type": "sync",
        "provider": "zhipu",
        "description": "Text‑to‑speech (returns audio file)",
        "requires_zai": False,
    },
    "Gemini 1.5 Flash": {
        "model": DEFAULT_GEMINI_MODEL,
        "type": "sync",
        "provider": "gemini",
        "description": "Google Gemini fallback",
        "requires_zai": False,
    }
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
# Cached Clients
# ----------------------------------------------------------------------
@st.cache_resource(show_spinner=False)
def get_zhipu_text_client() -> Optional[ChatOpenAI]:
    """Return a cached ChatOpenAI client for Zhipu text models."""
    api_key = get_zhipu_api_key()
    if not api_key:
        return None
    return ChatOpenAI(
        model="placeholder",   # will be overridden per call
        api_key=api_key,
        base_url=ZHIPU_BASE_URL,
        temperature=0.1,
        max_retries=2,
        request_timeout=30,
    )

@st.cache_resource(show_spinner=False)
def get_gemini_client() -> Optional[genai.Client]:
    """Return a cached Gemini client."""
    api_key = get_google_api_key()
    if not api_key or not GEMINI_SDK_AVAILABLE:
        return None
    return genai.Client(api_key=api_key)

# ----------------------------------------------------------------------
# Direct Zhipu API Helpers (for image, audio)
# ----------------------------------------------------------------------
def call_zhipu_direct(endpoint: str, payload: dict) -> dict:
    """Make a direct POST request to Zhipu API and return JSON response."""
    api_key = get_zhipu_api_key()
    if not api_key:
        raise RuntimeError("Zhipu API key not configured.")
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json"
    }
    url = f"{ZHIPU_BASE_URL.rstrip('/')}/{endpoint.lstrip('/')}"
    try:
        resp = requests.post(url, json=payload, headers=headers, timeout=60)
        resp.raise_for_status()
        return resp.json()
    except requests.exceptions.RequestException as e:
        logger.exception(f"Zhipu direct API call failed: {e}")
        raise RuntimeError(f"Zhipu API error: {e}")

def call_zhipu_audio(payload: dict) -> bytes:
    """Make a POST request to /audio/speech and return the binary content."""
    api_key = get_zhipu_api_key()
    if not api_key:
        raise RuntimeError("Zhipu API key not configured.")
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json"
    }
    url = f"{ZHIPU_BASE_URL.rstrip('/')}/audio/speech"
    try:
        resp = requests.post(url, json=payload, headers=headers, stream=True, timeout=60)
        resp.raise_for_status()
        return resp.content
    except requests.exceptions.RequestException as e:
        logger.exception(f"Zhipu audio API call failed: {e}")
        raise RuntimeError(f"Zhipu audio API error: {e}")

# ----------------------------------------------------------------------
# Async Task Queue for Video (SQLite + background polling)
# ----------------------------------------------------------------------
class AsyncTaskQueue:
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
        if not ZAI_AVAILABLE:
            raise RuntimeError("Video generation requires zai-sdk.")
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

# Instantiate the queue (global)
_task_queue = AsyncTaskQueue(db_path=str(Path(__file__).parent.parent / "yukti_tasks.db"))

# ----------------------------------------------------------------------
# Public Queue Functions
# ----------------------------------------------------------------------
def get_active_tasks() -> List[Tuple[str, str, str, int]]:
    return _task_queue.get_active_tasks()

def get_task_status(task_id: str) -> Optional[Dict[str, Any]]:
    return _task_queue.get_task(task_id)

# ----------------------------------------------------------------------
# Provider Clients (Zhipu, Gemini)
# ----------------------------------------------------------------------
class ZhipuClient:
    """Handles all synchronous Zhipu API calls (text, image, audio)."""
    def __init__(self, api_key: str):
        self.api_key = api_key

    def text(self, model: str, prompt: str, temperature: float = 0.1) -> str:
        """Invoke a Zhipu text model via ChatOpenAI."""
        # We can reuse the cached client but need to pass our key and URL.
        # The cached client is just a convenience; we create a new instance per call
        # with the correct model. That's fine because it's lightweight.
        llm = ChatOpenAI(
            model=model,
            api_key=self.api_key,
            base_url=ZHIPU_BASE_URL,
            temperature=temperature,
            max_retries=2,
        )
        return llm.invoke(prompt).content

    def image(self, model: str, prompt: str) -> str:
        """Generate an image and return its URL."""
        payload = {"model": model, "prompt": prompt}
        data = call_zhipu_direct("images/generations", payload)
        return data["data"][0]["url"]

    def audio(self, model: str, prompt: str, voice: str = "female") -> str:
        """Generate speech and return path to a temporary file."""
        payload = {
            "model": model,
            "input": prompt,
            "voice": voice,
            "response_format": "wav"
        }
        audio_bytes = call_zhipu_audio(payload)
        fd, path = tempfile.mkstemp(suffix=".wav", prefix="yukti_audio_")
        with os.fdopen(fd, "wb") as f:
            f.write(audio_bytes)
        return path

class GeminiClient:
    """Handles Gemini text generation via google-genai SDK."""
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
# Model Availability Check
# ----------------------------------------------------------------------
def get_available_models() -> List[str]:
    """Return list of models that are actually usable given keys/dependencies."""
    available = []
    for name, config in MODELS.items():
        if config["provider"] == "zhipu":
            if not ZHIPU_AVAILABLE:
                continue
            if config.get("requires_zai") and not ZAI_AVAILABLE:
                continue
            available.append(name)
        elif config["provider"] == "gemini":
            if GEMINI_AVAILABLE:
                available.append(name)
    return available

def get_model_config(model_key: str) -> Optional[Dict[str, Any]]:
    return MODELS.get(model_key)

# ----------------------------------------------------------------------
# YuktiModel Class – returned by load_model
# ----------------------------------------------------------------------
class YuktiModel:
    def __init__(self, model_key: str):
        self.config = get_model_config(model_key)
        if not self.config:
            raise ValueError(f"Unknown model key: {model_key}")
        self.model_key = model_key
        self.model_name = self.config["model"]
        self.zhipu_api_key = get_zhipu_api_key()
        self.gemini_api_key = get_google_api_key()

    def invoke(self, prompt: str, **kwargs) -> Any:
        """Main entry point for model invocation."""
        if self.config["type"] == "async":
            # Video only
            return _task_queue.submit_async(
                variant=self.model_key,
                model=self.model_name,
                prompt=prompt,
                **kwargs
            )

        # Synchronous models
        # Try primary provider (Zhipu) if applicable
        if self.config["provider"] == "zhipu":
            try:
                return self._call_zhipu(prompt, **kwargs)
            except Exception as e:
                logger.warning(f"Zhipu failed, attempting Gemini fallback: {e}")
                # Fallback to Gemini (if available and model is sync)
                if self._can_fallback_to_gemini():
                    return self._call_gemini(prompt, **kwargs)
                else:
                    raise  # re-raise original error

        elif self.config["provider"] == "gemini":
            # Direct Gemini call
            return self._call_gemini(prompt, **kwargs)

        else:
            raise ValueError(f"Unsupported provider: {self.config['provider']}")

    def _can_fallback_to_gemini(self) -> bool:
        """Check if Gemini fallback is possible."""
        return GEMINI_AVAILABLE and self.config["type"] == "sync"

    def _call_zhipu(self, prompt: str, **kwargs) -> Any:
        """Call Zhipu for text, image, or audio."""
        client = ZhipuClient(self.zhipu_api_key)
        if self.model_name in ["glm-4-flash", "glm-5"]:
            return client.text(self.model_name, prompt, temperature=kwargs.get("temperature", 0.1))
        elif self.model_name == "cogview-4":
            return client.image(self.model_name, prompt)
        elif self.model_name == "glm-tts":
            voice = kwargs.get("voice", "female")
            return client.audio(self.model_name, prompt, voice=voice)
        else:
            raise ValueError(f"Unsupported Zhipu model: {self.model_name}")

    def _call_gemini(self, prompt: str, **kwargs) -> str:
        """Fallback to Gemini."""
        if not GEMINI_AVAILABLE:
            raise RuntimeError("Gemini fallback not available (missing SDK or API key).")
        client = GeminiClient(self.gemini_api_key)
        # Use model from config (which may be the Gemini model name)
        return client.text(self.model_name, prompt, temperature=kwargs.get("temperature", 0.1))

# ----------------------------------------------------------------------
# Public Loader
# ----------------------------------------------------------------------
def load_model(model_key: str) -> YuktiModel:
    return YuktiModel(model_key)

# ----------------------------------------------------------------------
# Expose flags for UI
# ----------------------------------------------------------------------
__all__ = [
    "get_available_models",
    "MODELS",
    "get_active_tasks",
    "get_task_status",
    "ZHIPU_AVAILABLE",
    "GEMINI_AVAILABLE",
    "load_model",
    "get_model_config",
]
