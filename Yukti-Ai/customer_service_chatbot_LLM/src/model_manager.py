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

# Attempt to import zai-sdk for async video (optional)
try:
    from zai import ZhipuAiClient
    ZAI_AVAILABLE = True
except ImportError:
    ZAI_AVAILABLE = False
    logging.warning("zai-sdk not installed; video generation disabled.")

# Attempt to import Gemini (optional)
try:
    from langchain_google_genai import ChatGoogleGenerativeAI
    GEMINI_AVAILABLE = True
except ImportError:
    GEMINI_AVAILABLE = False
    logging.warning("langchain-google-genai not installed; Gemini fallback disabled.")

logger = logging.getLogger(__name__)

# ----------------------------------------------------------------------
# Model configurations – all use the same OpenAI‑compatible endpoint
# ----------------------------------------------------------------------
ZHIPU_BASE_URL = "https://api.z.ai/api/paas/v4/"  # new global endpoint

MODELS = {
    "Yukti‑Flash": {
        "model": "glm-4-flash",
        "type": "sync",
        "description": "Fast text & reasoning",
        "depends_on_zhipu": True,
        "requires_zai": False,
    },
    "Yukti‑Quantum": {
        "model": "glm-5",
        "type": "sync",
        "description": "Deep research & complex reasoning",
        "depends_on_zhipu": True,
        "requires_zai": False,
    },
    "Yukti‑Image": {
        "model": "cogview-4",
        "type": "sync",
        "description": "Image generation (returns URL)",
        "depends_on_zhipu": True,
        "requires_zai": False,
    },
    "Yukti‑Video": {
        "model": "cogvideox-3",
        "type": "async",
        "description": "Video generation (queued)",
        "depends_on_zhipu": True,
        "requires_zai": True,
    },
    "Yukti‑Audio": {
        "model": "glm-tts",
        "type": "sync",
        "description": "Text‑to‑speech (returns audio file)",
        "depends_on_zhipu": True,
        "requires_zai": False,
    },
    "Gemini 2.0 Flash": {
        "model": "gemini-2.0-flash-exp",
        "type": "sync",
        "description": "Google Gemini fallback (fast)",
        "depends_on_zhipu": False,
        "requires_zai": False,
    }
}

# ----------------------------------------------------------------------
# API key management
# ----------------------------------------------------------------------
def get_zhipu_api_key() -> Optional[str]:
    """Return Zhipu API key from secrets or environment."""
    return st.secrets.get("ZHIPU_API_KEY") or os.getenv("ZHIPU_API_KEY")

def get_google_api_key() -> Optional[str]:
    """Return Google API key from secrets or environment."""
    return st.secrets.get("GOOGLE_API_KEY") or os.getenv("GOOGLE_API_KEY")

# Determine Zhipu availability
ZHIPU_AVAILABLE = get_zhipu_api_key() is not None

# ----------------------------------------------------------------------
# Synchronous client for Zhipu text models (OpenAI‑compatible)
# ----------------------------------------------------------------------
@st.cache_resource
def get_zhipu_text_client() -> Optional[ChatOpenAI]:
    """Return a cached ChatOpenAI client for Zhipu text models."""
    api_key = get_zhipu_api_key()
    if not api_key:
        return None
    return ChatOpenAI(
        model="placeholder",  # will be overridden per call
        api_key=api_key,
        base_url=ZHIPU_BASE_URL,
        temperature=0.1,
        max_retries=2,
        request_timeout=30,
    )

# ----------------------------------------------------------------------
# Helper for direct Zhipu API calls (image, audio)
# ----------------------------------------------------------------------
def call_zhipu_direct(endpoint: str, payload: dict) -> dict:
    """
    Make a direct POST request to Zhipu API with proper error handling.
    Returns parsed JSON response.
    """
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

# ----------------------------------------------------------------------
# Async video queue using zai-sdk
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
                client = ZhipuAiClient(api_key=api_key)
                # Build arguments for video generation
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
                # Call the video generation API
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
                        api_key = get_zhipu_api_key()
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
# Public functions to access task queue
# ----------------------------------------------------------------------
def get_active_tasks() -> List[Tuple[str, str, str, int]]:
    """Return list of active async tasks (video)."""
    return _task_queue.get_active_tasks()

def get_task_status(task_id: str) -> Optional[Dict[str, Any]]:
    """Return status of a specific task."""
    return _task_queue.get_task(task_id)

# ----------------------------------------------------------------------
# Model availability check
# ----------------------------------------------------------------------
def get_available_models() -> List[str]:
    """Return list of models actually available given dependencies and keys."""
    available = []
    for name, config in MODELS.items():
        if config.get("depends_on_zhipu", False):
            # Zhipu model – needs API key
            if not get_zhipu_api_key():
                continue
            if config.get("requires_zai", False) and not ZAI_AVAILABLE:
                continue
            available.append(name)
        else:
            # Gemini model – needs Google API key and gemini library
            if GEMINI_AVAILABLE and get_google_api_key():
                available.append(name)
    return available

def get_model_config(model_key: str) -> Optional[Dict[str, Any]]:
    """Return configuration for a given model key."""
    return MODELS.get(model_key)

# ----------------------------------------------------------------------
# Core model invocation class
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
        Invoke the model. For sync models returns appropriate result (text, image URL, audio path).
        For async returns task_id.
        """
        if self.config.get("type") == "async":
            # Video only
            if not ZAI_AVAILABLE:
                raise RuntimeError("Video generation requires zai-sdk.")
            return _task_queue.submit_async(
                variant=self.model_key,
                model=self.model_name,
                prompt=prompt,
                **kwargs
            )

        # Sync models
        if self.config.get("depends_on_zhipu", False):
            # Primary: Zhipu
            return self._call_zhipu(prompt, **kwargs)
        else:
            # Fallback: Gemini
            return self._call_gemini(prompt, **kwargs)

    def _call_zhipu(self, prompt: str, **kwargs) -> Any:
        """Call Zhipu model (text, image, audio)."""
        # Text models (glm-4-flash, glm-5)
        if self.model_name in ["glm-4-flash", "glm-5"]:
            client = get_zhipu_text_client()
            if client is None:
                raise RuntimeError("Zhipu API key not configured.")
            # Override model per call
            llm = ChatOpenAI(
                model=self.model_name,
                api_key=client.api_key,
                base_url=client.base_url,
                temperature=kwargs.get("temperature", 0.1),
                max_retries=2
            )
            return llm.invoke(prompt)

        # Image generation (cogview-4)
        if self.model_name == "cogview-4":
            payload = {"model": self.model_name, "prompt": prompt}
            data = call_zhipu_direct("images/generations", payload)
            return data["data"][0]["url"]

        # Audio generation (glm-tts)
        if self.model_name == "glm-tts":
            payload = {
                "model": self.model_name,
                "input": prompt,
                "voice": kwargs.get("voice", "female"),
                "response_format": "wav"
            }
            resp = call_zhipu_direct("audio/speech", payload)
            # The response is binary audio; we need to download and save.
            # But call_zhipu_direct returns JSON; we need to handle stream differently.
            # Actually, the /audio/speech endpoint returns binary audio directly.
            # We'll do a separate request with stream=True.
            api_key = get_zhipu_api_key()
            headers = {"Authorization": f"Bearer {api_key}"}
            url = f"{ZHIPU_BASE_URL.rstrip('/')}/audio/speech"
            http_resp = requests.post(url, json=payload, headers=headers, stream=True, timeout=60)
            http_resp.raise_for_status()
            fd, path = tempfile.mkstemp(suffix=".wav", prefix="yukti_audio_")
            with os.fdopen(fd, "wb") as f:
                for chunk in http_resp.iter_content(chunk_size=8192):
                    f.write(chunk)
            return path

        raise ValueError(f"Unsupported Zhipu model: {self.model_name}")

    def _call_gemini(self, prompt: str, **kwargs) -> Any:
        """Fallback to Gemini."""
        if not GEMINI_AVAILABLE:
            raise RuntimeError("Gemini fallback not available (library missing).")
        api_key = get_google_api_key()
        if not api_key:
            raise RuntimeError("Google API key not configured for Gemini.")
        # Use a valid Gemini model name (update if needed)
        gemini_model = self.model_name if "gemini" in self.model_name else "gemini-2.0-flash-exp"
        llm = ChatGoogleGenerativeAI(
            model=gemini_model,
            google_api_key=api_key,
            temperature=kwargs.get("temperature", 0.1)
        )
        return llm.invoke(prompt)

# ----------------------------------------------------------------------
# Public interface
# ----------------------------------------------------------------------
def load_model(model_key: str) -> YuktiModel:
    """Return a YuktiModel instance for the given key."""
    return YuktiModel(model_key)

# ----------------------------------------------------------------------
# For standalone testing
# ----------------------------------------------------------------------
if __name__ == "__main__":
    print("Available models:", get_available_models())
    if get_available_models():
        model = load_model(get_available_models()[0])
        print(model.invoke("Hello, how are you?"))
