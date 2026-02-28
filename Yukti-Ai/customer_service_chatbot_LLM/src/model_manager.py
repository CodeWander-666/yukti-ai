"""
Yukti AI – Model Manager (2026 Edition)
Uses OpenAI‑compatible endpoint for all Zhipu models and zai-sdk for async video.
"""

import logging
import os
import time
import threading
import sqlite3
from datetime import datetime
from pathlib import Path
from typing import Dict, Any, Optional, List, Tuple

import streamlit as st
from langchain_openai import ChatOpenAI

# Attempt to import zai-sdk for async video (optional)
try:
    from zai import ZhipuAiClient
    ZHIPU_AVAILABLE = True          # <-- renamed from ZAI_AVAILABLE
except ImportError:
    ZHIPU_AVAILABLE = False
    logging.warning("zai-sdk not installed; video generation disabled.")

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
    },
    "Yukti‑Quantum": {
        "model": "glm-5",
        "type": "sync",
        "description": "Deep research & complex reasoning",
        "depends_on_zhipu": True,
    },
    "Yukti‑Image": {
        "model": "cogview-4",
        "type": "sync",
        "description": "Image generation (returns URL)",
        "depends_on_zhipu": True,
    },
    "Yukti‑Video": {
        "model": "cogvideox-3",
        "type": "async",
        "description": "Video generation (queued)",
        "depends_on_zhipu": True,
        "requires_zai": True,      # needs zai-sdk for polling
    },
    "Yukti‑Audio": {
        "model": "glm-tts",
        "type": "sync",
        "description": "Text‑to‑speech (returns audio file)",
        "depends_on_zhipu": True,
    },
    "Gemini 3 Flash": {
        "model": "gemini-3-flash",
        "type": "sync",
        "description": "Google Gemini fallback",
        "depends_on_zhipu": False,
    }
}

def get_available_models() -> List[str]:
    """Return list of models actually available given dependencies."""
    available = []
    for k, v in MODELS.items():
        if v.get("depends_on_zhipu", False):
            # For Zhipu models, we need an API key.
            if os.environ.get("ZHIPU_API_KEY") or st.secrets.get("ZHIPU_API_KEY"):
                if v.get("requires_zai") and not ZHIPU_AVAILABLE:   # <-- renamed
                    continue   # video requires zai-sdk
                available.append(k)
        else:
            available.append(k)   # Gemini always available if key present
    return available

def get_model_config(model_key: str) -> Optional[Dict[str, Any]]:
    return MODELS.get(model_key)

# ----------------------------------------------------------------------
# Synchronous client using ChatOpenAI (works for text, image, audio)
# ----------------------------------------------------------------------
@st.cache_resource
def get_sync_client() -> Optional[ChatOpenAI]:
    """Return a cached ChatOpenAI client for Zhipu models."""
    api_key = st.secrets.get("ZHIPU_API_KEY") or os.getenv("ZHIPU_API_KEY")
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
# Async queue for video tasks using zai-sdk
# ----------------------------------------------------------------------
if ZHIPU_AVAILABLE:   # <-- renamed
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
                api_key = st.secrets.get("ZHIPU_API_KEY") or os.getenv("ZHIPU_API_KEY")
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
                        api_key = st.secrets.get("ZHIPU_API_KEY") or os.getenv("ZHIPU_API_KEY")
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
            raise RuntimeError("Video generation requires zai-sdk. Install with: pip install zai-sdk")
        def get_active_tasks(self):
            return []
        def get_task(self, task_id):
            return None
    _task_queue = _DummyQueue()

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
        if self.config["type"] == "async":
            # Video only
            if not ZHIPU_AVAILABLE:   # <-- renamed
                raise RuntimeError("Video generation requires zai-sdk.")
            return _task_queue.submit_async(
                variant=self.model_key,
                model=self.model_name,
                prompt=prompt,
                **kwargs
            )

        # Sync models: use ChatOpenAI client
        client = get_sync_client()
        if client is None:
            # Fallback to Gemini if available
            from langchain_google_genai import ChatGoogleGenerativeAI
            gemini_key = st.secrets.get("GOOGLE_API_KEY") or os.getenv("GOOGLE_API_KEY")
            if not gemini_key:
                raise RuntimeError("No API key for Zhipu or Gemini.")
            llm = ChatGoogleGenerativeAI(
                model=self.model_name if "gemini" in self.model_name else "gemini-3-flash",
                google_api_key=gemini_key,
                temperature=0.1
            )
            return llm.invoke(prompt)

        # For text models
        if "glm" in self.model_name or "gemini" in self.model_name:
            # Use client with overridden model
            llm = ChatOpenAI(
                model=self.model_name,
                api_key=client.api_key,
                base_url=client.base_url,
                temperature=kwargs.get("temperature", 0.1),
                max_retries=2
            )
            return llm.invoke(prompt)

        # For image generation (cogview-4)
        if self.model_name == "cogview-4":
            # Zhipu's image API is not OpenAI‑compatible, so we use a direct request.
            import requests
            api_key = st.secrets.get("ZHIPU_API_KEY") or os.getenv("ZHIPU_API_KEY")
            headers = {"Authorization": f"Bearer {api_key}"}
            data = {"model": self.model_name, "prompt": prompt}
            resp = requests.post("https://api.z.ai/api/paas/v4/images/generations", json=data, headers=headers)
            resp.raise_for_status()
            return resp.json()["data"][0]["url"]

        # For audio generation (glm-tts)
        if self.model_name == "glm-tts":
            import requests, tempfile
            api_key = st.secrets.get("ZHIPU_API_KEY") or os.getenv("ZHIPU_API_KEY")
            headers = {"Authorization": f"Bearer {api_key}"}
            data = {
                "model": self.model_name,
                "input": prompt,
                "voice": kwargs.get("voice", "female"),
                "response_format": "wav"
            }
            resp = requests.post("https://api.z.ai/api/paas/v4/audio/speech", json=data, headers=headers, stream=True)
            resp.raise_for_status()
            fd, path = tempfile.mkstemp(suffix=".wav", prefix="yukti_audio_")
            with os.fdopen(fd, "wb") as f:
                for chunk in resp.iter_content(chunk_size=8192):
                    f.write(chunk)
            return path

        raise ValueError(f"Unknown model type: {self.model_name}")

# ----------------------------------------------------------------------
# Public interface
# ----------------------------------------------------------------------
def load_model(model_key: str) -> YuktiModel:
    return YuktiModel(model_key)

def get_active_tasks() -> List[Tuple[str, str, str, int]]:
    return _task_queue.get_active_tasks()

def get_task_status(task_id: str) -> Optional[Dict[str, Any]]:
    return _task_queue.get_task(task_id)
