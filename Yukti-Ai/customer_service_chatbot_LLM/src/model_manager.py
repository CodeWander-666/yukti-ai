"""
Yukti AI – Model Manager (High-End Multimodal Edition)
Integrates GLM-Image, CogVideoX-3, GLM-TTS using official zai-sdk.
Handles sync/async tasks, real‑time queue, and exhaustive error handling.
"""

import logging
import time
import threading
import sqlite3
import os
import tempfile
from datetime import datetime
from pathlib import Path
from typing import Dict, Any, Optional, List, Tuple

import streamlit as st

logger = logging.getLogger(__name__)

# ----------------------------------------------------------------------
# Attempt to import Zhipu zai-sdk; fallback gracefully
# ----------------------------------------------------------------------
ZHIPU_AVAILABLE = False
ZHIPU_SDK_IMPORT_ERROR = None
try:
    from zai import ZhipuAiClient
    from zai.exceptions import APIError, AuthenticationError, RateLimitError
    ZHIPU_AVAILABLE = True
except ImportError as e:
    ZHIPU_SDK_IMPORT_ERROR = str(e)
    logger.warning(f"Zhipu SDK not installed: {e}. Some models will be unavailable.")

# ----------------------------------------------------------------------
# Model configurations (always defined)
# ----------------------------------------------------------------------
MODELS = {
    "Yukti‑Flash": {
        "model": "glm-4-flash",
        "type": "sync",
        "api": "chat",
        "description": "Fast text & reasoning",
        "depends_on_zhipu": True,
    },
    "Yukti‑Quantum": {
        "model": "glm-5",
        "type": "sync",
        "api": "chat",
        "description": "Deep research & complex reasoning",
        "depends_on_zhipu": True,
    },
    "Yukti‑Image": {
        "model": "glm-image",
        "type": "sync",
        "api": "images.generations",
        "description": "Image generation (0.1元/次)",
        "depends_on_zhipu": True,
        "output_format": "url",
    },
    "Yukti‑Video": {
        "model": "cogvideox-3",
        "type": "async",
        "api": "videos.generations",
        "poll_api": "videos.retrieve_videos_result",
        "description": "Video generation (1元/次)",
        "depends_on_zhipu": True,
        "output_format": "video_url",
    },
    "Yukti‑Audio": {
        "model": "glm-tts",
        "type": "sync",
        "api": "audio.speech",
        "description": "Text‑to‑speech (multiple voices)",
        "depends_on_zhipu": True,
        "output_format": "audio_file",
    },
    "Gemini 3 Flash": {
        "model": "gemini-3-flash",
        "type": "sync",
        "api": "gemini",
        "description": "Google Gemini fallback (fast & free tier)",
        "depends_on_zhipu": False,
    }
}

def get_available_models() -> List[str]:
    """Return list of models actually available given dependencies."""
    if ZHIPU_AVAILABLE:
        return list(MODELS.keys())
    else:
        return [k for k, v in MODELS.items() if not v.get("depends_on_zhipu", False)]

def get_model_config(model_key: str) -> Optional[Dict[str, Any]]:
    return MODELS.get(model_key)

# ----------------------------------------------------------------------
# If Zhipu is unavailable, define dummy classes and functions
# ----------------------------------------------------------------------
if not ZHIPU_AVAILABLE:
    # Dummy client and queue that raise helpful errors
    class ZhipuAiClient:
        def __init__(self, *args, **kwargs):
            raise RuntimeError(f"Zhipu SDK not installed: {ZHIPU_SDK_IMPORT_ERROR}")

    class ZhipuTaskQueue:
        def __init__(self, *args, **kwargs):
            pass
        def submit_async(self, *args, **kwargs):
            raise RuntimeError("Async tasks require Zhipu SDK.")
        def get_active_tasks(self):
            return []
        def get_task(self, task_id):
            return None

    _task_queue = ZhipuTaskQueue()
    _sync_client = ZhipuAiClient()

    @st.cache_resource
    def get_zhipu_client():
        raise RuntimeError("Zhipu SDK not installed. Please install with: pip install zai-sdk==0.2.2")

    def get_active_tasks() -> List[Tuple[str, str, str, int]]:
        return []
    def get_task_status(task_id: str) -> Optional[Dict[str, Any]]:
        return None

# ----------------------------------------------------------------------
# If Zhipu is available, proceed with real implementation
# ----------------------------------------------------------------------
else:
    @st.cache_resource
    def get_zhipu_client() -> ZhipuAiClient:
        """Return a cached Zhipu client with API key from secrets."""
        api_key = st.secrets.get("ZHIPU_API_KEY") or os.getenv("ZHIPU_API_KEY")
        if not api_key:
            raise ValueError("ZHIPU_API_KEY not found in secrets or environment.")
        return ZhipuAiClient(api_key=api_key)

    # ------------------------------------------------------------------
    # Task Queue for Async Video Generation (CogVideoX-3)
    # ------------------------------------------------------------------
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
            """Submit an async video generation task and return local task ID."""
            local_id = f"{variant}_{int(time.time())}_{abs(hash(prompt)) % 10000}"
            self.add_task(local_id, variant, model)

            try:
                client = get_zhipu_client()
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
            except (APIError, AuthenticationError, RateLimitError) as e:
                error_msg = f"Zhipu API error: {e}"
                logger.exception(error_msg)
                self.update_task(local_id, status="failed", error=error_msg)
            except Exception as e:
                logger.exception("Unexpected error submitting async task")
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
                        client = get_zhipu_client()
                        # Poll the video result
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
                            # Some progress might be available; default to 50%
                            progress = getattr(status_data, 'progress', 50)
                            self.update_task(local_id, status="processing", progress=progress)
                        else:
                            # If no status field, assume still pending
                            pass
                    except Exception as e:
                        logger.error(f"Polling task {local_id} failed: {e}")
                time.sleep(5)

        def _start_poller(self):
            thread = threading.Thread(target=self._poll_tasks, daemon=True)
            thread.start()

    _task_queue = ZhipuTaskQueue(db_path=str(Path(__file__).parent.parent / "yukti_tasks.db"))

    # ------------------------------------------------------------------
    # Sync client wrapper
    # ------------------------------------------------------------------
    class ZhipuSyncClient:
        def __init__(self):
            self.client = get_zhipu_client()

        def chat(self, model: str, prompt: str, temperature: float = 0.1, max_tokens: int = 1024) -> str:
            """Sync chat completion."""
            try:
                response = self.client.chat.completions.create(
                    model=model,
                    messages=[{"role": "user", "content": prompt}],
                    temperature=temperature,
                    max_tokens=max_tokens
                )
                return response.choices[0].message.content
            except (APIError, AuthenticationError, RateLimitError) as e:
                logger.exception("Zhipu API error in chat")
                raise RuntimeError(f"Zhipu API error: {e}")
            except Exception as e:
                logger.exception("Unexpected error in chat")
                raise RuntimeError(f"Chat failed: {e}")

        def generate_image(self, model: str, prompt: str) -> str:
            """Generate image, return URL."""
            try:
                response = self.client.images.generations(model=model, prompt=prompt)
                return response.data[0].url
            except (APIError, AuthenticationError, RateLimitError) as e:
                logger.exception("Zhipu API error in image generation")
                raise RuntimeError(f"Image generation failed: {e}")
            except Exception as e:
                logger.exception("Unexpected error in image generation")
                raise RuntimeError(f"Image generation error: {e}")

        def generate_audio(self, model: str, prompt: str, voice: str = "female") -> str:
            """Generate speech, save to temporary file and return path."""
            try:
                response = self.client.audio.speech(
                    model=model,
                    input=prompt,
                    voice=voice,
                    response_format="wav"
                )
                # Save to temporary file
                fd, path = tempfile.mkstemp(suffix=".wav", prefix="yukti_audio_")
                os.close(fd)
                response.stream_to_file(path)
                return path
            except (APIError, AuthenticationError, RateLimitError) as e:
                logger.exception("Zhipu API error in TTS")
                raise RuntimeError(f"Audio generation failed: {e}")
            except Exception as e:
                logger.exception("Unexpected error in TTS")
                raise RuntimeError(f"TTS error: {e}")

    _sync_client = ZhipuSyncClient()

    # ------------------------------------------------------------------
    # Public task functions
    # ------------------------------------------------------------------
    def get_active_tasks() -> List[Tuple[str, str, str, int]]:
        return _task_queue.get_active_tasks()

    def get_task_status(task_id: str) -> Optional[Dict[str, Any]]:
        return _task_queue.get_task(task_id)

# ----------------------------------------------------------------------
# YuktiModel wrapper – works with or without Zhipu
# ----------------------------------------------------------------------
class YuktiModel:
    def __init__(self, model_key: str):
        self.config = get_model_config(model_key)
        if not self.config:
            raise ValueError(f"Unknown model key: {model_key}")
        self.model_key = model_key
        self.model_name = self.config["model"]
        self.depends_on_zhipu = self.config.get("depends_on_zhipu", False)

    def invoke(self, prompt: str, **kwargs) -> Any:
        """
        Invoke the model.
        For sync: returns result (string or URL or file path).
        For async: returns local task_id.
        """
        # If the model requires Zhipu but Zhipu is unavailable, raise clear error.
        if self.depends_on_zhipu and not ZHIPU_AVAILABLE:
            raise RuntimeError(
                f"Model '{self.model_key}' requires Zhipu SDK, which is not installed. "
                f"Please install it with: pip install zai-sdk==0.2.2\n"
                f"(Import error: {ZHIPU_SDK_IMPORT_ERROR})"
            )

        # If Zhipu is available, use it; otherwise fallback to Gemini (if configured)
        if self.depends_on_zhipu and ZHIPU_AVAILABLE:
            if self.config["type"] == "sync":
                # Route to appropriate sync method
                if self.config["api"] == "chat":
                    return _sync_client.chat(self.model_name, prompt, temperature=kwargs.get("temperature", 0.1))
                elif self.config["api"] == "images.generations":
                    return _sync_client.generate_image(self.model_name, prompt)
                elif self.config["api"] == "audio.speech":
                    voice = kwargs.get("voice", "female")
                    return _sync_client.generate_audio(self.model_name, prompt, voice)
                else:
                    raise ValueError(f"Unknown sync API: {self.config['api']}")
            else:
                # Async (video)
                return _task_queue.submit_async(
                    variant=self.model_key,
                    model=self.model_name,
                    prompt=prompt,
                    **kwargs
                )
        else:
            # Fallback to Gemini (if available)
            try:
                from langchain_google_genai import ChatGoogleGenerativeAI
                llm = ChatGoogleGenerativeAI(
                    model=self.model_name,
                    google_api_key=st.secrets.get("GOOGLE_API_KEY"),
                    temperature=0.1
                )
                return llm.invoke(prompt)
            except Exception as e:
                raise RuntimeError(f"Fallback to Gemini failed: {e}")

# ----------------------------------------------------------------------
# Public interface
# ----------------------------------------------------------------------
def load_model(model_key: str) -> YuktiModel:
    return YuktiModel(model_key)
