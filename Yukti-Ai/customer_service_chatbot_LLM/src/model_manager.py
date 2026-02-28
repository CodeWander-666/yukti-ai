"""
Yukti AI – Model Manager (Resilient Edition)
Handles all Yukti models with graceful degradation if dependencies are missing.
"""

import logging
import time
import threading
import sqlite3
import os
from datetime import datetime
from pathlib import Path
from typing import Dict, Any, Optional, List, Tuple

import streamlit as st

logger = logging.getLogger(__name__)

# ----------------------------------------------------------------------
# Attempt to import Zhipu SDK; if missing, we'll use dummy fallbacks
# ----------------------------------------------------------------------
ZHIPU_AVAILABLE = False
ZHIPU_SDK_IMPORT_ERROR = None
try:
    from zhipuai import ZhipuAI
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
        "description": "Fast text & reasoning (200 concurrent)",
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
        "type": "async",
        "description": "Image generation (queued)",
        "depends_on_zhipu": True,
    },
    "Yukti‑Video": {
        "model": "cogvideox",
        "type": "async",
        "description": "Video generation (queued)",
        "depends_on_zhipu": True,
    },
    "Yukti‑Audio": {
        "model": "glm-realtime",
        "type": "async",
        "description": "Audio generation (queued)",
        "depends_on_zhipu": True,
    },
    # Fallback model if Zhipu is missing (Gemini via langchain)
    "Gemini 3 Flash": {
        "model": "gemini-3-flash",
        "type": "sync",
        "description": "Google Gemini fallback (fast & free tier)",
        "depends_on_zhipu": False,
    }
}

def get_available_models() -> List[str]:
    """Return list of models that are actually available given current dependencies."""
    if ZHIPU_AVAILABLE:
        return list(MODELS.keys())
    else:
        # Only return non‑Zhipu models (Gemini)
        return [k for k, v in MODELS.items() if not v.get("depends_on_zhipu", False)]

def get_model_config(model_key: str) -> Optional[Dict[str, Any]]:
    return MODELS.get(model_key)

# ----------------------------------------------------------------------
# If Zhipu is unavailable, define dummy classes and functions
# ----------------------------------------------------------------------
if not ZHIPU_AVAILABLE:
    # Dummy client and queue that raise helpful errors
    class ZhipuAI:
        def __init__(self, *args, **kwargs):
            raise RuntimeError(f"Zhipu SDK not installed: {ZHIPU_SDK_IMPORT_ERROR}")

    class ZhipuTaskQueue:
        def __init__(self, *args, **kwargs):
            pass
        def submit_async(self, *args, **kwargs):
            raise RuntimeError("Async tasks require Zhipu SDK, which is not installed.")
        def get_active_tasks(self):
            return []
        def get_task(self, task_id):
            return None

    _task_queue = ZhipuTaskQueue()

    @st.cache_resource
    def get_zhipu_client():
        raise RuntimeError("Zhipu SDK not installed. Please install it with: pip install zhipuai")

    # For compatibility, define dummy task functions
    def get_active_tasks() -> List[Tuple[str, str, str, int]]:
        return []
    def get_task_status(task_id: str) -> Optional[Dict[str, Any]]:
        return None

# ----------------------------------------------------------------------
# If Zhipu is available, proceed with real implementation
# ----------------------------------------------------------------------
else:
    @st.cache_resource
    def get_zhipu_client() -> ZhipuAI:
        """Return a cached Zhipu client with API key from secrets."""
        api_key = st.secrets.get("ZHIPU_API_KEY") or os.getenv("ZHIPU_API_KEY")
        if not api_key:
            raise ValueError("ZHIPU_API_KEY not found in secrets or environment.")
        return ZhipuAI(api_key=api_key)

    # ------------------------------------------------------------------
    # Task Queue for Async Models
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

        def submit_async(self, variant: str, model: str, prompt: str) -> str:
            """Submit an async generation task and return local task ID."""
            local_id = f"{variant}_{int(time.time())}_{abs(hash(prompt)) % 10000}"
            self.add_task(local_id, variant, model)

            try:
                client = get_zhipu_client()
                # NOTE: The actual API methods depend on the Zhipu SDK version.
                # Below are placeholders; you must adapt to the actual SDK.
                if model.startswith("cogview"):
                    # Example: for image generation, the SDK might have:
                    resp = client.images.generate(model=model, prompt=prompt)
                    zhipu_task_id = resp.id  # adjust based on actual response
                elif model.startswith("cogvideox"):
                    resp = client.videos.generate(model=model, prompt=prompt)
                    zhipu_task_id = resp.id
                elif model == "glm-realtime":
                    resp = client.audio.generate(model=model, prompt=prompt)
                    zhipu_task_id = resp.id
                else:
                    raise ValueError(f"Unknown async model: {model}")

                with self.lock:
                    self.zhipu_map[local_id] = zhipu_task_id
                self.update_task(local_id, status="pending")
            except Exception as e:
                logger.exception("Failed to submit async task")
                self.update_task(local_id, status="failed", error=str(e))

            return local_id

        def _poll_tasks(self):
            """Background thread that polls all pending tasks."""
            while True:
                tasks = []
                with self.lock:
                    tasks = list(self.zhipu_map.items())
                for local_id, zhipu_id in tasks:
                    try:
                        client = get_zhipu_client()
                        # Example: retrieving file status – adjust to actual SDK.
                        status_data = client.files.retrieve(zhipu_id)
                        if status_data.status == "succeeded":
                            self.update_task(local_id, status="completed", progress=100,
                                             result_url=status_data.result_url)
                            with self.lock:
                                del self.zhipu_map[local_id]
                        elif status_data.status == "failed":
                            self.update_task(local_id, status="failed", error=status_data.error)
                            with self.lock:
                                del self.zhipu_map[local_id]
                        elif status_data.status == "processing":
                            progress = getattr(status_data, "progress", 50)
                            self.update_task(local_id, status="processing", progress=progress)
                    except Exception as e:
                        logger.error(f"Polling task {local_id} failed: {e}")
                time.sleep(5)

        def _start_poller(self):
            thread = threading.Thread(target=self._poll_tasks, daemon=True)
            thread.start()

    _task_queue = ZhipuTaskQueue(db_path=str(Path(__file__).parent.parent / "yukti_tasks.db"))

    # Public task functions
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
        """Invoke the model. For sync models returns answer string; for async returns local task_id."""
        # If the model requires Zhipu but Zhipu is unavailable, raise clear error.
        if self.depends_on_zhipu and not ZHIPU_AVAILABLE:
            raise RuntimeError(
                f"Model '{self.model_key}' requires Zhipu SDK, which is not installed. "
                f"Please install it with: pip install zhipuai\n"
                f"(Import error: {ZHIPU_SDK_IMPORT_ERROR})"
            )

        if self.config["type"] == "sync":
            # Use Zhipu client for sync models
            if ZHIPU_AVAILABLE:
                client = get_zhipu_client()
                max_retries = 3
                for attempt in range(max_retries):
                    try:
                        response = client.chat.completions.create(
                            model=self.model_name,
                            messages=[{"role": "user", "content": prompt}],
                            temperature=kwargs.get("temperature", 0.1),
                            max_tokens=kwargs.get("max_tokens", 1024),
                            timeout=30
                        )
                        return response.choices[0].message.content
                    except Exception as e:
                        logger.warning(f"Attempt {attempt+1} failed: {e}")
                        if attempt == max_retries - 1:
                            raise RuntimeError(f"Sync model failed after {max_retries} attempts: {e}")
                        time.sleep(2 ** attempt)
            else:
                # Fallback to Gemini (if available) – we'll try to import langchain_google_genai on the fly
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
        else:
            # Async – always requires Zhipu (already checked above)
            return _task_queue.submit_async(
                variant=self.model_key,
                model=self.model_name,
                prompt=prompt
            )

# ----------------------------------------------------------------------
# Public interface
# ----------------------------------------------------------------------
def load_model(model_key: str) -> YuktiModel:
    return YuktiModel(model_key)
