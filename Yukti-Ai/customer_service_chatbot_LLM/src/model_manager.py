"""
Yukti AI – Model Manager (High-End Zhipu SDK Edition)
Manages all Yukti models with sync/async, queuing, progress tracking, and robust error handling.
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
from zhipuai import ZhipuAI

logger = logging.getLogger(__name__)

# ----------------------------------------------------------------------
# Model configurations – map Yukti names to Zhipu model IDs
# ----------------------------------------------------------------------
MODELS = {
    "Yukti‑Flash": {
        "model": "glm-4-flash",
        "type": "sync",
        "concurrency": 200,  # not enforced by queue (sync)
        "description": "Fast text & reasoning (200 concurrent)",
    },
    "Yukti‑Quantum": {
        "model": "glm-5",
        "type": "sync",
        "concurrency": 3,
        "description": "Deep research & complex reasoning",
    },
    "Yukti‑Image": {
        "model": "cogview-4",
        "type": "async",
        "concurrency": 5,
        "description": "Image generation (queued)",
    },
    "Yukti‑Video": {
        "model": "cogvideox",
        "type": "async",
        "concurrency": 5,
        "description": "Video generation (queued)",
    },
    "Yukti‑Audio": {
        "model": "glm-realtime",
        "type": "async",
        "concurrency": 5,
        "description": "Audio generation (queued)",
    },
}

def get_available_models() -> List[str]:
    return list(MODELS.keys())

def get_model_config(model_key: str) -> Optional[Dict[str, Any]]:
    return MODELS.get(model_key)

# ----------------------------------------------------------------------
# Cached Zhipu client (shared across sessions)
# ----------------------------------------------------------------------
@st.cache_resource
def get_zhipu_client() -> ZhipuAI:
    """Return a cached Zhipu client with API key from secrets."""
    api_key = st.secrets.get("ZHIPU_API_KEY") or os.getenv("ZHIPU_API_KEY")
    if not api_key:
        raise ValueError("ZHIPU_API_KEY not found in secrets or environment.")
    return ZhipuAI(api_key=api_key)

# ----------------------------------------------------------------------
# Task Queue for Async Models (Image, Video, Audio)
# Uses SQLite for persistence and a background polling thread.
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

    def submit_async(self, variant: str, model: str, prompt: str) -> str:
        """Submit an async generation task and return local task ID."""
        local_id = f"{variant}_{int(time.time())}_{abs(hash(prompt)) % 10000}"
        self.add_task(local_id, variant, model)

        try:
            client = get_zhipu_client()
            # The exact API call depends on the model type
            if model.startswith("cogview"):
                # Image generation – returns a task ID
                resp = client.images.generate(model=model, prompt=prompt)
                # In Zhipu SDK, images.generate might return a task object
                # Adjust according to actual SDK (this is a placeholder)
                zhipu_task_id = resp.id  # Assuming it has .id
            elif model.startswith("cogvideox"):
                # Video generation – similar
                resp = client.videos.generate(model=model, prompt=prompt)
                zhipu_task_id = resp.id
            elif model == "glm-realtime":
                # Audio generation – similar
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
                    # Retrieve task status – exact method depends on model
                    # For images: client.files.retrieve(zhipu_id) etc.
                    # This is a placeholder; adjust to actual SDK.
                    status_data = client.files.retrieve(zhipu_id)  # Example
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
            time.sleep(5)  # poll every 5 seconds

    def _start_poller(self):
        thread = threading.Thread(target=self._poll_tasks, daemon=True)
        thread.start()

# Global queue instance (shared)
_task_queue = ZhipuTaskQueue(db_path=str(Path(__file__).parent.parent / "yukti_tasks.db"))

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
        self.client = get_zhipu_client()

    def invoke(self, prompt: str, **kwargs) -> Any:
        """Invoke the model. For sync models returns answer string; for async returns local task_id."""
        if self.config["type"] == "sync":
            # Sync chat with retry
            max_retries = 3
            for attempt in range(max_retries):
                try:
                    response = self.client.chat.completions.create(
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
                    time.sleep(2 ** attempt)  # exponential backoff
        else:
            # Async – submit to queue
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

def get_task_status(task_id: str) -> Optional[Dict[str, Any]]:
    """Retrieve status of an async task."""
    return _task_queue.get_task(task_id)

def get_active_tasks() -> List[Tuple[str, str, str, int]]:
    """Return list of active tasks (task_id, variant, status, progress)."""
    return _task_queue.get_active_tasks()
