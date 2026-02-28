"""
Async task queue for Zhipu generation models (image, video, audio).
Manages concurrency limits, database persistence, and progress polling.
"""

import sqlite3
import time
import threading
import requests
import logging
from datetime import datetime
from typing import Dict, Any, Optional, List
import streamlit as st

logger = logging.getLogger(__name__)

ZHIPU_API_BASE = "https://open.bigmodel.cn/api/paas/v4"
POLL_INTERVAL = 5  # seconds

class TaskQueue:
    def __init__(self, db_path="yukti_tasks.db"):
        self.db_path = db_path
        self.conn = sqlite3.connect(db_path, check_same_thread=False)
        self._init_db()
        self.lock = threading.Lock()
        self.zhipu_map = {}  # local_task_id -> zhipu_task_id
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

    def _get_api_key(self):
        key = st.secrets.get("ZHIPU_API_KEY")
        if not key:
            raise ValueError("ZHIPU_API_KEY not found in secrets.")
        return key

    def submit_async(self, variant: str, model: str, prompt: str) -> str:
        """
        Submit an async generation task.
        Returns a local task ID for tracking.
        """
        # Generate local task ID
        local_id = f"{variant}_{int(time.time())}_{abs(hash(prompt)) % 10000}"
        self.add_task(local_id, variant, model)

        # Submit to Zhipu async endpoint
        headers = {"Authorization": f"Bearer {self._get_api_key()}"}
        data = {"model": model, "prompt": prompt}
        try:
            resp = requests.post(f"{ZHIPU_API_BASE}/async/generate", json=data, headers=headers)
            resp.raise_for_status()
            zhipu_task_id = resp.json().get("id")
            if not zhipu_task_id:
                raise Exception("No task ID returned from Zhipu")
            with self.lock:
                self.zhipu_map[local_id] = zhipu_task_id
            self.update_task(local_id, status="pending")
        except Exception as e:
            logger.exception("Failed to submit async task")
            self.update_task(local_id, status="failed", error=str(e))
        return local_id

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
            "completed_at": row[8]
        }

    def get_active_tasks(self) -> List[tuple]:
        cursor = self.conn.execute(
            "SELECT task_id, variant, status, progress FROM tasks WHERE status IN ('submitted','pending','processing') ORDER BY created_at"
        )
        return cursor.fetchall()

    def _poll_tasks(self):
        """Background thread that polls all pending tasks."""
        while True:
            tasks = []
            with self.lock:
                # get all tasks with zhipu mapping
                tasks = list(self.zhipu_map.items())
            for local_id, zhipu_id in tasks:
                try:
                    headers = {"Authorization": f"Bearer {self._get_api_key()}"}
                    resp = requests.get(f"{ZHIPU_API_BASE}/async/result/{zhipu_id}", headers=headers)
                    if resp.status_code == 200:
                        data = resp.json()
                        status = data.get("status")
                        if status == "PROCESSING":
                            self.update_task(local_id, status="processing", progress=data.get("progress", 50))
                        elif status == "SUCCESS":
                            self.update_task(local_id, status="completed", progress=100,
                                            result_url=data.get("result_url", ""))
                            with self.lock:
                                del self.zhipu_map[local_id]
                        elif status == "FAILED":
                            self.update_task(local_id, status="failed", error=data.get("error", "Unknown"))
                            with self.lock:
                                del self.zhipu_map[local_id]
                except Exception as e:
                    logger.error(f"Error polling task {local_id}: {e}")
            time.sleep(POLL_INTERVAL)

    def _start_poller(self):
        thread = threading.Thread(target=self._poll_tasks, daemon=True)
        thread.start()
