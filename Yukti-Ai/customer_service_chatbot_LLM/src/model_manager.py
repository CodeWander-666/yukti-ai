"""
Yukti AI – Model Manager with robust path handling.
Attempts to import Zhipu, falls back to Gemini if unavailable.
"""

import logging
import sys
from pathlib import Path

logger = logging.getLogger(__name__)

# Add project root to path (if not already)
project_root = Path(__file__).parent.parent.parent.parent  # yukti-ai/
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

try:
    from LLM.zhipu import ZhipuSyncClient, TaskQueue, MODELS, get_model_config
    logger.info("✅ Zhipu models loaded successfully.")
    _zhipu_available = True
except ImportError as e:
    logger.warning(f"Zhipu import failed: {e}. Falling back to Gemini.")
    from langchain_google_genai import ChatGoogleGenerativeAI
    MODELS = {
        "Gemini 3 Flash": {
            "model": "gemini-3-flash",
            "type": "sync",
            "description": "Google Gemini fallback (fast & free tier)",
        }
    }
    def get_model_config(key):
        return MODELS.get(key)
    # Dummy placeholders for async tasks
    class ZhipuSyncClient:
        def chat(self, model, prompt, **kwargs):
            llm = ChatGoogleGenerativeAI(model=model, temperature=kwargs.get("temperature", 0.1))
            return llm.invoke(prompt)
    class TaskQueue:
        def __init__(self, db_path=None): pass
        def submit_async(self, *args, **kwargs):
            raise NotImplementedError("Async tasks require Zhipu.")
        def get_active_tasks(self):
            return []
    _zhipu_available = False

# Global instances
_task_queue = TaskQueue(db_path=Path(__file__).parent.parent / "yukti_tasks.db")
_sync_client = ZhipuSyncClient()

class YuktiModel:
    def __init__(self, model_key: str):
        self.config = get_model_config(model_key)
        if not self.config:
            raise ValueError(f"Unknown model key: {model_key}")
        self.model_key = model_key
        self.model_name = self.config["model"]

    def invoke(self, prompt: str, **kwargs):
        if self.config["type"] == "sync":
            return _sync_client.chat(
                model=self.model_name,
                prompt=prompt,
                temperature=kwargs.get("temperature", 0.1)
            )
        else:
            if not _zhipu_available:
                raise RuntimeError("Async tasks require Zhipu, which is not available.")
            return _task_queue.submit_async(
                variant=self.model_key,
                model=self.model_name,
                prompt=prompt
            )

def load_model(model_key: str):
    return YuktiModel(model_key)

def get_available_models():
    return list(MODELS.keys())
