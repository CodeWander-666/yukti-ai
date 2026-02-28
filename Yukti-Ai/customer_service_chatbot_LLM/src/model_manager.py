"""
Yukti AI – Model Manager (Zhipu Integration)
Loads models from LLM.zhipu package.
"""

import logging
import sys
from pathlib import Path

# Add LLM package to path (if not already)
LLM_PATH = Path(__file__).parent.parent.parent / "LLM"
if str(LLM_PATH) not in sys.path:
    sys.path.insert(0, str(LLM_PATH))

from zhipu import ZhipuSyncClient, TaskQueue, MODELS, get_model_config

logger = logging.getLogger(__name__)

# Global queue instance (shared across sessions)
_task_queue = TaskQueue(db_path=Path(__file__).parent.parent / "yukti_tasks.db")
_sync_client = ZhipuSyncClient()

class YuktiModel:
    """Wrapper for a model – can be invoked (sync) or submit async tasks."""
    def __init__(self, model_key: str):
        self.config = get_model_config(model_key)
        if not self.config:
            raise ValueError(f"Unknown model key: {model_key}")
        self.model_key = model_key
        self.model_name = self.config["model"]

    def invoke(self, prompt: str, **kwargs):
        """For sync models: returns answer string. For async: returns task_id."""
        if self.config["type"] == "sync":
            return _sync_client.chat(
                model=self.model_name,
                prompt=prompt,
                temperature=kwargs.get("temperature", 0.1)
            )
        else:
            # async: submit to queue
            return _task_queue.submit_async(
                variant=self.model_key,
                model=self.model_name,
                prompt=prompt
            )

def load_model(model_key: str):
    """Return a callable object for the model."""
    return YuktiModel(model_key)

def get_available_models():
    return list(MODELS.keys())
