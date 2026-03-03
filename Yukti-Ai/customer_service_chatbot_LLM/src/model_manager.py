"""
Core model orchestration – manages Zhipu (sync/async) and Gemini models
with concurrency control, per‑user task isolation, and language integration.
Includes comprehensive error handling and fallbacks for missing dependencies.
"""

import os
import time
import json
import logging
import threading
import sqlite3
import requests
import tempfile
import base64
import re
from datetime import datetime
from pathlib import Path
from typing import Dict, Any, Optional, List, Tuple, Union

# Import configuration
from config import (
    ZHIPU_API_KEY,
    GOOGLE_API_KEY,
    MODEL_CONCURRENCY,
    TASK_POLL_INTERVAL,
    DB_PATH,
)

# Import language utilities
try:
    from language_utils import get_language_name
except ImportError:
    # Fallback if language_utils not yet created
    def get_language_name(code: str) -> str:
        names = {'hi': 'Hindi', 'en': 'English', 'hinglish': 'Hinglish'}
        return names.get(code, code)

logger = logging.getLogger(__name__)

# ----------------------------------------------------------------------
# Constants and configuration
# ----------------------------------------------------------------------
ZHIPU_BASE_URL = "https://open.bigmodel.cn/api/paas/v4"

# Model Registry – each model has provider, type, concurrency limit, description
_MODEL_REGISTRY = {
    "glm-4-flash": {
        "model_id": "glm-4-flash",
        "provider": "zhipu",
        "type": "sync",
        "concurrency_limit": MODEL_CONCURRENCY.get("glm-4-flash", 200),
        "description": "GLM-4 Flash (fast, high concurrency)"
    },
    "glm-4-plus": {
        "model_id": "glm-4-plus",
        "provider": "zhipu",
        "type": "sync",
        "concurrency_limit": MODEL_CONCURRENCY.get("glm-4-plus", 20),
        "description": "GLM-4 Plus (balanced)"
    },
    "glm-5": {
        "model_id": "glm-5",
        "provider": "zhipu",
        "type": "sync",
        "concurrency_limit": MODEL_CONCURRENCY.get("glm-5", 3),
        "description": "GLM-5 (deep reasoning)"
    },
    "cogview-3-flash": {
        "model_id": "cogview-3-flash",
        "provider": "zhipu",
        "type": "sync",
        "concurrency_limit": MODEL_CONCURRENCY.get("cogview-3-flash", 10),
        "description": "CogView-3 Flash (fast image)"
    },
    "cogview-4": {
        "model_id": "cogview-4",
        "provider": "zhipu",
        "type": "sync",
        "concurrency_limit": MODEL_CONCURRENCY.get("cogview-4", 5),
        "description": "CogView-4 (high quality)"
    },
    "cogvideox-3": {
        "model_id": "cogvideox-3",
        "provider": "zhipu",
        "type": "async",
        "concurrency_limit": MODEL_CONCURRENCY.get("cogvideox-3", 5),
        "description": "CogVideoX-3"
    },
    "cogvideox-flash": {
        "model_id": "cogvideox-flash",
        "provider": "zhipu",
        "type": "async",
        "concurrency_limit": MODEL_CONCURRENCY.get("cogvideox-flash", 3),
        "description": "CogVideoX Flash"
    },
    "glm-4-voice": {
        "model_id": "glm-4-voice",
        "provider": "zhipu",
        "type": "sync",
        "concurrency_limit": MODEL_CONCURRENCY.get("glm-4-voice", 5),
        "description": "GLM-4 Voice"
    },
    "gemini": {
        "model_id": "gemini-dynamic",
        "provider": "gemini",
        "type": "sync",
        "concurrency_limit": MODEL_CONCURRENCY.get("gemini", 60),
        "description": "Google Gemini (dynamic selection)"
    }
}

# Service definitions – priority lists of model keys
SERVICES = {
    "Yukti‑Flash": ["glm-4-flash", "glm-4-plus"],
    "Yukti‑Quantum": ["glm-5", "glm-4-plus"],
    "Yukti‑Image": ["cogview-3-flash", "cogview-4"],
    "Yukti‑Video": ["cogvideox-3", "cogvideox-flash"],
    "Yukti‑Audio": ["glm-4-voice"],
    "Gemini": ["gemini"]
}

# Availability flags (derived from API keys)
ZHIPU_AVAILABLE = bool(ZHIPU_API_KEY)
GEMINI_AVAILABLE = bool(GOOGLE_API_KEY)  # SDK presence checked later

# ----------------------------------------------------------------------
# Concurrency Tracker (for sync models)
# ----------------------------------------------------------------------
class ConcurrencyTracker:
    """Thread‑safe counter for active requests per model."""
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
# Zhipu Client (with robust error handling)
# ----------------------------------------------------------------------
class ZhipuClient:
    def __init__(self, api_key: str):
        if not api_key:
            raise ValueError("Zhipu API key is required.")
        self.api_key = api_key
        self.session = requests.Session()
        self.session.headers.update({
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json"
        })

    def _request_with_retry(self, method: str, url: str, **kwargs) -> requests.Response:
        """Make HTTP request with exponential backoff and retry."""
        max_retries = 3
        for attempt in range(max_retries):
            try:
                resp = self.session.request(method, url, **kwargs)
                if resp.status_code == 429:
                    wait = (2 ** attempt) + (0.5 * attempt)
                    logger.warning(f"Rate limit hit, retrying in {wait:.2f}s")
                    time.sleep(wait)
                    continue
                resp.raise_for_status()
                return resp
            except requests.exceptions.RequestException as e:
                logger.warning(f"Request failed (attempt {attempt+1}): {e}")
                if attempt == max_retries - 1:
                    raise
                time.sleep(2 ** attempt)
        raise RuntimeError("Max retries exceeded")

    def text(self, model: str, prompt: str, temperature: float = 0.1, language: str = None) -> str:
        """Generate text with optional language hint."""
        try:
            from langchain_openai import ChatOpenAI
        except ImportError:
            raise RuntimeError("langchain-openai not installed. Cannot generate text.")

        # Determine target language
        if language is None:
            try:
                from language_detector import detect_language
                lang_info = detect_language(prompt)
                target_lang = lang_info['language']
            except ImportError:
                target_lang = 'en'
        else:
            target_lang = language

        # Prepare language‑specific prompt
        if target_lang == 'hinglish':
            enhanced_prompt = f"हिंग्लिश में जवाब दें। सवाल: {prompt}"
        elif target_lang != 'en':
            lang_name = get_language_name(target_lang)
            enhanced_prompt = f"Respond in {lang_name}. Question: {prompt}"
        else:
            enhanced_prompt = prompt

        try:
            llm = ChatOpenAI(
                model=model,
                api_key=self.api_key,
                base_url=ZHIPU_BASE_URL,
                temperature=temperature,
                max_retries=2,
            )
            response = llm.invoke(enhanced_prompt)
            content = response.content if hasattr(response, 'content') else str(response)
            if not content:
                logger.warning("Empty response, retrying with higher temperature")
                llm.temperature = 0.7
                response2 = llm.invoke(enhanced_prompt)
                content2 = response2.content if hasattr(response2, 'content') else str(response2)
                if content2:
                    return content2
                return "(Empty response from model)"
            return content
        except Exception as e:
            logger.exception("Text generation failed")
            raise RuntimeError(f"Text generation error: {e}") from e

    def image(self, model: str, prompt: str) -> str:
        """Generate image and return URL."""
        url = f"{ZHIPU_BASE_URL}/images/generations"
        payload = {"model": model, "prompt": prompt}
        try:
            resp = self._request_with_retry("POST", url, json=payload)
            data = resp.json()
            return data["data"][0]["url"]
        except Exception as e:
            logger.exception("Image generation failed")
            raise RuntimeError(f"Image generation error: {e}") from e

    def audio(self, model: str, prompt: str, voice: str = None, language: str = None) -> str:
        """Generate audio with optional language hint."""
        if language and language != 'en':
            lang_name = get_language_name(language)
            prompt = f"[Generate audio with speech in {lang_name}]\n{prompt}"

        url = f"{ZHIPU_BASE_URL}/chat/completions"
        messages = [{"role": "user", "content": [{"type": "text", "text": prompt}]}]
        payload = {"model": model, "messages": messages, "temperature": 0.1, "stream": False}
        try:
            resp = self._request_with_retry("POST", url, json=payload)
            data = resp.json()
            try:
                audio_b64 = data["choices"][0]["message"]["audio"]["data"]
            except (KeyError, IndexError, TypeError) as e:
                logger.error(f"Unexpected audio response: {data}")
                raise RuntimeError("Audio generation failed: unexpected response format") from e
            audio_bytes = base64.b64decode(audio_b64)
            fd, path = tempfile.mkstemp(suffix=".wav", prefix="yukti_audio_")
            with os.fdopen(fd, "wb") as f:
                f.write(audio_bytes)
            return path
        except Exception as e:
            logger.exception("Audio generation failed")
            raise RuntimeError(f"Audio generation error: {e}") from e

    def video_submit(self, model: str, prompt: str, language: str = None, **kwargs) -> str:
        """Submit a video generation task with optional language instruction."""
        try:
            from zai import ZhipuAiClient
        except ImportError:
            raise RuntimeError("Video generation requires zai-sdk.")

        if language and language != 'en':
            lang_name = get_language_name(language)
            prompt = f"[Generate video with narration in {lang_name}]\n{prompt}"

        try:
            client = ZhipuAiClient(api_key=self.api_key)
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
            return response.id
        except Exception as e:
            logger.exception("Video submission failed")
            raise RuntimeError(f"Video submission error: {e}") from e

    def video_poll(self, task_id: str) -> dict:
        """Poll video task status."""
        try:
            from zai import ZhipuAiClient
        except ImportError:
            raise RuntimeError("Video generation requires zai-sdk.")
        try:
            client = ZhipuAiClient(api_key=self.api_key)
            return client.videos.retrieve_videos_result(task_id)
        except Exception as e:
            logger.exception("Video polling failed")
            raise RuntimeError(f"Video polling error: {e}") from e

# ----------------------------------------------------------------------
# Gemini Client (with dynamic model refresh and quota handling)
# ----------------------------------------------------------------------
class GeminiClient:
    def __init__(self, api_key: str):
        if not api_key:
            raise ValueError("Google API key is required.")
        try:
            from google import genai
            from google.genai import types
            self.genai = genai
            self.types = types
        except ImportError:
            raise RuntimeError("google-genai not installed. Gemini models disabled.")

        self.client = self.genai.Client(api_key=api_key)
        self._available_models = None
        self._selected_model = None
        self._last_refresh = 0
        self._refresh_interval = 3600  # refresh model list every hour

    def _refresh_models(self):
        """Refresh available Gemini models list."""
        now = time.time()
        if self._available_models is None or (now - self._last_refresh) > self._refresh_interval:
            try:
                models = self.client.models.list()
                self._available_models = [
                    m.name for m in models
                    if 'generateContent' in (getattr(m, 'supported_actions', []) or [])
                ]
                self._last_refresh = now
                logger.info(f"Refreshed Gemini models: {self._available_models}")
            except Exception as e:
                logger.warning(f"Failed to refresh Gemini models: {e}")
                if self._available_models is None:
                    self._available_models = []

    def _select_model(self) -> str:
        """Select best available Gemini model."""
        self._refresh_models()
        preferred = [
            "gemini-2.0-flash-exp",
            "gemini-1.5-pro",
            "gemini-1.5-flash",
            "gemini-1.0-pro",
        ]
        available = self._available_models
        if not available:
            raise RuntimeError("No Gemini models available")
        for pref in preferred:
            for av in available:
                if pref in av or av.endswith(pref):
                    logger.info(f"Selected Gemini model: {av}")
                    return av
        logger.warning(f"None of {preferred} found, using first available: {available[0]}")
        return available[0]

    def text(self, prompt: str, temperature: float = 0.1, language: str = None) -> str:
        """Generate text with optional language hint."""
        if self._selected_model is None:
            self._selected_model = self._select_model()

        # Determine target language
        if language is None:
            try:
                from language_detector import detect_language
                lang_info = detect_language(prompt)
                target_lang = lang_info['language']
            except ImportError:
                target_lang = 'en'
        else:
            target_lang = language

        # Add system instruction for language if needed
        if target_lang == 'hinglish':
            system_msg = "Respond in Hinglish (mix of Hindi and English)."
        elif target_lang != 'en':
            lang_name = get_language_name(target_lang)
            system_msg = f"Respond in {lang_name}."
        else:
            system_msg = None

        if system_msg:
            full_prompt = f"{system_msg}\n\nUser: {prompt}"
        else:
            full_prompt = prompt

        for attempt in range(2):
            try:
                response = self.client.models.generate_content(
                    model=self._selected_model,
                    contents=full_prompt,
                    config=self.types.GenerateContentConfig(
                        temperature=temperature,
                        max_output_tokens=2048
                    )
                )
                return response.text
            except Exception as e:
                if "429" in str(e) or "RESOURCE_EXHAUSTED" in str(e):
                    delay_match = re.search(r'retry in (\d+(\.\d+)?)s', str(e), re.IGNORECASE)
                    delay = float(delay_match.group(1)) if delay_match else 60
                    logger.warning(f"Quota exceeded for {self._selected_model}. Retrying in {delay:.1f}s")
                    time.sleep(delay)
                    if attempt == 0:
                        continue
                    else:
                        self._selected_model = None
                        self._selected_model = self._select_model()
                        logger.info(f"Switching to fallback model: {self._selected_model}")
                        return self.text(prompt, temperature, language)
                logger.exception("Gemini invocation failed")
                raise RuntimeError(f"Gemini error: {e}") from e

        raise RuntimeError("Gemini service unavailable after retries")

# ----------------------------------------------------------------------
# Async Task Queue (with concurrency control, SQLite persistence, and user isolation)
# ----------------------------------------------------------------------
# Semaphores for async tasks per model
_async_semaphores = {}
_async_lock = threading.Lock()

def _get_async_semaphore(model_key: str) -> threading.Semaphore:
    with _async_lock:
        if model_key not in _async_semaphores:
            limit = _MODEL_REGISTRY[model_key].get("concurrency_limit", 5)
            _async_semaphores[model_key] = threading.Semaphore(limit)
        return _async_semaphores[model_key]

# Check if zai-sdk is available
try:
    from zai import ZhipuAiClient
    ZAI_AVAILABLE = True
except ImportError:
    ZAI_AVAILABLE = False
    logger.warning("zai-sdk not installed; video generation disabled.")

# Ensure tasks table has user_id column (migration)
def _ensure_tasks_schema():
    conn = sqlite3.connect(str(DB_PATH))
    c = conn.cursor()
    # Check if user_id column exists
    c.execute("PRAGMA table_info(tasks)")
    columns = [col[1] for col in c.fetchall()]
    if 'user_id' not in columns:
        logger.info("Adding user_id column to tasks table...")
        c.execute("ALTER TABLE tasks ADD COLUMN user_id INTEGER")
        c.execute("CREATE INDEX IF NOT EXISTS idx_tasks_user_id ON tasks(user_id)")
        conn.commit()
        logger.info("tasks table updated with user_id.")
    conn.close()

_ensure_tasks_schema()

if ZAI_AVAILABLE and ZHIPU_AVAILABLE:
    class ZhipuTaskQueue:
        def __init__(self, db_path: str = str(DB_PATH)):
            self.db_path = db_path
            self.conn = sqlite3.connect(db_path, check_same_thread=False)
            self.conn.row_factory = sqlite3.Row
            self.lock = threading.Lock()
            self.zhipu_map: Dict[str, str] = {}  # local_id -> zhipu_task_id
            self.user_map: Dict[str, int] = {}    # local_id -> user_id
            self._start_poller()

        def _init_db(self):
            """Create tasks table if not exists (already done)."""
            pass

        def add_task(self, task_id: str, variant: str, model: str, user_id: int):
            """Insert a new task record."""
            with self.lock:
                self.conn.execute(
                    "INSERT INTO tasks (task_id, variant, model, status, progress, user_id, created_at) VALUES (?,?,?,?,?,?,?)",
                    (task_id, variant, model, "submitted", 0, user_id, datetime.now())
                )
                self.conn.commit()

        def update_task(self, task_id: str, **kwargs):
            """Update task fields."""
            with self.lock:
                fields = ", ".join([f"{k}=?" for k in kwargs])
                values = list(kwargs.values()) + [task_id]
                self.conn.execute(f"UPDATE tasks SET {fields} WHERE task_id=?", values)
                self.conn.commit()

        def get_task(self, task_id: str) -> Optional[Dict[str, Any]]:
            """Retrieve a task by ID."""
            cursor = self.conn.execute("SELECT * FROM tasks WHERE task_id=?", (task_id,))
            row = cursor.fetchone()
            if not row:
                return None
            return dict(row)

        def get_active_tasks(self, user_id: Optional[int] = None) -> List[Tuple[str, str, str, int]]:
            """
            Return list of active (not completed/failed) tasks.
            If user_id is given, return only tasks for that user.
            """
            if user_id is None:
                cursor = self.conn.execute(
                    "SELECT task_id, variant, status, progress FROM tasks WHERE status IN ('submitted','pending','processing') ORDER BY created_at"
                )
            else:
                cursor = self.conn.execute(
                    "SELECT task_id, variant, status, progress FROM tasks WHERE user_id=? AND status IN ('submitted','pending','processing') ORDER BY created_at",
                    (user_id,)
                )
            return cursor.fetchall()

        def submit_async(self, variant: str, model: str, prompt: str, user_id: int, language: str = None, **kwargs) -> str:
            """Submit an async task, respecting concurrency limits."""
            sem = _get_async_semaphore(model)
            acquired = sem.acquire(blocking=True, timeout=60)
            if not acquired:
                raise RuntimeError(f"Concurrency limit for {model} reached; try again later.")

            local_id = f"{variant}_{int(time.time())}_{abs(hash(prompt)) % 10000}"
            self.add_task(local_id, variant, model, user_id)
            try:
                client = ZhipuClient(ZHIPU_API_KEY)
                zhipu_task_id = client.video_submit(model, prompt, language=language, **kwargs)
                with self.lock:
                    self.zhipu_map[local_id] = zhipu_task_id
                    self.user_map[local_id] = user_id
                self.update_task(local_id, status="pending")
                logger.info(f"Submitted video task {local_id} (Zhipu ID: {zhipu_task_id}) for user {user_id}")
            except Exception as e:
                logger.exception("Failed to submit async task")
                self.update_task(local_id, status="failed", error=str(e))
            finally:
                sem.release()
            return local_id

        def _poll_tasks(self):
            """Background thread that polls all pending tasks."""
            while True:
                tasks = []
                with self.lock:
                    tasks = list(self.zhipu_map.items())
                for local_id, zhipu_id in tasks:
                    try:
                        client = ZhipuClient(ZHIPU_API_KEY)
                        status_data = client.video_poll(zhipu_id)
                        if hasattr(status_data, 'status') and status_data.status == "succeeded":
                            result_url = status_data.video_url
                            self.update_task(local_id, status="completed", progress=100,
                                             result_url=result_url, completed_at=datetime.now())
                            with self.lock:
                                del self.zhipu_map[local_id]
                                del self.user_map[local_id]
                            logger.info(f"Task {local_id} completed: {result_url}")
                        elif hasattr(status_data, 'status') and status_data.status == "failed":
                            error = status_data.error if hasattr(status_data, 'error') else "Unknown error"
                            self.update_task(local_id, status="failed", error=error, completed_at=datetime.now())
                            with self.lock:
                                del self.zhipu_map[local_id]
                                del self.user_map[local_id]
                            logger.error(f"Task {local_id} failed: {error}")
                        elif hasattr(status_data, 'status') and status_data.status == "processing":
                            progress = getattr(status_data, 'progress', 50)
                            self.update_task(local_id, status="processing", progress=progress)
                        else:
                            # Unknown status, keep as pending
                            logger.debug(f"Task {local_id} status: {getattr(status_data, 'status', 'unknown')}")
                    except Exception as e:
                        logger.error(f"Polling task {local_id} failed: {e}")
                time.sleep(TASK_POLL_INTERVAL)

        def _start_poller(self):
            thread = threading.Thread(target=self._poll_tasks, daemon=True)
            thread.start()

    _task_queue = ZhipuTaskQueue()
else:
    class _DummyQueue:
        def submit_async(self, *args, **kwargs):
            raise RuntimeError("Video generation requires zai-sdk and Zhipu API key.")
        def get_active_tasks(self, user_id=None):
            return []
        def get_task(self, task_id):
            return None
    _task_queue = _DummyQueue()

def get_active_tasks(user_id: Optional[int] = None) -> List[Tuple[str, str, str, int]]:
    """Return list of active async tasks, optionally filtered by user_id."""
    return _task_queue.get_active_tasks(user_id)

def get_task_status(task_id: str) -> Optional[Dict[str, Any]]:
    """Return status of a specific task."""
    return _task_queue.get_task(task_id)

# ----------------------------------------------------------------------
# Public API: YuktiModel and helper functions
# ----------------------------------------------------------------------
MODELS = {}
for service, models in SERVICES.items():
    if not models:
        continue
    first_model = models[0]
    desc = _MODEL_REGISTRY[first_model]["description"]
    MODELS[service] = {
        "description": desc,
        "models": models
    }

class YuktiModel:
    """Wrapper for a service that can invoke its models with fallback and concurrency control."""
    def __init__(self, service: str):
        self.service = service
        if service not in SERVICES:
            raise ValueError(f"Unknown service: {service}")
        first_model = SERVICES[service][0]
        self.provider = _MODEL_REGISTRY[first_model]["provider"]
        self.zhipu_api_key = ZHIPU_API_KEY
        self.gemini_api_key = GOOGLE_API_KEY

    def invoke(self, prompt: str, **kwargs) -> Any:
        """
        Invoke the service with fallback over its model list.
        Extra kwargs (e.g., language, user_id) are passed to the underlying model.
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
                    result = self._call_gemini_model(prompt, **kwargs)
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
        if not self.zhipu_api_key:
            raise RuntimeError("Zhipu API key not configured.")
        client = ZhipuClient(self.zhipu_api_key)
        model_id = _MODEL_REGISTRY[model_key]["model_id"]
        if model_id in ["glm-4-flash", "glm-4-plus", "glm-5"]:
            return client.text(model_id, prompt, temperature=kwargs.get("temperature", 0.1),
                               language=kwargs.get("language"))
        elif model_id in ["cogview-3-flash", "cogview-4"]:
            return client.image(model_id, prompt)
        elif model_id in ["cogvideox-3", "cogvideox-flash"]:
            # IMPORTANT: pop 'language' and 'user_id' to avoid duplicate argument
            lang = kwargs.pop('language', None)
            user_id = kwargs.pop('user_id', None)
            if user_id is None:
                raise RuntimeError("user_id is required for video tasks")
            return _task_queue.submit_async(
                variant=self.service,
                model=model_id,
                prompt=prompt,
                user_id=user_id,
                language=lang,
                **kwargs
            )
        elif model_id == "glm-4-voice":
            return client.audio(model_id, prompt, voice=kwargs.get("voice"),
                                language=kwargs.get("language"))
        else:
            raise ValueError(f"Unsupported model: {model_id}")

    def _call_gemini_model(self, prompt: str, **kwargs) -> str:
        if not self.gemini_api_key:
            raise RuntimeError("Gemini not available")
        client = GeminiClient(self.gemini_api_key)
        return client.text(prompt, temperature=kwargs.get("temperature", 0.1),
                           language=kwargs.get("language"))

def load_model(service: str) -> YuktiModel:
    """Return a YuktiModel instance for the given service."""
    return YuktiModel(service)

def get_available_models() -> List[str]:
    """Return list of service names that are currently available (based on API keys)."""
    available = []
    for service in SERVICES:
        if service in ["Yukti‑Flash", "Yukti‑Quantum", "Yukti‑Image", "Yukti‑Video", "Yukti‑Audio"]:
            if ZHIPU_AVAILABLE:
                available.append(service)
        elif service == "Gemini":
            if GEMINI_AVAILABLE:
                # Check if google-genai is installed
                try:
                    import google.genai
                    available.append(service)
                except ImportError:
                    pass
    return available

def get_service_config(service: str) -> Optional[Dict[str, Any]]:
    """Return configuration for a service."""
    if service in SERVICES:
        return {"models": SERVICES[service]}
    return None

def get_model_config(service: str) -> Optional[Dict[str, Any]]:
    """Return configuration for the primary model of a service."""
    if service not in SERVICES:
        return None
    first_model = SERVICES[service][0]
    config = _MODEL_REGISTRY[first_model].copy()
    config['service'] = service
    config['model_key'] = first_model
    return config

# ----------------------------------------------------------------------
# Exported symbols
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
