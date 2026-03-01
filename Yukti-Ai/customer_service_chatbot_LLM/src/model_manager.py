import logging
import os
import time
import threading
import sqlite3
import requests
import tempfile
import base64
import re
from datetime import datetime
from pathlib import Path
from typing import Dict, Any, Optional, List, Tuple

import streamlit as st
from langchain_openai import ChatOpenAI

try:
    from zai import ZhipuAiClient
    ZAI_AVAILABLE = True
except ImportError:
    ZAI_AVAILABLE = False
    logging.warning("zai-sdk not installed; video generation disabled.")

try:
    from google import genai
    from google.genai import types
    GEMINI_SDK_AVAILABLE = True
except ImportError:
    GEMINI_SDK_AVAILABLE = False
    logging.warning("google-genai not installed; Gemini models disabled.")

# NEW: Import language detector (optional – for auto-detection fallback)
from language_detector import detect_language

logger = logging.getLogger(__name__)

ZHIPU_BASE_URL = "https://open.bigmodel.cn/api/paas/v4"

_MODEL_REGISTRY = {
    "glm-4-flash": {
        "model_id": "glm-4-flash",
        "provider": "zhipu",
        "type": "sync",
        "concurrency_limit": 200,
        "description": "GLM-4 Flash (fast, high concurrency)"
    },
    "glm-4-plus": {
        "model_id": "glm-4-plus",
        "provider": "zhipu",
        "type": "sync",
        "concurrency_limit": 20,
        "description": "GLM-4 Plus (balanced)"
    },
    "glm-5": {
        "model_id": "glm-5",
        "provider": "zhipu",
        "type": "sync",
        "concurrency_limit": 3,
        "description": "GLM-5 (deep reasoning)"
    },
    "cogview-3-flash": {
        "model_id": "cogview-3-flash",
        "provider": "zhipu",
        "type": "sync",
        "concurrency_limit": 10,
        "description": "CogView-3 Flash (fast image)"
    },
    "cogview-4": {
        "model_id": "cogview-4",
        "provider": "zhipu",
        "type": "sync",
        "concurrency_limit": 5,
        "description": "CogView-4 (high quality)"
    },
    "cogvideox-3": {
        "model_id": "cogvideox-3",
        "provider": "zhipu",
        "type": "async",
        "concurrency_limit": 5,
        "description": "CogVideoX-3"
    },
    "cogvideox-flash": {
        "model_id": "cogvideox-flash",
        "provider": "zhipu",
        "type": "async",
        "concurrency_limit": 3,
        "description": "CogVideoX Flash"
    },
    "glm-4-voice": {
        "model_id": "glm-4-voice",
        "provider": "zhipu",
        "type": "sync",
        "concurrency_limit": 5,
        "description": "GLM-4 Voice"
    },
    "gemini": {
        "model_id": "gemini-dynamic",
        "provider": "gemini",
        "type": "sync",
        "concurrency_limit": 60,
        "description": "Google Gemini (dynamic selection)"
    }
}

SERVICES = {
    "Yukti‑Flash": ["glm-4-flash", "glm-4-plus"],
    "Yukti‑Quantum": ["glm-5", "glm-4-plus"],
    "Yukti‑Image": ["cogview-3-flash", "cogview-4"],
    "Yukti‑Video": ["cogvideox-3", "cogvideox-flash"],
    "Yukti‑Audio": ["glm-4-voice"],
    "Gemini": ["gemini"]
}

def get_zhipu_api_key() -> Optional[str]:
    return st.secrets.get("ZHIPU_API_KEY") or os.getenv("ZHIPU_API_KEY")

def get_google_api_key() -> Optional[str]:
    return st.secrets.get("GOOGLE_API_KEY") or os.getenv("GOOGLE_API_KEY")

ZHIPU_AVAILABLE = get_zhipu_api_key() is not None
GEMINI_AVAILABLE = GEMINI_SDK_AVAILABLE and get_google_api_key() is not None

class ConcurrencyTracker:
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

class ZhipuClient:
    def __init__(self, api_key: str):
        self.api_key = api_key
        self.session = requests.Session()
        self.session.headers.update({
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json"
        })

    def _request_with_retry(self, method: str, url: str, **kwargs) -> requests.Response:
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
                if attempt == max_retries - 1:
                    raise
                logger.warning(f"Request failed (attempt {attempt+1}): {e}, retrying...")
                time.sleep(2 ** attempt)
        raise RuntimeError("Max retries exceeded")

    def text(self, model: str, prompt: str, temperature: float = 0.1, language: str = None) -> str:
        """
        Generate text with optional language hint.
        If language is not provided, auto‑detect from prompt.
        """
        # Determine target language
        if language is None:
            lang_info = detect_language(prompt)
            target_lang = lang_info['language']
        else:
            target_lang = language

        # Prepare language‑specific prompt
        if target_lang == 'hinglish':
            enhanced_prompt = f"हिंग्लिश में जवाब दें। सवाल: {prompt}"
        elif target_lang != 'en':
            lang_name = get_language_name(target_lang)  # we'll define a simple map
            enhanced_prompt = f"Respond in {lang_name}. Question: {prompt}"
        else:
            enhanced_prompt = prompt

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

    def image(self, model: str, prompt: str) -> str:
        url = f"{ZHIPU_BASE_URL}/images/generations"
        payload = {"model": model, "prompt": prompt}
        resp = self._request_with_retry("POST", url, json=payload)
        return resp.json()["data"][0]["url"]

    def audio(self, model: str, prompt: str, voice: str = None) -> str:
        url = f"{ZHIPU_BASE_URL}/chat/completions"
        messages = [{"role": "user", "content": [{"type": "text", "text": prompt}]}]
        payload = {"model": model, "messages": messages, "temperature": 0.1, "stream": False}
        resp = self._request_with_retry("POST", url, json=payload)
        data = resp.json()
        try:
            audio_b64 = data["choices"][0]["message"]["audio"]["data"]
            audio_bytes = base64.b64decode(audio_b64)
        except (KeyError, IndexError) as e:
            logger.error(f"Unexpected response format: {data}")
            raise RuntimeError("Audio generation failed: unexpected response format") from e
        fd, path = tempfile.mkstemp(suffix=".wav", prefix="yukti_audio_")
        with os.fdopen(fd, "wb") as f:
            f.write(audio_bytes)
        return path

    def video_submit(self, model: str, prompt: str, **kwargs) -> str:
        if not ZAI_AVAILABLE:
            raise RuntimeError("Video generation requires zai-sdk.")
        from zai import ZhipuAiClient
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

    def video_poll(self, task_id: str) -> dict:
        from zai import ZhipuAiClient
        client = ZhipuAiClient(api_key=self.api_key)
        return client.videos.retrieve_videos_result(task_id)

class GeminiClient:
    def __init__(self, api_key: str):
        self.client = genai.Client(api_key=api_key)
        self._available_models = None
        self._selected_model = None

    def _get_available_models(self) -> List[str]:
        if self._available_models is None:
            try:
                models = self.client.models.list()
                self._available_models = [
                    m.name for m in models 
                    if 'generateContent' in (getattr(m, 'supported_actions', []) or [])
                ]
                logger.info(f"Available Gemini models: {self._available_models}")
            except Exception as e:
                logger.exception("Failed to list Gemini models")
                self._available_models = []
        return self._available_models

    def _select_model(self) -> str:
        preferred = [
            "gemini-3.1-pro-preview",
            "gemini-3-flash-preview",
            "gemini-2.5-flash",
            "gemini-1.5-pro",
            "gemini-1.5-flash",
            "gemini-1.0-pro",
        ]
        available = self._get_available_models()
        if not available:
            raise RuntimeError("No Gemini models available")
        for pref in preferred:
            for av in available:
                if av.endswith(pref) or av == pref or av == f"models/{pref}":
                    logger.info(f"Selected Gemini model: {av}")
                    return av
        logger.warning(f"None of {preferred} found, using first available: {available[0]}")
        return available[0]

    def text(self, prompt: str, temperature: float = 0.1, language: str = None) -> str:
        """
        Generate text with optional language hint.
        Gemini natively supports many languages; we add a system instruction if needed.
        """
        if self._selected_model is None:
            self._selected_model = self._select_model()

        # Determine target language (if not provided, auto‑detect)
        if language is None:
            lang_info = detect_language(prompt)
            target_lang = lang_info['language']
        else:
            target_lang = language

        # Add language instruction for non‑English (Gemini is good, but explicit helps)
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
                    config=types.GenerateContentConfig(
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
                raise RuntimeError(f"Gemini error: {e}")

        raise RuntimeError("Gemini service unavailable after retries")

# ----------------------------------------------------------------------
# Async Task Queue (unchanged)
# ----------------------------------------------------------------------
if ZAI_AVAILABLE and ZHIPU_AVAILABLE:
    class ZhipuTaskQueue:
        def __init__(self, db_path: str = "yukti_tasks.db"):
            self.db_path = db_path
            self.conn = sqlite3.connect(db_path, check_same_thread=False)
            self._init_db()
            self.lock = threading.Lock()
            self.zhipu_map: Dict[str, str] = {}
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
            local_id = f"{variant}_{int(time.time())}_{abs(hash(prompt)) % 10000}"
            self.add_task(local_id, variant, model)
            try:
                api_key = get_zhipu_api_key()
                if not api_key:
                    raise RuntimeError("Zhipu API key not configured.")
                client = ZhipuClient(api_key)
                zhipu_task_id = client.video_submit(model, prompt, **kwargs)
                with self.lock:
                    self.zhipu_map[local_id] = zhipu_task_id
                self.update_task(local_id, status="pending")
            except Exception as e:
                logger.exception("Failed to submit async task")
                self.update_task(local_id, status="failed", error=str(e))
            return local_id

        def _poll_tasks(self):
            while True:
                tasks = []
                with self.lock:
                    tasks = list(self.zhipu_map.items())
                for local_id, zhipu_id in tasks:
                    try:
                        api_key = get_zhipu_api_key()
                        client = ZhipuClient(api_key)
                        status_data = client.video_poll(zhipu_id)
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

def get_active_tasks() -> List[Tuple[str, str, str, int]]:
    return _task_queue.get_active_tasks()

def get_task_status(task_id: str) -> Optional[Dict[str, Any]]:
    return _task_queue.get_task(task_id)

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
    def __init__(self, service: str):
        self.service = service
        if service not in SERVICES:
            raise ValueError(f"Unknown service: {service}")
        first_model = SERVICES[service][0]
        self.provider = _MODEL_REGISTRY[first_model]["provider"]
        self.zhipu_api_key = get_zhipu_api_key()
        self.gemini_api_key = get_google_api_key()

    def invoke(self, prompt: str, **kwargs) -> Any:
        """
        Passes any extra kwargs (e.g., language) to the underlying model.
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
        client = ZhipuClient(self.zhipu_api_key)
        model_id = _MODEL_REGISTRY[model_key]["model_id"]
        if model_id in ["glm-4-flash", "glm-4-plus", "glm-5"]:
            # Pass language from kwargs if present
            return client.text(model_id, prompt, temperature=kwargs.get("temperature", 0.1),
                               language=kwargs.get("language"))
        elif model_id in ["cogview-3-flash", "cogview-4"]:
            return client.image(model_id, prompt)
        elif model_id in ["cogvideox-3", "cogvideox-flash"]:
            return _task_queue.submit_async(
                variant=self.service,
                model=model_id,
                prompt=prompt,
                **kwargs
            )
        elif model_id == "glm-4-voice":
            return client.audio(model_id, prompt, voice=kwargs.get("voice"))
        else:
            raise ValueError(f"Unsupported model: {model_id}")

    def _call_gemini_model(self, prompt: str, **kwargs) -> str:
        if not GEMINI_AVAILABLE:
            raise RuntimeError("Gemini not available")
        client = GeminiClient(self.gemini_api_key)
        return client.text(prompt, temperature=kwargs.get("temperature", 0.1),
                           language=kwargs.get("language"))

def load_model(service: str) -> YuktiModel:
    return YuktiModel(service)

def get_available_models() -> List[str]:
    available = []
    for service in SERVICES:
        if service in ["Yukti‑Flash", "Yukti‑Quantum", "Yukti‑Image", "Yukti‑Video", "Yukti‑Audio"]:
            if ZHIPU_AVAILABLE:
                available.append(service)
        elif service == "Gemini":
            if GEMINI_AVAILABLE:
                available.append(service)
    return available

def get_service_config(service: str) -> Optional[Dict[str, Any]]:
    if service in SERVICES:
        return {"models": SERVICES[service]}
    return None

def get_model_config(service: str) -> Optional[Dict[str, Any]]:
    if service not in SERVICES:
        return None
    first_model = SERVICES[service][0]
    config = _MODEL_REGISTRY[first_model].copy()
    config['service'] = service
    config['model_key'] = first_model
    return config

# Simple helper to map language codes to names
def get_language_name(code: str) -> str:
    names = {
        'hi': 'Hindi',
        'en': 'English',
        'ur': 'Urdu',
        'bn': 'Bengali',
        'te': 'Telugu',
        'ta': 'Tamil',
        'mr': 'Marathi',
        'gu': 'Gujarati',
        'kn': 'Kannada',
        'ml': 'Malayalam',
        'pa': 'Punjabi',
        'or': 'Odia',
        'as': 'Assamese',
        'mai': 'Maithili',
        'sat': 'Santali',
        'ks': 'Kashmiri',
        'sd': 'Sindhi',
        'ne': 'Nepali',
        'doi': 'Dogri',
        'mni': 'Manipuri',
        'bodo': 'Bodo',
    }
    return names.get(code, code)

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
