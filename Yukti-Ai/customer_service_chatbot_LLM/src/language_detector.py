"""
Advanced language detection with multiple methods (FastText, script, Hinglish wordlist, transformer).
Handles explicit instructions, code‑mixed text, and 100+ languages.
Includes graceful fallbacks and comprehensive error handling.
"""

import os
import re
import json
import logging
import urllib.request
from pathlib import Path
from typing import Dict, Any, Optional, Tuple

# Optional imports – handle gracefully if missing
try:
    import fasttext
    FASTTEXT_AVAILABLE = True
except ImportError:
    FASTTEXT_AVAILABLE = False

try:
    from transformers import pipeline
    TRANSFORMERS_AVAILABLE = True
except ImportError:
    TRANSFORMERS_AVAILABLE = False

from config import ENABLE_TRANSFORMER
from language_utils import normalize_language_code

logger = logging.getLogger(__name__)

# ----------------------------------------------------------------------
# Load Hinglish wordlist from external JSON file (if exists)
# ----------------------------------------------------------------------
WORDLIST_PATH = Path(__file__).parent / "hinglish_words.json"
HINGLISH_HINDI_WORDS = set()
if WORDLIST_PATH.exists():
    try:
        with open(WORDLIST_PATH, 'r', encoding='utf-8') as f:
            HINGLISH_HINDI_WORDS = set(json.load(f))
        logger.info(f"Loaded {len(HINGLISH_HINDI_WORDS)} Hinglish words.")
    except Exception as e:
        logger.warning(f"Failed to load hinglish_words.json: {e}")
else:
    logger.warning("hinglish_words.json not found; Hinglish detection disabled.")

# ----------------------------------------------------------------------
# FastText model loading (if available)
# ----------------------------------------------------------------------
FASTTEXT_MODEL_PATH = Path.home() / ".cache" / "yukti" / "lid.176.bin"

def download_fasttext_model():
    """Download FastText language identification model (176 languages)."""
    if FASTTEXT_MODEL_PATH.exists():
        return
    FASTTEXT_MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
    url = "https://dl.fbaipublicfiles.com/fasttext/supervised-models/lid.176.bin"
    logger.info("Downloading FastText language model (176 languages)...")
    try:
        urllib.request.urlretrieve(url, FASTTEXT_MODEL_PATH)
        logger.info("Download complete.")
    except Exception as e:
        logger.error(f"Failed to download FastText model: {e}")
        raise

_fasttext_model = None
if FASTTEXT_AVAILABLE:
    try:
        if not FASTTEXT_MODEL_PATH.exists():
            download_fasttext_model()
        _fasttext_model = fasttext.load_model(str(FASTTEXT_MODEL_PATH))
        logger.info("FastText language model loaded (176 languages).")
    except Exception as e:
        logger.warning(f"Failed to load FastText model: {e}")
        _fasttext_model = None

# ----------------------------------------------------------------------
# Unicode script ranges for Indian languages
# ----------------------------------------------------------------------
SCRIPTS = {
    'devanagari': (0x0900, 0x097F, ['hi', 'mr', 'ne', 'mai', 'sat', 'bodo', 'doi']),
    'bengali': (0x0980, 0x09FF, ['bn', 'as']),
    'gurmukhi': (0x0A00, 0x0A7F, ['pa']),
    'gujarati': (0x0A80, 0x0AFF, ['gu']),
    'odia': (0x0B00, 0x0B7F, ['or']),
    'tamil': (0x0B80, 0x0BFF, ['ta']),
    'telugu': (0x0C00, 0x0C7F, ['te']),
    'kannada': (0x0C80, 0x0CFF, ['kn']),
    'malayalam': (0x0D00, 0x0D7F, ['ml']),
    'sinhala': (0x0D80, 0x0DFF, ['si']),
    'arabic': (0x0600, 0x06FF, ['ur', 'ar', 'fa']),
    'thai': (0x0E00, 0x0E7F, ['th']),
    'lao': (0x0E80, 0x0EFF, ['lo']),
    'tibetan': (0x0F00, 0x0FFF, ['bo']),
    'myanmar': (0x1000, 0x109F, ['my']),
    'georgian': (0x10A0, 0x10FF, ['ka']),
    'hangul': (0xAC00, 0xD7AF, ['ko']),
    'hiragana': (0x3040, 0x309F, ['ja']),
    'katakana': (0x30A0, 0x30FF, ['ja']),
    'han': (0x4E00, 0x9FFF, ['zh', 'ja']),
    'cyrillic': (0x0400, 0x04FF, ['ru', 'uk', 'bg', 'sr']),
    'greek': (0x0370, 0x03FF, ['el']),
}

def detect_script(text: str) -> Optional[str]:
    """Identify the dominant script in the text."""
    if not text:
        return None
    counts = {script: 0 for script in SCRIPTS}
    for char in text:
        code = ord(char)
        for script, (start, end, _) in SCRIPTS.items():
            if start <= code <= end:
                counts[script] += 1
                break
    max_script = max(counts, key=counts.get)
    if counts[max_script] > 0:
        return max_script
    return None

def script_to_languages(script: str) -> list:
    """Return list of language codes for a given script."""
    return SCRIPTS.get(script, (None, None, []))[2]

# ----------------------------------------------------------------------
# FastText detection wrapper
# ----------------------------------------------------------------------
def fasttext_detect(text: str) -> Optional[Dict[str, Any]]:
    """Use FastText model for language identification."""
    if not _fasttext_model or not text.strip():
        return None
    try:
        # FastText expects text with newlines; predict returns list of (lang, prob)
        pred = _fasttext_model.predict(text.replace('\n', ' '), k=1)
        lang = pred[0][0].replace('__label__', '')
        prob = float(pred[1][0])
        return {'code': lang, 'confidence': prob, 'method': 'fasttext'}
    except Exception as e:
        logger.debug(f"FastText detection failed: {e}")
        return None

# ----------------------------------------------------------------------
# Transformer detector (optional, controlled by config)
# ----------------------------------------------------------------------
class TransformerDetector:
    def __init__(self):
        self._lang_detector = None
        if TRANSFORMERS_AVAILABLE and ENABLE_TRANSFORMER:
            try:
                self._lang_detector = pipeline(
                    "text-classification",
                    model="papluca/xlm-roberta-base-language-detection"
                )
                logger.info("Transformer language detector loaded")
            except Exception as e:
                logger.warning(f"Failed to load transformer: {e}")

    def detect(self, text: str) -> Optional[Dict[str, Any]]:
        if not self._lang_detector or not text.strip():
            return None
        try:
            result = self._lang_detector(text[:512])[0]
            return {
                'code': result['label'],
                'confidence': result['score'],
                'method': 'transformer'
            }
        except Exception as e:
            logger.debug(f"Transformer detection failed: {e}")
            return None

_transformer = TransformerDetector()

# ----------------------------------------------------------------------
# Hinglish detection using wordlist
# ----------------------------------------------------------------------
def is_hinglish(text: str) -> Tuple[bool, float]:
    """
    Detect if text is Hinglish (Roman script Hindi + English mix).
    Returns (is_hinglish, confidence)
    """
    if not HINGLISH_HINDI_WORDS:
        return False, 0.0
    # If it has Devanagari, it's pure Hindi, not Hinglish
    if any('\u0900' <= c <= '\u097F' for c in text):
        return False, 0.0

    words = re.findall(r'\b[a-zA-Z]+\b', text.lower())
    if not words:
        return False, 0.0

    hindi_word_count = sum(1 for w in words if w in HINGLISH_HINDI_WORDS)
    total_words = len(words)
    if total_words == 0:
        return False, 0.0

    hindi_ratio = hindi_word_count / total_words

    # Hinglish typically has 15-85% Hindi words mixed with English
    if 0.15 < hindi_ratio < 0.85:
        return True, hindi_ratio
    elif hindi_ratio >= 0.85:
        # Could be transliterated Hindi (all words Hindi but Roman script)
        return True, 0.7
    else:
        return False, hindi_ratio

# ----------------------------------------------------------------------
# Explicit language instruction detection
# ----------------------------------------------------------------------
def detect_explicit_language_instruction(text: str) -> Optional[str]:
    """
    Detect if user explicitly requests a language.
    Supports many language names.
    """
    text_lower = text.lower()

    # Language name mapping (extend as needed)
    lang_names = {
        'hindi': 'hi', 'hind': 'hi',
        'hinglish': 'hinglish',
        'english': 'en', 'eng': 'en',
        'urdu': 'ur',
        'bengali': 'bn', 'bangla': 'bn',
        'telugu': 'te',
        'tamil': 'ta',
        'marathi': 'mr',
        'gujarati': 'gu',
        'kannada': 'kn',
        'malayalam': 'ml',
        'punjabi': 'pa',
        'odia': 'or', 'oriya': 'or',
        'assamese': 'as',
        'maithili': 'mai',
        'santali': 'sat',
        'kashmiri': 'ks',
        'sindhi': 'sd',
        'nepali': 'ne',
        'dogri': 'doi',
        'manipuri': 'mni',
        'bodo': 'bodo',
        'spanish': 'es', 'french': 'fr', 'german': 'de', 'chinese': 'zh',
        'japanese': 'ja', 'korean': 'ko', 'russian': 'ru', 'arabic': 'ar',
    }

    # Patterns for explicit instructions
    patterns = [
        (r'\bin\s+(\w+)\b', 1),                       # "in English"
        (r'(\w+)\s+(?:mein|mai|main|me)\b', 1),       # "Hindi mein"
        (r'(?:bolo|batao|likho|karo)\s+(\w+)\b', 1),  # "bolo Hindi"
        (r'^(?:speak|answer|respond)\s+(\w+)\b', 1),  # "speak Hindi"
        (r'(\w+)\s+(?:language|language\s+mein)\b', 1), # "Hindi language"
        (r'can you (?:speak|understand)\s+(\w+)\?', 1), # "can you speak Hindi?"
    ]

    for pattern, group in patterns:
        match = re.search(pattern, text_lower)
        if match:
            lang_word = match.group(group)
            for name, code in lang_names.items():
                if name in lang_word or lang_word in name:
                    return code
    return None

# ----------------------------------------------------------------------
# Main language detection function (combines all methods)
# ----------------------------------------------------------------------
def detect_language(text: str) -> Dict[str, Any]:
    """
    Comprehensive language detection.
    Returns dict with keys:
        language: language code (ISO 639-1 or custom like 'hinglish')
        confidence: float 0-1
        method: detection method used
        explicit_instruction: language code if user explicitly requested, else None
    """
    if not text or not text.strip():
        return {
            'language': 'en',
            'confidence': 1.0,
            'method': 'default',
            'explicit_instruction': None
        }

    # Step 0: Check for explicit instruction
    explicit = detect_explicit_language_instruction(text)
    if explicit:
        return {
            'language': explicit,
            'confidence': 1.0,
            'method': 'explicit',
            'explicit_instruction': explicit
        }

    # Step 1: Script detection
    script = detect_script(text)
    if script:
        candidates = script_to_languages(script)
        if candidates:
            # For scripts with multiple languages, pick first (most common)
            lang = candidates[0]
            return {
                'language': lang,
                'confidence': 0.9,
                'method': f'script_{script}',
                'explicit_instruction': None
            }

    # Step 2: FastText (if available)
    if _fasttext_model:
        ft_res = fasttext_detect(text)
        if ft_res and ft_res['confidence'] > 0.7:
            # Sanity check for Urdu (if no Arabic script, downgrade)
            if ft_res['code'] == 'ur' and not any('\u0600' <= c <= '\u06FF' for c in text):
                ft_res['confidence'] *= 0.5
                if ft_res['confidence'] < 0.6:
                    ft_res['code'] = 'en'
            return {
                'language': ft_res['code'],
                'confidence': ft_res['confidence'],
                'method': ft_res['method'],
                'explicit_instruction': None
            }

    # Step 3: Hinglish detection
    is_hing, conf = is_hinglish(text)
    if is_hing:
        return {
            'language': 'hinglish',
            'confidence': conf,
            'method': 'hinglish_wordlist',
            'explicit_instruction': None
        }

    # Step 4: Transformer (if available and enabled)
    if _transformer:
        trans_res = _transformer.detect(text)
        if trans_res and trans_res['confidence'] > 0.7:
            # Sanity check for Urdu
            if trans_res['code'] == 'ur' and not any('\u0600' <= c <= '\u06FF' for c in text):
                trans_res['confidence'] *= 0.5
                if trans_res['confidence'] < 0.6:
                    trans_res['code'] = 'en'
            return {
                'language': trans_res['code'],
                'confidence': trans_res['confidence'],
                'method': trans_res['method'],
                'explicit_instruction': None
            }

    # Step 5: Default to English
    return {
        'language': 'en',
        'confidence': 0.8,
        'method': 'default',
        'explicit_instruction': None
    }

# ----------------------------------------------------------------------
# Convenience functions
# ----------------------------------------------------------------------
def get_language_code(text: str) -> str:
    """Return just the language code."""
    return detect_language(text)['language']

def get_response_language(prompt: str, user_preferred: Optional[str] = None) -> str:
    """
    Determine the language for the response.
    If user explicitly requested a language, use that.
    Otherwise, if user_preferred is set (and not 'auto'), use that.
    Otherwise, use the detected language.
    """
    detected = detect_language(prompt)
    if detected['explicit_instruction']:
        return detected['explicit_instruction']
    if user_preferred and user_preferred != 'auto':
        return user_preferred
    return detected['language']

# ----------------------------------------------------------------------
# Explicit exports
# ----------------------------------------------------------------------
__all__ = [
    "detect_language",
    "get_language_code",
    "get_response_language",
]
