import re
import json
import logging
from pathlib import Path
from typing import Dict, Any, Optional

# Load Hinglish wordlist from external JSON file
WORDLIST_PATH = Path(__file__).parent / "hinglish_words.json"
if WORDLIST_PATH.exists():
    with open(WORDLIST_PATH, 'r', encoding='utf-8') as f:
        HINGLISH_HINDI_WORDS = set(json.load(f))
else:
    HINGLISH_HINDI_WORDS = set()  # fallback empty
    logging.warning("hinglish_words.json not found; Hinglish detection disabled.")

# ... (rest of the script detection, FastText, transformer code remains, but transformer loading made optional)

# At the bottom, near the end, add a conditional for transformer:
_transformer = None
if os.getenv("ENABLE_TRANSFORMER", "false").lower() == "true":
    _transformer = TransformerDetector()
    logging.info("Transformer language detector enabled.")
else:
    logging.info("Transformer language detector disabled (set ENABLE_TRANSFORMER=true to enable).")
