# Mapping from language codes to human‑readable names
LANGUAGE_NAMES = {
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
    'hinglish': 'Hinglish',
}

def get_language_name(code: str) -> str:
    """Return the human‑readable name for a language code."""
    return LANGUAGE_NAMES.get(code, code)
