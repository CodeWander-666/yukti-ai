"""
Model configurations for Zhipu GLM models.
Each model has type (sync/async), concurrency limit, and description.
"""

MODELS = {
    "Yukti‑Flash": {
        "model": "glm-4-flash",
        "type": "sync",
        "concurrency": 200,
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
        "description": "Image generation",
    },
    "Yukti‑Video": {
        "model": "cogvideox",
        "type": "async",
        "concurrency": 5,
        "description": "Video generation",
    },
    "Yukti‑Audio": {
        "model": "glm-realtime",
        "type": "async",
        "concurrency": 5,
        "description": "Audio generation",
    },
}

def get_model_config(model_key: str):
    """Return config dict for given model key, or None if not found."""
    return MODELS.get(model_key)
