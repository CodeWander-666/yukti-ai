# knowledge_updater/config.py (enhanced)

SOURCES = {
    "rss": [...],
    "api": [...],
    "websites": [
        {
            "name": "Documentation Site",
            "url": "https://docs.example.com",
            "enabled": False,
            "crawl_depth": 50,
            "extract_selectors": {  # Optional: specific data to extract
                "title": "h1",
                "content": "article"
            },
            "use_javascript": False,  # Static vs dynamic
            "respect_robots": True,
            "rate_limit": (2, 5)  # Min/max seconds between requests
        }
    ]
}
