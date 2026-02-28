SOURCES = [
    {
        "type": "csv",
        "path": "/path/to/your/dataset/dataset.csv",
        "name": "Original Dataset",
        "columns": ["prompt", "response"],
        "content_template": "Q: {prompt}\nA: {response}"
    },
    {
        "type": "csv",
        "path": "/path/to/your/dataset/faq.csv",
        "name": "FAQs",
        "columns": ["prompt", "response"],
        "content_template": "Q: {prompt}\nA: {response}"
    },
    # Add more sources here (RSS, API, etc.)
]
