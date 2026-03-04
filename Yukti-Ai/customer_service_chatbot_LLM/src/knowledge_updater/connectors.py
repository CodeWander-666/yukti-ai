"""
Data source connectors for knowledge updater.
Supports CSV files, RSS feeds, REST APIs, and local uploads.
"""
import logging
import pandas as pd
import requests
import feedparser
from pathlib import Path
from typing import List, Dict, Any
from langchain_core.documents import Document
from langchain_community.document_loaders import CSVLoader, TextLoader, PyPDFLoader

from .config import DATASET_PATH, UPLOADS_PATH, SOURCES

logger = logging.getLogger(__name__)

def fetch_csv(file_path: Path) -> List[Document]:
    """Read a CSV file with multiple encoding fallbacks."""
    if not file_path.exists():
        logger.warning(f"CSV file not found: {file_path}")
        return []
    encodings = ['utf-8', 'cp1252', 'latin-1', 'iso-8859-1']
    for enc in encodings:
        try:
            loader = CSVLoader(str(file_path), encoding=enc)
            docs = loader.load()
            logger.info(f"Loaded {len(docs)} documents from {file_path.name} (encoding {enc})")
            return docs
        except UnicodeDecodeError:
            continue
        except Exception as e:
            logger.warning(f"Failed to load {file_path} with {enc}: {e}")
    logger.error(f"Could not read {file_path} with any encoding.")
    return []

def fetch_uploads() -> List[Document]:
    """Scan UPLOADS_PATH for supported files and load them."""
    supported_extensions = {'.csv': CSVLoader, '.txt': TextLoader, '.pdf': PyPDFLoader}
    docs = []
    for file_path in UPLOADS_PATH.glob("*"):
        if file_path.is_file() and file_path.suffix.lower() in supported_extensions:
            loader_class = supported_extensions[file_path.suffix.lower()]
            try:
                loader = loader_class(str(file_path))
                file_docs = loader.load()
                docs.extend(file_docs)
                logger.info(f"Loaded {len(file_docs)} docs from {file_path.name}")
            except Exception as e:
                logger.error(f"Failed to load {file_path.name}: {e}")
    return docs

def fetch_rss(feed_config: Dict[str, Any]) -> List[Document]:
    """Fetch entries from an RSS feed."""
    if not feed_config.get("enabled", False):
        return []
    url = feed_config["url"]
    try:
        feed = feedparser.parse(url)
        docs = []
        for entry in feed.entries:
            content = entry.get("summary", entry.get("description", ""))
            if content:
                doc = Document(
                    page_content=content,
                    metadata={"source": url, "title": entry.get("title"), "published": entry.get("published")}
                )
                docs.append(doc)
        logger.info(f"Fetched {len(docs)} items from RSS {url}")
        return docs
    except Exception as e:
        logger.error(f"RSS fetch failed for {url}: {e}")
        return []

def fetch_api(api_config: Dict[str, Any]) -> List[Document]:
    """Fetch data from a REST API (expects JSON array)."""
    if not api_config.get("enabled", False):
        return []
    url = api_config["url"]
    try:
        resp = requests.get(url, timeout=10)
        resp.raise_for_status()
        data = resp.json()
        docs = []
        if isinstance(data, list):
            for item in data:
                content = str(item)
                doc = Document(page_content=content, metadata={"source": url})
                docs.append(doc)
        logger.info(f"Fetched {len(docs)} items from API {url}")
        return docs
    except Exception as e:
        logger.error(f"API fetch failed for {url}: {e}")
        return []

def fetch_all_sources() -> List[Document]:
    """Fetch documents from all enabled sources."""
    all_docs = []

    # Main CSV
    all_docs.extend(fetch_csv(DATASET_PATH))

    # Uploads
    all_docs.extend(fetch_uploads())

    # RSS feeds
    for feed in SOURCES.get("rss", []):
        all_docs.extend(fetch_rss(feed))

    # APIs
    for api in SOURCES.get("api", []):
        all_docs.extend(fetch_api(api))

    return all_docs
