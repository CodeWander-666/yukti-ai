"""
Connectors for various data sources (CSV, RSS, API).
Each function returns a list of LangChain Document objects.
Includes comprehensive error handling and logging.
"""

import os
import logging
import requests
import pandas as pd
import feedparser
from datetime import datetime
from typing import List, Dict, Any
from langchain_core.documents import Document

logger = logging.getLogger(__name__)

# ----------------------------------------------------------------------
# CSV Connector – robust encoding fallback and error handling
# ----------------------------------------------------------------------
def fetch_csv(source: Dict[str, Any]) -> List[Document]:
    """
    Reads a CSV file and converts each row into a LangChain Document.
    Handles common encodings and malformed lines gracefully.
    """
    path = source["path"]
    if not os.path.exists(path):
        logger.warning(f"CSV file not found: {path}")
        return []

    # Try multiple encodings in order of likelihood
    encodings = ['utf-8', 'cp1252', 'latin-1', 'iso-8859-1']
    df = None
    last_error = None
    for enc in encodings:
        try:
            df = pd.read_csv(path, encoding=enc, on_bad_lines='skip')
            logger.info(f"Successfully read {path} with encoding {enc}")
            break
        except UnicodeDecodeError as e:
            last_error = e
            continue
        except pd.errors.EmptyDataError as e:
            logger.error(f"CSV file {path} is empty: {e}")
            return []
        except pd.errors.ParserError as e:
            logger.error(f"CSV parsing error in {path}: {e}")
            return []
        except Exception as e:
            logger.warning(f"Unexpected error reading {path} with {enc}: {e}")
            last_error = e
            continue

    if df is None:
        logger.error(f"Could not read {path} with any tried encoding. Last error: {last_error}")
        return []

    # Validate required columns
    columns = source.get("columns", ["prompt", "response"])
    missing = [c for c in columns if c not in df.columns]
    if missing:
        logger.error(f"CSV {path} missing required columns: {missing}")
        return []

    template = source.get("content_template", "Q: {prompt}\nA: {response}")
    documents = []
    skipped = 0
    for idx, row in df.iterrows():
        try:
            content = template.format(**{c: row[c] for c in columns})
            doc = Document(
                page_content=content,
                metadata={
                    "source": str(path),
                    "row": int(idx),
                    "source_name": source.get("name", "CSV"),
                    "fetched_at": datetime.now().isoformat()
                }
            )
            documents.append(doc)
        except KeyError as e:
            logger.warning(f"Row {idx} missing column {e} – skipping")
            skipped += 1
        except Exception as e:
            logger.warning(f"Row {idx} formatting error: {e}")
            skipped += 1

    if skipped:
        logger.info(f"Skipped {skipped} rows due to errors in {source.get('name', path)}")
    logger.info(f"Loaded {len(documents)} documents from {source.get('name', path)}")
    return documents


# ----------------------------------------------------------------------
# RSS Connector – with timeout and error handling
# ----------------------------------------------------------------------
def fetch_rss(source: Dict[str, Any]) -> List[Document]:
    """Fetch and parse an RSS feed."""
    url = source["url"]
    try:
        feed = feedparser.parse(url)
        if feed.bozo:
            logger.warning(f"RSS feed may have issues: {feed.bozo_exception}")
    except Exception as e:
        logger.exception(f"Failed to fetch RSS feed {url}")
        return []

    template = source.get("content_template", "{title}\n{link}\n{summary}")
    documents = []
    for entry in feed.entries[:50]:  # limit to last 50 entries
        try:
            fields = {
                "title": entry.get("title", ""),
                "link": entry.get("link", ""),
                "summary": entry.get("summary", ""),
                "published": entry.get("published", ""),
            }
            content = template.format(**fields)
            doc = Document(
                page_content=content,
                metadata={
                    "source": url,
                    "source_name": source.get("name", "RSS"),
                    "published": fields["published"],
                    "fetched_at": datetime.now().isoformat()
                }
            )
            documents.append(doc)
        except Exception as e:
            logger.warning(f"Skipping RSS entry: {e}")
            continue

    logger.info(f"Fetched {len(documents)} entries from RSS {source.get('name')}")
    return documents


# ----------------------------------------------------------------------
# API Connector – with retries and JSON parsing
# ----------------------------------------------------------------------
def fetch_api(source: Dict[str, Any]) -> List[Document]:
    """Fetch data from a REST API. Supports simple date substitution."""
    url_template = source["url"]
    # Replace {{date}} with today's date in YYYY/MM/DD format
    today = datetime.now().strftime("%Y/%m/%d")
    url = url_template.replace("{{date}}", today)

    # Retry logic
    max_retries = 3
    timeout = 10
    for attempt in range(max_retries):
        try:
            resp = requests.get(url, timeout=timeout)
            resp.raise_for_status()
            data = resp.json()
            break
        except requests.exceptions.RequestException as e:
            logger.warning(f"API request failed (attempt {attempt+1}): {e}")
            if attempt == max_retries - 1:
                logger.error(f"All {max_retries} attempts failed for {url}")
                return []
            time.sleep(2 ** attempt)  # exponential backoff

    # Generic parser – adapt to your actual API response structure
    # Example: Wikipedia featured content
    items = []
    if "tfa" in data:  # Today's featured article
        items.append(data["tfa"])
    if "news" in data:  # In the news
        items.extend(data.get("news", []))
    if "onthisday" in data:  # On this day
        items.extend(data.get("onthisday", []))

    # If no known structure, try to extract a list from the top level
    if not items and isinstance(data, list):
        items = data
    elif not items and isinstance(data, dict):
        # Try common keys that might contain lists
        for key in ['items', 'results', 'data', 'articles']:
            if key in data and isinstance(data[key], list):
                items = data[key]
                break

    template = source.get("content_template", "{title}\n{extract}")
    documents = []
    for item in items:
        try:
            fields = {
                "title": item.get("title", ""),
                "extract": item.get("extract", item.get("description", item.get("content", ""))),
                "url": item.get("url", item.get("link", "")),
            }
            content = template.format(**fields)
            doc = Document(
                page_content=content,
                metadata={
                    "source": url,
                    "source_name": source.get("name", "API"),
                    "fetched_at": datetime.now().isoformat()
                }
            )
            documents.append(doc)
        except Exception as e:
            logger.warning(f"Skipping API item: {e}")
            continue

    logger.info(f"Fetched {len(documents)} items from API {source.get('name')}")
    return documents
