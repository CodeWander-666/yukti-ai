import os
import logging
import pandas as pd
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
    for enc in encodings:
        try:
            df = pd.read_csv(path, encoding=enc, on_bad_lines='skip')
            logger.info(f"Successfully read {path} with encoding {enc}")
            break
        except UnicodeDecodeError:
            continue
        except Exception as e:
            logger.error(f"Unexpected error reading {path} with {enc}: {e}")
            continue

    if df is None:
        logger.error(f"Could not read {path} with any tried encoding.")
        return []

    # Validate required columns
    columns = source.get("columns", ["prompt", "response"])
    missing = [c for c in columns if c not in df.columns]
    if missing:
        logger.error(f"CSV {path} missing required columns: {missing}")
        return []

    template = source.get("content_template", "Q: {prompt}\nA: {response}")
    documents = []
    for idx, row in df.iterrows():
        try:
            content = template.format(**{c: row[c] for c in columns})
            doc = Document(
                page_content=content,
                metadata={
                    "source": path,
                    "row": idx,
                    "source_name": source.get("name", "CSV")
                }
            )
            documents.append(doc)
        except Exception as e:
            logger.warning(f"Skipping row {idx} in {path}: {e}")
            continue

    logger.info(f"Loaded {len(documents)} documents from {source.get('name', path)}")
    return documents


# ----------------------------------------------------------------------
# RSS Connector (placeholder – implement when needed)
# ----------------------------------------------------------------------
def fetch_rss(source: Dict[str, Any]) -> List[Document]:
    """Fetch and parse an RSS feed."""
    # TODO: implement with feedparser
    logger.warning("RSS connector not implemented yet.")
    return []


# ----------------------------------------------------------------------
# API Connector (placeholder)
# ----------------------------------------------------------------------
def fetch_api(source: Dict[str, Any]) -> List[Document]:
    """Fetch data from a REST API."""
    # TODO: implement with requests
    logger.warning("API connector not implemented yet.")
    return []
