import logging
from langchain_community.vectorstores import FAISS
from langchain_helper import get_embeddings  # reuse the same embedding function
from .config import SOURCES, VECTORDB_PATH
from . import connectors

logger = logging.getLogger(__name__)

def rebuild_index() -> bool:
    """
    Rebuild the FAISS index from all configured sources.
    Returns True on success, False on failure.
    """
    logger.info("Starting knowledge base rebuild...")
    all_docs = []

    # Collect documents from each source
    for source in SOURCES:
        src_type = source.get("type")
        try:
            if src_type == "csv":
                docs = connectors.fetch_csv(source)
            elif src_type == "rss":
                docs = connectors.fetch_rss(source)
            elif src_type == "api":
                docs = connectors.fetch_api(source)
            else:
                logger.warning(f"Unknown source type '{src_type}' – skipping")
                continue

            all_docs.extend(docs)
        except Exception as e:
            logger.exception(f"Error fetching from source {source.get('name', 'unknown')}: {e}")
            # Decide whether to abort or continue – here we continue but log the error
            continue

    if not all_docs:
        logger.warning("No documents fetched – index not updated.")
        return False

    logger.info(f"Total documents fetched: {len(all_docs)}")

    # Generate embeddings and build the index
    try:
        embeddings = get_embeddings()
        vectordb = FAISS.from_documents(all_docs, embeddings)
        vectordb.save_local(str(VECTORDB_PATH))
        logger.info(f"FAISS index saved to {VECTORDB_PATH} with {vectordb.index.ntotal} vectors.")
        return True
    except Exception as e:
        logger.exception("Failed to build or save FAISS index.")
        return False
