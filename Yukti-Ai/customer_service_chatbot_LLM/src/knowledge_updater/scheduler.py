import logging
from .builder import rebuild_index

logger = logging.getLogger(__name__)

def run_once():
    """
    Single execution of the knowledge base update.
    Intended to be called by a cron job or manually.
    """
    logger.info("Knowledge updater started.")
    success = rebuild_index()
    if success:
        logger.info("Knowledge base updated successfully.")
    else:
        logger.error("Knowledge base update failed.")
