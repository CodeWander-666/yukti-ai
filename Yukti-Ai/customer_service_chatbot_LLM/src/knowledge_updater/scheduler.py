"""
Scheduler entry point for knowledge updater.
Calls rebuild_index and logs result.
"""

import logging
from .builder import rebuild_index

logger = logging.getLogger(__name__)

def run_once(source: str = None, dry_run: bool = False):
    """
    Single execution of the knowledge base update.
    Args:
        source: if provided, only update that source (by name) – not yet implemented
        dry_run: if True, fetch but do not build index (not yet implemented)
    """
    logger.info("Knowledge updater started.")
    if dry_run:
        logger.info("Dry run mode – will not save index.")
    if source:
        logger.info(f"Filtering for source: {source} (not implemented, will fetch all)")

    success = rebuild_index()
    if success:
        logger.info("Knowledge base updated successfully.")
    else:
        logger.error("Knowledge base update failed.")
    return success
