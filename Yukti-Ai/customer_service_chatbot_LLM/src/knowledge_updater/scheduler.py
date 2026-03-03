"""
Scheduler entry point – can be called by cron or manually.
"""

import logging
from .builder import rebuild_index

logger = logging.getLogger(__name__)

def run_once(source: str = None, dry_run: bool = False) -> bool:
    """
    Run the updater once.
    If source is specified, only that source? (Not implemented; we always rebuild all.)
    """
    logger.info("Running knowledge updater...")
    if dry_run:
        logger.info("Dry run – would rebuild index.")
        return True
    success = rebuild_index()
    return success
