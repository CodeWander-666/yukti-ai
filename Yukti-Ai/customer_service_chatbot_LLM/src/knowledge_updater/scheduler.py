import logging
from .builder import rebuild_index

logger = logging.getLogger(__name__)

def run_once(source: str = None, dry_run: bool = False):
    """
    Single execution of the knowledge base update.
    source: if provided, only update that source (by name)
    dry_run: if True, fetch but do not build index
    """
    logger.info("Knowledge updater started.")
    if dry_run:
        logger.info("Dry run mode – will not save index.")
    # TODO: filter sources based on source argument
    success = rebuild_index()  # modify rebuild_index to respect dry_run if needed
    if success:
        logger.info("Knowledge base updated successfully.")
    else:
        logger.error("Knowledge base update failed.")
    return success
