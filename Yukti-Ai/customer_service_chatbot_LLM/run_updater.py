#!/usr/bin/env python3
"""
Cron entry point for knowledge updater.
Adds src to path, acquires a lock, and runs the scheduler.
"""

import os
import sys
import time
import logging
import argparse
import fcntl
from pathlib import Path

# Configure logging to both file and console
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("updater.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# Add src directory to Python path so we can import our modules
src_dir = Path(__file__).parent / "src"
sys.path.insert(0, str(src_dir))

# Also add project root for config imports
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

LOCK_FILE = "/tmp/yukti_updater.lock"

def acquire_lock():
    """
    Acquire a file lock to prevent concurrent runs.
    Returns the lock file descriptor if successful, None otherwise.
    """
    try:
        lock_fd = open(LOCK_FILE, 'w')
        fcntl.flock(lock_fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
        return lock_fd
    except (IOError, OSError) as e:
        logger.error(f"Could not acquire lock: {e}")
        return None

def main():
    parser = argparse.ArgumentParser(description="Yukti Knowledge Updater")
    parser.add_argument("--source", help="Run only for a specific source (by name)")
    parser.add_argument("--dry-run", action="store_true", help="Fetch but do not build index")
    args = parser.parse_args()

    # Ensure API keys are in environment (not st.secrets)
    if not os.getenv("ZHIPU_API_KEY") and not os.getenv("GOOGLE_API_KEY"):
        logger.error("No API keys found in environment. Set ZHIPU_API_KEY or GOOGLE_API_KEY.")
        sys.exit(1)

    # Acquire lock
    lock_fd = acquire_lock()
    if lock_fd is None:
        logger.error("Another updater instance is already running. Exiting.")
        sys.exit(1)

    try:
        # Import here so that missing dependencies don't break argument parsing
        from knowledge_updater.scheduler import run_once
        success = run_once(source=args.source, dry_run=args.dry_run)
        if success:
            logger.info("Knowledge base updated successfully.")
        else:
            logger.error("Knowledge base update failed.")
            sys.exit(1)
    except ImportError as e:
        logger.exception(f"Failed to import updater module: {e}")
        sys.exit(1)
    except Exception as e:
        logger.exception("Unhandled exception in updater")
        sys.exit(1)
    finally:
        # Release lock
        fcntl.flock(lock_fd, fcntl.LOCK_UN)
        lock_fd.close()

if __name__ == "__main__":
    main()
