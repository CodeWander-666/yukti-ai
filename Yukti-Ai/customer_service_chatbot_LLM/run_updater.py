#!/usr/bin/env python3
import sys
import os
import logging
import argparse
import fcntl
import time
from pathlib import Path

# Add src to path
src_dir = Path(__file__).parent / "src"
sys.path.insert(0, str(src_dir))

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

LOCK_FILE = "/tmp/yukti_updater.lock"

def acquire_lock():
    """Acquire a file lock to prevent concurrent runs."""
    try:
        lock_fd = open(LOCK_FILE, 'w')
        fcntl.flock(lock_fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
        return lock_fd
    except (IOError, OSError):
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

    lock_fd = acquire_lock()
    if lock_fd is None:
        logger.error("Another updater instance is already running. Exiting.")
        sys.exit(1)

    try:
        from knowledge_updater.scheduler import run_once
        # Pass args to run_once (modify scheduler to accept them)
        success = run_once(source=args.source, dry_run=args.dry_run)
        if success:
            logger.info("Knowledge base updated successfully.")
        else:
            logger.error("Knowledge base update failed.")
            sys.exit(1)
    finally:
        fcntl.flock(lock_fd, fcntl.LOCK_UN)
        lock_fd.close()

if __name__ == "__main__":
    main()
