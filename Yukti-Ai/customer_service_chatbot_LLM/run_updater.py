import sys
import logging
from pathlib import Path

# Configure logging to console (can be redirected to file by cron)
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)

# Add src directory to Python path so we can import our modules
src_dir = Path(__file__).parent / "src"
sys.path.insert(0, str(src_dir))

try:
    from knowledge_updater.scheduler import run_once
    run_once()
except ImportError as e:
    logging.error(f"Failed to import updater module: {e}")
    sys.exit(1)
except Exception as e:
    logging.exception("Unhandled exception in updater")
    sys.exit(1)
