"""
Example script to run live proctoring system
Demonstrates real-time face, person, and phone detection with alert logic
"""

import logging
import sys
from pathlib import Path

# Add app directory to path
sys.path.insert(0, str(Path(__file__).parent))

from app.services.ai.live_proctoring import LiveProctoring, alert_handler

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler('logs/proctoring.log')
    ]
)

logger = logging.getLogger(__name__)


def main():
    """Run live proctoring example"""
    logger.info("Starting AI Proctoring System...")
    logger.info("Press 'q' to quit")
    
    # Initialize and start live proctoring
    proctoring = LiveProctoring(
        camera_source=0,  # Use default camera (change to 1, 2, etc. for other cameras)
        alert_callback=alert_handler,  # Custom alert handler
        display=True  # Show video window
    )
    
    try:
        proctoring.start()
    except KeyboardInterrupt:
        logger.info("Interrupted by user")
    except Exception as e:
        logger.error(f"Error: {e}", exc_info=True)
    finally:
        proctoring.stop()
        logger.info("Proctoring system stopped")


if __name__ == "__main__":
    main()

