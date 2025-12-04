"""
Main application entry point
Production-ready AI Proctoring System
"""

import cv2
import logging
import sys
from pathlib import Path
from typing import Optional

from app.core.config import settings
from app.core.logging_config import setup_logging, get_logger
from app.inference import ProctoringPipeline
from app.utils.video import VideoCapture, FrameProcessor

logger = get_logger(__name__)


def alert_handler(result: dict):
    """
    Alert callback handler
    
    Args:
        result: Detection result with violations
    """
    logger.warning("=" * 60)
    logger.warning("PROCTORING ALERT TRIGGERED!")
    logger.warning(f"Timestamp: {result.get('timestamp')}")
    
    for violation in result.get('violations', []):
        logger.warning(f"Type: {violation.get('type')}")
        logger.warning(f"Severity: {violation.get('severity')}")
        logger.warning(f"Message: {violation.get('message')}")
    
    logger.warning("=" * 60)


def main():
    """Main application entry point"""
    # Setup logging
    setup_logging()
    logger.info(f"Starting {settings.APP_NAME} v{settings.APP_VERSION}")
    
    # Initialize pipeline
    logger.info("Initializing proctoring pipeline...")
    pipeline = ProctoringPipeline(alert_callback=alert_handler)
    
    # Initialize video capture
    video_capture = VideoCapture(
        source=settings.VIDEO_SOURCE,
        width=settings.VIDEO_WIDTH,
        height=settings.VIDEO_HEIGHT,
        fps=settings.VIDEO_FPS
    )
    
    if not video_capture.start():
        logger.error("Failed to start video capture")
        sys.exit(1)
    
    # Frame processor
    frame_processor = FrameProcessor(frame_skip=settings.FRAME_SKIP)
    
    logger.info("Proctoring system ready. Press 'q' to quit.")
    
    try:
        frame_count = 0
        
        while video_capture.is_running:
            # Read frame
            frame = video_capture.read()
            if frame is None:
                logger.warning("Failed to read frame")
                break
            
            # Process frame
            if frame_count % settings.FRAME_SKIP == 0:
                result = pipeline.process_frame(frame)
                
                # Draw results
                annotated_frame = pipeline.draw_results(frame, result)
                
                # Display
                cv2.imshow('AI Proctoring System', annotated_frame)
                
                # Check for quit
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    logger.info("Quit requested by user")
                    break
            
            frame_count += 1
    
    except KeyboardInterrupt:
        logger.info("Interrupted by user")
    except Exception as e:
        logger.error(f"Error in main loop: {e}", exc_info=True)
    finally:
        video_capture.stop()
        cv2.destroyAllWindows()
        logger.info("Proctoring system stopped")


if __name__ == "__main__":
    main()

