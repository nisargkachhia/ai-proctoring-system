"""
Test script to verify detection services work correctly
"""

import sys
from pathlib import Path
import logging

sys.path.insert(0, str(Path(__file__).parent))

from app.services.ai.object_detection import ObjectDetectionService
from app.services.ai.face_detection import FaceDetectionService
import cv2
import numpy as np

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def test_object_detection():
    """Test object detection service"""
    logger.info("Testing Object Detection Service...")
    
    try:
        # Initialize service
        detector = ObjectDetectionService(confidence_threshold=0.5)
        logger.info("Object detection service initialized")
        
        # Create a test frame (black image)
        test_frame = np.zeros((480, 640, 3), dtype=np.uint8)
        
        # Test detection
        results = detector.detect_objects(test_frame)
        logger.info(f"Detection results: {results}")
        logger.info("Object detection test passed!")
        
    except Exception as e:
        logger.error(f"Object detection test failed: {e}")
        raise


def test_face_detection():
    """Test face detection service"""
    logger.info("Testing Face Detection Service...")
    
    try:
        # Initialize service
        detector = FaceDetectionService()
        logger.info("Face detection service initialized")
        
        # Create a test frame (black image)
        test_frame = np.zeros((480, 640, 3), dtype=np.uint8)
        
        # Test detection
        results = detector.detect_faces(test_frame)
        logger.info(f"Face detection results: {results}")
        logger.info("Face detection test passed!")
        
    except Exception as e:
        logger.error(f"Face detection test failed: {e}")
        raise


if __name__ == "__main__":
    logger.info("Running detection service tests...")
    
    try:
        test_object_detection()
        test_face_detection()
        logger.info("All tests passed!")
    except Exception as e:
        logger.error(f"Tests failed: {e}")
        sys.exit(1)

