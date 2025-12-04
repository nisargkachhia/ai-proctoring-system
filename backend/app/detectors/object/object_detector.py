"""
Object Detection Module
Main interface for object detection using SSD or ResNet models
"""

import numpy as np
from typing import List, Dict, Optional
import logging

from app.core.config import settings
from app.core.logging_config import get_logger
from .ssd_detector import SSDDetector
from .resnet_detector import ResNetDetector

logger = get_logger(__name__)


class ObjectDetector:
    """
    Unified object detection interface
    Supports both SSD and ResNet-based detection
    """
    
    # COCO class IDs for common objects
    COCO_CLASSES = {
        1: "person",
        77: "cell phone",
        73: "laptop",
        67: "cell phone",  # Alternative class ID
    }
    
    def __init__(
        self,
        model_type: str = "ssd",  # "ssd" or "resnet"
        confidence_threshold: Optional[float] = None,
        device: Optional[str] = None
    ):
        """
        Initialize object detector
        
        Args:
            model_type: Model type ("ssd" or "resnet")
            confidence_threshold: Minimum confidence for detections
            device: Device to run on ("cuda", "cpu", or None for auto)
        """
        self.model_type = model_type
        self.confidence_threshold = confidence_threshold or settings.SSD_CONFIDENCE_THRESHOLD
        
        # Initialize appropriate detector
        if model_type == "ssd":
            self.detector = SSDDetector(
                confidence_threshold=self.confidence_threshold,
                device=device
            )
        elif model_type == "resnet":
            self.detector = ResNetDetector(
                confidence_threshold=self.confidence_threshold,
                device=device
            )
        else:
            raise ValueError(f"Unknown model type: {model_type}")
        
        logger.info(f"Object detector initialized with {model_type} model")
    
    def detect(self, frame: np.ndarray) -> Dict[str, List[Dict]]:
        """
        Detect objects in frame
        
        Args:
            frame: Input frame (BGR format)
            
        Returns:
            Dictionary with detection results:
            {
                "persons": [...],
                "phones": [...],
                "laptops": [...],
                "all": [...]
            }
        """
        return self.detector.detect(frame)
    
    def detect_persons(self, frame: np.ndarray) -> List[Dict]:
        """
        Detect only persons in frame
        
        Args:
            frame: Input frame
            
        Returns:
            List of person detections
        """
        results = self.detect(frame)
        return results.get("persons", [])
    
    def detect_phones(self, frame: np.ndarray) -> List[Dict]:
        """
        Detect only phones in frame
        
        Args:
            frame: Input frame
            
        Returns:
            List of phone detections
        """
        results = self.detect(frame)
        return results.get("phones", [])
    
    def draw_detections(self, frame: np.ndarray, detections: Dict[str, List[Dict]]) -> np.ndarray:
        """
        Draw detection bounding boxes on frame
        
        Args:
            frame: Input frame
            detections: Detection results from detect()
            
        Returns:
            Frame with drawn bounding boxes
        """
        return self.detector.draw_detections(frame, detections)
    
    def count_objects(self, frame: np.ndarray, object_type: str = "person") -> int:
        """
        Count specific objects in frame
        
        Args:
            frame: Input frame
            object_type: Type of object to count ("person", "phone", etc.)
            
        Returns:
            Number of objects detected
        """
        results = self.detect(frame)
        return len(results.get(object_type + "s", []))

