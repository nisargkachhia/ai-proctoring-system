"""
Multi-Person Detection Module
Tracks and counts multiple persons in video frames
"""

import cv2
import numpy as np
from typing import List, Dict, Optional, Tuple
from collections import deque
import logging

from app.core.config import settings
from app.core.logging_config import get_logger
from ..object.object_detector import ObjectDetector

logger = get_logger(__name__)


class MultiPersonDetector:
    """
    Detects and tracks multiple persons in video
    Uses object detection + optional tracking for person counting
    """
    
    def __init__(
        self,
        object_detector: Optional[ObjectDetector] = None,
        max_allowed_persons: Optional[int] = None,
        tracking_enabled: bool = True,
        tracking_history_size: int = 30
    ):
        """
        Initialize multi-person detector
        
        Args:
            object_detector: Object detector instance
            max_allowed_persons: Maximum allowed persons (default from settings)
            tracking_enabled: Enable person tracking
            tracking_history_size: Size of tracking history buffer
        """
        self.object_detector = object_detector or ObjectDetector()
        self.max_allowed_persons = max_allowed_persons or settings.MAX_ALLOWED_PERSONS
        self.tracking_enabled = tracking_enabled and settings.PERSON_TRACKING_ENABLED
        
        # Tracking
        if self.tracking_enabled:
            self.tracker = cv2.TrackerCSRT_create()
            self.tracked_persons = {}
            self.tracking_history = deque(maxlen=tracking_history_size)
        else:
            self.tracker = None
            self.tracked_persons = {}
            self.tracking_history = deque(maxlen=tracking_history_size)
        
        logger.info(f"Multi-person detector initialized (max_allowed={self.max_allowed_persons})")
    
    def detect_persons(self, frame: np.ndarray) -> Dict:
        """
        Detect and count persons in frame
        
        Args:
            frame: Input frame
            
        Returns:
            Dictionary with detection results:
            {
                "count": int,
                "detections": List[Dict],
                "violation": bool,
                "message": str
            }
        """
        # Detect persons using object detector
        person_detections = self.object_detector.detect_persons(frame)
        person_count = len(person_detections)
        
        # Check for violation
        violation = person_count > self.max_allowed_persons
        message = None
        
        if violation:
            message = f"Multiple persons detected: {person_count} (max allowed: {self.max_allowed_persons})"
            logger.warning(message)
        
        # Update tracking if enabled
        if self.tracking_enabled:
            self._update_tracking(frame, person_detections)
        
        return {
            "count": person_count,
            "detections": person_detections,
            "violation": violation,
            "message": message,
            "max_allowed": self.max_allowed_persons
        }
    
    def _update_tracking(self, frame: np.ndarray, detections: List[Dict]):
        """
        Update person tracking
        
        Args:
            frame: Current frame
            detections: Current person detections
        """
        # Simple tracking implementation
        # In production, use more sophisticated tracking (DeepSORT, etc.)
        self.tracking_history.append({
            "frame": len(self.tracking_history),
            "detections": detections,
            "count": len(detections)
        })
    
    def get_person_count(self, frame: np.ndarray) -> int:
        """
        Get person count in frame
        
        Args:
            frame: Input frame
            
        Returns:
            Number of persons detected
        """
        result = self.detect_persons(frame)
        return result["count"]
    
    def is_violation(self, frame: np.ndarray) -> bool:
        """
        Check if multiple persons violation exists
        
        Args:
            frame: Input frame
            
        Returns:
            True if violation detected
        """
        result = self.detect_persons(frame)
        return result["violation"]
    
    def draw_detections(self, frame: np.ndarray, result: Dict) -> np.ndarray:
        """
        Draw person detections and violation indicators
        
        Args:
            frame: Input frame
            result: Detection result from detect_persons()
            
        Returns:
            Frame with drawn detections
        """
        result_frame = frame.copy()
        
        # Draw person bounding boxes
        for detection in result["detections"]:
            x1, y1, x2, y2 = detection["bbox"]
            confidence = detection.get("confidence", 1.0)
            
            # Use red color if violation, green otherwise
            color = (0, 0, 255) if result["violation"] else (0, 255, 0)
            cv2.rectangle(result_frame, (x1, y1), (x2, y2), color, 2)
            
            label = f"Person {confidence:.2f}"
            cv2.putText(result_frame, label, (x1, y1 - 10),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
        
        # Draw violation indicator
        if result["violation"]:
            h, w = result_frame.shape[:2]
            cv2.rectangle(result_frame, (0, 0), (w, h), (0, 0, 255), 5)
            
            text = f"VIOLATION: {result['count']} persons detected!"
            text_size = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 1, 2)[0]
            text_x = (w - text_size[0]) // 2
            cv2.putText(result_frame, text, (text_x, 40),
                       cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
        
        # Draw count
        count_text = f"Persons: {result['count']}/{self.max_allowed_persons}"
        cv2.putText(result_frame, count_text, (10, result_frame.shape[0] - 20),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        
        return result_frame
    
    def get_tracking_history(self) -> List[Dict]:
        """Get tracking history"""
        return list(self.tracking_history)

