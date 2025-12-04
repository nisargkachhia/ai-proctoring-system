"""
Face Detection Module
Detects faces in video frames using OpenCV and deep learning models
"""

import cv2
import numpy as np
from typing import List, Dict, Optional, Tuple
from pathlib import Path
import logging

from app.core.config import settings
from app.core.logging_config import get_logger

logger = get_logger(__name__)


class FaceDetector:
    """
    Face detection using multiple methods:
    - OpenCV Haar Cascade (fast, lightweight)
    - DNN-based models (more accurate)
    """
    
    def __init__(
        self,
        method: str = "haar",  # "haar" or "dnn"
        cascade_path: Optional[str] = None,
        confidence_threshold: Optional[float] = None,
        min_face_size: Optional[Tuple[int, int]] = None
    ):
        """
        Initialize face detector
        
        Args:
            method: Detection method ("haar" or "dnn")
            cascade_path: Path to Haar Cascade XML file
            confidence_threshold: Minimum confidence for detection
            min_face_size: Minimum face size (width, height)
        """
        self.method = method
        self.confidence_threshold = confidence_threshold or settings.FACE_CONFIDENCE_THRESHOLD
        self.min_face_size = min_face_size or settings.FACE_MIN_SIZE
        
        # Initialize detector based on method
        if method == "haar":
            self._init_haar_cascade(cascade_path)
        elif method == "dnn":
            self._init_dnn_model()
        else:
            raise ValueError(f"Unknown detection method: {method}")
        
        logger.info(f"Face detector initialized with method: {method}")
    
    def _init_haar_cascade(self, cascade_path: Optional[str] = None):
        """Initialize Haar Cascade classifier"""
        if cascade_path is None:
            cascade_path = cv2.data.haarcascades + settings.FACE_DETECTION_MODEL
        
        self.cascade = cv2.CascadeClassifier(cascade_path)
        
        if self.cascade.empty():
            raise ValueError(f"Failed to load cascade from {cascade_path}")
        
        logger.info("Haar Cascade classifier loaded")
    
    def _init_dnn_model(self):
        """Initialize DNN-based face detector"""
        # TODO: Implement DNN-based face detection
        # Example: OpenCV DNN with face detection models
        # model_path = settings.MODEL_DIR / "opencv_face_detector_uint8.pb"
        # config_path = settings.MODEL_DIR / "opencv_face_detector.pbtxt"
        # self.net = cv2.dnn.readNetFromTensorflow(str(model_path), str(config_path))
        logger.warning("DNN face detection not yet implemented, falling back to Haar")
        self._init_haar_cascade()
    
    def detect(self, frame: np.ndarray) -> List[Dict]:
        """
        Detect faces in a frame
        
        Args:
            frame: Input frame (BGR format)
            
        Returns:
            List of face detections with bounding boxes and confidence
        """
        if self.method == "haar":
            return self._detect_haar(frame)
        elif self.method == "dnn":
            return self._detect_dnn(frame)
        else:
            return []
    
    def _detect_haar(self, frame: np.ndarray) -> List[Dict]:
        """Detect faces using Haar Cascade"""
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        
        faces = self.cascade.detectMultiScale(
            gray,
            scaleFactor=1.1,
            minNeighbors=5,
            minSize=self.min_face_size,
            flags=cv2.CASCADE_SCALE_IMAGE
        )
        
        detections = []
        for (x, y, w, h) in faces:
            detections.append({
                "bbox": [int(x), int(y), int(x + w), int(y + h)],
                "confidence": 1.0,  # Haar doesn't provide confidence
                "method": "haar",
                "width": int(w),
                "height": int(h)
            })
        
        return detections
    
    def _detect_dnn(self, frame: np.ndarray) -> List[Dict]:
        """Detect faces using DNN model"""
        # TODO: Implement DNN detection
        # This is a placeholder for future implementation
        return self._detect_haar(frame)
    
    def draw_detections(self, frame: np.ndarray, detections: List[Dict]) -> np.ndarray:
        """
        Draw face bounding boxes on frame
        
        Args:
            frame: Input frame
            detections: List of face detections
            
        Returns:
            Frame with drawn bounding boxes
        """
        result = frame.copy()
        
        for detection in detections:
            x1, y1, x2, y2 = detection["bbox"]
            confidence = detection.get("confidence", 1.0)
            
            # Draw bounding box
            cv2.rectangle(result, (x1, y1), (x2, y2), (0, 255, 0), 2)
            
            # Draw label
            label = f"Face {confidence:.2f}" if confidence < 1.0 else "Face"
            cv2.putText(
                result, label, (x1, y1 - 10),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2
            )
        
        return result
    
    def count_faces(self, frame: np.ndarray) -> int:
        """
        Count number of faces in frame
        
        Args:
            frame: Input frame
            
        Returns:
            Number of faces detected
        """
        detections = self.detect(frame)
        return len(detections)

