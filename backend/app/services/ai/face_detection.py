"""
Face Detection Service using OpenCV
Detects faces in video frames
"""

import cv2
import numpy as np
from typing import List, Dict, Optional, Tuple
import logging

logger = logging.getLogger(__name__)


class FaceDetectionService:
    """Service for detecting faces using OpenCV Haar Cascade"""
    
    def __init__(self, cascade_path: Optional[str] = None):
        """
        Initialize the face detection service
        
        Args:
            cascade_path: Path to Haar Cascade XML file. If None, uses default.
        """
        self.face_cascade = None
        self._load_cascade(cascade_path)
    
    def _load_cascade(self, cascade_path: Optional[str] = None):
        """Load Haar Cascade classifier for face detection"""
        try:
            if cascade_path is None:
                # Use OpenCV's built-in Haar Cascade
                cascade_path = cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
            
            self.face_cascade = cv2.CascadeClassifier(cascade_path)
            
            if self.face_cascade.empty():
                raise ValueError(f"Failed to load cascade from {cascade_path}")
            
            logger.info("Face detection cascade loaded successfully")
        except Exception as e:
            logger.error(f"Error loading face cascade: {e}")
            raise
    
    def detect_faces(self, frame: np.ndarray, scale_factor: float = 1.1, 
                    min_neighbors: int = 5, min_size: Tuple[int, int] = (30, 30)) -> List[Dict]:
        """
        Detect faces in a video frame
        
        Args:
            frame: Input frame as numpy array (BGR format from OpenCV)
            scale_factor: Parameter specifying how much the image size is reduced at each scale
            min_neighbors: Minimum number of neighbors required for detection
            min_size: Minimum possible object size
            
        Returns:
            List of dictionaries containing face detection info
        """
        if self.face_cascade is None:
            raise RuntimeError("Cascade not loaded. Call _load_cascade() first.")
        
        try:
            # Convert to grayscale for face detection
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            
            # Detect faces
            faces = self.face_cascade.detectMultiScale(
                gray,
                scaleFactor=scale_factor,
                minNeighbors=min_neighbors,
                minSize=min_size
            )
            
            # Convert to list of dictionaries
            face_detections = []
            for (x, y, w, h) in faces:
                face_detections.append({
                    'bbox': [int(x), int(y), int(x + w), int(y + h)],
                    'confidence': 1.0,  # Haar Cascade doesn't provide confidence scores
                    'width': int(w),
                    'height': int(h)
                })
            
            return face_detections
            
        except Exception as e:
            logger.error(f"Error during face detection: {e}")
            return []
    
    def draw_faces(self, frame: np.ndarray, faces: List[Dict]) -> np.ndarray:
        """
        Draw face bounding boxes on frame
        
        Args:
            frame: Input frame
            faces: Face detection results from detect_faces()
            
        Returns:
            Frame with drawn bounding boxes
        """
        result_frame = frame.copy()
        
        for face in faces:
            x1, y1, x2, y2 = face['bbox']
            cv2.rectangle(result_frame, (x1, y1), (x2, y2), (255, 0, 0), 2)
            cv2.putText(result_frame, "Face", (x1, y1 - 10), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)
        
        return result_frame

