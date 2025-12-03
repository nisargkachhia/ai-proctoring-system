"""
Object Detection Service using COCO SSD model with PyTorch
Detects phones, persons, and other objects in video frames
"""

import torch
import torchvision.transforms as transforms
from torchvision.models.detection import ssd300_vgg16, SSD300_VGG16_Weights
import cv2
import numpy as np
from typing import List, Dict, Tuple, Optional
import logging

logger = logging.getLogger(__name__)


class ObjectDetectionService:
    """Service for detecting objects using COCO SSD model"""
    
    # COCO class IDs
    COCO_CLASSES = {
        1: 'person',
        77: 'cell phone',  # Note: COCO uses 'cell phone' as class 77
    }
    
    def __init__(self, device: Optional[str] = None, confidence_threshold: float = 0.5):
        """
        Initialize the object detection service
        
        Args:
            device: Device to run model on ('cuda', 'cpu', or None for auto)
            confidence_threshold: Minimum confidence score for detections
        """
        self.confidence_threshold = confidence_threshold
        self.device = device if device else ('cuda' if torch.cuda.is_available() else 'cpu')
        self.model = None
        self.transform = None
        self._load_model()
        
    def _load_model(self):
        """Load COCO SSD model with PyTorch"""
        try:
            logger.info(f"Loading COCO SSD model on device: {self.device}")
            
            # Load pre-trained SSD300 model with VGG16 backbone
            weights = SSD300_VGG16_Weights.DEFAULT
            self.model = ssd300_vgg16(weights=weights)
            self.model.to(self.device)
            self.model.eval()
            
            # Get the transform for preprocessing
            self.transform = weights.transforms()
            
            logger.info("COCO SSD model loaded successfully")
        except Exception as e:
            logger.error(f"Error loading COCO SSD model: {e}")
            raise
    
    def detect_objects(self, frame: np.ndarray) -> Dict[str, List[Dict]]:
        """
        Detect objects in a video frame
        
        Args:
            frame: Input frame as numpy array (BGR format from OpenCV)
            
        Returns:
            Dictionary with 'persons' and 'phones' lists containing detection info
        """
        if self.model is None:
            raise RuntimeError("Model not loaded. Call _load_model() first.")
        
        try:
            # Convert BGR to RGB
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            
            # Convert to PIL Image then to tensor
            from PIL import Image
            pil_image = Image.fromarray(rgb_frame)
            
            # Apply transforms and add batch dimension
            input_tensor = self.transform(pil_image).unsqueeze(0).to(self.device)
            
            # Run inference
            with torch.no_grad():
                predictions = self.model(input_tensor)[0]
            
            # Extract detections
            boxes = predictions['boxes'].cpu().numpy()
            scores = predictions['scores'].cpu().numpy()
            labels = predictions['labels'].cpu().numpy()
            
            persons = []
            phones = []
            
            for box, score, label in zip(boxes, scores, labels):
                if score < self.confidence_threshold:
                    continue
                
                # Convert box coordinates to integers
                x1, y1, x2, y2 = map(int, box)
                
                detection_info = {
                    'bbox': [x1, y1, x2, y2],
                    'confidence': float(score),
                    'class_id': int(label)
                }
                
                # Check for person (class 1)
                if label == 1:
                    persons.append(detection_info)
                # Check for cell phone (class 77)
                elif label == 77:
                    phones.append(detection_info)
            
            return {
                'persons': persons,
                'phones': phones
            }
            
        except Exception as e:
            logger.error(f"Error during object detection: {e}")
            return {'persons': [], 'phones': []}
    
    def draw_detections(self, frame: np.ndarray, detections: Dict[str, List[Dict]]) -> np.ndarray:
        """
        Draw detection boxes on frame
        
        Args:
            frame: Input frame
            detections: Detection results from detect_objects()
            
        Returns:
            Frame with drawn bounding boxes
        """
        result_frame = frame.copy()
        
        # Draw person detections in green
        for person in detections['persons']:
            x1, y1, x2, y2 = person['bbox']
            cv2.rectangle(result_frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(result_frame, f"Person {person['confidence']:.2f}", 
                       (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
        
        # Draw phone detections in red
        for phone in detections['phones']:
            x1, y1, x2, y2 = phone['bbox']
            cv2.rectangle(result_frame, (x1, y1), (x2, y2), (0, 0, 255), 2)
            cv2.putText(result_frame, f"Phone {phone['confidence']:.2f}", 
                       (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
        
        return result_frame

