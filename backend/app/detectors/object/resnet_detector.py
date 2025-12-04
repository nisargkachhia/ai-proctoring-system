"""
ResNet-based Object Detection
Uses Faster R-CNN with ResNet backbone for more accurate detection
"""

import torch
from torchvision.models.detection import (
    fasterrcnn_resnet50_fpn,
    FasterRCNN_ResNet50_FPN_Weights
)
import cv2
import numpy as np
from typing import List, Dict, Optional
from PIL import Image
import logging

from app.core.config import settings
from app.core.logging_config import get_logger

logger = get_logger(__name__)


class ResNetDetector:
    """
    ResNet-based object detection using Faster R-CNN
    More accurate but slower than SSD
    """
    
    PERSON_CLASS = 1
    CELL_PHONE_CLASS = 77
    
    def __init__(
        self,
        model_type: str = "fasterrcnn_resnet50_fpn",
        confidence_threshold: float = 0.5,
        device: Optional[str] = None
    ):
        """
        Initialize ResNet detector
        
        Args:
            model_type: ResNet model type
            confidence_threshold: Minimum confidence for detections
            device: Device to run on
        """
        self.model_type = model_type
        self.confidence_threshold = confidence_threshold
        self.device = device if device else ("cuda" if torch.cuda.is_available() else "cpu")
        
        self.model = None
        self.transform = None
        
        self._load_model()
    
    def _load_model(self):
        """Load ResNet model with PyTorch"""
        try:
            logger.info(f"Loading {self.model_type} model on {self.device}...")
            
            if self.model_type == "fasterrcnn_resnet50_fpn":
                weights = FasterRCNN_ResNet50_FPN_Weights.DEFAULT
                self.model = fasterrcnn_resnet50_fpn(weights=weights)
            else:
                raise ValueError(f"Unknown ResNet model type: {self.model_type}")
            
            self.model.to(self.device)
            self.model.eval()
            self.transform = weights.transforms()
            
            logger.info(f"ResNet model loaded successfully on {self.device}")
        except Exception as e:
            logger.error(f"Error loading ResNet model: {e}")
            raise
    
    def detect(self, frame: np.ndarray) -> Dict[str, List[Dict]]:
        """
        Detect objects in frame using ResNet
        
        Args:
            frame: Input frame (BGR format)
            
        Returns:
            Dictionary with detection results
        """
        if self.model is None:
            raise RuntimeError("Model not loaded")
        
        try:
            # Convert BGR to RGB
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            pil_image = Image.fromarray(rgb_frame)
            
            # Preprocess
            input_tensor = self.transform(pil_image).unsqueeze(0).to(self.device)
            
            # Inference
            with torch.no_grad():
                predictions = self.model(input_tensor)[0]
            
            # Extract detections
            boxes = predictions['boxes'].cpu().numpy()
            scores = predictions['scores'].cpu().numpy()
            labels = predictions['labels'].cpu().numpy()
            
            # Filter by confidence and class
            persons = []
            phones = []
            all_detections = []
            
            for box, score, label in zip(boxes, scores, labels):
                if score < self.confidence_threshold:
                    continue
                
                x1, y1, x2, y2 = map(int, box)
                detection = {
                    "bbox": [x1, y1, x2, y2],
                    "confidence": float(score),
                    "class_id": int(label),
                    "class_name": self._get_class_name(int(label))
                }
                
                all_detections.append(detection)
                
                if label == self.PERSON_CLASS:
                    persons.append(detection)
                elif label == self.CELL_PHONE_CLASS:
                    phones.append(detection)
            
            return {
                "persons": persons,
                "phones": phones,
                "all": all_detections
            }
            
        except Exception as e:
            logger.error(f"Error during ResNet detection: {e}")
            return {"persons": [], "phones": [], "all": []}
    
    def _get_class_name(self, class_id: int) -> str:
        """Get class name from COCO class ID"""
        class_names = {
            1: "person",
            77: "cell phone",
            73: "laptop"
        }
        return class_names.get(class_id, f"class_{class_id}")
    
    def draw_detections(self, frame: np.ndarray, detections: Dict[str, List[Dict]]) -> np.ndarray:
        """Draw detection bounding boxes"""
        result = frame.copy()
        
        colors = {
            "person": (0, 255, 0),
            "phone": (0, 0, 255),
            "default": (255, 0, 0)
        }
        
        for person in detections.get("persons", []):
            x1, y1, x2, y2 = person["bbox"]
            cv2.rectangle(result, (x1, y1), (x2, y2), colors["person"], 2)
            label = f"Person {person['confidence']:.2f}"
            cv2.putText(result, label, (x1, y1 - 10),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, colors["person"], 2)
        
        for phone in detections.get("phones", []):
            x1, y1, x2, y2 = phone["bbox"]
            cv2.rectangle(result, (x1, y1), (x2, y2), colors["phone"], 2)
            label = f"Phone {phone['confidence']:.2f}"
            cv2.putText(result, label, (x1, y1 - 10),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, colors["phone"], 2)
        
        return result

