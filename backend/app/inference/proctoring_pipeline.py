"""
Proctoring Pipeline Module
Main pipeline that combines all detection modules and decision logic
"""

import cv2
import numpy as np
from typing import Dict, Optional, Callable
import logging
from datetime import datetime

from app.core.config import settings
from app.core.logging_config import get_logger
from app.detectors.face import FaceDetector
from app.detectors.object import ObjectDetector
from app.detectors.multi_person import MultiPersonDetector
from app.decision import DecisionEngine, ViolationRules

logger = get_logger(__name__)


class ProctoringPipeline:
    """
    Main proctoring pipeline that orchestrates all detection and decision modules
    """
    
    def __init__(
        self,
        face_detector: Optional[FaceDetector] = None,
        object_detector: Optional[ObjectDetector] = None,
        multi_person_detector: Optional[MultiPersonDetector] = None,
        decision_engine: Optional[DecisionEngine] = None,
        alert_callback: Optional[Callable] = None
    ):
        """
        Initialize proctoring pipeline
        
        Args:
            face_detector: Face detector instance
            object_detector: Object detector instance
            multi_person_detector: Multi-person detector instance
            decision_engine: Decision engine instance
            alert_callback: Callback function for alerts
        """
        # Initialize detectors
        self.face_detector = face_detector or FaceDetector()
        self.object_detector = object_detector or ObjectDetector()
        self.multi_person_detector = multi_person_detector or MultiPersonDetector(
            object_detector=self.object_detector
        )
        
        # Initialize decision engine
        self.decision_engine = decision_engine or DecisionEngine(
            alert_callback=alert_callback
        )
        
        logger.info("Proctoring pipeline initialized")
    
    def process_frame(self, frame: np.ndarray) -> Dict:
        """
        Process a single frame through the complete pipeline
        
        Args:
            frame: Input video frame (BGR format)
            
        Returns:
            Dictionary with complete processing results:
            {
                "timestamp": str,
                "face_detection": Dict,
                "object_detection": Dict,
                "multi_person": Dict,
                "decision": Dict,
                "violations": List[Dict],
                "alert_triggered": bool
            }
        """
        timestamp = datetime.now().isoformat()
        
        try:
            # Step 1: Face detection
            face_detections = self.face_detector.detect(frame)
            face_detected = len(face_detections) > 0
            face_count = len(face_detections)
            
            # Step 2: Object detection (persons, phones, etc.)
            object_detections = self.object_detector.detect(frame)
            phone_detected = len(object_detections.get("phones", [])) > 0
            person_detections = object_detections.get("persons", [])
            
            # Step 3: Multi-person detection
            multi_person_result = self.multi_person_detector.detect_persons(frame)
            person_count = multi_person_result["count"]
            
            # Step 4: Decision engine evaluation
            decision_result = self.decision_engine.evaluate(
                face_detected=face_detected,
                phone_detected=phone_detected,
                person_count=person_count,
                frame_data={
                    "face_detections": face_detections,
                    "object_detections": object_detections,
                    "multi_person_result": multi_person_result
                }
            )
            
            # Compile results
            result = {
                "timestamp": timestamp,
                "face_detection": {
                    "detected": face_detected,
                    "count": face_count,
                    "detections": face_detections
                },
                "object_detection": {
                    "persons": person_detections,
                    "phones": object_detections.get("phones", []),
                    "all": object_detections.get("all", [])
                },
                "multi_person": multi_person_result,
                "decision": decision_result,
                "violations": decision_result.get("violations", []),
                "alert_triggered": decision_result.get("alert_triggered", False)
            }
            
            return result
            
        except Exception as e:
            logger.error(f"Error processing frame: {e}", exc_info=True)
            return {
                "timestamp": timestamp,
                "error": str(e),
                "violations": [],
                "alert_triggered": False
            }
    
    def draw_results(self, frame: np.ndarray, result: Dict) -> np.ndarray:
        """
        Draw all detection results and violations on frame
        
        Args:
            frame: Input frame
            result: Processing result from process_frame()
            
        Returns:
            Frame with all visualizations
        """
        result_frame = frame.copy()
        
        # Draw face detections
        if result.get("face_detection", {}).get("detections"):
            result_frame = self.face_detector.draw_detections(
                result_frame,
                result["face_detection"]["detections"]
            )
        
        # Draw object detections
        if result.get("object_detection"):
            result_frame = self.object_detector.draw_detections(
                result_frame,
                result["object_detection"]
            )
        
        # Draw multi-person violations
        if result.get("multi_person", {}).get("violation"):
            result_frame = self.multi_person_detector.draw_detections(
                result_frame,
                result["multi_person"]
            )
        
        # Draw violation indicators
        if result.get("alert_triggered"):
            h, w = result_frame.shape[:2]
            cv2.rectangle(result_frame, (0, 0), (w, h), (0, 0, 255), 5)
            
            text = "ALERT: VIOLATION DETECTED!"
            text_size = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 1, 2)[0]
            text_x = (w - text_size[0]) // 2
            cv2.putText(result_frame, text, (text_x, 40),
                       cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
        
        # Draw status info
        self._draw_status_info(result_frame, result)
        
        return result_frame
    
    def _draw_status_info(self, frame: np.ndarray, result: Dict):
        """Draw status information on frame"""
        h = frame.shape[0]
        y_offset = h - 60
        
        # Face status
        face_status = f"Faces: {result.get('face_detection', {}).get('count', 0)}"
        cv2.putText(frame, face_status, (10, y_offset),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        
        # Person status
        person_status = f"Persons: {result.get('multi_person', {}).get('count', 0)}"
        cv2.putText(frame, person_status, (150, y_offset),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        
        # Phone status
        phone_count = len(result.get('object_detection', {}).get('phones', []))
        phone_status = f"Phones: {phone_count}"
        cv2.putText(frame, phone_status, (280, y_offset),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        
        # Violation count
        violation_count = len(result.get('violations', []))
        violation_status = f"Violations: {violation_count}"
        color = (0, 0, 255) if violation_count > 0 else (255, 255, 255)
        cv2.putText(frame, violation_status, (410, y_offset),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)

