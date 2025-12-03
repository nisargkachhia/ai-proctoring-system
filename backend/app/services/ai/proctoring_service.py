"""
Proctoring Service - Main service that combines face detection, object detection,
and implements alert logic for proctoring violations
"""

import cv2
import numpy as np
from typing import Dict, List, Optional, Callable
import logging
from datetime import datetime

from .face_detection import FaceDetectionService
from .object_detection import ObjectDetectionService

logger = logging.getLogger(__name__)


class ProctoringService:
    """
    Main proctoring service that monitors video for violations:
    - Phone detected + face missing
    - Person count > 1
    """
    
    def __init__(self, 
                 face_detector: Optional[FaceDetectionService] = None,
                 object_detector: Optional[ObjectDetectionService] = None,
                 alert_callback: Optional[Callable] = None):
        """
        Initialize proctoring service
        
        Args:
            face_detector: Face detection service instance
            object_detector: Object detection service instance
            alert_callback: Callback function called when alert is raised
        """
        self.face_detector = face_detector or FaceDetectionService()
        self.object_detector = object_detector or ObjectDetectionService()
        self.alert_callback = alert_callback
        
        # Alert state tracking
        self.alert_history = []
        self.current_alerts = []
    
    def process_frame(self, frame: np.ndarray) -> Dict:
        """
        Process a single frame and check for violations
        
        Args:
            frame: Input video frame
            
        Returns:
            Dictionary containing detection results and alert status
        """
        try:
            # Detect faces
            faces = self.face_detector.detect_faces(frame)
            face_count = len(faces)
            face_detected = face_count > 0
            
            # Detect objects (persons and phones)
            object_detections = self.object_detector.detect_objects(frame)
            person_count = len(object_detections['persons'])
            phone_count = len(object_detections['phones'])
            phone_detected = phone_count > 0
            
            # Check for violations
            alerts = self._check_violations(
                face_detected=face_detected,
                phone_detected=phone_detected,
                person_count=person_count
            )
            
            # Create result dictionary
            result = {
                'timestamp': datetime.now().isoformat(),
                'faces': {
                    'count': face_count,
                    'detected': face_detected,
                    'detections': faces
                },
                'persons': {
                    'count': person_count,
                    'detections': object_detections['persons']
                },
                'phones': {
                    'count': phone_count,
                    'detected': phone_detected,
                    'detections': object_detections['phones']
                },
                'alerts': alerts,
                'violation_detected': len(alerts) > 0
            }
            
            # Trigger alert callback if violations detected
            if alerts and self.alert_callback:
                self.alert_callback(result)
            
            # Update alert history
            if alerts:
                self.alert_history.append({
                    'timestamp': result['timestamp'],
                    'alerts': alerts,
                    'details': result
                })
                self.current_alerts = alerts
            
            return result
            
        except Exception as e:
            logger.error(f"Error processing frame: {e}")
            return {
                'timestamp': datetime.now().isoformat(),
                'error': str(e),
                'violation_detected': False
            }
    
    def _check_violations(self, face_detected: bool, phone_detected: bool, 
                         person_count: int) -> List[Dict]:
        """
        Check for proctoring violations based on detection results
        
        Alert conditions:
        1. Phone detected + Face missing
        2. Person count > 1
        
        Args:
            face_detected: Whether a face was detected
            phone_detected: Whether a phone was detected
            person_count: Number of persons detected
            
        Returns:
            List of alert dictionaries
        """
        alerts = []
        
        # Alert 1: Phone detected + Face missing
        if phone_detected and not face_detected:
            alerts.append({
                'type': 'phone_without_face',
                'severity': 'high',
                'message': 'Phone detected but face is not visible',
                'details': {
                    'phone_detected': True,
                    'face_detected': False
                }
            })
            logger.warning("ALERT: Phone detected but face is missing")
        
        # Alert 2: Multiple persons detected
        if person_count > 1:
            alerts.append({
                'type': 'multiple_persons',
                'severity': 'high',
                'message': f'Multiple persons detected ({person_count} persons)',
                'details': {
                    'person_count': person_count
                }
            })
            logger.warning(f"ALERT: Multiple persons detected ({person_count})")
        
        return alerts
    
    def draw_detections(self, frame: np.ndarray, result: Dict) -> np.ndarray:
        """
        Draw all detections and alerts on frame
        
        Args:
            frame: Input frame
            result: Result dictionary from process_frame()
            
        Returns:
            Frame with drawn detections and alert indicators
        """
        result_frame = frame.copy()
        
        # Draw face detections
        if result.get('faces', {}).get('detections'):
            result_frame = self.face_detector.draw_faces(
                result_frame, 
                result['faces']['detections']
            )
        
        # Draw object detections
        object_detections = {
            'persons': result.get('persons', {}).get('detections', []),
            'phones': result.get('phones', {}).get('detections', [])
        }
        result_frame = self.object_detector.draw_detections(
            result_frame, 
            object_detections
        )
        
        # Draw alert indicators
        if result.get('violation_detected'):
            # Draw red border and alert text
            h, w = result_frame.shape[:2]
            cv2.rectangle(result_frame, (0, 0), (w, h), (0, 0, 255), 5)
            
            # Draw alert text at top
            alert_text = "VIOLATION DETECTED!"
            text_size = cv2.getTextSize(alert_text, cv2.FONT_HERSHEY_SIMPLEX, 1, 2)[0]
            text_x = (w - text_size[0]) // 2
            cv2.putText(result_frame, alert_text, (text_x, 40),
                       cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
            
            # Draw alert details
            y_offset = 80
            for alert in result.get('alerts', []):
                alert_msg = alert.get('message', '')
                cv2.putText(result_frame, alert_msg, (10, y_offset),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
                y_offset += 30
        
        # Draw status info
        status_y = h - 20
        face_status = f"Faces: {result.get('faces', {}).get('count', 0)}"
        person_status = f"Persons: {result.get('persons', {}).get('count', 0)}"
        phone_status = f"Phones: {result.get('phones', {}).get('count', 0)}"
        
        cv2.putText(result_frame, face_status, (10, status_y),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        cv2.putText(result_frame, person_status, (150, status_y),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        cv2.putText(result_frame, phone_status, (280, status_y),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        
        return result_frame
    
    def get_alert_history(self, limit: int = 100) -> List[Dict]:
        """
        Get alert history
        
        Args:
            limit: Maximum number of alerts to return
            
        Returns:
            List of alert records
        """
        return self.alert_history[-limit:]
    
    def clear_alert_history(self):
        """Clear alert history"""
        self.alert_history = []
        self.current_alerts = []

