"""
Live Proctoring - Real-time video processing with OpenCV and SSD model
Main entry point for live proctoring system
"""

import cv2
import logging
from typing import Optional, Callable
import time

from .proctoring_service import ProctoringService
from .face_detection import FaceDetectionService
from .object_detection import ObjectDetectionService
from ..video.video_capture import VideoCaptureService

logger = logging.getLogger(__name__)


class LiveProctoring:
    """
    Main class for live proctoring with real-time detection
    Combines video capture, face detection, and object detection
    """
    
    def __init__(self, 
                 camera_source: int = 0,
                 alert_callback: Optional[Callable] = None,
                 display: bool = True):
        """
        Initialize live proctoring system
        
        Args:
            camera_source: Camera source index (default 0)
            alert_callback: Callback function for alerts
            display: Whether to display video window
        """
        self.camera_source = camera_source
        self.display = display
        self.is_running = False
        
        # Initialize services
        logger.info("Initializing proctoring services...")
        self.face_detector = FaceDetectionService()
        self.object_detector = ObjectDetectionService()
        self.proctoring_service = ProctoringService(
            face_detector=self.face_detector,
            object_detector=self.object_detector,
            alert_callback=alert_callback
        )
        self.video_capture = VideoCaptureService(source=camera_source)
        
        logger.info("Proctoring services initialized")
    
    def start(self):
        """Start live proctoring"""
        if self.is_running:
            logger.warning("Proctoring is already running")
            return
        
        if not self.video_capture.start():
            logger.error("Failed to start video capture")
            return
        
        self.is_running = True
        logger.info("Live proctoring started")
        
        try:
            self._run_detection_loop()
        except KeyboardInterrupt:
            logger.info("Stopped by user")
        finally:
            self.stop()
    
    def _run_detection_loop(self):
        """Main detection loop"""
        frame_count = 0
        fps_start_time = time.time()
        
        while self.is_running:
            # Read frame
            frame = self.video_capture.read_frame()
            if frame is None:
                logger.warning("Failed to read frame")
                break
            
            # Process frame
            result = self.proctoring_service.process_frame(frame)
            
            # Draw detections
            annotated_frame = self.proctoring_service.draw_detections(frame, result)
            
            # Calculate FPS
            frame_count += 1
            if frame_count % 30 == 0:
                fps = 30 / (time.time() - fps_start_time)
                fps_start_time = time.time()
                logger.info(f"Processing FPS: {fps:.2f}")
            
            # Display frame
            if self.display:
                cv2.imshow('AI Proctoring System', annotated_frame)
                
                # Break on 'q' key press
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    logger.info("Stopped by user (q key)")
                    break
    
    def stop(self):
        """Stop live proctoring"""
        self.is_running = False
        self.video_capture.stop()
        if self.display:
            cv2.destroyAllWindows()
        logger.info("Live proctoring stopped")
    
    def get_statistics(self) -> dict:
        """
        Get proctoring statistics
        
        Returns:
            Dictionary with statistics
        """
        alert_history = self.proctoring_service.get_alert_history()
        
        return {
            'total_alerts': len(alert_history),
            'current_alerts': len(self.proctoring_service.current_alerts),
            'alert_history': alert_history[-10:]  # Last 10 alerts
        }


def alert_handler(result: dict):
    """
    Example alert callback function
    
    Args:
        result: Detection result dictionary with alerts
    """
    logger.warning("=" * 50)
    logger.warning("PROCTORING ALERT TRIGGERED!")
    logger.warning(f"Timestamp: {result.get('timestamp')}")
    
    for alert in result.get('alerts', []):
        logger.warning(f"Alert Type: {alert.get('type')}")
        logger.warning(f"Severity: {alert.get('severity')}")
        logger.warning(f"Message: {alert.get('message')}")
    
    logger.warning("=" * 50)


def main():
    """Main function to run live proctoring"""
    # Configure logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    # Create and start live proctoring
    proctoring = LiveProctoring(
        camera_source=0,
        alert_callback=alert_handler,
        display=True
    )
    
    try:
        proctoring.start()
    except Exception as e:
        logger.error(f"Error in main: {e}")
    finally:
        proctoring.stop()


if __name__ == "__main__":
    main()

