"""
Video Capture Service for live video processing
"""

import cv2
import numpy as np
from typing import Optional, Callable, Generator
import logging
import threading
import time

logger = logging.getLogger(__name__)


class VideoCaptureService:
    """Service for capturing and processing live video streams"""
    
    def __init__(self, source: int = 0, width: int = 640, height: int = 480, fps: int = 30):
        """
        Initialize video capture service
        
        Args:
            source: Video source (0 for default camera, or path to video file)
            width: Frame width
            height: Frame height
            fps: Frames per second
        """
        self.source = source
        self.width = width
        self.height = height
        self.fps = fps
        self.cap = None
        self.is_running = False
        self.current_frame = None
        self.frame_lock = threading.Lock()
    
    def start(self) -> bool:
        """
        Start video capture
        
        Returns:
            True if successful, False otherwise
        """
        try:
            self.cap = cv2.VideoCapture(self.source)
            
            if not self.cap.isOpened():
                logger.error(f"Failed to open video source: {self.source}")
                return False
            
            # Set camera properties
            self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.width)
            self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.height)
            self.cap.set(cv2.CAP_PROP_FPS, self.fps)
            
            self.is_running = True
            logger.info(f"Video capture started from source: {self.source}")
            return True
            
        except Exception as e:
            logger.error(f"Error starting video capture: {e}")
            return False
    
    def read_frame(self) -> Optional[np.ndarray]:
        """
        Read a single frame from the video source
        
        Returns:
            Frame as numpy array, or None if failed
        """
        if not self.cap or not self.cap.isOpened():
            return None
        
        ret, frame = self.cap.read()
        
        if ret:
            with self.frame_lock:
                self.current_frame = frame.copy()
            return frame
        
        return None
    
    def get_current_frame(self) -> Optional[np.ndarray]:
        """
        Get the most recently captured frame
        
        Returns:
            Current frame as numpy array, or None
        """
        with self.frame_lock:
            return self.current_frame.copy() if self.current_frame is not None else None
    
    def stream_frames(self) -> Generator[np.ndarray, None, None]:
        """
        Generator that yields frames continuously
        
        Yields:
            Video frames as numpy arrays
        """
        while self.is_running:
            frame = self.read_frame()
            if frame is not None:
                yield frame
            else:
                logger.warning("Failed to read frame")
                break
            time.sleep(1.0 / self.fps)
    
    def stop(self):
        """Stop video capture"""
        self.is_running = False
        if self.cap:
            self.cap.release()
            logger.info("Video capture stopped")
    
    def __enter__(self):
        """Context manager entry"""
        self.start()
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit"""
        self.stop()

