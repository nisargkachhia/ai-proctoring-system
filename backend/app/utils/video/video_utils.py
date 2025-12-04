"""
Video utility functions for capture and processing
"""

import cv2
import numpy as np
from typing import Optional, Callable, Generator
import logging
import threading
import time

from app.core.config import settings
from app.core.logging_config import get_logger

logger = get_logger(__name__)


class VideoCapture:
    """
    Enhanced video capture with frame buffering and processing
    """
    
    def __init__(
        self,
        source: int = 0,
        width: Optional[int] = None,
        height: Optional[int] = None,
        fps: Optional[int] = None
    ):
        """
        Initialize video capture
        
        Args:
            source: Video source (camera index or file path)
            width: Frame width
            height: Frame height
            fps: Frames per second
        """
        self.source = source
        self.width = width or settings.VIDEO_WIDTH
        self.height = height or settings.VIDEO_HEIGHT
        self.fps = fps or settings.VIDEO_FPS
        
        self.cap = None
        self.is_running = False
        
    def start(self) -> bool:
        """Start video capture"""
        try:
            self.cap = cv2.VideoCapture(self.source)
            
            if not self.cap.isOpened():
                logger.error(f"Failed to open video source: {self.source}")
                return False
            
            # Set properties
            self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.width)
            self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.height)
            self.cap.set(cv2.CAP_PROP_FPS, self.fps)
            
            self.is_running = True
            logger.info(f"Video capture started: {self.width}x{self.height} @ {self.fps}fps")
            return True
            
        except Exception as e:
            logger.error(f"Error starting video capture: {e}")
            return False
    
    def read(self) -> Optional[np.ndarray]:
        """Read a frame"""
        if not self.cap or not self.cap.isOpened():
            return None
        
        ret, frame = self.cap.read()
        return frame if ret else None
    
    def stop(self):
        """Stop video capture"""
        self.is_running = False
        if self.cap:
            self.cap.release()
            logger.info("Video capture stopped")
    
    def __enter__(self):
        self.start()
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        self.stop()


class FrameProcessor:
    """
    Frame processor with frame skipping and processing callbacks
    """
    
    def __init__(
        self,
        frame_skip: int = 1,
        process_callback: Optional[Callable] = None
    ):
        """
        Initialize frame processor
        
        Args:
            frame_skip: Process every Nth frame
            process_callback: Callback function for processing
        """
        self.frame_skip = frame_skip
        self.process_callback = process_callback
        self.frame_count = 0
    
    def process(self, frame: np.ndarray) -> Optional[np.ndarray]:
        """
        Process frame with skipping logic
        
        Args:
            frame: Input frame
            
        Returns:
            Processed frame or None if skipped
        """
        self.frame_count += 1
        
        if self.frame_count % self.frame_skip != 0:
            return None
        
        if self.process_callback:
            return self.process_callback(frame)
        
        return frame
    
    def reset(self):
        """Reset frame counter"""
        self.frame_count = 0

