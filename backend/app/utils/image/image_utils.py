"""
Image utility functions for preprocessing and manipulation
"""

import cv2
import numpy as np
from typing import Tuple, Optional

from app.core.config import settings


def preprocess_frame(
    frame: np.ndarray,
    target_size: Optional[Tuple[int, int]] = None,
    normalize: bool = False
) -> np.ndarray:
    """
    Preprocess frame for model inference
    
    Args:
        frame: Input frame
        target_size: Target size (width, height)
        normalize: Whether to normalize pixel values
        
    Returns:
        Preprocessed frame
    """
    result = frame.copy()
    
    # Resize if needed
    if target_size:
        result = resize_frame(result, target_size)
    
    # Normalize if needed
    if normalize:
        result = normalize_frame(result)
    
    return result


def resize_frame(frame: np.ndarray, size: Tuple[int, int]) -> np.ndarray:
    """
    Resize frame to target size
    
    Args:
        frame: Input frame
        size: Target size (width, height)
        
    Returns:
        Resized frame
    """
    return cv2.resize(frame, size, interpolation=cv2.INTER_LINEAR)


def normalize_frame(frame: np.ndarray) -> np.ndarray:
    """
    Normalize frame pixel values to [0, 1]
    
    Args:
        frame: Input frame
        
    Returns:
        Normalized frame
    """
    return frame.astype(np.float32) / 255.0


def enhance_frame(frame: np.ndarray) -> np.ndarray:
    """
    Enhance frame quality (contrast, brightness)
    
    Args:
        frame: Input frame
        
    Returns:
        Enhanced frame
    """
    # Convert to LAB color space
    lab = cv2.cvtColor(frame, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)
    
    # Apply CLAHE to L channel
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    l = clahe.apply(l)
    
    # Merge channels and convert back
    enhanced = cv2.merge([l, a, b])
    enhanced = cv2.cvtColor(enhanced, cv2.COLOR_LAB2BGR)
    
    return enhanced

