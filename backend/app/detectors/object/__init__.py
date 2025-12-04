"""
Object detection module
"""

from .object_detector import ObjectDetector
from .ssd_detector import SSDDetector
from .resnet_detector import ResNetDetector

__all__ = ["ObjectDetector", "SSDDetector", "ResNetDetector"]

