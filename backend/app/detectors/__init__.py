"""
Detection modules for AI Proctoring System
"""

from .face.face_detector import FaceDetector
from .object.object_detector import ObjectDetector
from .multi_person.multi_person_detector import MultiPersonDetector

__all__ = [
    "FaceDetector",
    "ObjectDetector",
    "MultiPersonDetector",
]

