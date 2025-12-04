"""
Violation Rules Module
Defines rules for proctoring violations
"""

from typing import Dict, Optional
import logging

from app.core.config import settings
from app.core.logging_config import get_logger

logger = get_logger(__name__)


class ViolationRules:
    """
    Defines and evaluates violation rules for proctoring
    """
    
    def __init__(self, max_allowed_persons: Optional[int] = None):
        """
        Initialize violation rules
        
        Args:
            max_allowed_persons: Maximum allowed persons (default from settings)
        """
        self.max_allowed_persons = max_allowed_persons or settings.MAX_ALLOWED_PERSONS
        logger.info(f"Violation rules initialized (max_persons={self.max_allowed_persons})")
    
    def check_phone_no_face(self, phone_detected: bool, face_detected: bool) -> bool:
        """
        Check if phone is detected without face
        
        Rule: Phone detected + Face missing = Violation
        
        Args:
            phone_detected: Whether phone was detected
            face_detected: Whether face was detected
            
        Returns:
            True if violation detected
        """
        return phone_detected and not face_detected
    
    def check_multiple_persons(self, person_count: int) -> bool:
        """
        Check if multiple persons are detected
        
        Rule: Person count > max_allowed = Violation
        
        Args:
            person_count: Number of persons detected
            
        Returns:
            True if violation detected
        """
        return person_count > self.max_allowed_persons
    
    def check_face_missing(self, face_detected: bool, required: bool = True) -> bool:
        """
        Check if face is missing when required
        
        Args:
            face_detected: Whether face was detected
            required: Whether face is required
            
        Returns:
            True if violation (face missing when required)
        """
        return required and not face_detected
    
    def evaluate_all(
        self,
        phone_detected: bool,
        face_detected: bool,
        person_count: int
    ) -> Dict[str, bool]:
        """
        Evaluate all violation rules
        
        Args:
            phone_detected: Whether phone was detected
            face_detected: Whether face was detected
            person_count: Number of persons detected
            
        Returns:
            Dictionary with all rule evaluations
        """
        return {
            "phone_without_face": self.check_phone_no_face(phone_detected, face_detected),
            "multiple_persons": self.check_multiple_persons(person_count),
            "face_missing": self.check_face_missing(face_detected)
        }

