"""
Decision Engine Module
Main decision logic for proctoring violations
"""

from typing import Dict, List, Optional, Callable
from datetime import datetime, timedelta
from collections import deque
import logging

from app.core.config import settings
from app.core.logging_config import get_logger
from .violation_rules import ViolationRules

logger = get_logger(__name__)


class DecisionEngine:
    """
    Decision engine that evaluates detection results and triggers alerts
    Implements violation rules and alert cooldown logic
    """
    
    def __init__(
        self,
        violation_rules: Optional[ViolationRules] = None,
        alert_callback: Optional[Callable] = None,
        cooldown_seconds: Optional[int] = None
    ):
        """
        Initialize decision engine
        
        Args:
            violation_rules: Violation rules instance
            alert_callback: Callback function for alerts
            cooldown_seconds: Alert cooldown period in seconds
        """
        self.violation_rules = violation_rules or ViolationRules()
        self.alert_callback = alert_callback
        self.cooldown_seconds = cooldown_seconds or settings.ALERT_COOLDOWN_SECONDS
        
        # Alert tracking
        self.alert_history = []
        self.last_alert_time = {}
        self.violation_buffer = deque(maxlen=settings.VIOLATION_THRESHOLD)
        
        logger.info("Decision engine initialized")
    
    def evaluate(
        self,
        face_detected: bool,
        phone_detected: bool,
        person_count: int,
        frame_data: Optional[Dict] = None
    ) -> Dict:
        """
        Evaluate detection results and determine violations
        
        Args:
            face_detected: Whether face was detected
            phone_detected: Whether phone was detected
            person_count: Number of persons detected
            frame_data: Additional frame data
            
        Returns:
            Dictionary with evaluation results:
            {
                "violations": List[Dict],
                "alert_triggered": bool,
                "timestamp": str
            }
        """
        violations = []
        
        # Check phone + no face violation
        if self.violation_rules.check_phone_no_face(phone_detected, face_detected):
            violations.append({
                "type": "phone_without_face",
                "severity": "high",
                "message": "Phone detected but face is not visible",
                "details": {
                    "phone_detected": phone_detected,
                    "face_detected": face_detected
                }
            })
        
        # Check multiple persons violation
        if self.violation_rules.check_multiple_persons(person_count):
            violations.append({
                "type": "multiple_persons",
                "severity": "high",
                "message": f"Multiple persons detected: {person_count}",
                "details": {
                    "person_count": person_count,
                    "max_allowed": self.violation_rules.max_allowed_persons
                }
            })
        
        # Add to violation buffer
        has_violation = len(violations) > 0
        self.violation_buffer.append(has_violation)
        
        # Check if we should trigger alert (threshold-based)
        alert_triggered = False
        if has_violation:
            consecutive_violations = sum(self.violation_buffer)
            if consecutive_violations >= settings.VIOLATION_THRESHOLD:
                alert_triggered = self._should_trigger_alert(violations)
        
        result = {
            "violations": violations,
            "alert_triggered": alert_triggered,
            "timestamp": datetime.now().isoformat(),
            "frame_data": frame_data
        }
        
        # Trigger alert callback if needed
        if alert_triggered and self.alert_callback:
            self.alert_callback(result)
            self._record_alert(result)
        
        return result
    
    def _should_trigger_alert(self, violations: List[Dict]) -> bool:
        """
        Determine if alert should be triggered (with cooldown)
        
        Args:
            violations: List of violations
            
        Returns:
            True if alert should be triggered
        """
        now = datetime.now()
        
        # Check cooldown for each violation type
        for violation in violations:
            violation_type = violation["type"]
            last_alert = self.last_alert_time.get(violation_type)
            
            if last_alert is None:
                return True
            
            time_since_last = (now - last_alert).total_seconds()
            if time_since_last >= self.cooldown_seconds:
                return True
        
        return False
    
    def _record_alert(self, result: Dict):
        """Record alert in history"""
        self.alert_history.append(result)
        
        # Update last alert time for each violation type
        for violation in result["violations"]:
            violation_type = violation["type"]
            self.last_alert_time[violation_type] = datetime.now()
        
        # Keep only recent history
        if len(self.alert_history) > 1000:
            self.alert_history = self.alert_history[-1000:]
    
    def get_alert_history(self, limit: int = 100) -> List[Dict]:
        """
        Get alert history
        
        Args:
            limit: Maximum number of alerts to return
            
        Returns:
            List of alert records
        """
        return self.alert_history[-limit:]
    
    def clear_history(self):
        """Clear alert history"""
        self.alert_history = []
        self.last_alert_time = {}
        self.violation_buffer.clear()
        logger.info("Alert history cleared")

