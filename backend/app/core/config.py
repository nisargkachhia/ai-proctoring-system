"""
Configuration management for AI Proctoring System
Handles environment variables, model paths, and system settings
"""

import os
from pathlib import Path
from typing import Optional
from pydantic_settings import BaseSettings
from pydantic import Field


class Settings(BaseSettings):
    """Application settings"""
    
    # Application
    APP_NAME: str = "AI Proctoring System"
    APP_VERSION: str = "1.0.0"
    DEBUG: bool = Field(default=False, env="DEBUG")
    
    # Model Configuration
    MODEL_DIR: Path = Field(default=Path("data/models"), env="MODEL_DIR")
    USE_GPU: bool = Field(default=True, env="USE_GPU")
    CONFIDENCE_THRESHOLD: float = Field(default=0.5, env="CONFIDENCE_THRESHOLD")
    
    # Face Detection
    FACE_DETECTION_MODEL: str = Field(default="haarcascade_frontalface_default.xml", env="FACE_DETECTION_MODEL")
    FACE_CONFIDENCE_THRESHOLD: float = Field(default=0.5, env="FACE_CONFIDENCE_THRESHOLD")
    FACE_MIN_SIZE: tuple = (30, 30)
    
    # Object Detection (SSD)
    SSD_MODEL_TYPE: str = Field(default="ssd300_vgg16", env="SSD_MODEL_TYPE")  # ssd300_vgg16 or ssdlite320_mobilenet
    SSD_CONFIDENCE_THRESHOLD: float = Field(default=0.5, env="SSD_CONFIDENCE_THRESHOLD")
    
    # Object Detection (ResNet)
    RESNET_MODEL_TYPE: str = Field(default="fasterrcnn_resnet50_fpn", env="RESNET_MODEL_TYPE")
    RESNET_CONFIDENCE_THRESHOLD: float = Field(default=0.5, env="RESNET_CONFIDENCE_THRESHOLD")
    
    # Multi-Person Detection
    MAX_ALLOWED_PERSONS: int = Field(default=1, env="MAX_ALLOWED_PERSONS")
    PERSON_TRACKING_ENABLED: bool = Field(default=True, env="PERSON_TRACKING_ENABLED")
    
    # Video Processing
    VIDEO_SOURCE: int = Field(default=0, env="VIDEO_SOURCE")
    VIDEO_WIDTH: int = Field(default=640, env="VIDEO_WIDTH")
    VIDEO_HEIGHT: int = Field(default=480, env="VIDEO_HEIGHT")
    VIDEO_FPS: int = Field(default=30, env="VIDEO_FPS")
    FRAME_SKIP: int = Field(default=1, env="FRAME_SKIP")  # Process every Nth frame
    
    # Decision Logic
    ALERT_COOLDOWN_SECONDS: int = Field(default=5, env="ALERT_COOLDOWN_SECONDS")
    VIOLATION_THRESHOLD: int = Field(default=3, env="VIOLATION_THRESHOLD")  # Consecutive violations before alert
    
    # Logging
    LOG_LEVEL: str = Field(default="INFO", env="LOG_LEVEL")
    LOG_FILE: Path = Field(default=Path("logs/app.log"), env="LOG_FILE")
    LOG_FORMAT: str = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    
    # Paths
    BASE_DIR: Path = Path(__file__).parent.parent.parent
    DATA_DIR: Path = BASE_DIR / "data"
    LOGS_DIR: Path = BASE_DIR / "logs"
    
    class Config:
        env_file = ".env"
        env_file_encoding = "utf-8"
        case_sensitive = True


# Global settings instance
settings = Settings()


def get_settings() -> Settings:
    """Get application settings"""
    return settings

