# Production-Ready AI Proctoring System Structure

## Overview

This is a production-ready folder structure for an AI Proctoring System using Python, OpenCV, and pretrained SSD/ResNet models. The system is modular, scalable, and follows best practices for production deployment.

## Folder Structure

```
backend/
├── app/
│   ├── core/                      # Core configuration and utilities
│   │   ├── config.py              # Application settings and configuration
│   │   └── logging_config.py      # Logging setup and configuration
│   │
│   ├── detectors/                 # Detection modules
│   │   ├── face/                  # Face detection
│   │   │   ├── __init__.py
│   │   │   └── face_detector.py   # Face detection using OpenCV/DNN
│   │   │
│   │   ├── object/                # Object detection (SSD/ResNet)
│   │   │   ├── __init__.py
│   │   │   ├── object_detector.py    # Unified object detection interface
│   │   │   ├── ssd_detector.py       # SSD model implementation
│   │   │   └── resnet_detector.py    # ResNet model implementation
│   │   │
│   │   └── multi_person/          # Multi-person detection
│   │       ├── __init__.py
│   │       └── multi_person_detector.py  # Person counting and tracking
│   │
│   ├── decision/                  # Decision logic
│   │   ├── __init__.py
│   │   ├── decision_engine.py    # Main decision engine
│   │   └── violation_rules.py   # Violation rule definitions
│   │
│   ├── inference/                 # Inference pipeline
│   │   ├── __init__.py
│   │   └── proctoring_pipeline.py  # Complete proctoring pipeline
│   │
│   ├── utils/                     # Utility functions
│   │   ├── image/                 # Image utilities
│   │   │   ├── __init__.py
│   │   │   └── image_utils.py    # Image preprocessing functions
│   │   │
│   │   └── video/                 # Video utilities
│   │       ├── __init__.py
│   │       └── video_utils.py    # Video capture and processing
│   │
│   └── main.py                    # Main application entry point
│
├── config/                        # Configuration files
├── logs/                          # Application logs
├── data/                          # Data and models
│   ├── models/                    # ML model files
│   └── samples/                   # Sample data
├── tests/                         # Test suite
│   ├── unit/                      # Unit tests
│   ├── integration/                # Integration tests
│   └── fixtures/                  # Test fixtures
│
├── requirements.txt               # Python dependencies
├── .env.example                   # Environment variables template
└── PRODUCTION_STRUCTURE.md        # This file
```

## Module Descriptions

### 1. Core Module (`app/core/`)

**Purpose**: Application-wide configuration and utilities

- **`config.py`**: Centralized configuration using Pydantic Settings
  - Model paths and settings
  - Detection thresholds
  - Video processing parameters
  - Logging configuration

- **`logging_config.py`**: Logging setup
  - Console and file handlers
  - Log rotation
  - Configurable log levels

### 2. Detectors Module (`app/detectors/`)

**Purpose**: All detection functionality

#### Face Detection (`detectors/face/`)
- **`face_detector.py`**: Face detection using:
  - OpenCV Haar Cascade (fast, lightweight)
  - DNN-based models (more accurate)
- Features:
  - Configurable detection methods
  - Bounding box visualization
  - Face counting

#### Object Detection (`detectors/object/`)
- **`object_detector.py`**: Unified interface for object detection
- **`ssd_detector.py`**: SSD model implementation
  - SSD300 with VGG16 backbone
  - SSDLite320 with MobileNet
  - Detects: persons, phones, laptops
- **`resnet_detector.py`**: ResNet model implementation
  - Faster R-CNN with ResNet50-FPN
  - More accurate but slower than SSD

#### Multi-Person Detection (`detectors/multi_person/`)
- **`multi_person_detector.py`**: Person counting and tracking
  - Uses object detector for person detection
  - Optional person tracking
  - Violation detection (multiple persons)

### 3. Decision Module (`app/decision/`)

**Purpose**: Violation detection and alert logic

- **`violation_rules.py`**: Rule definitions
  - Phone detected + Face missing
  - Multiple persons detected
  - Face missing when required

- **`decision_engine.py`**: Main decision engine
  - Evaluates detection results
  - Triggers alerts based on rules
  - Alert cooldown management
  - Violation threshold logic

### 4. Inference Module (`app/inference/`)

**Purpose**: Complete proctoring pipeline

- **`proctoring_pipeline.py`**: Orchestrates all modules
  - Face detection
  - Object detection
  - Multi-person detection
  - Decision evaluation
  - Result visualization

### 5. Utils Module (`app/utils/`)

**Purpose**: Utility functions

- **`image/`**: Image preprocessing and manipulation
- **`video/`**: Video capture and frame processing

## Usage

### Basic Usage

```python
from app.main import main

if __name__ == "__main__":
    main()
```

### Custom Usage

```python
from app.inference import ProctoringPipeline
from app.utils.video import VideoCapture
import cv2

# Initialize pipeline
pipeline = ProctoringPipeline()

# Start video capture
video = VideoCapture(source=0)
video.start()

# Process frames
while True:
    frame = video.read()
    if frame is None:
        break
    
    result = pipeline.process_frame(frame)
    annotated = pipeline.draw_results(frame, result)
    
    cv2.imshow('Proctoring', annotated)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

video.stop()
```

## Configuration

All configuration is managed through `app/core/config.py` and environment variables:

```bash
# .env file
VIDEO_SOURCE=0
VIDEO_WIDTH=640
VIDEO_HEIGHT=480
CONFIDENCE_THRESHOLD=0.5
MAX_ALLOWED_PERSONS=1
LOG_LEVEL=INFO
```

## Model Loading

Models are automatically downloaded on first use:
- **SSD Models**: Via torchvision (COCO pretrained)
- **ResNet Models**: Via torchvision (COCO pretrained)
- **Face Detection**: OpenCV built-in Haar Cascade

## Features

✅ **Modular Design**: Each module is independent and testable
✅ **Production Ready**: Error handling, logging, configuration management
✅ **Multiple Models**: Support for SSD and ResNet models
✅ **Real-time Processing**: Optimized for live video streams
✅ **Configurable**: All parameters configurable via settings
✅ **Extensible**: Easy to add new detection methods or rules

## Testing

```bash
# Run tests
pytest tests/

# Run with coverage
pytest --cov=app tests/
```

## Deployment

1. Install dependencies: `pip install -r requirements.txt`
2. Configure environment: Copy `.env.example` to `.env`
3. Run: `python -m app.main`

## Performance Tips

- Use GPU for faster inference (CUDA-enabled PyTorch)
- Adjust `FRAME_SKIP` to process fewer frames
- Use SSD models for speed, ResNet for accuracy
- Lower video resolution for better performance

