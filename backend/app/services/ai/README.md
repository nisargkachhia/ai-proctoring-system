# AI Proctoring Services

This directory contains the AI/ML services for the proctoring system.

## Services

### 1. Object Detection Service (`object_detection.py`)
- Uses **COCO SSD model with PyTorch** (SSD300 with VGG16 backbone)
- Detects:
  - **Persons** (COCO class 1)
  - **Cell Phones** (COCO class 77)
- Features:
  - Configurable confidence threshold
  - Automatic device selection (CUDA/CPU)
  - Bounding box visualization

### 2. Face Detection Service (`face_detection.py`)
- Uses **OpenCV Haar Cascade** for face detection
- Fast and lightweight face detection
- Configurable detection parameters

### 3. Proctoring Service (`proctoring_service.py`)
- Main service that combines face and object detection
- Implements **alert logic**:
  - **Alert 1**: Phone detected + Face missing
  - **Alert 2**: Person count > 1
- Tracks alert history
- Provides visualization functions

### 4. Live Proctoring (`live_proctoring.py`)
- Real-time video processing
- Combines all services for live monitoring
- Features:
  - Live video capture
  - Real-time detection
  - Alert callbacks
  - FPS monitoring
  - Video display

## Usage

### Basic Usage

```python
from app.services.ai.live_proctoring import LiveProctoring

# Start live proctoring
proctoring = LiveProctoring(camera_source=0, display=True)
proctoring.start()
```

### Custom Alert Handler

```python
def my_alert_handler(result):
    print(f"Alert: {result['alerts']}")

proctoring = LiveProctoring(
    camera_source=0,
    alert_callback=my_alert_handler
)
proctoring.start()
```

### Run Example

```bash
cd backend
python example_live_proctoring.py
```

## Model Loading

The COCO SSD model is automatically downloaded on first use via PyTorch's model weights system. The model will be cached for subsequent runs.

## Requirements

- PyTorch >= 2.1.0
- torchvision >= 0.16.0
- OpenCV >= 4.8.0
- NumPy >= 1.24.0
- Pillow >= 10.1.0

