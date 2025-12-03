# Usage Guide - AI Proctoring System

## Quick Start

### 1. Install Dependencies

```bash
cd backend
pip install -r requirements.txt
```

**Note**: The first run will download the COCO SSD model (~100MB) automatically.

### 2. Run Live Proctoring

```bash
python example_live_proctoring.py
```

Or use the module directly:

```bash
python -m app.services.ai.live_proctoring
```

### 3. Controls

- Press **'q'** to quit
- The system will automatically detect violations and log alerts

## Detection Features

### Real-time Detection

The system performs real-time detection of:

1. **Faces** - Using OpenCV Haar Cascade
2. **Persons** - Using COCO SSD model (PyTorch)
3. **Cell Phones** - Using COCO SSD model (PyTorch)

### Alert Logic

The system raises alerts when:

1. **Phone detected + Face missing**: When a phone is detected but no face is visible
2. **Multiple persons**: When more than 1 person is detected in the frame

## Code Examples

### Basic Usage

```python
from app.services.ai.live_proctoring import LiveProctoring

# Start proctoring
proctoring = LiveProctoring(camera_source=0, display=True)
proctoring.start()
```

### Custom Alert Handler

```python
def my_alert_handler(result):
    """Custom function to handle alerts"""
    for alert in result['alerts']:
        print(f"ALERT: {alert['message']}")
        # Send email, save to database, etc.

proctoring = LiveProctoring(
    camera_source=0,
    alert_callback=my_alert_handler,
    display=True
)
proctoring.start()
```

### Process Single Frame

```python
from app.services.ai.proctoring_service import ProctoringService
from app.services.ai.face_detection import FaceDetectionService
from app.services.ai.object_detection import ObjectDetectionService
import cv2

# Initialize services
face_detector = FaceDetectionService()
object_detector = ObjectDetectionService()
proctoring = ProctoringService(face_detector, object_detector)

# Read frame
frame = cv2.imread('test_image.jpg')

# Process frame
result = proctoring.process_frame(frame)

# Check for violations
if result['violation_detected']:
    print("Violation detected!")
    for alert in result['alerts']:
        print(f"  - {alert['message']}")
```

### Use Different Camera

```python
# Use camera index 1 instead of default (0)
proctoring = LiveProctoring(camera_source=1, display=True)
proctoring.start()
```

## Model Information

### COCO SSD Model

- **Model**: SSD300 with VGG16 backbone
- **Dataset**: COCO (Common Objects in Context)
- **Classes Detected**:
  - Person (class 1)
  - Cell Phone (class 77)
- **Framework**: PyTorch / torchvision
- **Device**: Automatically uses CUDA if available, otherwise CPU

### Face Detection

- **Method**: Haar Cascade Classifier
- **Library**: OpenCV
- **Model**: Built-in `haarcascade_frontalface_default.xml`

## Performance Tips

1. **GPU Acceleration**: Install CUDA-enabled PyTorch for faster inference
2. **Frame Skipping**: Process every Nth frame for lower CPU usage
3. **Resolution**: Lower camera resolution for faster processing
4. **Confidence Threshold**: Adjust in `ObjectDetectionService` initialization

## Troubleshooting

### Camera Not Found

```python
# Try different camera indices
for i in range(5):
    try:
        proctoring = LiveProctoring(camera_source=i, display=True)
        proctoring.start()
        break
    except:
        continue
```

### Model Download Issues

The model downloads automatically on first use. If it fails:
- Check internet connection
- Verify PyTorch installation: `python -c "import torch; print(torch.__version__)"`
- Manually download weights if needed

### Low FPS

- Reduce camera resolution
- Use GPU if available
- Process every 2nd or 3rd frame
- Lower confidence threshold

## API Reference

See individual service files for detailed API documentation:
- `app/services/ai/object_detection.py`
- `app/services/ai/face_detection.py`
- `app/services/ai/proctoring_service.py`
- `app/services/ai/live_proctoring.py`

