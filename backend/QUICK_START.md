# Quick Start Guide - Production AI Proctoring System

## 🚀 Getting Started

### 1. Installation

```bash
cd backend
pip install -r requirements.txt
```

### 2. Configuration

```bash
cp .env.example .env
# Edit .env with your settings
```

### 3. Run

```bash
python -m app.main
```

## 📁 Module Overview

### Detection Modules

#### Face Detection
```python
from app.detectors.face import FaceDetector

detector = FaceDetector(method="haar")
faces = detector.detect(frame)
```

#### Object Detection (SSD/ResNet)
```python
from app.detectors.object import ObjectDetector

# SSD (fast)
detector = ObjectDetector(model_type="ssd")
results = detector.detect(frame)

# ResNet (accurate)
detector = ObjectDetector(model_type="resnet")
results = detector.detect(frame)
```

#### Multi-Person Detection
```python
from app.detectors.multi_person import MultiPersonDetector

detector = MultiPersonDetector()
result = detector.detect_persons(frame)
```

### Decision Logic

```python
from app.decision import DecisionEngine, ViolationRules

rules = ViolationRules(max_allowed_persons=1)
engine = DecisionEngine(violation_rules=rules)

result = engine.evaluate(
    face_detected=True,
    phone_detected=False,
    person_count=1
)
```

### Complete Pipeline

```python
from app.inference import ProctoringPipeline

pipeline = ProctoringPipeline()
result = pipeline.process_frame(frame)
annotated = pipeline.draw_results(frame, result)
```

## 🎯 Key Features

- ✅ **Face Detection**: OpenCV Haar Cascade or DNN
- ✅ **Object Detection**: SSD (fast) or ResNet (accurate)
- ✅ **Multi-Person Detection**: Person counting and tracking
- ✅ **Decision Engine**: Violation detection with alert logic
- ✅ **Production Ready**: Logging, error handling, configuration

## 📊 Detection Capabilities

- **Faces**: Real-time face detection
- **Persons**: Person detection and counting
- **Phones**: Cell phone detection
- **Violations**: 
  - Phone + No Face
  - Multiple Persons

## ⚙️ Configuration

All settings in `app/core/config.py`:

- Model selection (SSD/ResNet)
- Confidence thresholds
- Video settings
- Alert cooldown
- Violation thresholds

## 📝 Example Usage

See `example_live_proctoring.py` for a complete example.

