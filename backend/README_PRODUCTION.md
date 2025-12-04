# Production-Ready AI Proctoring System

## Quick Start

```bash
# 1. Install dependencies
pip install -r requirements.txt

# 2. Configure environment
cp .env.example .env
# Edit .env with your settings

# 3. Run the system
python -m app.main
```

## Architecture

The system follows a modular architecture:

1. **Detectors**: Face, object, and multi-person detection
2. **Decision Engine**: Evaluates violations and triggers alerts
3. **Pipeline**: Orchestrates all components
4. **Utils**: Helper functions for image/video processing

## Key Modules

### Detection Modules

- **Face Detection**: `app/detectors/face/face_detector.py`
- **Object Detection**: `app/detectors/object/object_detector.py`
- **Multi-Person**: `app/detectors/multi_person/multi_person_detector.py`

### Decision Logic

- **Rules**: `app/decision/violation_rules.py`
- **Engine**: `app/decision/decision_engine.py`

### Main Pipeline

- **Pipeline**: `app/inference/proctoring_pipeline.py`
- **Entry Point**: `app/main.py`

## Configuration

See `app/core/config.py` for all configuration options.

## Models

- **SSD**: Fast, good for real-time (default)
- **ResNet**: More accurate, slower
- **Face**: OpenCV Haar Cascade (fast) or DNN (accurate)

## License

[Add your license here]

