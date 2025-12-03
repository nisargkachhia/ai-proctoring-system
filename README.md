# AI Proctoring System

An AI-based proctoring system with Python backend and React frontend for automated exam monitoring.

## Project Structure

```
ai-proctoring-system/
├── backend/                 # Python FastAPI backend
│   ├── app/
│   │   ├── api/           # API endpoints
│   │   │   └── v1/        # API version 1
│   │   ├── core/          # Core configuration
│   │   ├── models/        # Database models
│   │   ├── services/      # Business logic
│   │   │   ├── ai/        # AI/ML services
│   │   │   ├── video/     # Video processing
│   │   │   └── audio/     # Audio processing
│   │   ├── utils/         # Utility functions
│   │   └── middleware/    # Custom middleware
│   ├── tests/             # Test suite
│   │   ├── unit/          # Unit tests
│   │   ├── integration/   # Integration tests
│   │   └── fixtures/      # Test fixtures
│   ├── config/            # Configuration files
│   ├── logs/              # Application logs
│   ├── static/            # Static files
│   ├── uploads/           # Uploaded files
│   ├── requirements.txt   # Python dependencies
│   └── .env.example       # Environment variables template
│
├── frontend/              # React frontend (optional)
│   ├── src/
│   │   ├── components/    # React components
│   │   │   ├── common/    # Common/reusable components
│   │   │   └── proctoring/# Proctoring-specific components
│   │   ├── pages/         # Page components
│   │   ├── services/      # API services
│   │   ├── hooks/         # Custom React hooks
│   │   ├── utils/         # Utility functions
│   │   ├── context/       # React context providers
│   │   └── assets/        # Static assets
│   │       ├── images/
│   │       └── icons/
│   ├── public/            # Public assets
│   ├── package.json       # Node dependencies
│   └── vite.config.js     # Vite configuration
│
├── docs/                  # Documentation
├── scripts/               # Utility scripts
├── data/                  # Data files
│   ├── models/            # ML model files
│   └── samples/           # Sample data
└── README.md              # This file
```

## Features

- **AI-Powered Monitoring**: Face detection, eye tracking, and behavior analysis
- **Video Processing**: Real-time video stream analysis
- **Audio Processing**: Audio monitoring and analysis
- **Secure Authentication**: JWT-based authentication
- **RESTful API**: FastAPI-based backend
- **Modern Frontend**: React with Vite (optional)

## Getting Started

### Backend Setup

1. Navigate to the backend directory:
   ```bash
   cd backend
   ```

2. Create a virtual environment:
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

4. Copy environment variables:
   ```bash
   cp .env.example .env
   ```

5. Update `.env` with your configuration

6. Run the application:
   ```bash
   uvicorn app.main:app --reload
   ```

### Frontend Setup (Optional)

1. Navigate to the frontend directory:
   ```bash
   cd frontend
   ```

2. Install dependencies:
   ```bash
   npm install
   ```

3. Copy environment variables:
   ```bash
   cp .env.example .env
   ```

4. Update `.env` with your API URL

5. Run the development server:
   ```bash
   npm run dev
   ```

## Technology Stack

### Backend
- **FastAPI**: Modern Python web framework
- **SQLAlchemy**: ORM for database operations
- **OpenCV**: Computer vision and image processing
- **MediaPipe**: Face detection and tracking
- **TensorFlow/PyTorch**: Deep learning models

### Frontend
- **React**: UI library
- **Vite**: Build tool and dev server
- **React Router**: Routing
- **Axios**: HTTP client

## License

[Add your license here]

