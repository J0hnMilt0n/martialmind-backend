# MartialMind AI Backend

AI-powered performance analysis and injury risk detection system for combat sports athletes.

## 🏗️ Architecture

```
Dojo Republic (Next.js Frontend)
        ↓
    API Call
        ↓
MartialMind AI (FastAPI Backend)
        ↓
   [MediaPipe] → [Pose Estimation]
        ↓
   [Biomechanics Analysis]
        ↓
   [Performance Scoring + Injury Risk Detection]
```

## 🚀 Quick Start

### Prerequisites

- Python 3.9 or higher
- pip (Python package manager)
- Virtual environment (recommended)

### Installation

1. **Create virtual environment:**

```bash
python -m venv venv

# Windows
venv\Scripts\activate

# Linux/Mac
source venv/bin/activate
```

2. **Install dependencies:**

```bash
pip install -r requirements.txt
```

3. **Run the server:**

```bash
uvicorn main:app --reload --host 0.0.0.0 --port 8000
```

4. **Test the API:**

```bash
# Health check
curl http://localhost:8000/health

# API documentation
# Open in browser: http://localhost:8000/docs
```

## 📁 Project Structure

```
martialmind-backend/
├── main.py                    # FastAPI application entry point
├── requirements.txt           # Python dependencies
├── config.py                  # Configuration settings
├── models/
│   ├── __init__.py
│   ├── pose_estimator.py     # MediaPipe pose estimation
│   ├── performance_scorer.py # Performance scoring logic
│   └── injury_detector.py    # Injury risk detection
├── utils/
│   ├── __init__.py
│   ├── video_processor.py    # Video handling utilities
│   └── biomechanics.py       # Biomechanics calculations
├── tests/
│   ├── __init__.py
│   └── test_api.py           # API tests
└── README.md                  # This file
```

## 🔧 Environment Variables

Create a `.env` file in the root directory:

```env
# Server Configuration
HOST=0.0.0.0
PORT=8000
DEBUG=True

# Performance Settings
MAX_VIDEO_SIZE_MB=100
VIDEO_TIMEOUT_SECONDS=120

# AI Model Configuration
POSE_MODEL_COMPLEXITY=1  # 0, 1, or 2 (higher = more accurate but slower)
MIN_DETECTION_CONFIDENCE=0.5
MIN_TRACKING_CONFIDENCE=0.5

# Injury Risk Thresholds
KNEE_ALIGNMENT_THRESHOLD=15  # degrees
HIP_ROTATION_MIN=30          # degrees
BALANCE_STABILITY_MIN=0.7    # 0-1 score
```

## 📡 API Endpoints

### POST /analyze-video

Analyzes a video and returns performance metrics and injury risk assessment.

**Request:**
- Method: `POST`
- Content-Type: `multipart/form-data`
- Body: `video` (file)

**Response:**

```json
{
  "score": 7.2,
  "issues": [
    "Low hip rotation",
    "Balance instability",
    "Late retraction"
  ],
  "injury_risk": {
    "level": "Medium",
    "area": "Knee",
    "reason": "Knee misalignment during landing",
    "risk_type": "Movement-based instability"
  },
  "drills": [
    "Hip turn drill 3x10",
    "Stance stability drill",
    "Knee alignment practice"
  ],
  "prevention_advice": [
    "Strengthen glutes",
    "Improve landing control",
    "Work on knee alignment"
  ]
}
```

### GET /health

Health check endpoint.

**Response:**

```json
{
  "status": "healthy",
  "version": "1.0.0",
  "timestamp": "2026-03-04T10:30:00Z"
}
```

## 🧠 How It Works

### 1. Pose Estimation (MediaPipe)

- Extracts 33 body landmarks per frame
- Tracks joint positions in 3D space
- Provides visibility confidence scores

### 2. Performance Scoring

Analyzes:
- **Technique Quality**: Form accuracy, movement efficiency
- **Balance**: Center of mass stability, weight distribution
- **Power Generation**: Hip rotation, kinetic chain
- **Speed**: Strike velocity, retraction time
- **Precision**: Target accuracy, follow-through

### 3. Injury Risk Detection

**Movement-Based Risk Factors:**

| Risk Area | Detection Method | Severity |
|-----------|-----------------|----------|
| Knee | Joint alignment during landing | Low/Medium/High |
| Hip | Overextension patterns | Low/Medium/High |
| Ankle | Balance instability | Low/Medium/High |
| Shoulder | Overrotation during strikes | Low/Medium/High |
| Lower Back | Forward lean angles | Low/Medium/High |

**Important:** This is NOT medical diagnosis. It detects biomechanical movement patterns that correlate with injury risk.

### 4. Drill Recommendations

Based on identified issues, the system suggests:
- Corrective exercises
- Technique drills
- Strength and conditioning work
- Injury prevention strategies

## 🔬 Technical Details

### MediaPipe Pose Landmarks

```python
# Key landmarks used for analysis:
0:  Nose
11: Left Shoulder
12: Right Shoulder
23: Left Hip
24: Right Hip
25: Left Knee
26: Right Knee
27: Left Ankle
28: Right Ankle
```

### Biomechanics Calculations

```python
# Example: Knee alignment check
def check_knee_alignment(hip, knee, ankle):
    # Calculate angle between hip-knee-ankle
    angle = calculate_angle(hip, knee, ankle)
    
    # Ideal range: 170-180 degrees (nearly straight)
    if angle < 165:
        return {"risk": "High", "reason": "Knee valgus collapse"}
    elif angle < 170:
        return {"risk": "Medium", "reason": "Slight knee misalignment"}
    else:
        return {"risk": "Low", "reason": "Good alignment"}
```

## 📊 Performance Optimization

### For Development:
- Use `POSE_MODEL_COMPLEXITY=0` for faster processing
- Limit video length to 10 seconds
- Use lower resolution (720p)

### For Production:
- Use `POSE_MODEL_COMPLEXITY=1` for balanced performance
- Consider GPU acceleration (CUDA)
- Implement video frame sampling (analyze every Nth frame)
- Use async processing with task queues (Celery + Redis)

## 🚢 Deployment Options

### Option 1: Railway

```bash
# Install Railway CLI
npm install -g @railway/cli

# Login and deploy
railway login
railway init
railway up
```

### Option 2: Render

1. Connect your GitHub repository
2. Create new Web Service
3. Set build command: `pip install -r requirements.txt`
4. Set start command: `uvicorn main:app --host 0.0.0.0 --port $PORT`

### Option 3: AWS EC2

```bash
# SSH into EC2 instance
ssh -i key.pem ubuntu@your-instance-ip

# Install dependencies
sudo apt update
sudo apt install python3-pip python3-venv

# Clone repository and setup
git clone your-repo
cd martialmind-backend
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt

# Run with Gunicorn
gunicorn main:app -w 4 -k uvicorn.workers.UvicornWorker -b 0.0.0.0:8000
```

### Option 4: Docker

```dockerfile
FROM python:3.9-slim

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]
```

```bash
docker build -t martialmind-ai .
docker run -p 8000:8000 martialmind-ai
```

## 🔒 Security Considerations

- Implement rate limiting (10 requests/minute per user)
- Validate file types and sizes
- Sanitize file names
- Use CORS middleware for frontend communication
- Implement API key authentication for production
- Delete uploaded videos after processing

## 📈 Future Enhancements

### Phase 2:
- [ ] Athlete history tracking
- [ ] Comparative analysis (current vs. baseline)
- [ ] Fatigue detection over time
- [ ] Real-time video processing

### Phase 3:
- [ ] True predictive ML models (trained on historical data)
- [ ] Multi-angle video support
- [ ] Wearable sensor integration
- [ ] Coach dashboard with analytics

## 🧪 Testing

```bash
# Run tests
pytest tests/

# With coverage
pytest tests/ --cov=./ --cov-report=html
```

## 📝 License

MIT License - See LICENSE file for details

## 🤝 Contributing

This is part of the Dojo Republic MSc thesis project.

## 📞 Support

For issues or questions, contact: [your-email@example.com]

---

**Built with:**
- FastAPI
- MediaPipe
- OpenCV
- NumPy
- Pydantic

**Powered by Dojo Republic**
