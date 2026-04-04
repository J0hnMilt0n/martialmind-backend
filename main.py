from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from datetime import datetime
import os
import tempfile
import shutil
from typing import Dict, List

from models.pose_estimator import PoseEstimator
from models.performance_scorer import PerformanceScorer
from models.injury_detector import InjuryDetector
from utils.video_processor import VideoProcessor

# Initialize FastAPI app
app = FastAPI(
    title="MartialMind AI",
    description="AI-powered performance analysis and injury risk detection for combat sports",
    version="1.0.0"
)

# Configure CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # In production, specify your frontend URL
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Lazy initialization of AI components
_pose_estimator = None
_performance_scorer = None
_injury_detector = None
_video_processor = None


def get_pose_estimator() -> PoseEstimator:
    """Lazy initialization of PoseEstimator"""
    global _pose_estimator
    if _pose_estimator is None:
        _pose_estimator = PoseEstimator()
    return _pose_estimator


def get_performance_scorer() -> PerformanceScorer:
    """Lazy initialization of PerformanceScorer"""
    global _performance_scorer
    if _performance_scorer is None:
        _performance_scorer = PerformanceScorer()
    return _performance_scorer


def get_injury_detector() -> InjuryDetector:
    """Lazy initialization of InjuryDetector"""
    global _injury_detector
    if _injury_detector is None:
        _injury_detector = InjuryDetector()
    return _injury_detector


def get_video_processor() -> VideoProcessor:
    """Lazy initialization of VideoProcessor"""
    global _video_processor
    if _video_processor is None:
        _video_processor = VideoProcessor()
    return _video_processor

# Configuration
MAX_VIDEO_SIZE = 100 * 1024 * 1024  # 100MB
ALLOWED_EXTENSIONS = {'.mp4', '.mov', '.avi'}


@app.get("/")
async def root():
    """Root endpoint"""
    return {
        "message": "MartialMind AI - Performance Analysis API",
        "version": "1.0.0",
        "status": "operational"
    }


@app.get("/health")
async def health_check():
    """Health check endpoint"""
    return {
        "status": "healthy",
        "version": "1.0.0",
        "timestamp": datetime.utcnow().isoformat() + "Z",
        "components": {
            "pose_estimator": "ready",
            "performance_scorer": "ready",
            "injury_detector": "ready"
        }
    }


@app.post("/analyze-video")
async def analyze_video(video: UploadFile = File(...)):
    """
    Analyze a video for performance and injury risk.
    
    Args:
        video: Video file (MP4, MOV, or AVI)
        
    Returns:
        Analysis results including performance score, issues, injury risk, and recommendations
    """
    
    # Validate file
    if not video.filename:
        raise HTTPException(status_code=400, detail="No filename provided")
    
    # Check file extension
    file_ext = os.path.splitext(video.filename)[1].lower()
    if file_ext not in ALLOWED_EXTENSIONS:
        raise HTTPException(
            status_code=400, 
            detail=f"Invalid file type. Allowed: {', '.join(ALLOWED_EXTENSIONS)}"
        )
    
    # Create temporary file
    temp_file = None
    try:
        # Save uploaded file to temporary location
        with tempfile.NamedTemporaryFile(delete=False, suffix=file_ext) as temp_file:
            shutil.copyfileobj(video.file, temp_file)
            temp_file_path = temp_file.name
        
        # Process video
        print(f"Processing video: {video.filename}")
        
        # Get AI components (lazy initialization)
        pose_estimator = get_pose_estimator()
        performance_scorer = get_performance_scorer()
        injury_detector = get_injury_detector()
        video_processor = get_video_processor()
        
        # Extract frames and poses
        frames, poses = video_processor.extract_poses(temp_file_path, pose_estimator)
        
        if not poses or len(poses) == 0:
            raise HTTPException(
                status_code=422,
                detail="Could not detect person in video. Please ensure the athlete is clearly visible."
            )
        
        # Analyze performance
        performance_result = performance_scorer.score_performance(poses, frames)
        
        # Detect injury risks
        injury_result = injury_detector.assess_risk(poses, frames)
        
        # Generate recommendations
        drills = performance_scorer.recommend_drills(performance_result['issues'])
        prevention_advice = injury_detector.get_prevention_advice(injury_result)
        
        # Compile results
        response = {
            "score": round(performance_result['score'], 1),
            "issues": performance_result['issues'],
            "injury_risk": {
                "level": injury_result['level'],
                "area": injury_result['area'],
                "reason": injury_result['reason'],
                "risk_type": injury_result['risk_type']
            },
            "drills": drills,
            "prevention_advice": prevention_advice,
            "metadata": {
                "frames_analyzed": len(frames),
                "duration_seconds": len(frames) / 30,  # Assuming 30 FPS
                "timestamp": datetime.utcnow().isoformat() + "Z"
            }
        }
        
        print(f"Analysis complete: Score={response['score']}, Risk={response['injury_risk']['level']}")
        
        return JSONResponse(content=response)
        
    except HTTPException:
        raise
    except Exception as e:
        print(f"Error processing video: {str(e)}")
        raise HTTPException(
            status_code=500,
            detail=f"Error processing video: {str(e)}"
        )
    finally:
        # Clean up temporary file
        if temp_file and os.path.exists(temp_file_path):
            try:
                os.unlink(temp_file_path)
            except Exception as e:
                print(f"Error deleting temp file: {str(e)}")
        
        # Close uploaded file
        await video.close()


@app.get("/test")
async def test_endpoint():
    """
    Test endpoint with mock data for frontend development
    """
    return {
        "score": 7.2,
        "issues": [
            "Low hip rotation (15° instead of 45°)",
            "Balance instability during landing",
            "Late retraction after strike"
        ],
        "injury_risk": {
            "level": "Medium",
            "area": "Knee",
            "reason": "Knee misalignment during landing - valgus collapse detected",
            "risk_type": "Movement-based instability"
        },
        "drills": [
            "Hip turn drill 3x10 reps",
            "Stance stability drill with resistance band",
            "Knee alignment practice with mirror"
        ],
        "prevention_advice": [
            "Strengthen glutes and hip abductors",
            "Improve landing control with plyometric exercises",
            "Work on knee alignment during all lower body movements"
        ],
        "metadata": {
            "frames_analyzed": 150,
            "duration_seconds": 5.0,
            "timestamp": datetime.utcnow().isoformat() + "Z"
        }
    }


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
