from pydantic_settings import BaseSettings
from typing import Optional


class Settings(BaseSettings):
    # Server Configuration
    HOST: str = "0.0.0.0"
    PORT: int = 8000
    DEBUG: bool = True
    
    # Performance Settings
    MAX_VIDEO_SIZE_MB: int = 100
    VIDEO_TIMEOUT_SECONDS: int = 120
    
    # AI Model Configuration
    POSE_MODEL_COMPLEXITY: int = 1  # 0, 1, or 2
    MIN_DETECTION_CONFIDENCE: float = 0.5
    MIN_TRACKING_CONFIDENCE: float = 0.5
    
    # Injury Risk Thresholds
    KNEE_ALIGNMENT_THRESHOLD: float = 15.0  # degrees
    HIP_ROTATION_MIN: float = 30.0  # degrees
    BALANCE_STABILITY_MIN: float = 0.7  # 0-1 score
    
    class Config:
        env_file = ".env"
        case_sensitive = True


settings = Settings()
