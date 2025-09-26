"""
Configuration settings for the Hybrid Identity Monitoring & Deepfake-Resistant Verification System
"""
import os
from typing import Optional
from pydantic import BaseSettings

class Settings(BaseSettings):
    # Application settings
    app_name: str = "Hybrid Identity Monitoring System"
    version: str = "1.0.0"
    debug: bool = False
    
    # Security settings
    secret_key: str = "your-secret-key-change-in-production"
    algorithm: str = "HS256"
    access_token_expire_minutes: int = 30
    
    # Database settings
    database_url: str = "sqlite:///./identity_monitoring.db"
    
    # Redis settings for caching
    redis_url: str = "redis://localhost:6379"
    
    # Deepfake detection settings
    deepfake_model_path: str = "./models/deepfake_detector.h5"
    face_detection_model: str = "mediapipe"  # or "opencv"
    confidence_threshold: float = 0.8
    
    # Video processing settings
    max_video_size_mb: int = 100
    max_video_duration_seconds: int = 300
    frame_extraction_interval: int = 30  # Extract frame every N frames
    
    # Hybrid deployment settings
    deployment_mode: str = "hybrid"  # "on-prem", "cloud", "hybrid"
    cloud_endpoint: Optional[str] = None
    on_prem_endpoint: str = "http://localhost:8000"
    
    # Monitoring settings
    monitoring_interval_seconds: int = 60
    alert_threshold_anomaly_score: float = 0.7
    
    # API settings
    api_host: str = "0.0.0.0"
    api_port: int = 8000
    
    class Config:
        env_file = ".env"

settings = Settings()
