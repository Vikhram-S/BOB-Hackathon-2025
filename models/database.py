"""
Database models for the Hybrid Identity Monitoring System
"""
from sqlalchemy import Column, Integer, String, DateTime, Float, Boolean, Text, JSON
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker
from sqlalchemy import create_engine
from datetime import datetime
from config import settings

Base = declarative_base()

class User(Base):
    __tablename__ = "users"
    
    id = Column(Integer, primary_key=True, index=True)
    user_id = Column(String, unique=True, index=True)
    name = Column(String)
    email = Column(String, unique=True, index=True)
    phone = Column(String)
    created_at = Column(DateTime, default=datetime.utcnow)
    is_verified = Column(Boolean, default=False)
    verification_status = Column(String, default="pending")

class IdentityVerification(Base):
    __tablename__ = "identity_verifications"
    
    id = Column(Integer, primary_key=True, index=True)
    user_id = Column(String, index=True)
    verification_type = Column(String)  # "video_kyc", "biometric", "document"
    status = Column(String)  # "pending", "approved", "rejected"
    deepfake_score = Column(Float)
    confidence_score = Column(Float)
    verification_data = Column(JSON)
    created_at = Column(DateTime, default=datetime.utcnow)
    completed_at = Column(DateTime)

class DeepfakeDetection(Base):
    __tablename__ = "deepfake_detections"
    
    id = Column(Integer, primary_key=True, index=True)
    user_id = Column(String, index=True)
    video_path = Column(String)
    deepfake_score = Column(Float)
    is_deepfake = Column(Boolean)
    detection_metadata = Column(JSON)
    created_at = Column(DateTime, default=datetime.utcnow)

class IdentityMonitoring(Base):
    __tablename__ = "identity_monitoring"
    
    id = Column(Integer, primary_key=True, index=True)
    user_id = Column(String, index=True)
    activity_type = Column(String)  # "login", "transaction", "verification"
    anomaly_score = Column(Float)
    is_anomalous = Column(Boolean)
    monitoring_data = Column(JSON)
    created_at = Column(DateTime, default=datetime.utcnow)

class SystemAlert(Base):
    __tablename__ = "system_alerts"
    
    id = Column(Integer, primary_key=True, index=True)
    alert_type = Column(String)  # "deepfake_detected", "anomaly_detected", "security_breach"
    severity = Column(String)  # "low", "medium", "high", "critical"
    message = Column(Text)
    user_id = Column(String, index=True)
    alert_data = Column(JSON)
    is_resolved = Column(Boolean, default=False)
    created_at = Column(DateTime, default=datetime.utcnow)
    resolved_at = Column(DateTime)

# Database setup
engine = create_engine(settings.database_url)
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)

def create_tables():
    Base.metadata.create_all(bind=engine)

def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()
