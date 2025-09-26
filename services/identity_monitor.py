"""
Continuous Identity Monitoring and Anomaly Detection System
"""
import numpy as np
import pandas as pd
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler
from typing import Dict, List, Optional, Tuple
import logging
from datetime import datetime, timedelta
import json
from sqlalchemy.orm import Session
from models.database import IdentityMonitoring, SystemAlert, User

logger = logging.getLogger(__name__)

class IdentityMonitor:
    def __init__(self):
        self.anomaly_detector = IsolationForest(
            contamination=0.1,
            random_state=42,
            n_estimators=100
        )
        self.scaler = StandardScaler()
        self.is_fitted = False
        
    def extract_behavioral_features(self, user_activities: List[Dict]) -> np.ndarray:
        """Extract behavioral features from user activities"""
        features = []
        
        for activity in user_activities:
            feature_vector = [
                # Time-based features
                activity.get('hour_of_day', 0),
                activity.get('day_of_week', 0),
                activity.get('is_weekend', 0),
                
                # Location features
                activity.get('location_lat', 0),
                activity.get('location_lng', 0),
                activity.get('location_accuracy', 0),
                
                # Device features
                activity.get('device_type', 0),  # 0=mobile, 1=desktop, 2=tablet
                activity.get('browser_type', 0),
                activity.get('os_type', 0),
                
                # Behavioral features
                activity.get('session_duration', 0),
                activity.get('click_rate', 0),
                activity.get('scroll_rate', 0),
                activity.get('typing_speed', 0),
                
                # Transaction features
                activity.get('transaction_amount', 0),
                activity.get('transaction_frequency', 0),
                activity.get('payment_method', 0),
                
                # Biometric features
                activity.get('face_confidence', 0),
                activity.get('voice_confidence', 0),
                activity.get('fingerprint_confidence', 0),
            ]
            features.append(feature_vector)
        
        return np.array(features)
    
    def train_anomaly_detector(self, user_activities: List[Dict]) -> bool:
        """Train the anomaly detection model on user activities"""
        try:
            if len(user_activities) < 10:
                logger.warning("Insufficient data for training anomaly detector")
                return False
            
            features = self.extract_behavioral_features(user_activities)
            
            # Normalize features
            features_scaled = self.scaler.fit_transform(features)
            
            # Train the anomaly detector
            self.anomaly_detector.fit(features_scaled)
            self.is_fitted = True
            
            logger.info(f"Anomaly detector trained on {len(user_activities)} activities")
            return True
            
        except Exception as e:
            logger.error(f"Error training anomaly detector: {e}")
            return False
    
    def detect_anomalies(self, user_activities: List[Dict]) -> Dict:
        """Detect anomalies in user behavior"""
        if not self.is_fitted:
            return {
                "anomaly_score": 0.0,
                "is_anomalous": False,
                "anomalies": [],
                "error": "Model not trained"
            }
        
        try:
            features = self.extract_behavioral_features(user_activities)
            features_scaled = self.scaler.transform(features)
            
            # Predict anomalies
            anomaly_scores = self.anomaly_detector.decision_function(features_scaled)
            predictions = self.anomaly_detector.predict(features_scaled)
            
            # Calculate overall anomaly score
            avg_anomaly_score = np.mean(anomaly_scores)
            is_anomalous = np.any(predictions == -1)
            
            # Identify specific anomalies
            anomalies = []
            if is_anomalous:
                anomalies.append("behavioral_anomaly")
            
            # Check for specific patterns
            if len(user_activities) > 0:
                recent_activity = user_activities[-1]
                
                # Time-based anomalies
                current_hour = datetime.now().hour
                if recent_activity.get('hour_of_day', 0) != current_hour:
                    anomalies.append("unusual_time_access")
                
                # Location anomalies
                if recent_activity.get('location_accuracy', 0) < 0.5:
                    anomalies.append("low_location_accuracy")
                
                # Device anomalies
                if recent_activity.get('device_type', 0) not in [0, 1, 2]:
                    anomalies.append("unknown_device")
                
                # Biometric anomalies
                if recent_activity.get('face_confidence', 0) < 0.7:
                    anomalies.append("low_biometric_confidence")
            
            return {
                "anomaly_score": float(avg_anomaly_score),
                "is_anomalous": bool(is_anomalous),
                "anomalies": anomalies,
                "individual_scores": anomaly_scores.tolist(),
                "timestamp": datetime.utcnow().isoformat()
            }
            
        except Exception as e:
            logger.error(f"Error detecting anomalies: {e}")
            return {
                "anomaly_score": 0.0,
                "is_anomalous": False,
                "anomalies": [],
                "error": str(e)
            }
    
    def continuous_monitoring(self, user_id: str, current_activity: Dict, db: Session) -> Dict:
        """Perform continuous identity monitoring"""
        try:
            # Get recent user activities (last 24 hours)
            cutoff_time = datetime.utcnow() - timedelta(hours=24)
            
            recent_activities = db.query(IdentityMonitoring).filter(
                IdentityMonitoring.user_id == user_id,
                IdentityMonitoring.created_at >= cutoff_time
            ).all()
            
            # Convert to list of dictionaries
            activities_data = []
            for activity in recent_activities:
                activity_dict = {
                    "hour_of_day": activity.created_at.hour,
                    "day_of_week": activity.created_at.weekday(),
                    "is_weekend": activity.created_at.weekday() >= 5,
                    **activity.monitoring_data
                }
                activities_data.append(activity_dict)
            
            # Add current activity
            current_activity_dict = {
                "hour_of_day": datetime.utcnow().hour,
                "day_of_week": datetime.utcnow().weekday(),
                "is_weekend": datetime.utcnow().weekday() >= 5,
                **current_activity
            }
            activities_data.append(current_activity_dict)
            
            # Train model if not fitted or if we have enough new data
            if not self.is_fitted and len(activities_data) >= 10:
                self.train_anomaly_detector(activities_data)
            
            # Detect anomalies
            anomaly_result = self.detect_anomalies(activities_data)
            
            # Store monitoring result
            monitoring_record = IdentityMonitoring(
                user_id=user_id,
                activity_type=current_activity.get('activity_type', 'unknown'),
                anomaly_score=anomaly_result['anomaly_score'],
                is_anomalous=anomaly_result['is_anomalous'],
                monitoring_data=current_activity
            )
            db.add(monitoring_record)
            
            # Create alert if anomaly detected
            if anomaly_result['is_anomalous']:
                alert = SystemAlert(
                    alert_type="anomaly_detected",
                    severity="medium",
                    message=f"Anomalous behavior detected for user {user_id}",
                    user_id=user_id,
                    alert_data=anomaly_result
                )
                db.add(alert)
            
            db.commit()
            
            return anomaly_result
            
        except Exception as e:
            logger.error(f"Error in continuous monitoring: {e}")
            return {
                "anomaly_score": 0.0,
                "is_anomalous": False,
                "anomalies": [],
                "error": str(e)
            }
    
    def risk_assessment(self, user_id: str, db: Session) -> Dict:
        """Perform comprehensive risk assessment for a user"""
        try:
            # Get user's recent activities and verifications
            recent_activities = db.query(IdentityMonitoring).filter(
                IdentityMonitoring.user_id == user_id,
                IdentityMonitoring.created_at >= datetime.utcnow() - timedelta(days=7)
            ).all()
            
            recent_verifications = db.query(IdentityVerification).filter(
                IdentityVerification.user_id == user_id,
                IdentityVerification.created_at >= datetime.utcnow() - timedelta(days=30)
            ).all()
            
            # Calculate risk factors
            risk_factors = []
            risk_score = 0.0
            
            # Anomaly-based risk
            if recent_activities:
                anomaly_scores = [a.anomaly_score for a in recent_activities]
                avg_anomaly_score = np.mean(anomaly_scores)
                if avg_anomaly_score > 0.5:
                    risk_factors.append("high_anomaly_score")
                    risk_score += 0.3
            
            # Verification-based risk
            if recent_verifications:
                failed_verifications = [v for v in recent_verifications if v.status == "rejected"]
                if len(failed_verifications) > 2:
                    risk_factors.append("multiple_failed_verifications")
                    risk_score += 0.2
                
                deepfake_scores = [v.deepfake_score for v in recent_verifications if v.deepfake_score is not None]
                if deepfake_scores and np.mean(deepfake_scores) > 0.7:
                    risk_factors.append("high_deepfake_probability")
                    risk_score += 0.4
            
            # Time-based risk
            recent_activity_count = len(recent_activities)
            if recent_activity_count > 100:  # Unusually high activity
                risk_factors.append("excessive_activity")
                risk_score += 0.1
            
            # Determine risk level
            if risk_score >= 0.7:
                risk_level = "high"
            elif risk_score >= 0.4:
                risk_level = "medium"
            else:
                risk_level = "low"
            
            return {
                "risk_score": float(risk_score),
                "risk_level": risk_level,
                "risk_factors": risk_factors,
                "assessment_timestamp": datetime.utcnow().isoformat()
            }
            
        except Exception as e:
            logger.error(f"Error in risk assessment: {e}")
            return {
                "risk_score": 0.0,
                "risk_level": "unknown",
                "risk_factors": [],
                "error": str(e)
            }
