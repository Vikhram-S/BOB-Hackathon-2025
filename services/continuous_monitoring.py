"""
Continuous Identity Assurance Monitoring System
"""
import asyncio
import logging
from datetime import datetime, timedelta
from typing import Dict, List, Optional
import json
from sqlalchemy.orm import Session
from sqlalchemy import and_, or_
from models.database import (
    User, IdentityVerification, IdentityMonitoring, 
    SystemAlert, DeepfakeDetection
)
from services.identity_monitor import IdentityMonitor
from services.deepfake_detector import DeepfakeDetector

logger = logging.getLogger(__name__)

class ContinuousIdentityMonitor:
    def __init__(self, identity_monitor: IdentityMonitor, deepfake_detector: DeepfakeDetector):
        self.identity_monitor = identity_monitor
        self.deepfake_detector = deepfake_detector
        self.monitoring_active = False
        self.monitoring_tasks = {}
    
    async def start_continuous_monitoring(self, user_id: str, db: Session):
        """Start continuous monitoring for a user"""
        try:
            if user_id in self.monitoring_tasks:
                logger.warning(f"Monitoring already active for user {user_id}")
                return
            
            self.monitoring_active = True
            task = asyncio.create_task(self._monitor_user_continuously(user_id, db))
            self.monitoring_tasks[user_id] = task
            
            logger.info(f"Started continuous monitoring for user {user_id}")
            
        except Exception as e:
            logger.error(f"Error starting continuous monitoring for user {user_id}: {e}")
    
    async def stop_continuous_monitoring(self, user_id: str):
        """Stop continuous monitoring for a user"""
        try:
            if user_id in self.monitoring_tasks:
                task = self.monitoring_tasks[user_id]
                task.cancel()
                del self.monitoring_tasks[user_id]
                logger.info(f"Stopped continuous monitoring for user {user_id}")
            
        except Exception as e:
            logger.error(f"Error stopping continuous monitoring for user {user_id}: {e}")
    
    async def _monitor_user_continuously(self, user_id: str, db: Session):
        """Continuously monitor user identity and behavior"""
        while self.monitoring_active:
            try:
                # Perform monitoring checks
                await self._perform_monitoring_checks(user_id, db)
                
                # Wait before next check
                await asyncio.sleep(60)  # Check every minute
                
            except asyncio.CancelledError:
                logger.info(f"Continuous monitoring cancelled for user {user_id}")
                break
            except Exception as e:
                logger.error(f"Error in continuous monitoring for user {user_id}: {e}")
                await asyncio.sleep(30)  # Wait before retry
    
    async def _perform_monitoring_checks(self, user_id: str, db: Session):
        """Perform various monitoring checks"""
        try:
            # Get user's recent activities
            recent_activities = self._get_recent_activities(user_id, db)
            
            if not recent_activities:
                return
            
            # 1. Behavioral anomaly detection
            anomaly_result = self.identity_monitor.detect_anomalies(recent_activities)
            
            if anomaly_result.get("is_anomalous", False):
                await self._handle_anomaly_detection(user_id, anomaly_result, db)
            
            # 2. Risk assessment
            risk_assessment = self.identity_monitor.risk_assessment(user_id, db)
            
            if risk_assessment.get("risk_level") in ["high", "medium"]:
                await self._handle_risk_assessment(user_id, risk_assessment, db)
            
            # 3. Verification status monitoring
            await self._monitor_verification_status(user_id, db)
            
            # 4. Deepfake detection monitoring
            await self._monitor_deepfake_activity(user_id, db)
            
        except Exception as e:
            logger.error(f"Error in monitoring checks for user {user_id}: {e}")
    
    def _get_recent_activities(self, user_id: str, db: Session, hours: int = 24) -> List[Dict]:
        """Get recent user activities"""
        try:
            cutoff_time = datetime.utcnow() - timedelta(hours=hours)
            
            activities = db.query(IdentityMonitoring).filter(
                and_(
                    IdentityMonitoring.user_id == user_id,
                    IdentityMonitoring.created_at >= cutoff_time
                )
            ).all()
            
            activities_data = []
            for activity in activities:
                activity_dict = {
                    "activity_type": activity.activity_type,
                    "anomaly_score": activity.anomaly_score,
                    "is_anomalous": activity.is_anomalous,
                    "timestamp": activity.created_at.isoformat(),
                    **activity.monitoring_data
                }
                activities_data.append(activity_dict)
            
            return activities_data
            
        except Exception as e:
            logger.error(f"Error getting recent activities for user {user_id}: {e}")
            return []
    
    async def _handle_anomaly_detection(self, user_id: str, anomaly_result: Dict, db: Session):
        """Handle detected anomalies"""
        try:
            # Create high-priority alert
            alert = SystemAlert(
                alert_type="behavioral_anomaly",
                severity="high" if anomaly_result.get("anomaly_score", 0) > 0.8 else "medium",
                message=f"Behavioral anomaly detected for user {user_id}",
                user_id=user_id,
                alert_data=anomaly_result
            )
            db.add(alert)
            
            # Update user verification status if anomaly is severe
            if anomaly_result.get("anomaly_score", 0) > 0.9:
                user = db.query(User).filter(User.user_id == user_id).first()
                if user:
                    user.verification_status = "suspended"
                    user.is_verified = False
            
            db.commit()
            
            logger.warning(f"Behavioral anomaly detected for user {user_id}: {anomaly_result}")
            
        except Exception as e:
            logger.error(f"Error handling anomaly detection for user {user_id}: {e}")
    
    async def _handle_risk_assessment(self, user_id: str, risk_assessment: Dict, db: Session):
        """Handle risk assessment results"""
        try:
            risk_level = risk_assessment.get("risk_level", "low")
            risk_score = risk_assessment.get("risk_score", 0.0)
            
            if risk_level == "high":
                # Create critical alert
                alert = SystemAlert(
                    alert_type="high_risk_user",
                    severity="critical",
                    message=f"High risk detected for user {user_id}",
                    user_id=user_id,
                    alert_data=risk_assessment
                )
                db.add(alert)
                
                # Suspend user if risk is critical
                user = db.query(User).filter(User.user_id == user_id).first()
                if user:
                    user.verification_status = "suspended"
                    user.is_verified = False
                
            elif risk_level == "medium":
                # Create medium priority alert
                alert = SystemAlert(
                    alert_type="medium_risk_user",
                    severity="medium",
                    message=f"Medium risk detected for user {user_id}",
                    user_id=user_id,
                    alert_data=risk_assessment
                )
                db.add(alert)
            
            db.commit()
            
            logger.info(f"Risk assessment for user {user_id}: {risk_level} (score: {risk_score})")
            
        except Exception as e:
            logger.error(f"Error handling risk assessment for user {user_id}: {e}")
    
    async def _monitor_verification_status(self, user_id: str, db: Session):
        """Monitor verification status and detect issues"""
        try:
            # Get recent verifications
            recent_verifications = db.query(IdentityVerification).filter(
                and_(
                    IdentityVerification.user_id == user_id,
                    IdentityVerification.created_at >= datetime.utcnow() - timedelta(days=7)
                )
            ).all()
            
            if not recent_verifications:
                return
            
            # Check for failed verifications
            failed_verifications = [v for v in recent_verifications if v.status == "rejected"]
            
            if len(failed_verifications) >= 3:
                # Multiple failed verifications
                alert = SystemAlert(
                    alert_type="multiple_failed_verifications",
                    severity="high",
                    message=f"Multiple failed verifications for user {user_id}",
                    user_id=user_id,
                    alert_data={
                        "failed_count": len(failed_verifications),
                        "verification_ids": [v.id for v in failed_verifications]
                    }
                )
                db.add(alert)
                db.commit()
            
            # Check for low confidence scores
            low_confidence_verifications = [
                v for v in recent_verifications 
                if v.confidence_score is not None and v.confidence_score < 0.5
            ]
            
            if len(low_confidence_verifications) >= 2:
                alert = SystemAlert(
                    alert_type="low_confidence_verifications",
                    severity="medium",
                    message=f"Low confidence verifications for user {user_id}",
                    user_id=user_id,
                    alert_data={
                        "low_confidence_count": len(low_confidence_verifications),
                        "avg_confidence": sum(v.confidence_score for v in low_confidence_verifications) / len(low_confidence_verifications)
                    }
                )
                db.add(alert)
                db.commit()
            
        except Exception as e:
            logger.error(f"Error monitoring verification status for user {user_id}: {e}")
    
    async def _monitor_deepfake_activity(self, user_id: str, db: Session):
        """Monitor for deepfake-related activity"""
        try:
            # Get recent deepfake detections
            recent_detections = db.query(DeepfakeDetection).filter(
                and_(
                    DeepfakeDetection.user_id == user_id,
                    DeepfakeDetection.created_at >= datetime.utcnow() - timedelta(days=7)
                )
            ).all()
            
            if not recent_detections:
                return
            
            # Check for high deepfake scores
            high_deepfake_scores = [
                d for d in recent_detections 
                if d.deepfake_score is not None and d.deepfake_score > 0.7
            ]
            
            if len(high_deepfake_scores) >= 2:
                alert = SystemAlert(
                    alert_type="repeated_deepfake_attempts",
                    severity="critical",
                    message=f"Repeated deepfake attempts detected for user {user_id}",
                    user_id=user_id,
                    alert_data={
                        "high_score_count": len(high_deepfake_scores),
                        "avg_deepfake_score": sum(d.deepfake_score for d in high_deepfake_scores) / len(high_deepfake_scores)
                    }
                )
                db.add(alert)
                
                # Suspend user for repeated deepfake attempts
                user = db.query(User).filter(User.user_id == user_id).first()
                if user:
                    user.verification_status = "suspended"
                    user.is_verified = False
                
                db.commit()
            
        except Exception as e:
            logger.error(f"Error monitoring deepfake activity for user {user_id}: {e}")
    
    async def get_monitoring_dashboard_data(self, user_id: str, db: Session) -> Dict:
        """Get comprehensive monitoring dashboard data for a user"""
        try:
            # Get user info
            user = db.query(User).filter(User.user_id == user_id).first()
            if not user:
                return {"error": "User not found"}
            
            # Get recent activities (last 7 days)
            recent_activities = self._get_recent_activities(user_id, db, hours=24*7)
            
            # Get verification history
            verifications = db.query(IdentityVerification).filter(
                IdentityVerification.user_id == user_id
            ).order_by(IdentityVerification.created_at.desc()).limit(10).all()
            
            # Get alerts
            alerts = db.query(SystemAlert).filter(
                SystemAlert.user_id == user_id
            ).order_by(SystemAlert.created_at.desc()).limit(10).all()
            
            # Get deepfake detections
            deepfake_detections = db.query(DeepfakeDetection).filter(
                DeepfakeDetection.user_id == user_id
            ).order_by(DeepfakeDetection.created_at.desc()).limit(10).all()
            
            # Calculate statistics
            total_activities = len(recent_activities)
            anomalous_activities = sum(1 for a in recent_activities if a.get("is_anomalous", False))
            avg_anomaly_score = sum(a.get("anomaly_score", 0) for a in recent_activities) / max(1, total_activities)
            
            verification_success_rate = sum(1 for v in verifications if v.status == "approved") / max(1, len(verifications))
            
            return {
                "user_info": {
                    "user_id": user.user_id,
                    "name": user.name,
                    "email": user.email,
                    "is_verified": user.is_verified,
                    "verification_status": user.verification_status
                },
                "activity_stats": {
                    "total_activities": total_activities,
                    "anomalous_activities": anomalous_activities,
                    "avg_anomaly_score": avg_anomaly_score
                },
                "verification_stats": {
                    "total_verifications": len(verifications),
                    "success_rate": verification_success_rate,
                    "recent_verifications": [
                        {
                            "id": v.id,
                            "type": v.verification_type,
                            "status": v.status,
                            "confidence_score": v.confidence_score,
                            "deepfake_score": v.deepfake_score,
                            "created_at": v.created_at.isoformat()
                        } for v in verifications
                    ]
                },
                "alerts": [
                    {
                        "id": a.id,
                        "type": a.alert_type,
                        "severity": a.severity,
                        "message": a.message,
                        "is_resolved": a.is_resolved,
                        "created_at": a.created_at.isoformat()
                    } for a in alerts
                ],
                "deepfake_detections": [
                    {
                        "id": d.id,
                        "deepfake_score": d.deepfake_score,
                        "is_deepfake": d.is_deepfake,
                        "created_at": d.created_at.isoformat()
                    } for d in deepfake_detections
                ],
                "monitoring_status": {
                    "is_monitored": user_id in self.monitoring_tasks,
                    "monitoring_active": self.monitoring_active
                }
            }
            
        except Exception as e:
            logger.error(f"Error getting monitoring dashboard data for user {user_id}: {e}")
            return {"error": str(e)}
    
    async def resolve_alert(self, alert_id: int, resolution_notes: str, db: Session) -> bool:
        """Resolve a system alert"""
        try:
            alert = db.query(SystemAlert).filter(SystemAlert.id == alert_id).first()
            if not alert:
                return False
            
            alert.is_resolved = True
            alert.resolved_at = datetime.utcnow()
            alert.alert_data = alert.alert_data or {}
            alert.alert_data["resolution_notes"] = resolution_notes
            
            db.commit()
            return True
            
        except Exception as e:
            logger.error(f"Error resolving alert {alert_id}: {e}")
            return False
