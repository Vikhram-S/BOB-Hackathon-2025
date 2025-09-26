"""
Main FastAPI Application for Hybrid Identity Monitoring & Deepfake-Resistant Verification
"""
from fastapi import FastAPI, HTTPException, Depends, UploadFile, File, Form
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import HTMLResponse, JSONResponse
from sqlalchemy.orm import Session
from typing import Optional, List
import logging
import asyncio
from datetime import datetime
import json

# Import our modules
from config import settings
from models.database import create_tables, get_db, User, IdentityVerification, SystemAlert
from services.deepfake_detector import DeepfakeDetector
from services.identity_monitor import IdentityMonitor
from services.video_kyc import VideoKYCVerifier
from services.continuous_monitoring import ContinuousIdentityMonitor
from deployment.hybrid_deployment import HybridDeploymentManager

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Initialize FastAPI app
app = FastAPI(
    title="Hybrid Identity Monitoring & Deepfake-Resistant Verification",
    description="Bank of Baroda Hackathon Solution",
    version="1.0.0"
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Initialize services
deepfake_detector = DeepfakeDetector(settings.deepfake_model_path)
identity_monitor = IdentityMonitor()
video_kyc_verifier = VideoKYCVerifier(deepfake_detector, identity_monitor)
continuous_monitor = ContinuousIdentityMonitor(identity_monitor, deepfake_detector)
hybrid_deployment = HybridDeploymentManager()

# Create database tables
create_tables()

@app.on_event("startup")
async def startup_event():
    """Initialize services on startup"""
    try:
        # Initialize hybrid deployment
        await hybrid_deployment.initialize_hybrid_deployment()
        logger.info("Application started successfully")
    except Exception as e:
        logger.error(f"Error during startup: {e}")

@app.get("/")
async def root():
    """Root endpoint with API information"""
    return {
        "message": "Hybrid Identity Monitoring & Deepfake-Resistant Verification API",
        "version": "1.0.0",
        "status": "active",
        "endpoints": {
            "health": "/health",
            "docs": "/docs",
            "video_kyc": "/api/v1/kyc/video",
            "deepfake_detection": "/api/v1/deepfake/detect",
            "monitoring": "/api/v1/monitoring",
            "dashboard": "/dashboard"
        }
    }

@app.get("/health")
async def health_check():
    """Health check endpoint"""
    return {
        "status": "healthy",
        "timestamp": datetime.utcnow().isoformat(),
        "services": {
            "deepfake_detector": "active",
            "identity_monitor": "active",
            "video_kyc": "active",
            "continuous_monitoring": "active"
        }
    }

@app.get("/ready")
async def readiness_check():
    """Readiness check endpoint"""
    return {"status": "ready"}

# User Management Endpoints
@app.post("/api/v1/users/register")
async def register_user(
    user_id: str = Form(...),
    name: str = Form(...),
    email: str = Form(...),
    phone: str = Form(...),
    db: Session = Depends(get_db)
):
    """Register a new user"""
    try:
        # Check if user already exists
        existing_user = db.query(User).filter(User.user_id == user_id).first()
        if existing_user:
            raise HTTPException(status_code=400, detail="User already exists")
        
        # Create new user
        user = User(
            user_id=user_id,
            name=name,
            email=email,
            phone=phone
        )
        db.add(user)
        db.commit()
        
        return {
            "message": "User registered successfully",
            "user_id": user_id,
            "status": "pending_verification"
        }
        
    except Exception as e:
        logger.error(f"Error registering user: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/api/v1/users/{user_id}")
async def get_user(user_id: str, db: Session = Depends(get_db)):
    """Get user information"""
    try:
        user = db.query(User).filter(User.user_id == user_id).first()
        if not user:
            raise HTTPException(status_code=404, detail="User not found")
        
        return {
            "user_id": user.user_id,
            "name": user.name,
            "email": user.email,
            "phone": user.phone,
            "is_verified": user.is_verified,
            "verification_status": user.verification_status,
            "created_at": user.created_at.isoformat()
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error getting user: {e}")
        raise HTTPException(status_code=500, detail=str(e))

# Video KYC Endpoints
@app.post("/api/v1/kyc/video")
async def process_video_kyc(
    user_id: str = Form(...),
    video_file: UploadFile = File(...),
    document_image: Optional[UploadFile] = File(None),
    db: Session = Depends(get_db)
):
    """Process Video KYC verification"""
    try:
        # Save uploaded video
        video_path = f"data/videos/{user_id}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.mp4"
        with open(video_path, "wb") as buffer:
            content = await video_file.read()
            buffer.write(content)
        
        # Process document image if provided
        document_img = None
        if document_image:
            document_content = await document_image.read()
            import cv2
            import numpy as np
            nparr = np.frombuffer(document_content, np.uint8)
            document_img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        
        # Process Video KYC
        result = video_kyc_verifier.process_video_kyc(user_id, video_path, document_img, db)
        
        return result
        
    except Exception as e:
        logger.error(f"Error processing video KYC: {e}")
        raise HTTPException(status_code=500, detail=str(e))

# Deepfake Detection Endpoints
@app.post("/api/v1/deepfake/detect")
async def detect_deepfake(
    video_file: UploadFile = File(...),
    user_id: Optional[str] = Form(None)
):
    """Detect deepfake in uploaded video"""
    try:
        # Save uploaded video
        video_path = f"data/videos/deepfake_check_{datetime.now().strftime('%Y%m%d_%H%M%S')}.mp4"
        with open(video_path, "wb") as buffer:
            content = await video_file.read()
            buffer.write(content)
        
        # Detect deepfake
        result = deepfake_detector.detect_deepfake(video_path)
        
        return result
        
    except Exception as e:
        logger.error(f"Error detecting deepfake: {e}")
        raise HTTPException(status_code=500, detail=str(e))

# Identity Monitoring Endpoints
@app.post("/api/v1/monitoring/check")
async def check_identity_monitoring(
    user_id: str = Form(...),
    activity_data: str = Form(...),
    db: Session = Depends(get_db)
):
    """Check identity monitoring for user activity"""
    try:
        activity_dict = json.loads(activity_data)
        
        # Perform monitoring check
        result = identity_monitor.continuous_monitoring(user_id, activity_dict, db)
        
        return result
        
    except Exception as e:
        logger.error(f"Error in identity monitoring: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/api/v1/monitoring/dashboard/{user_id}")
async def get_monitoring_dashboard(user_id: str, db: Session = Depends(get_db)):
    """Get monitoring dashboard data for user"""
    try:
        dashboard_data = await continuous_monitor.get_monitoring_dashboard_data(user_id, db)
        return dashboard_data
        
    except Exception as e:
        logger.error(f"Error getting monitoring dashboard: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/api/v1/monitoring/start/{user_id}")
async def start_continuous_monitoring(user_id: str, db: Session = Depends(get_db)):
    """Start continuous monitoring for user"""
    try:
        await continuous_monitor.start_continuous_monitoring(user_id, db)
        return {"message": f"Continuous monitoring started for user {user_id}"}
        
    except Exception as e:
        logger.error(f"Error starting continuous monitoring: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/api/v1/monitoring/stop/{user_id}")
async def stop_continuous_monitoring(user_id: str):
    """Stop continuous monitoring for user"""
    try:
        await continuous_monitor.stop_continuous_monitoring(user_id)
        return {"message": f"Continuous monitoring stopped for user {user_id}"}
        
    except Exception as e:
        logger.error(f"Error stopping continuous monitoring: {e}")
        raise HTTPException(status_code=500, detail=str(e))

# Alerts Management
@app.get("/api/v1/alerts")
async def get_alerts(
    user_id: Optional[str] = None,
    severity: Optional[str] = None,
    limit: int = 50,
    db: Session = Depends(get_db)
):
    """Get system alerts"""
    try:
        query = db.query(SystemAlert)
        
        if user_id:
            query = query.filter(SystemAlert.user_id == user_id)
        if severity:
            query = query.filter(SystemAlert.severity == severity)
        
        alerts = query.order_by(SystemAlert.created_at.desc()).limit(limit).all()
        
        return [
            {
                "id": alert.id,
                "alert_type": alert.alert_type,
                "severity": alert.severity,
                "message": alert.message,
                "user_id": alert.user_id,
                "is_resolved": alert.is_resolved,
                "created_at": alert.created_at.isoformat(),
                "resolved_at": alert.resolved_at.isoformat() if alert.resolved_at else None
            }
            for alert in alerts
        ]
        
    except Exception as e:
        logger.error(f"Error getting alerts: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/api/v1/alerts/{alert_id}/resolve")
async def resolve_alert(
    alert_id: int,
    resolution_notes: str = Form(...),
    db: Session = Depends(get_db)
):
    """Resolve a system alert"""
    try:
        success = await continuous_monitor.resolve_alert(alert_id, resolution_notes, db)
        
        if success:
            return {"message": "Alert resolved successfully"}
        else:
            raise HTTPException(status_code=404, detail="Alert not found")
            
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error resolving alert: {e}")
        raise HTTPException(status_code=500, detail=str(e))

# Deployment Status
@app.get("/api/v1/deployment/status")
async def get_deployment_status():
    """Get hybrid deployment status"""
    try:
        status = await hybrid_deployment.get_deployment_status()
        return status
        
    except Exception as e:
        logger.error(f"Error getting deployment status: {e}")
        raise HTTPException(status_code=500, detail=str(e))

# Dashboard Web Interface
@app.get("/dashboard", response_class=HTMLResponse)
async def dashboard():
    """Serve the monitoring dashboard"""
    return """
    <!DOCTYPE html>
    <html>
    <head>
        <title>Identity Monitoring Dashboard</title>
        <script src="https://cdn.jsdelivr.net/npm/chart.js"></script>
        <style>
            body { font-family: Arial, sans-serif; margin: 0; padding: 20px; background-color: #f5f5f5; }
            .container { max-width: 1200px; margin: 0 auto; }
            .header { background: #2c3e50; color: white; padding: 20px; border-radius: 8px; margin-bottom: 20px; }
            .card { background: white; padding: 20px; border-radius: 8px; box-shadow: 0 2px 4px rgba(0,0,0,0.1); margin-bottom: 20px; }
            .grid { display: grid; grid-template-columns: repeat(auto-fit, minmax(300px, 1fr)); gap: 20px; }
            .status-indicator { display: inline-block; width: 12px; height: 12px; border-radius: 50%; margin-right: 8px; }
            .status-healthy { background-color: #27ae60; }
            .status-warning { background-color: #f39c12; }
            .status-critical { background-color: #e74c3c; }
            .btn { background: #3498db; color: white; border: none; padding: 10px 20px; border-radius: 4px; cursor: pointer; }
            .btn:hover { background: #2980b9; }
            .alert { padding: 15px; border-radius: 4px; margin-bottom: 10px; }
            .alert-critical { background: #f8d7da; border: 1px solid #f5c6cb; color: #721c24; }
            .alert-warning { background: #fff3cd; border: 1px solid #ffeaa7; color: #856404; }
            .alert-info { background: #d1ecf1; border: 1px solid #bee5eb; color: #0c5460; }
        </style>
    </head>
    <body>
        <div class="container">
            <div class="header">
                <h1>🛡️ Hybrid Identity Monitoring Dashboard</h1>
                <p>Bank of Baroda Hackathon - Deepfake-Resistant Verification System</p>
            </div>
            
            <div class="grid">
                <div class="card">
                    <h3>System Status</h3>
                    <div id="system-status">
                        <p><span class="status-indicator status-healthy"></span>Deepfake Detector: Active</p>
                        <p><span class="status-indicator status-healthy"></span>Identity Monitor: Active</p>
                        <p><span class="status-indicator status-healthy"></span>Video KYC: Active</p>
                        <p><span class="status-indicator status-healthy"></span>Continuous Monitoring: Active</p>
                    </div>
                </div>
                
                <div class="card">
                    <h3>Quick Actions</h3>
                    <button class="btn" onclick="startVideoKYC()">Start Video KYC</button>
                    <button class="btn" onclick="checkDeepfake()">Check Deepfake</button>
                    <button class="btn" onclick="viewAlerts()">View Alerts</button>
                    <button class="btn" onclick="refreshDashboard()">Refresh</button>
                </div>
                
                <div class="card">
                    <h3>Recent Activity</h3>
                    <div id="recent-activity">
                        <p>Loading recent activity...</p>
                    </div>
                </div>
                
                <div class="card">
                    <h3>Security Metrics</h3>
                    <canvas id="security-chart" width="400" height="200"></canvas>
                </div>
            </div>
            
            <div class="card">
                <h3>System Alerts</h3>
                <div id="alerts-container">
                    <p>Loading alerts...</p>
                </div>
            </div>
        </div>
        
        <script>
            // Initialize dashboard
            document.addEventListener('DOMContentLoaded', function() {
                loadDashboardData();
                loadAlerts();
                initializeChart();
            });
            
            async function loadDashboardData() {
                try {
                    const response = await fetch('/api/v1/deployment/status');
                    const data = await response.json();
                    console.log('Dashboard data:', data);
                } catch (error) {
                    console.error('Error loading dashboard data:', error);
                }
            }
            
            async function loadAlerts() {
                try {
                    const response = await fetch('/api/v1/alerts?limit=10');
                    const alerts = await response.json();
                    
                    const container = document.getElementById('alerts-container');
                    if (alerts.length === 0) {
                        container.innerHTML = '<p>No alerts found</p>';
                        return;
                    }
                    
                    container.innerHTML = alerts.map(alert => `
                        <div class="alert alert-${alert.severity}">
                            <strong>${alert.alert_type}</strong> - ${alert.message}
                            <br><small>User: ${alert.user_id} | ${new Date(alert.created_at).toLocaleString()}</small>
                        </div>
                    `).join('');
                } catch (error) {
                    console.error('Error loading alerts:', error);
                }
            }
            
            function initializeChart() {
                const ctx = document.getElementById('security-chart').getContext('2d');
                new Chart(ctx, {
                    type: 'line',
                    data: {
                        labels: ['Mon', 'Tue', 'Wed', 'Thu', 'Fri', 'Sat', 'Sun'],
                        datasets: [{
                            label: 'Deepfake Detections',
                            data: [12, 19, 3, 5, 2, 3, 7],
                            borderColor: '#e74c3c',
                            backgroundColor: 'rgba(231, 76, 60, 0.1)'
                        }, {
                            label: 'Identity Verifications',
                            data: [8, 15, 12, 18, 14, 6, 11],
                            borderColor: '#27ae60',
                            backgroundColor: 'rgba(39, 174, 96, 0.1)'
                        }]
                    },
                    options: {
                        responsive: true,
                        scales: {
                            y: {
                                beginAtZero: true
                            }
                        }
                    }
                });
            }
            
            function startVideoKYC() {
                alert('Video KYC functionality would be implemented here');
            }
            
            function checkDeepfake() {
                alert('Deepfake detection functionality would be implemented here');
            }
            
            function viewAlerts() {
                loadAlerts();
            }
            
            function refreshDashboard() {
                loadDashboardData();
                loadAlerts();
            }
        </script>
    </body>
    </html>
    """

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host=settings.api_host, port=settings.api_port)
