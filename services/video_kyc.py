"""
Video KYC (Know Your Customer) Verification System with Deepfake Resistance
"""
import cv2
import numpy as np
import mediapipe as mp
from typing import Dict, List, Optional, Tuple
import logging
from datetime import datetime, timedelta
import json
import base64
from pathlib import Path
from sqlalchemy.orm import Session
from models.database import IdentityVerification, User
from services.deepfake_detector import DeepfakeDetector
from services.identity_monitor import IdentityMonitor

logger = logging.getLogger(__name__)

class VideoKYCVerifier:
    def __init__(self, deepfake_detector: DeepfakeDetector, identity_monitor: IdentityMonitor):
        self.deepfake_detector = deepfake_detector
        self.identity_monitor = identity_monitor
        
        # Initialize MediaPipe for face detection and landmarks
        self.mp_face_detection = mp.solutions.face_detection
        self.mp_face_mesh = mp.solutions.face_mesh
        self.mp_drawing = mp.solutions.drawing_utils
        
        self.face_detection = self.mp_face_detection.FaceDetection(
            model_selection=1, min_detection_confidence=0.7
        )
        self.face_mesh = self.mp_face_mesh.FaceMesh(
            static_image_mode=False,
            max_num_faces=1,
            refine_landmarks=True,
            min_detection_confidence=0.7,
            min_tracking_confidence=0.7
        )
    
    def extract_frames_from_video(self, video_path: str, target_frames: int = 30) -> List[np.ndarray]:
        """Extract frames from video for analysis"""
        frames = []
        cap = cv2.VideoCapture(video_path)
        
        if not cap.isOpened():
            raise ValueError("Could not open video file")
        
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        fps = cap.get(cv2.CAP_PROP_FPS)
        duration = total_frames / fps if fps > 0 else 0
        
        # Check video duration (should be between 10-60 seconds for KYC)
        if duration < 10 or duration > 60:
            cap.release()
            raise ValueError(f"Video duration {duration:.1f}s is not suitable for KYC (should be 10-60 seconds)")
        
        frame_interval = max(1, total_frames // target_frames)
        frame_count = 0
        
        while cap.isOpened() and len(frames) < target_frames:
            ret, frame = cap.read()
            if not ret:
                break
            
            if frame_count % frame_interval == 0:
                # Convert BGR to RGB for MediaPipe
                rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                frames.append(rgb_frame)
            
            frame_count += 1
        
        cap.release()
        return frames
    
    def detect_face_quality(self, frame: np.ndarray) -> Dict:
        """Assess face quality for KYC verification"""
        try:
            # Detect face
            results = self.face_detection.process(frame)
            
            if not results.detections:
                return {
                    "face_detected": False,
                    "quality_score": 0.0,
                    "issues": ["no_face_detected"]
                }
            
            detection = results.detections[0]
            bbox = detection.location_data.relative_bounding_box
            
            # Calculate face area
            face_area = bbox.width * bbox.height
            
            # Check face size (should be reasonable for KYC)
            if face_area < 0.1:  # Face too small
                return {
                    "face_detected": True,
                    "quality_score": 0.3,
                    "issues": ["face_too_small"]
                }
            
            # Check face position (should be centered)
            center_x = bbox.xmin + bbox.width / 2
            center_y = bbox.ymin + bbox.height / 2
            
            position_score = 1.0
            if center_x < 0.3 or center_x > 0.7:
                position_score -= 0.3
            if center_y < 0.3 or center_y > 0.7:
                position_score -= 0.3
            
            # Check face angle using landmarks
            landmarks_results = self.face_mesh.process(frame)
            angle_score = 1.0
            
            if landmarks_results.multi_face_landmarks:
                # Calculate face angle (simplified)
                face_landmarks = landmarks_results.multi_face_landmarks[0]
                h, w, _ = frame.shape
                
                # Get key facial points
                nose_tip = face_landmarks.landmark[1]  # Nose tip
                left_eye = face_landmarks.landmark[33]
                right_eye = face_landmarks.landmark[263]
                
                # Calculate face angle
                eye_distance = abs(left_eye.x - right_eye.x) * w
                if eye_distance > 0:
                    # Simple angle calculation
                    angle = abs(nose_tip.x - (left_eye.x + right_eye.x) / 2) * w / eye_distance
                    if angle > 0.3:  # Face turned too much
                        angle_score = 0.5
            
            # Calculate overall quality score
            quality_score = (face_area * 2 + position_score + angle_score) / 4
            
            issues = []
            if face_area < 0.15:
                issues.append("small_face")
            if position_score < 0.7:
                issues.append("poor_positioning")
            if angle_score < 0.7:
                issues.append("face_angle")
            
            return {
                "face_detected": True,
                "quality_score": float(quality_score),
                "face_area": float(face_area),
                "position_score": float(position_score),
                "angle_score": float(angle_score),
                "issues": issues
            }
            
        except Exception as e:
            logger.error(f"Error in face quality detection: {e}")
            return {
                "face_detected": False,
                "quality_score": 0.0,
                "issues": ["detection_error"]
            }
    
    def verify_liveness(self, frames: List[np.ndarray]) -> Dict:
        """Verify that the person in the video is alive and not a static image"""
        if len(frames) < 5:
            return {
                "liveness_score": 0.0,
                "is_live": False,
                "issues": ["insufficient_frames"]
            }
        
        try:
            # Check for movement between frames
            movement_scores = []
            for i in range(1, len(frames)):
                # Calculate optical flow between consecutive frames
                prev_gray = cv2.cvtColor(frames[i-1], cv2.COLOR_RGB2GRAY)
                curr_gray = cv2.cvtColor(frames[i], cv2.COLOR_RGB2GRAY)
                
                # Calculate frame difference
                diff = cv2.absdiff(prev_gray, curr_gray)
                movement = np.mean(diff)
                movement_scores.append(movement)
            
            avg_movement = np.mean(movement_scores)
            
            # Check for blinking patterns
            blink_scores = []
            for frame in frames:
                landmarks_results = self.face_mesh.process(frame)
                if landmarks_results.multi_face_landmarks:
                    face_landmarks = landmarks_results.multi_face_landmarks[0]
                    
                    # Calculate eye aspect ratio (simplified)
                    left_eye_landmarks = [face_landmarks.landmark[i] for i in [33, 7, 163, 144, 145, 153, 154, 155, 133, 173, 157, 158, 159, 160, 161, 246]]
                    right_eye_landmarks = [face_landmarks.landmark[i] for i in [362, 382, 381, 380, 374, 373, 390, 249, 263, 466, 388, 387, 386, 385, 384, 398]]
                    
                    # Simple eye openness calculation
                    left_eye_open = self._calculate_eye_openness(left_eye_landmarks)
                    right_eye_open = self._calculate_eye_openness(right_eye_landmarks)
                    
                    avg_eye_openness = (left_eye_open + right_eye_open) / 2
                    blink_scores.append(avg_eye_openness)
            
            # Analyze blinking pattern
            blink_variance = np.var(blink_scores) if blink_scores else 0
            
            # Check for natural head movement
            head_movement_scores = []
            for frame in frames:
                landmarks_results = self.face_mesh.process(frame)
                if landmarks_results.multi_face_landmarks:
                    face_landmarks = landmarks_results.multi_face_landmarks[0]
                    
                    # Get key facial points for head pose
                    nose_tip = face_landmarks.landmark[1]
                    chin = face_landmarks.landmark[175]
                    left_ear = face_landmarks.landmark[234]
                    right_ear = face_landmarks.landmark[454]
                    
                    # Calculate head pose (simplified)
                    head_pose = abs(nose_tip.x - (left_ear.x + right_ear.x) / 2)
                    head_movement_scores.append(head_pose)
            
            head_movement_variance = np.var(head_movement_scores) if head_movement_scores else 0
            
            # Calculate overall liveness score
            movement_score = min(1.0, avg_movement / 10)  # Normalize movement
            blink_score = min(1.0, blink_variance * 10)  # Normalize blink variance
            head_score = min(1.0, head_movement_variance * 10)  # Normalize head movement
            
            liveness_score = (movement_score + blink_score + head_score) / 3
            
            issues = []
            if movement_score < 0.1:
                issues.append("no_movement")
            if blink_score < 0.1:
                issues.append("no_blinking")
            if head_score < 0.1:
                issues.append("no_head_movement")
            
            return {
                "liveness_score": float(liveness_score),
                "is_live": liveness_score > 0.3,
                "movement_score": float(movement_score),
                "blink_score": float(blink_score),
                "head_movement_score": float(head_score),
                "issues": issues
            }
            
        except Exception as e:
            logger.error(f"Error in liveness verification: {e}")
            return {
                "liveness_score": 0.0,
                "is_live": False,
                "issues": ["verification_error"]
            }
    
    def _calculate_eye_openness(self, eye_landmarks: List) -> float:
        """Calculate eye openness from landmarks"""
        if len(eye_landmarks) < 6:
            return 0.5
        
        # Get key eye points
        top = eye_landmarks[1].y
        bottom = eye_landmarks[4].y
        left = eye_landmarks[0].x
        right = eye_landmarks[3].x
        
        # Calculate eye aspect ratio
        vertical_distance = abs(top - bottom)
        horizontal_distance = abs(left - right)
        
        if horizontal_distance > 0:
            ear = vertical_distance / horizontal_distance
            return min(1.0, ear * 2)  # Normalize
        
        return 0.5
    
    def verify_document_consistency(self, video_frames: List[np.ndarray], document_image: np.ndarray) -> Dict:
        """Verify consistency between video and document photo"""
        try:
            if not video_frames or document_image is None:
                return {
                    "consistency_score": 0.0,
                    "is_consistent": False,
                    "issues": ["missing_data"]
                }
            
            # Extract face from document
            doc_results = self.face_detection.process(document_image)
            if not doc_results.detections:
                return {
                    "consistency_score": 0.0,
                    "is_consistent": False,
                    "issues": ["no_face_in_document"]
                }
            
            # Extract faces from video frames
            video_faces = []
            for frame in video_frames:
                frame_results = self.face_detection.process(frame)
                if frame_results.detections:
                    detection = frame_results.detections[0]
                    bbox = detection.location_data.relative_bounding_box
                    
                    # Extract face region
                    h, w, _ = frame.shape
                    x = int(bbox.xmin * w)
                    y = int(bbox.ymin * h)
                    width = int(bbox.width * w)
                    height = int(bbox.height * h)
                    
                    face_region = frame[y:y+height, x:x+width]
                    if face_region.size > 0:
                        video_faces.append(face_region)
            
            if not video_faces:
                return {
                    "consistency_score": 0.0,
                    "is_consistent": False,
                    "issues": ["no_faces_in_video"]
                }
            
            # Simple face comparison (in production, use more sophisticated methods)
            consistency_scores = []
            for video_face in video_faces:
                # Resize faces to same size
                video_face_resized = cv2.resize(video_face, (100, 100))
                doc_face_resized = cv2.resize(document_image, (100, 100))
                
                # Convert to grayscale
                video_gray = cv2.cvtColor(video_face_resized, cv2.COLOR_RGB2GRAY)
                doc_gray = cv2.cvtColor(doc_face_resized, cv2.COLOR_RGB2GRAY)
                
                # Calculate similarity (simplified)
                diff = cv2.absdiff(video_gray, doc_gray)
                similarity = 1.0 - (np.mean(diff) / 255.0)
                consistency_scores.append(similarity)
            
            avg_consistency = np.mean(consistency_scores)
            
            issues = []
            if avg_consistency < 0.6:
                issues.append("low_face_similarity")
            
            return {
                "consistency_score": float(avg_consistency),
                "is_consistent": avg_consistency > 0.6,
                "face_count": len(video_faces),
                "issues": issues
            }
            
        except Exception as e:
            logger.error(f"Error in document consistency verification: {e}")
            return {
                "consistency_score": 0.0,
                "is_consistent": False,
                "issues": ["verification_error"]
            }
    
    def process_video_kyc(self, user_id: str, video_path: str, document_image: np.ndarray, db: Session) -> Dict:
        """Main Video KYC processing function"""
        try:
            # Extract frames from video
            frames = self.extract_frames_from_video(video_path)
            
            if not frames:
                return {
                    "verification_status": "failed",
                    "confidence_score": 0.0,
                    "error": "No frames extracted from video"
                }
            
            # Step 1: Deepfake detection
            deepfake_result = self.deepfake_detector.detect_deepfake(video_path)
            
            if deepfake_result.get("is_deepfake", False):
                # Create alert for deepfake detection
                from models.database import SystemAlert
                alert = SystemAlert(
                    alert_type="deepfake_detected",
                    severity="high",
                    message=f"Deepfake detected in KYC video for user {user_id}",
                    user_id=user_id,
                    alert_data=deepfake_result
                )
                db.add(alert)
                db.commit()
                
                return {
                    "verification_status": "rejected",
                    "confidence_score": 0.0,
                    "reason": "deepfake_detected",
                    "deepfake_score": deepfake_result.get("confidence", 0.0)
                }
            
            # Step 2: Face quality assessment
            quality_scores = []
            for frame in frames:
                quality_result = self.detect_face_quality(frame)
                quality_scores.append(quality_result.get("quality_score", 0.0))
            
            avg_quality_score = np.mean(quality_scores)
            
            if avg_quality_score < 0.5:
                return {
                    "verification_status": "failed",
                    "confidence_score": avg_quality_score,
                    "reason": "poor_face_quality"
                }
            
            # Step 3: Liveness verification
            liveness_result = self.verify_liveness(frames)
            
            if not liveness_result.get("is_live", False):
                return {
                    "verification_status": "failed",
                    "confidence_score": liveness_result.get("liveness_score", 0.0),
                    "reason": "liveness_failed"
                }
            
            # Step 4: Document consistency (if document provided)
            consistency_score = 1.0
            if document_image is not None:
                consistency_result = self.verify_document_consistency(frames, document_image)
                consistency_score = consistency_result.get("consistency_score", 0.0)
                
                if not consistency_result.get("is_consistent", False):
                    return {
                        "verification_status": "failed",
                        "confidence_score": consistency_score,
                        "reason": "document_inconsistency"
                    }
            
            # Step 5: Calculate overall confidence score
            overall_confidence = (
                (1 - deepfake_result.get("confidence", 0.0)) * 0.4 +  # Deepfake resistance
                avg_quality_score * 0.3 +  # Face quality
                liveness_result.get("liveness_score", 0.0) * 0.2 +  # Liveness
                consistency_score * 0.1  # Document consistency
            )
            
            # Step 6: Store verification result
            verification = IdentityVerification(
                user_id=user_id,
                verification_type="video_kyc",
                status="approved" if overall_confidence > 0.7 else "rejected",
                deepfake_score=deepfake_result.get("confidence", 0.0),
                confidence_score=overall_confidence,
                verification_data={
                    "face_quality_score": avg_quality_score,
                    "liveness_score": liveness_result.get("liveness_score", 0.0),
                    "consistency_score": consistency_score,
                    "deepfake_analysis": deepfake_result,
                    "frame_count": len(frames)
                }
            )
            db.add(verification)
            db.commit()
            
            # Step 7: Update user verification status
            user = db.query(User).filter(User.user_id == user_id).first()
            if user:
                user.is_verified = overall_confidence > 0.7
                user.verification_status = "approved" if overall_confidence > 0.7 else "rejected"
                db.commit()
            
            return {
                "verification_status": "approved" if overall_confidence > 0.7 else "rejected",
                "confidence_score": overall_confidence,
                "detailed_scores": {
                    "face_quality": avg_quality_score,
                    "liveness": liveness_result.get("liveness_score", 0.0),
                    "consistency": consistency_score,
                    "deepfake_resistance": 1 - deepfake_result.get("confidence", 0.0)
                },
                "verification_id": verification.id
            }
            
        except Exception as e:
            logger.error(f"Error in video KYC processing: {e}")
            return {
                "verification_status": "failed",
                "confidence_score": 0.0,
                "error": str(e)
            }
