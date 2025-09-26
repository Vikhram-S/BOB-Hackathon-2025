"""
Advanced Deepfake Detection System using multiple AI techniques
"""
import cv2
import numpy as np
import tensorflow as tf
import mediapipe as mp
from typing import List, Tuple, Dict, Optional
import logging
from pathlib import Path
import json
from datetime import datetime

logger = logging.getLogger(__name__)

class DeepfakeDetector:
    def __init__(self, model_path: str = None):
        self.model_path = model_path
        self.mp_face_detection = mp.solutions.face_detection
        self.mp_face_mesh = mp.solutions.face_mesh
        self.mp_drawing = mp.solutions.drawing_utils
        
        # Initialize models
        self.face_detection = self.mp_face_detection.FaceDetection(
            model_selection=1, min_detection_confidence=0.5
        )
        self.face_mesh = self.mp_face_mesh.FaceMesh(
            static_image_mode=True,
            max_num_faces=1,
            refine_landmarks=True,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5
        )
        
        # Load deepfake detection model if available
        self.deepfake_model = None
        if model_path and Path(model_path).exists():
            try:
                self.deepfake_model = tf.keras.models.load_model(model_path)
                logger.info("Deepfake detection model loaded successfully")
            except Exception as e:
                logger.warning(f"Could not load deepfake model: {e}")
    
    def extract_frames(self, video_path: str, max_frames: int = 30) -> List[np.ndarray]:
        """Extract frames from video for analysis"""
        frames = []
        cap = cv2.VideoCapture(video_path)
        
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        frame_interval = max(1, total_frames // max_frames)
        
        frame_count = 0
        while cap.isOpened() and len(frames) < max_frames:
            ret, frame = cap.read()
            if not ret:
                break
                
            if frame_count % frame_interval == 0:
                frames.append(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
            
            frame_count += 1
        
        cap.release()
        return frames
    
    def detect_facial_landmarks(self, frame: np.ndarray) -> Optional[List[Tuple[int, int]]]:
        """Detect facial landmarks using MediaPipe"""
        try:
            results = self.face_mesh.process(frame)
            if results.multi_face_landmarks:
                landmarks = []
                for face_landmarks in results.multi_face_landmarks:
                    h, w, _ = frame.shape
                    for landmark in face_landmarks.landmark:
                        x = int(landmark.x * w)
                        y = int(landmark.y * h)
                        landmarks.append((x, y))
                return landmarks
        except Exception as e:
            logger.error(f"Error detecting landmarks: {e}")
        return None
    
    def analyze_facial_consistency(self, frames: List[np.ndarray]) -> Dict:
        """Analyze facial consistency across frames"""
        if len(frames) < 2:
            return {"consistency_score": 0.0, "anomalies": []}
        
        landmarks_history = []
        consistency_scores = []
        
        for frame in frames:
            landmarks = self.detect_facial_landmarks(frame)
            if landmarks:
                landmarks_history.append(landmarks)
        
        if len(landmarks_history) < 2:
            return {"consistency_score": 0.0, "anomalies": ["insufficient_landmarks"]}
        
        # Calculate consistency between consecutive frames
        for i in range(1, len(landmarks_history)):
            prev_landmarks = np.array(landmarks_history[i-1])
            curr_landmarks = np.array(landmarks_history[i])
            
            if len(prev_landmarks) == len(curr_landmarks):
                # Calculate Euclidean distance between corresponding landmarks
                distances = np.linalg.norm(prev_landmarks - curr_landmarks, axis=1)
                avg_distance = np.mean(distances)
                
                # Normalize consistency score (lower distance = higher consistency)
                consistency = max(0, 1 - (avg_distance / 50))  # 50 is threshold
                consistency_scores.append(consistency)
        
        avg_consistency = np.mean(consistency_scores) if consistency_scores else 0.0
        
        anomalies = []
        if avg_consistency < 0.7:
            anomalies.append("low_facial_consistency")
        
        return {
            "consistency_score": float(avg_consistency),
            "anomalies": anomalies,
            "frame_count": len(landmarks_history)
        }
    
    def detect_eye_blinking_patterns(self, frames: List[np.ndarray]) -> Dict:
        """Detect unnatural eye blinking patterns"""
        eye_ratios = []
        
        for frame in frames:
            landmarks = self.detect_facial_landmarks(frame)
            if landmarks and len(landmarks) >= 468:  # Full face mesh has 468 landmarks
                # Extract eye landmarks (approximate indices for left and right eyes)
                left_eye_indices = [33, 7, 163, 144, 145, 153, 154, 155, 133, 173, 157, 158, 159, 160, 161, 246]
                right_eye_indices = [362, 382, 381, 380, 374, 373, 390, 249, 263, 466, 388, 387, 386, 385, 384, 398]
                
                # Calculate eye aspect ratio (EAR)
                def calculate_ear(eye_landmarks):
                    if len(eye_landmarks) < 6:
                        return 0
                    
                    # Vertical eye landmarks
                    A = np.linalg.norm(eye_landmarks[1] - eye_landmarks[5])
                    B = np.linalg.norm(eye_landmarks[2] - eye_landmarks[4])
                    # Horizontal eye landmark
                    C = np.linalg.norm(eye_landmarks[0] - eye_landmarks[3])
                    
                    ear = (A + B) / (2.0 * C)
                    return ear
                
                left_eye_landmarks = [landmarks[i] for i in left_eye_indices if i < len(landmarks)]
                right_eye_landmarks = [landmarks[i] for i in right_eye_indices if i < len(landmarks)]
                
                if len(left_eye_landmarks) >= 6 and len(right_eye_landmarks) >= 6:
                    left_ear = calculate_ear(left_eye_landmarks)
                    right_ear = calculate_ear(right_eye_landmarks)
                    avg_ear = (left_ear + right_ear) / 2
                    eye_ratios.append(avg_ear)
        
        if not eye_ratios:
            return {"blink_score": 0.0, "anomalies": ["no_eye_data"]}
        
        # Analyze blinking patterns
        blink_threshold = 0.25
        blinks = sum(1 for ratio in eye_ratios if ratio < blink_threshold)
        total_frames = len(eye_ratios)
        blink_rate = blinks / total_frames if total_frames > 0 else 0
        
        anomalies = []
        if blink_rate < 0.1:  # Very low blink rate
            anomalies.append("unnatural_blink_pattern")
        elif blink_rate > 0.5:  # Very high blink rate
            anomalies.append("excessive_blinking")
        
        return {
            "blink_score": float(blink_rate),
            "total_blinks": blinks,
            "anomalies": anomalies
        }
    
    def detect_face_swapping_artifacts(self, frames: List[np.ndarray]) -> Dict:
        """Detect face swapping artifacts"""
        artifacts_detected = []
        
        for i, frame in enumerate(frames):
            # Convert to grayscale for edge detection
            gray = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)
            
            # Detect edges
            edges = cv2.Canny(gray, 50, 150)
            
            # Look for unnatural boundaries around face area
            # This is a simplified approach - in practice, you'd use more sophisticated methods
            face_region = edges[100:400, 100:400]  # Approximate face region
            
            # Count edge pixels in face region
            edge_density = np.sum(face_region > 0) / (face_region.shape[0] * face_region.shape[1])
            
            if edge_density > 0.3:  # High edge density might indicate artifacts
                artifacts_detected.append(i)
        
        artifact_score = len(artifacts_detected) / len(frames) if frames else 0
        
        anomalies = []
        if artifact_score > 0.3:
            anomalies.append("face_swapping_artifacts")
        
        return {
            "artifact_score": float(artifact_score),
            "artifacts_detected": artifacts_detected,
            "anomalies": anomalies
        }
    
    def predict_with_ml_model(self, frames: List[np.ndarray]) -> float:
        """Use ML model to predict deepfake probability"""
        if not self.deepfake_model or not frames:
            return 0.5  # Neutral score if no model or frames
        
        try:
            # Preprocess frames for the model
            processed_frames = []
            for frame in frames:
                # Resize to model input size (assuming 224x224)
                resized = cv2.resize(frame, (224, 224))
                normalized = resized / 255.0
                processed_frames.append(normalized)
            
            # Convert to numpy array and add batch dimension
            X = np.array(processed_frames)
            if len(X.shape) == 3:
                X = np.expand_dims(X, axis=0)
            
            # Make prediction
            predictions = self.deepfake_model.predict(X, verbose=0)
            return float(np.mean(predictions))
        
        except Exception as e:
            logger.error(f"Error in ML model prediction: {e}")
            return 0.5
    
    def detect_deepfake(self, video_path: str) -> Dict:
        """Main deepfake detection function"""
        try:
            # Extract frames
            frames = self.extract_frames(video_path)
            
            if not frames:
                return {
                    "is_deepfake": False,
                    "confidence": 0.0,
                    "error": "No frames extracted from video"
                }
            
            # Run various detection methods
            facial_consistency = self.analyze_facial_consistency(frames)
            blinking_patterns = self.detect_eye_blinking_patterns(frames)
            face_artifacts = self.detect_face_swapping_artifacts(frames)
            ml_prediction = self.predict_with_ml_model(frames)
            
            # Combine all scores
            consistency_score = facial_consistency["consistency_score"]
            blink_score = blinking_patterns["blink_score"]
            artifact_score = face_artifacts["artifact_score"]
            
            # Weighted combination of scores
            weights = {
                "consistency": 0.3,
                "blinking": 0.2,
                "artifacts": 0.2,
                "ml_model": 0.3
            }
            
            # Calculate final deepfake probability
            deepfake_probability = (
                (1 - consistency_score) * weights["consistency"] +
                (1 - blink_score) * weights["blinking"] +
                artifact_score * weights["artifacts"] +
                ml_prediction * weights["ml_model"]
            )
            
            # Collect all anomalies
            all_anomalies = []
            all_anomalies.extend(facial_consistency.get("anomalies", []))
            all_anomalies.extend(blinking_patterns.get("anomalies", []))
            all_anomalies.extend(face_artifacts.get("anomalies", []))
            
            # Determine if video is likely a deepfake
            is_deepfake = deepfake_probability > 0.6
            
            return {
                "is_deepfake": is_deepfake,
                "confidence": float(deepfake_probability),
                "detailed_scores": {
                    "facial_consistency": facial_consistency,
                    "blinking_patterns": blinking_patterns,
                    "face_artifacts": face_artifacts,
                    "ml_prediction": ml_prediction
                },
                "anomalies": all_anomalies,
                "frame_count": len(frames),
                "timestamp": datetime.utcnow().isoformat()
            }
        
        except Exception as e:
            logger.error(f"Error in deepfake detection: {e}")
            return {
                "is_deepfake": False,
                "confidence": 0.0,
                "error": str(e)
            }
