"""
Test cases for deepfake detection functionality
"""
import pytest
import numpy as np
import cv2
from pathlib import Path
import tempfile
import os

from services.deepfake_detector import DeepfakeDetector

class TestDeepfakeDetector:
    """Test cases for DeepfakeDetector class"""
    
    @pytest.fixture
    def detector(self):
        """Create a DeepfakeDetector instance for testing"""
        return DeepfakeDetector()
    
    @pytest.fixture
    def sample_video_path(self):
        """Create a sample video file for testing"""
        # Create a temporary video file
        temp_file = tempfile.NamedTemporaryFile(suffix='.mp4', delete=False)
        temp_file.close()
        
        # Create a simple test video
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(temp_file.name, fourcc, 20.0, (640, 480))
        
        # Write some frames
        for i in range(60):  # 3 seconds at 20 fps
            frame = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)
            out.write(frame)
        
        out.release()
        
        yield temp_file.name
        
        # Cleanup
        os.unlink(temp_file.name)
    
    def test_detector_initialization(self, detector):
        """Test detector initialization"""
        assert detector is not None
        assert detector.mp_face_detection is not None
        assert detector.mp_face_mesh is not None
    
    def test_extract_frames(self, detector, sample_video_path):
        """Test frame extraction from video"""
        frames = detector.extract_frames(sample_video_path, max_frames=10)
        
        assert len(frames) > 0
        assert len(frames) <= 10
        assert all(isinstance(frame, np.ndarray) for frame in frames)
        assert all(frame.shape[2] == 3 for frame in frames)  # RGB frames
    
    def test_detect_facial_landmarks(self, detector):
        """Test facial landmark detection"""
        # Create a test frame with a face-like pattern
        frame = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)
        
        landmarks = detector.detect_facial_landmarks(frame)
        
        # Should return None or a list of landmarks
        assert landmarks is None or isinstance(landmarks, list)
    
    def test_analyze_facial_consistency(self, detector):
        """Test facial consistency analysis"""
        # Create test frames
        frames = [
            np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)
            for _ in range(5)
        ]
        
        result = detector.analyze_facial_consistency(frames)
        
        assert isinstance(result, dict)
        assert "consistency_score" in result
        assert "anomalies" in result
        assert 0 <= result["consistency_score"] <= 1
    
    def test_detect_eye_blinking_patterns(self, detector):
        """Test eye blinking pattern detection"""
        # Create test frames
        frames = [
            np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)
            for _ in range(10)
        ]
        
        result = detector.detect_eye_blinking_patterns(frames)
        
        assert isinstance(result, dict)
        assert "blink_score" in result
        assert "anomalies" in result
        assert 0 <= result["blink_score"] <= 1
    
    def test_detect_face_swapping_artifacts(self, detector):
        """Test face swapping artifact detection"""
        # Create test frames
        frames = [
            np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)
            for _ in range(5)
        ]
        
        result = detector.detect_face_swapping_artifacts(frames)
        
        assert isinstance(result, dict)
        assert "artifact_score" in result
        assert "anomalies" in result
        assert 0 <= result["artifact_score"] <= 1
    
    def test_detect_deepfake_complete(self, detector, sample_video_path):
        """Test complete deepfake detection pipeline"""
        result = detector.detect_deepfake(sample_video_path)
        
        assert isinstance(result, dict)
        assert "is_deepfake" in result
        assert "confidence" in result
        assert isinstance(result["is_deepfake"], bool)
        assert 0 <= result["confidence"] <= 1
    
    def test_detect_deepfake_invalid_video(self, detector):
        """Test deepfake detection with invalid video path"""
        result = detector.detect_deepfake("nonexistent_video.mp4")
        
        assert isinstance(result, dict)
        assert "error" in result
        assert result["is_deepfake"] is False
        assert result["confidence"] == 0.0
    
    def test_detect_deepfake_empty_video(self, detector):
        """Test deepfake detection with empty video"""
        # Create an empty video file
        temp_file = tempfile.NamedTemporaryFile(suffix='.mp4', delete=False)
        temp_file.close()
        
        try:
            result = detector.detect_deepfake(temp_file.name)
            
            assert isinstance(result, dict)
            assert "error" in result or "is_deepfake" in result
        finally:
            os.unlink(temp_file.name)
    
    @pytest.mark.parametrize("max_frames", [5, 10, 30, 50])
    def test_extract_frames_different_limits(self, detector, sample_video_path, max_frames):
        """Test frame extraction with different frame limits"""
        frames = detector.extract_frames(sample_video_path, max_frames=max_frames)
        
        assert len(frames) <= max_frames
        assert all(isinstance(frame, np.ndarray) for frame in frames)
    
    def test_ml_model_prediction_without_model(self, detector):
        """Test ML model prediction when no model is loaded"""
        # Create test frames
        frames = [
            np.random.randint(0, 255, (224, 224, 3), dtype=np.uint8)
            for _ in range(3)
        ]
        
        prediction = detector.predict_with_ml_model(frames)
        
        # Should return neutral score (0.5) when no model is available
        assert prediction == 0.5
    
    def test_detect_deepfake_with_ml_model(self, detector):
        """Test deepfake detection with ML model (if available)"""
        # This test will pass even if no ML model is loaded
        # as the detector handles missing models gracefully
        
        # Create a test video
        temp_file = tempfile.NamedTemporaryFile(suffix='.mp4', delete=False)
        temp_file.close()
        
        try:
            # Create a simple test video
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            out = cv2.VideoWriter(temp_file.name, fourcc, 20.0, (640, 480))
            
            for i in range(30):  # 1.5 seconds at 20 fps
                frame = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)
                out.write(frame)
            
            out.release()
            
            result = detector.detect_deepfake(temp_file.name)
            
            assert isinstance(result, dict)
            assert "is_deepfake" in result
            assert "confidence" in result
            assert isinstance(result["is_deepfake"], bool)
            assert 0 <= result["confidence"] <= 1
            
        finally:
            os.unlink(temp_file.name)
