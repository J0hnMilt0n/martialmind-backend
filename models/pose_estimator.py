import numpy as np
import cv2
import urllib.request
import os
import tflite_runtime.interpreter as tflite
from typing import List, Tuple, Optional
from config import settings


class PoseEstimator:
    """
    Lightweight pose estimation using MoveNet via TensorFlow Lite.
    Extracts 17 body landmarks from video frames.
    Much smaller footprint than MediaPipe (~50MB vs ~500MB).
    """
    
    # MoveNet model URL - using direct download link to get actual TFLite file
    # The tfhub.dev URL returns HTML, so we use direct file URLs
    MODEL_URL = "https://storage.googleapis.com/download.tensorflow.org/models/tflite/posenet/movenet_singlepose_lightning_ptq_v5_128x128_float16_1.tflite"
    MODEL_PATH = "/tmp/movenet_lightning_f16.tflite"
    
    # Fallback URLs if the primary fails (different input sizes but compatible)
    FALLBACK_MODEL_URLS = [
        # 192x192 version
        "https://storage.googleapis.com/download.tensorflow.org/models/tflite/posenet/movenet_singlepose_lightning_ptq_v5_192x192_float16_1.tflite",
        # 256x256 version  
        "https://storage.googleapis.com/download.tensorflow.org/models/tflite/posenet/movenet_singlepose_lightning_ptq_v5_256x256_float16_1.tflite",
    ]
    
    # MoveNet landmark names (17 keypoints)
    LANDMARK_NAMES = [
        'nose', 'left_eye', 'right_eye', 'left_ear', 'right_ear',
        'left_shoulder', 'right_shoulder', 'left_elbow', 'right_elbow',
        'left_wrist', 'right_wrist', 'left_hip', 'right_hip',
        'left_knee', 'right_knee', 'left_ankle', 'right_ankle'
    ]
    
    def __init__(self):
        # Download model if not exists
        if not os.path.exists(self.MODEL_PATH):
            self._download_model()
        
        # Initialize TensorFlow Lite interpreter
        self.interpreter = tflite.Interpreter(model_path=self.MODEL_PATH)
        self.interpreter.allocate_tensors()
        
        # Get input and output details
        self.input_details = self.interpreter.get_input_details()
        self.output_details = self.interpreter.get_output_details()
        
        # Get model input shape
        self.input_height = self.input_details[0]['shape'][1]
        self.input_width = self.input_details[0]['shape'][2]
    
    def _download_model(self):
        """Download MoveNet model from a reliable source"""
        print(f"Downloading MoveNet model to {self.MODEL_PATH}...")
        
        # Try primary URL first, then fallbacks
        urls_to_try = [self.MODEL_URL] + self.FALLBACK_MODEL_URLS
        
        for url in urls_to_try:
            try:
                print(f"  Trying: {url}")
                urllib.request.urlretrieve(url, self.MODEL_PATH)
                
                # Verify the downloaded file is a valid TFLite model
                with open(self.MODEL_PATH, 'rb') as f:
                    magic = f.read(4)
                    if magic == b'TFL3':
                        print("MoveNet model downloaded successfully")
                        return
                    else:
                        print(f"  Invalid model file (magic bytes: {magic}), trying next URL...")
                        continue
            except Exception as e:
                print(f"  Failed to download from {url}: {e}")
                continue
        
        raise RuntimeError(
            "Failed to download MoveNet model from all sources. "
            "Please ensure the model file is available or check network connectivity."
        )
    
    def estimate_pose(self, frame: np.ndarray) -> Optional[dict]:
        """
        Estimate pose landmarks from a single frame.
        
        Args:
            frame: BGR image from OpenCV
            
        Returns:
            Dictionary containing landmarks and metadata, or None if no person detected
        """
        # Convert BGR to RGB
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        
        # Resize and normalize input
        input_frame = cv2.resize(frame_rgb, (self.input_width, self.input_height))
        input_frame = input_frame.astype(np.float32) / 127.5 - 1.0
        input_frame = np.expand_dims(input_frame, axis=0)
        
        # Run inference
        self.interpreter.set_tensor(self.input_details[0]['index'], input_frame)
        self.interpreter.invoke()
        
        # Get output
        keypoints = self.interpreter.get_tensor(self.output_details[0]['index'])[0]
        scores = self.interpreter.get_tensor(self.output_details[1]['index'])[0]
        
        # Filter keypoints by confidence threshold
        confidence_threshold = settings.MIN_DETECTION_CONFIDENCE
        landmarks = []
        
        for i, (keypoint, score) in enumerate(zip(keypoints, scores)):
            if score >= confidence_threshold:
                landmarks.append({
                    'id': i,
                    'name': self.LANDMARK_NAMES[i],
                    'x': float(keypoint[1]),  # x coordinate (normalized)
                    'y': float(keypoint[0]),  # y coordinate (normalized)
                    'z': float(keypoint[2]),  # z coordinate (depth, less accurate in 2D)
                    'visibility': float(score)
                })
        
        if not landmarks:
            return None
        
        # Convert normalized coordinates to pixel coordinates
        height, width = frame.shape[:2]
        for landmark in landmarks:
            landmark['x'] = landmark['x'] * width
            landmark['y'] = landmark['y'] * height
        
        return {
            'landmarks': landmarks,
            'scores': scores.tolist(),
            'num_landmarks': len(landmarks)
        }
    
    def get_landmark(self, pose_data: dict, landmark_id: int) -> Optional[dict]:
        """
        Get specific landmark by ID.
        
        Args:
            pose_data: Pose data dictionary from estimate_pose()
            landmark_id: MoveNet landmark ID (0-16)
            
        Returns:
            Landmark dictionary or None
        """
        if not pose_data or landmark_id >= len(pose_data['landmarks']):
            return None
        
        return pose_data['landmarks'][landmark_id]
    
    def close(self):
        """Release resources"""
        # TensorFlow Lite doesn't require explicit cleanup
        pass


# MoveNet Pose Landmark IDs (17 keypoints)
POSE_LANDMARKS = {
    'NOSE': 0,
    'LEFT_EYE': 1,
    'RIGHT_EYE': 2,
    'LEFT_EAR': 3,
    'RIGHT_EAR': 4,
    'LEFT_SHOULDER': 5,
    'RIGHT_SHOULDER': 6,
    'LEFT_ELBOW': 7,
    'RIGHT_ELBOW': 8,
    'LEFT_WRIST': 9,
    'RIGHT_WRIST': 10,
    'LEFT_HIP': 11,
    'RIGHT_HIP': 12,
    'LEFT_KNEE': 13,
    'RIGHT_KNEE': 14,
    'LEFT_ANKLE': 15,
    'RIGHT_ANKLE': 16
}