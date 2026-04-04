import numpy as np
import cv2
import mediapipe as mp
from typing import List, Tuple, Optional
from config import settings


class PoseEstimator:
    """
    Pose estimation using MediaPipe Pose.
    Extracts 33 body landmarks from video frames.
    More robust than MoveNet but slightly larger footprint.
    """
    
    # MediaPipe landmark names (33 keypoints)
    LANDMARK_NAMES = [
        'nose', 'left_eye_inner', 'left_eye', 'left_eye_outer', 'right_eye_inner',
        'right_eye', 'right_eye_outer', 'left_ear', 'right_ear', 'mouth_left',
        'mouth_right', 'left_shoulder', 'right_shoulder', 'left_elbow', 'right_elbow',
        'left_wrist', 'right_wrist', 'left_pinky', 'right_pinky', 'left_index',
        'right_index', 'left_thumb', 'right_thumb', 'left_hip', 'right_hip',
        'left_knee', 'right_knee', 'left_ankle', 'right_ankle', 'left_heel',
        'right_heel', 'left_foot_index', 'right_foot_index'
    ]
    
    def __init__(self):
        # Initialize MediaPipe Pose
        self.pose = mp.solutions.pose.Pose(
            static_image_mode=False,
            model_complexity=1,
            smooth_landmarks=True,
            enable_segmentation=False,
            min_detection_confidence=settings.MIN_DETECTION_CONFIDENCE,
            min_tracking_confidence=0.5
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
        
        # Process the frame
        results = self.pose.process(frame_rgb)
        
        if not results.pose_landmarks:
            return None
        
        # Extract landmarks
        landmarks = []
        scores = []
        height, width = frame.shape[:2]
        
        for i, landmark in enumerate(results.pose_landmarks.landmark):
            # Get visibility score
            visibility = landmark.visibility if hasattr(landmark, 'visibility') else 1.0
            
            # Filter by confidence threshold
            if visibility >= settings.MIN_DETECTION_CONFIDENCE:
                landmarks.append({
                    'id': i,
                    'name': self.LANDMARK_NAMES[i] if i < len(self.LANDMARK_NAMES) else f'point_{i}',
                    'x': landmark.x * width,
                    'y': landmark.y * height,
                    'z': landmark.z if landmark.z else 0.0,
                    'visibility': float(visibility)
                })
            
            scores.append(float(visibility))
        
        if not landmarks:
            return None
        
        return {
            'landmarks': landmarks,
            'scores': scores,
            'num_landmarks': len(landmarks)
        }
    
    def get_landmark(self, pose_data: dict, landmark_id: int) -> Optional[dict]:
        """
        Get specific landmark by ID.
        
        Args:
            pose_data: Pose data dictionary from estimate_pose()
            landmark_id: MediaPipe landmark ID (0-32)
            
        Returns:
            Landmark dictionary or None
        """
        if not pose_data or landmark_id >= len(pose_data['landmarks']):
            return None
        
        return pose_data['landmarks'][landmark_id]
    
    def close(self):
        """Release resources"""
        self.pose.close()


# MediaPipe Pose Landmark IDs (33 keypoints)
POSE_LANDMARKS = {
    'NOSE': 0,
    'LEFT_EYE_INNER': 1,
    'LEFT_EYE': 2,
    'LEFT_EYE_OUTER': 3,
    'RIGHT_EYE_INNER': 4,
    'RIGHT_EYE': 5,
    'RIGHT_EYE_OUTER': 6,
    'LEFT_EAR': 7,
    'RIGHT_EAR': 8,
    'MOUTH_LEFT': 9,
    'MOUTH_RIGHT': 10,
    'LEFT_SHOULDER': 11,
    'RIGHT_SHOULDER': 12,
    'LEFT_ELBOW': 13,
    'RIGHT_ELBOW': 14,
    'LEFT_WRIST': 15,
    'RIGHT_WRIST': 16,
    'LEFT_PINKY': 17,
    'RIGHT_PINKY': 18,
    'LEFT_INDEX': 19,
    'RIGHT_INDEX': 20,
    'LEFT_THUMB': 21,
    'RIGHT_THUMB': 22,
    'LEFT_HIP': 23,
    'RIGHT_HIP': 24,
    'LEFT_KNEE': 25,
    'RIGHT_KNEE': 26,
    'LEFT_ANKLE': 27,
    'RIGHT_ANKLE': 28,
    'LEFT_HEEL': 29,
    'RIGHT_HEEL': 30,
    'LEFT_FOOT_INDEX': 31,
    'RIGHT_FOOT_INDEX': 32
}