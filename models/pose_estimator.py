import mediapipe as mp
import cv2
import numpy as np
from typing import List, Tuple, Optional
from config import settings


class PoseEstimator:
    """
    Wrapper for MediaPipe Pose estimation.
    Extracts 33 body landmarks from video frames.
    """
    
    def __init__(self):
        self.mp_pose = mp.solutions.pose
        self.mp_drawing = mp.solutions.drawing_utils
        
        # Initialize pose estimator with configuration
        self.pose = self.mp_pose.Pose(
            model_complexity=settings.POSE_MODEL_COMPLEXITY,
            min_detection_confidence=settings.MIN_DETECTION_CONFIDENCE,
            min_tracking_confidence=settings.MIN_TRACKING_CONFIDENCE,
            enable_segmentation=False,
            smooth_landmarks=True
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
        
        # Process frame
        results = self.pose.process(frame_rgb)
        
        if not results.pose_landmarks:
            return None
        
        # Extract landmarks
        landmarks = []
        for landmark in results.pose_landmarks.landmark:
            landmarks.append({
                'x': landmark.x,
                'y': landmark.y,
                'z': landmark.z,
                'visibility': landmark.visibility
            })
        
        return {
            'landmarks': landmarks,
            'world_landmarks': results.pose_world_landmarks
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


# MediaPipe Pose Landmark IDs (for reference)
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
