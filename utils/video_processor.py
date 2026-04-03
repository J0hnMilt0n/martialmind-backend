import cv2
import numpy as np
from typing import List, Tuple, Optional


class VideoProcessor:
    """
    Handles video file processing and frame extraction.
    """
    
    def __init__(self, max_frames: int = 300):
        """
        Args:
            max_frames: Maximum number of frames to process (for performance)
        """
        self.max_frames = max_frames
    
    def extract_frames(self, video_path: str, sample_rate: int = 1) -> List[np.ndarray]:
        """
        Extract frames from video file.
        
        Args:
            video_path: Path to video file
            sample_rate: Extract every Nth frame (1 = all frames)
            
        Returns:
            List of frames as numpy arrays
        """
        frames = []
        
        cap = cv2.VideoCapture(video_path)
        
        if not cap.isOpened():
            raise ValueError(f"Could not open video file: {video_path}")
        
        frame_count = 0
        extracted_count = 0
        
        while True:
            ret, frame = cap.read()
            
            if not ret:
                break
            
            # Sample frames based on sample_rate
            if frame_count % sample_rate == 0:
                frames.append(frame)
                extracted_count += 1
                
                # Stop if we've reached max frames
                if extracted_count >= self.max_frames:
                    break
            
            frame_count += 1
        
        cap.release()
        
        print(f"Extracted {len(frames)} frames from {frame_count} total frames")
        
        return frames
    
    def extract_poses(self, video_path: str, pose_estimator) -> Tuple[List[np.ndarray], List[dict]]:
        """
        Extract frames and estimate poses.
        
        Args:
            video_path: Path to video file
            pose_estimator: PoseEstimator instance
            
        Returns:
            Tuple of (frames, poses)
        """
        # Extract frames
        frames = self.extract_frames(video_path, sample_rate=2)  # Every 2nd frame for performance
        
        if not frames:
            raise ValueError("No frames could be extracted from video")
        
        # Estimate poses
        poses = []
        for i, frame in enumerate(frames):
            pose_data = pose_estimator.estimate_pose(frame)
            
            if pose_data is not None:
                poses.append(pose_data)
            
            # Progress indicator
            if (i + 1) % 10 == 0:
                print(f"Processed {i + 1}/{len(frames)} frames")
        
        print(f"Successfully detected poses in {len(poses)}/{len(frames)} frames")
        
        return frames, poses
    
    def get_video_info(self, video_path: str) -> dict:
        """
        Get video metadata.
        
        Args:
            video_path: Path to video file
            
        Returns:
            Dictionary with video information
        """
        cap = cv2.VideoCapture(video_path)
        
        if not cap.isOpened():
            raise ValueError(f"Could not open video file: {video_path}")
        
        info = {
            'frame_count': int(cap.get(cv2.CAP_PROP_FRAME_COUNT)),
            'fps': cap.get(cv2.CAP_PROP_FPS),
            'width': int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)),
            'height': int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)),
            'duration_seconds': None
        }
        
        if info['fps'] > 0:
            info['duration_seconds'] = info['frame_count'] / info['fps']
        
        cap.release()
        
        return info
    
    def resize_frame(self, frame: np.ndarray, max_width: int = 640) -> np.ndarray:
        """
        Resize frame while maintaining aspect ratio.
        
        Args:
            frame: Input frame
            max_width: Maximum width in pixels
            
        Returns:
            Resized frame
        """
        height, width = frame.shape[:2]
        
        if width <= max_width:
            return frame
        
        ratio = max_width / width
        new_height = int(height * ratio)
        
        resized = cv2.resize(frame, (max_width, new_height), interpolation=cv2.INTER_AREA)
        
        return resized
