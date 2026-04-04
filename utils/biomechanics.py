import numpy as np
import math
from typing import Tuple, Optional, List


def calculate_angle(point1: Tuple[float, float], 
                   point2: Tuple[float, float], 
                   point3: Tuple[float, float]) -> float:
    """
    Calculate angle at point2 formed by point1-point2-point3.
    
    Args:
        point1: First point (x, y)
        point2: Vertex point (x, y)
        point3: Third point (x, y)
        
    Returns:
        Angle in degrees
    """
    # Vector from point2 to point1
    vector1 = np.array([point1[0] - point2[0], point1[1] - point2[1]])
    
    # Vector from point2 to point3
    vector2 = np.array([point3[0] - point2[0], point3[1] - point2[1]])
    
    # Calculate angle using dot product
    dot_product = np.dot(vector1, vector2)
    
    # Calculate magnitudes
    magnitude1 = np.linalg.norm(vector1)
    magnitude2 = np.linalg.norm(vector2)
    
    # Avoid division by zero
    if magnitude1 == 0 or magnitude2 == 0:
        return 0.0
    
    # Calculate cosine of angle
    cos_angle = dot_product / (magnitude1 * magnitude2)
    
    # Clamp to [-1, 1] to avoid numerical errors with arccos
    cos_angle = np.clip(cos_angle, -1.0, 1.0)
    
    # Calculate angle in radians then convert to degrees
    angle_radians = np.arccos(cos_angle)
    angle_degrees = math.degrees(angle_radians)
    
    return angle_degrees


def calculate_distance(point1: Tuple[float, float], 
                      point2: Tuple[float, float]) -> float:
    """
    Calculate Euclidean distance between two points.
    
    Args:
        point1: First point (x, y)
        point2: Second point (x, y)
        
    Returns:
        Distance between points
    """
    return math.sqrt((point2[0] - point1[0])**2 + (point2[1] - point1[1])**2)


def calculate_center_of_mass(landmarks: List[dict]) -> Optional[Tuple[float, float]]:
    """
    Calculate approximate center of mass from body landmarks.
    
    Args:
        landmarks: List of landmark dictionaries with 'x' and 'y' coordinates
        
    Returns:
        Center of mass (x, y) or None if insufficient data
    """
    if not landmarks or len(landmarks) < 4:
        return None
    
    # Use hip and shoulder landmarks for COM estimation
    # Our landmark indices: 2=right_shoulder, 5=left_shoulder, 8=right_hip, 11=left_hip
    try:
        # Try to use available landmarks
        shoulder_points = []
        hip_points = []
        
        # Check for shoulders (indices 2 and 5)
        if 2 < len(landmarks):
            shoulder_points.append(landmarks[2])
        if 5 < len(landmarks):
            shoulder_points.append(landmarks[5])
        
        # Check for hips (indices 8 and 11)
        if 8 < len(landmarks):
            hip_points.append(landmarks[8])
        if 11 < len(landmarks):
            hip_points.append(landmarks[11])
        
        all_points = shoulder_points + hip_points
        
        if len(all_points) < 2:
            return None
        
        # Average of key points
        com_x = sum(p['x'] for p in all_points) / len(all_points)
        com_y = sum(p['y'] for p in all_points) / len(all_points)
        
        return (com_x, com_y)
    except (IndexError, KeyError):
        return None


def calculate_velocity(point_curr: dict, 
                      point_next: dict, 
                      time_delta: float) -> float:
    """
    Calculate velocity between two landmark positions.
    
    Args:
        point_curr: Current landmark position with 'x' and 'y'
        point_next: Next landmark position with 'x' and 'y'
        time_delta: Time between frames in seconds
        
    Returns:
        Velocity (distance/time)
    """
    if time_delta == 0:
        return 0.0
    
    distance = calculate_distance(
        (point_curr['x'], point_curr['y']),
        (point_next['x'], point_next['y'])
    )
    
    velocity = distance / time_delta
    
    return velocity


def calculate_acceleration(velocity_curr: float, 
                          velocity_next: float, 
                          time_delta: float) -> float:
    """
    Calculate acceleration between two velocity measurements.
    
    Args:
        velocity_curr: Current velocity
        velocity_next: Next velocity
        time_delta: Time between measurements in seconds
        
    Returns:
        Acceleration (velocity change/time)
    """
    if time_delta == 0:
        return 0.0
    
    acceleration = (velocity_next - velocity_curr) / time_delta
    
    return acceleration


def normalize_landmarks(landmarks: List[dict], 
                       reference_distance: Optional[float] = None) -> List[dict]:
    """
    Normalize landmarks to account for different distances from camera.
    
    Args:
        landmarks: List of landmark dictionaries
        reference_distance: Optional reference distance for normalization
        
    Returns:
        Normalized landmarks
    """
    if not landmarks or len(landmarks) < 6:
        return landmarks
    
    # Use shoulder width as reference distance if not provided
    if reference_distance is None:
        try:
            # Our landmark indices: 2=right_shoulder, 5=left_shoulder
            right_shoulder = landmarks[2]
            left_shoulder = landmarks[5]
            reference_distance = calculate_distance(
                (left_shoulder['x'], left_shoulder['y']),
                (right_shoulder['x'], right_shoulder['y'])
            )
        except (IndexError, KeyError):
            return landmarks
    
    if reference_distance == 0:
        return landmarks
    
    # Normalize all landmarks
    normalized = []
    for landmark in landmarks:
        normalized.append({
            'x': landmark['x'] / reference_distance,
            'y': landmark['y'] / reference_distance,
            'z': landmark.get('z', 0) / reference_distance,
            'visibility': landmark.get('visibility', 1.0)
        })
    
    return normalized


def smooth_landmarks(landmark_sequence: List[List[dict]], 
                    window_size: int = 3) -> List[List[dict]]:
    """
    Apply smoothing to landmark sequence to reduce jitter.
    
    Args:
        landmark_sequence: List of frames, each containing list of landmarks
        window_size: Size of smoothing window (odd number)
        
    Returns:
        Smoothed landmark sequence
    """
    if len(landmark_sequence) < window_size:
        return landmark_sequence
    
    smoothed = []
    half_window = window_size // 2
    
    for i in range(len(landmark_sequence)):
        # Get window range
        start_idx = max(0, i - half_window)
        end_idx = min(len(landmark_sequence), i + half_window + 1)
        
        # Get landmarks in window
        window = landmark_sequence[start_idx:end_idx]
        
        # Average landmarks
        averaged_landmarks = []
        for landmark_idx in range(len(landmark_sequence[i])):
            x_values = [frame[landmark_idx]['x'] for frame in window]
            y_values = [frame[landmark_idx]['y'] for frame in window]
            z_values = [frame[landmark_idx].get('z', 0) for frame in window]
            
            averaged_landmarks.append({
                'x': np.mean(x_values),
                'y': np.mean(y_values),
                'z': np.mean(z_values),
                'visibility': landmark_sequence[i][landmark_idx].get('visibility', 1.0)
            })
        
        smoothed.append(averaged_landmarks)
    
    return smoothed
