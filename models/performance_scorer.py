import numpy as np
from typing import List, Dict, Tuple
from models.pose_estimator import POSE_LANDMARKS
from utils.biomechanics import (
    calculate_angle,
    calculate_distance,
    calculate_center_of_mass,
    calculate_velocity
)


class PerformanceScorer:
    """
    Analyzes pose data to score combat sports technique performance.
    Evaluates: technique quality, balance, power, speed, precision.
    """
    
    def __init__(self):
        self.weights = {
            'technique': 0.30,
            'balance': 0.25,
            'power': 0.20,
            'speed': 0.15,
            'precision': 0.10
        }
    
    def score_performance(self, poses: List[dict], frames: List[np.ndarray]) -> Dict:
        """
        Calculate overall performance score from pose sequence.
        
        Args:
            poses: List of pose data dictionaries
            frames: List of video frames
            
        Returns:
            Dictionary with score, subscores, and identified issues
        """
        if not poses or len(poses) == 0:
            return {
                'score': 0.0,
                'subscores': {},
                'issues': ['No pose data available']
            }
        
        # Calculate individual component scores
        technique_score, technique_issues = self._score_technique(poses)
        balance_score, balance_issues = self._score_balance(poses)
        power_score, power_issues = self._score_power(poses)
        speed_score, speed_issues = self._score_speed(poses)
        precision_score, precision_issues = self._score_precision(poses)
        
        # Calculate weighted total score
        total_score = (
            technique_score * self.weights['technique'] +
            balance_score * self.weights['balance'] +
            power_score * self.weights['power'] +
            speed_score * self.weights['speed'] +
            precision_score * self.weights['precision']
        )
        
        # Compile all issues
        all_issues = (
            technique_issues + 
            balance_issues + 
            power_issues + 
            speed_issues + 
            precision_issues
        )
        
        return {
            'score': total_score,
            'subscores': {
                'technique': technique_score,
                'balance': balance_score,
                'power': power_score,
                'speed': speed_score,
                'precision': precision_score
            },
            'issues': all_issues[:5]  # Return top 5 issues
        }
    
    def _score_technique(self, poses: List[dict]) -> Tuple[float, List[str]]:
        """Score technique quality (form, alignment, execution)"""
        issues = []
        score = 10.0
        
        # Analyze key frames (start, mid, end)
        if len(poses) < 3:
            return 5.0, ["Video too short for technique analysis"]
        
        mid_pose = poses[len(poses) // 2]
        landmarks = mid_pose['landmarks']
        
        # Check hip rotation (key for power generation)
        left_hip = landmarks[POSE_LANDMARKS['LEFT_HIP']]
        right_hip = landmarks[POSE_LANDMARKS['RIGHT_HIP']]
        hip_rotation = abs(left_hip['z'] - right_hip['z'])
        
        if hip_rotation < 0.05:  # Threshold for minimal rotation
            issues.append("Low hip rotation (weak power generation)")
            score -= 2.0
        
        # Check shoulder alignment
        left_shoulder = landmarks[POSE_LANDMARKS['LEFT_SHOULDER']]
        right_shoulder = landmarks[POSE_LANDMARKS['RIGHT_SHOULDER']]
        shoulder_level = abs(left_shoulder['y'] - right_shoulder['y'])
        
        if shoulder_level > 0.1:
            issues.append("Uneven shoulder position")
            score -= 1.5
        
        return max(0.0, score), issues
    
    def _score_balance(self, poses: List[dict]) -> Tuple[float, List[str]]:
        """Score balance and stability"""
        issues = []
        score = 10.0
        
        # Calculate center of mass stability across frames
        com_positions = []
        for pose in poses:
            com = calculate_center_of_mass(pose['landmarks'])
            if com:
                com_positions.append(com)
        
        if len(com_positions) < 2:
            return 5.0, ["Insufficient data for balance analysis"]
        
        # Calculate COM variance (lower = more stable)
        com_variance = np.var([pos[0] for pos in com_positions])
        
        if com_variance > 0.02:
            issues.append("Balance instability detected")
            score -= 2.5
        
        # Check base of support
        mid_pose = poses[len(poses) // 2]
        landmarks = mid_pose['landmarks']
        left_ankle = landmarks[POSE_LANDMARKS['LEFT_ANKLE']]
        right_ankle = landmarks[POSE_LANDMARKS['RIGHT_ANKLE']]
        
        stance_width = calculate_distance(
            (left_ankle['x'], left_ankle['y']),
            (right_ankle['x'], right_ankle['y'])
        )
        
        if stance_width < 0.15:
            issues.append("Narrow stance affecting balance")
            score -= 1.5
        
        return max(0.0, score), issues
    
    def _score_power(self, poses: List[dict]) -> Tuple[float, List[str]]:
        """Score power generation (kinetic chain, hip drive)"""
        issues = []
        score = 10.0
        
        if len(poses) < 5:
            return 5.0, ["Video too short for power analysis"]
        
        # Analyze hip drive
        hip_velocities = []
        for i in range(len(poses) - 1):
            hip_curr = poses[i]['landmarks'][POSE_LANDMARKS['LEFT_HIP']]
            hip_next = poses[i + 1]['landmarks'][POSE_LANDMARKS['LEFT_HIP']]
            velocity = calculate_velocity(hip_curr, hip_next, 1/30)  # Assuming 30 FPS
            hip_velocities.append(velocity)
        
        avg_hip_velocity = np.mean(hip_velocities) if hip_velocities else 0
        
        if avg_hip_velocity < 0.5:
            issues.append("Insufficient hip drive for power generation")
            score -= 2.0
        
        return max(0.0, score), issues
    
    def _score_speed(self, poses: List[dict]) -> Tuple[float, List[str]]:
        """Score movement speed and retraction"""
        issues = []
        score = 10.0
        
        if len(poses) < 5:
            return 5.0, ["Video too short for speed analysis"]
        
        # Analyze strike speed (using wrist movement as proxy)
        wrist_velocities = []
        for i in range(len(poses) - 1):
            wrist_curr = poses[i]['landmarks'][POSE_LANDMARKS['RIGHT_WRIST']]
            wrist_next = poses[i + 1]['landmarks'][POSE_LANDMARKS['RIGHT_WRIST']]
            velocity = calculate_velocity(wrist_curr, wrist_next, 1/30)
            wrist_velocities.append(velocity)
        
        max_velocity = max(wrist_velocities) if wrist_velocities else 0
        
        if max_velocity < 1.0:
            issues.append("Slow strike velocity")
            score -= 1.5
        
        # Check retraction (velocity should decrease after peak)
        if len(wrist_velocities) > 2:
            peak_idx = wrist_velocities.index(max_velocity)
            if peak_idx < len(wrist_velocities) - 2:
                retraction_velocity = np.mean(wrist_velocities[peak_idx+1:])
                if retraction_velocity > max_velocity * 0.5:
                    issues.append("Late retraction after strike")
                    score -= 1.0
        
        return max(0.0, score), issues
    
    def _score_precision(self, poses: List[dict]) -> Tuple[float, List[str]]:
        """Score movement precision and control"""
        issues = []
        score = 10.0
        
        # Analyze path consistency
        wrist_positions = []
        for pose in poses:
            wrist = pose['landmarks'][POSE_LANDMARKS['RIGHT_WRIST']]
            wrist_positions.append((wrist['x'], wrist['y']))
        
        # Calculate path straightness (simplified)
        if len(wrist_positions) > 2:
            path_variance = np.var([pos[1] for pos in wrist_positions])
            if path_variance > 0.05:
                issues.append("Inconsistent strike path")
                score -= 1.5
        
        return max(0.0, score), issues
    
    def recommend_drills(self, issues: List[str]) -> List[str]:
        """Generate drill recommendations based on identified issues"""
        drills = []
        
        issue_text = ' '.join(issues).lower()
        
        if 'hip rotation' in issue_text or 'hip drive' in issue_text:
            drills.append("Hip turn drill 3x10 reps")
            drills.append("Medicine ball rotational throws")
        
        if 'balance' in issue_text or 'stability' in issue_text:
            drills.append("Single-leg stance drill 3x30 seconds")
            drills.append("Balance board exercises")
        
        if 'speed' in issue_text or 'velocity' in issue_text:
            drills.append("Speed bag training 3x2 minutes")
            drills.append("Plyometric punching drills")
        
        if 'retraction' in issue_text:
            drills.append("Snap-back drill with resistance band")
        
        if 'shoulder' in issue_text:
            drills.append("Shoulder mobility and stability exercises")
        
        if 'stance' in issue_text:
            drills.append("Stance work with mirror feedback")
        
        # Default recommendations if no specific issues
        if not drills:
            drills = [
                "Continue practicing current form",
                "Focus on consistency and repetition",
                "Record videos regularly to track progress"
            ]
        
        return drills[:5]  # Return top 5 drills
