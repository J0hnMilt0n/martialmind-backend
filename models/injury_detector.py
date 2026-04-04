import numpy as np
from typing import List, Dict, Tuple
from models.pose_estimator import POSE_LANDMARKS
from utils.biomechanics import calculate_angle, calculate_distance
from config import settings


class InjuryDetector:
    """
    Detects movement-based injury risk patterns.
    
    IMPORTANT: This is NOT medical diagnosis. 
    It identifies biomechanical movement patterns that correlate with injury risk.
    """
    
    def __init__(self):
        self.risk_factors = {
            'knee': [],
            'hip': [],
            'ankle': [],
            'shoulder': [],
            'lower_back': []
        }
    
    def assess_risk(self, poses: List[dict], frames: List[np.ndarray]) -> Dict:
        """
        Assess injury risk from pose sequence.
        
        Args:
            poses: List of pose data dictionaries
            frames: List of video frames
            
        Returns:
            Dictionary with risk level, area, reason, and type
        """
        if not poses or len(poses) == 0:
            return self._default_result()
        
        # Analyze different body regions
        knee_risk = self._check_knee_alignment(poses)
        hip_risk = self._check_hip_overextension(poses)
        ankle_risk = self._check_ankle_stability(poses)
        shoulder_risk = self._check_shoulder_overrotation(poses)
        back_risk = self._check_lower_back(poses)
        
        # Determine highest risk
        all_risks = [knee_risk, hip_risk, ankle_risk, shoulder_risk, back_risk]
        highest_risk = max(all_risks, key=lambda x: self._risk_level_to_score(x['level']))
        
        return highest_risk
    
    def _check_knee_alignment(self, poses: List[dict]) -> Dict:
        """Check for knee valgus (inward collapse) or hyperextension"""
        max_misalignment = 0
        problematic_frames = 0
        
        for pose in poses:
            landmarks = pose['landmarks']
            
            # Check right leg (can be extended for both legs)
            hip = landmarks[POSE_LANDMARKS['RIGHT_HIP']]
            knee = landmarks[POSE_LANDMARKS['RIGHT_KNEE']]
            ankle = landmarks[POSE_LANDMARKS['RIGHT_ANKLE']]
            
            # Calculate knee angle
            angle = calculate_angle(
                (hip['x'], hip['y']),
                (knee['x'], knee['y']),
                (ankle['x'], ankle['y'])
            )
            
            # Ideal range: 170-180 degrees when standing
            # Less than 165: valgus collapse
            # More than 185: hyperextension
            
            if angle < 165:
                misalignment = 165 - angle
                max_misalignment = max(max_misalignment, misalignment)
                problematic_frames += 1
            elif angle > 185:
                misalignment = angle - 185
                max_misalignment = max(max_misalignment, misalignment)
                problematic_frames += 1
        
        risk_percentage = (problematic_frames / len(poses)) * 100
        
        if risk_percentage > 40 or max_misalignment > settings.KNEE_ALIGNMENT_THRESHOLD:
            return {
                'level': 'High',
                'area': 'Knee',
                'reason': f'Knee misalignment during landing - valgus collapse detected',
                'risk_type': 'Movement-based instability'
            }
        elif risk_percentage > 15 or max_misalignment > 10:
            return {
                'level': 'Medium',
                'area': 'Knee',
                'reason': 'Occasional knee alignment issues during movement',
                'risk_type': 'Movement-based instability'
            }
        else:
            return {
                'level': 'Low',
                'area': 'Knee',
                'reason': 'Good knee alignment maintained',
                'risk_type': 'N/A'
            }
    
    def _check_hip_overextension(self, poses: List[dict]) -> Dict:
        """Check for hip overextension that could strain muscles/ligaments"""
        overextension_count = 0
        
        for pose in poses:
            landmarks = pose['landmarks']
            
            shoulder = landmarks[POSE_LANDMARKS['RIGHT_SHOULDER']]
            hip = landmarks[POSE_LANDMARKS['RIGHT_HIP']]
            knee = landmarks[POSE_LANDMARKS['RIGHT_KNEE']]
            
            # Calculate hip angle
            angle = calculate_angle(
                (shoulder['x'], shoulder['y']),
                (hip['x'], hip['y']),
                (knee['x'], knee['y'])
            )
            
            # Check for excessive extension
            if angle > 200:  # Hyperextension threshold
                overextension_count += 1
        
        risk_percentage = (overextension_count / len(poses)) * 100
        
        if risk_percentage > 30:
            return {
                'level': 'High',
                'area': 'Hip',
                'reason': 'Excessive hip hyperextension detected',
                'risk_type': 'Overextension pattern'
            }
        elif risk_percentage > 10:
            return {
                'level': 'Medium',
                'area': 'Hip',
                'reason': 'Occasional hip overextension during kicks',
                'risk_type': 'Overextension pattern'
            }
        else:
            return {
                'level': 'Low',
                'area': 'Hip',
                'reason': 'Hip range of motion within safe limits',
                'risk_type': 'N/A'
            }
    
    def _check_ankle_stability(self, poses: List[dict]) -> Dict:
        """Check ankle stability and balance"""
        unstable_frames = 0
        
        for pose in poses:
            landmarks = pose['landmarks']
            
            left_ankle = landmarks[POSE_LANDMARKS['LEFT_ANKLE']]
            right_ankle = landmarks[POSE_LANDMARKS['RIGHT_ANKLE']]
            
            # Check ankle visibility (low visibility = unstable/twisted position)
            if left_ankle['visibility'] < 0.5 or right_ankle['visibility'] < 0.5:
                unstable_frames += 1
                continue
            
            # Check for excessive lateral movement
            ankle_separation = abs(left_ankle['x'] - right_ankle['x'])
            if ankle_separation < 0.05:  # Too narrow
                unstable_frames += 1
        
        risk_percentage = (unstable_frames / len(poses)) * 100
        
        if risk_percentage > 40:
            return {
                'level': 'High',
                'area': 'Ankle',
                'reason': 'Significant ankle instability detected',
                'risk_type': 'Balance and stability issue'
            }
        elif risk_percentage > 20:
            return {
                'level': 'Medium',
                'area': 'Ankle',
                'reason': 'Occasional ankle instability',
                'risk_type': 'Balance and stability issue'
            }
        else:
            return {
                'level': 'Low',
                'area': 'Ankle',
                'reason': 'Good ankle stability maintained',
                'risk_type': 'N/A'
            }
    
    def _check_shoulder_overrotation(self, poses: List[dict]) -> Dict:
        """Check for excessive shoulder rotation during strikes"""
        overrotation_count = 0
        
        for pose in poses:
            landmarks = pose['landmarks']
            
            left_shoulder = landmarks[POSE_LANDMARKS['LEFT_SHOULDER']]
            right_shoulder = landmarks[POSE_LANDMARKS['RIGHT_SHOULDER']]
            
            # Check shoulder rotation (z-axis difference)
            rotation = abs(left_shoulder['z'] - right_shoulder['z'])
            
            if rotation > 0.15:  # Excessive rotation threshold
                overrotation_count += 1
        
        risk_percentage = (overrotation_count / len(poses)) * 100
        
        if risk_percentage > 35:
            return {
                'level': 'Medium',
                'area': 'Shoulder',
                'reason': 'Excessive shoulder rotation during strikes',
                'risk_type': 'Overrotation pattern'
            }
        else:
            return {
                'level': 'Low',
                'area': 'Shoulder',
                'reason': 'Shoulder rotation within safe range',
                'risk_type': 'N/A'
            }
    
    def _check_lower_back(self, poses: List[dict]) -> Dict:
        """Check for excessive forward lean that stresses lower back"""
        forward_lean_count = 0
        
        for pose in poses:
            landmarks = pose['landmarks']
            
            shoulder = landmarks[POSE_LANDMARKS['LEFT_SHOULDER']]
            hip = landmarks[POSE_LANDMARKS['LEFT_HIP']]
            
            # Calculate forward lean (shoulder ahead of hip)
            forward_distance = shoulder['z'] - hip['z']
            
            if forward_distance > 0.1:  # Excessive forward lean
                forward_lean_count += 1
        
        risk_percentage = (forward_lean_count / len(poses)) * 100
        
        if risk_percentage > 50:
            return {
                'level': 'High',
                'area': 'Lower Back',
                'reason': 'Excessive forward lean stressing lower back',
                'risk_type': 'Postural stress'
            }
        elif risk_percentage > 25:
            return {
                'level': 'Medium',
                'area': 'Lower Back',
                'reason': 'Occasional forward lean detected',
                'risk_type': 'Postural stress'
            }
        else:
            return {
                'level': 'Low',
                'area': 'Lower Back',
                'reason': 'Good spinal alignment maintained',
                'risk_type': 'N/A'
            }
    
    def _risk_level_to_score(self, level: str) -> int:
        """Convert risk level to numeric score for comparison"""
        levels = {'Low': 1, 'Medium': 2, 'High': 3}
        return levels.get(level, 0)
    
    def _default_result(self) -> Dict:
        """Return default low-risk result"""
        return {
            'level': 'Low',
            'area': 'Overall',
            'reason': 'Insufficient data for risk assessment',
            'risk_type': 'N/A'
        }
    
    def get_prevention_advice(self, risk_result: Dict) -> List[str]:
        """Generate injury prevention advice based on risk assessment"""
        advice = []
        
        risk_area = risk_result['area'].lower()
        risk_level = risk_result['level']
        
        if risk_level == 'Low':
            return [
                "Continue current training regimen",
                "Maintain proper warm-up and cool-down routines",
                "Focus on movement quality over quantity"
            ]
        
        # Specific advice based on risk area
        if 'knee' in risk_area:
            advice.extend([
                "Strengthen glutes and hip abductors (clamshells, side leg raises)",
                "Practice proper landing mechanics with plyometric exercises",
                "Work on knee alignment during all lower body movements",
                "Consider knee stability exercises with resistance bands"
            ])
        
        if 'hip' in risk_area:
            advice.extend([
                "Improve hip flexor flexibility with dynamic stretching",
                "Strengthen hip stabilizers (single-leg bridges, hip thrusts)",
                "Practice controlled kicks with focus on range of motion limits"
            ])
        
        if 'ankle' in risk_area:
            advice.extend([
                "Perform ankle strengthening exercises (calf raises, toe walks)",
                "Practice balance drills on unstable surfaces",
                "Ensure proper footwear with adequate ankle support"
            ])
        
        if 'shoulder' in risk_area:
            advice.extend([
                "Strengthen rotator cuff muscles",
                "Improve shoulder mobility with dynamic stretches",
                "Practice strikes with controlled rotation"
            ])
        
        if 'back' in risk_area:
            advice.extend([
                "Strengthen core muscles (planks, dead bugs)",
                "Maintain neutral spine position during movements",
                "Work on hip hinge mechanics"
            ])
        
        # General advice
        advice.append("Consider consultation with sports medicine professional if pain develops")
        
        return advice[:5]  # Return top 5 pieces of advice