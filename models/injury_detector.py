import numpy as np
from typing import List, Dict, Tuple, Optional
from models.pose_estimator import POSE_LANDMARKS
from utils.biomechanics import calculate_angle, calculate_distance
from config import settings


def get_landmark_by_id(landmarks: List[dict], landmark_name: str) -> Optional[dict]:
    """
    Find a landmark by its name/ID in the landmarks list.
    The landmarks list may be filtered, so we need to search by 'id' field.
    """
    landmark_id = POSE_LANDMARKS.get(landmark_name)
    if landmark_id is None:
        return None
    for landmark in landmarks:
        if landmark.get('id') == landmark_id:
            return landmark
    return None


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
    
    def _detect_movement_type(self, poses: List[dict]) -> str:
        """
        Detect the type of movement being performed (punching, kicking, or general).
        
        Key insight: The difference is NOT absolute knee angle (fighters keep knees bent),
        but rather the VARIANCE in ankle/knee movement:
        - Punching: Knees stay at consistent angle, ankles don't move much
        - Kicking: Knee angle changes dramatically during kick extension/retraction
        """
        if not poses or len(poses) < 3:
            return 'general'
        
        # Analyze movement patterns
        wrist_distances = []
        ankle_distances = []
        knee_angles = []
        
        for i in range(len(poses) - 1):
            curr_landmarks = poses[i]['landmarks']
            next_landmarks = poses[i + 1]['landmarks']
            
            # Calculate wrist movement
            curr_wrist = get_landmark_by_id(curr_landmarks, 'RIGHT_WRIST')
            next_wrist = get_landmark_by_id(next_landmarks, 'RIGHT_WRIST')
            if curr_wrist and next_wrist:
                wrist_dist = calculate_distance(
                    (curr_wrist['x'], curr_wrist['y']),
                    (next_wrist['x'], next_wrist['y'])
                )
                wrist_distances.append(wrist_dist)
            
            # Calculate ankle movement
            curr_ankle = get_landmark_by_id(curr_landmarks, 'RIGHT_ANKLE')
            next_ankle = get_landmark_by_id(next_landmarks, 'RIGHT_ANKLE')
            if curr_ankle and next_ankle:
                ankle_dist = calculate_distance(
                    (curr_ankle['x'], curr_ankle['y']),
                    (next_ankle['x'], next_ankle['y'])
                )
                ankle_distances.append(ankle_dist)
        
        # Calculate knee angles for all poses
        for pose in poses:
            landmarks = pose['landmarks']
            hip = get_landmark_by_id(landmarks, 'RIGHT_HIP')
            knee = get_landmark_by_id(landmarks, 'RIGHT_KNEE')
            ankle = get_landmark_by_id(landmarks, 'RIGHT_ANKLE')
            
            if hip and knee and ankle:
                knee_angle = calculate_angle(
                    (hip['x'], hip['y']),
                    (knee['x'], knee['y']),
                    (ankle['x'], ankle['y'])
                )
                knee_angles.append(knee_angle)
        
        if not wrist_distances or not ankle_distances:
            print(f"[DEBUG] No valid wrist/ankle data, returning 'general'")
            return 'general'
        
        avg_wrist_dist = np.mean(wrist_distances)
        avg_ankle_dist = np.mean(ankle_distances)
        max_wrist_dist = max(wrist_distances)
        max_ankle_dist = max(ankle_distances)
        
        print(f"[DEBUG] Movement - wrist(avg:{avg_wrist_dist:.1f}, max:{max_wrist_dist:.1f}) vs ankle(avg:{avg_ankle_dist:.1f}, max:{max_ankle_dist:.1f})")
        
        # Calculate knee angle variance (KEY DIFFERENTIATOR)
        if len(knee_angles) > 1:
            knee_angle_std = np.std(knee_angles)
            knee_angle_range = max(knee_angles) - min(knee_angles)
        else:
            knee_angle_std = 0
            knee_angle_range = 0
        
        avg_knee_angle = np.mean(knee_angles) if knee_angles else 180
        min_knee_angle = min(knee_angles) if knee_angles else 180
        
        print(f"[DEBUG] Knee - avg:{avg_knee_angle:.1f}, min:{min_knee_angle:.1f}, std:{knee_angle_std:.1f}, range:{knee_angle_range:.1f}")
        
        # KICKING detection: High ankle movement is the key indicator
        # During a kick, the ankle moves significantly more than in punching
        # Use ankle movement as primary indicator, knee variance as secondary
        if max_ankle_dist > 100 or avg_ankle_dist > 50:
            print(f"[DEBUG] Detected KICKING (high ankle movement: max={max_ankle_dist:.1f}, avg={avg_ankle_dist:.1f})")
            return 'kicking'
        
        # Also detect kicking if knee variance is very high (clear kicking motion)
        if knee_angle_std > 20 and knee_angle_range > 50:
            print(f"[DEBUG] Detected KICKING (knee variance: std={knee_angle_std:.1f}, range={knee_angle_range:.1f})")
            return 'kicking'
        
        # PUNCHING detection: Wrist movement present, lower ankle movement
        if max_wrist_dist > 3 and max_ankle_dist < 100:
            print(f"[DEBUG] Detected PUNCHING (wrist moving, ankle stable)")
            return 'punching'
        
        # Fallback: If we have wrist movement and low ankle movement
        if max_wrist_dist > 5 and avg_ankle_dist < 50:
            print(f"[DEBUG] Detected PUNCHING (fallback - wrist > ankle)")
            return 'punching'
        
        # Fallback based on movement ratios
        if avg_ankle_dist > 0 and avg_wrist_dist / avg_ankle_dist > 0.3:
            print(f"[DEBUG] Detected PUNCHING (wrist/ankle ratio: {avg_wrist_dist/avg_ankle_dist:.2f})")
            return 'punching'
        
        print(f"[DEBUG] No specific pattern, returning 'general'")
        return 'general'
    
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
        
        # Detect movement type to adjust risk assessment priorities
        movement_type = self._detect_movement_type(poses)
        
        # Analyze different body regions
        knee_risk = self._check_knee_alignment(poses)
        hip_risk = self._check_hip_overextension(poses)
        ankle_risk = self._check_ankle_stability(poses)
        shoulder_risk = self._check_shoulder_overrotation(poses)
        back_risk = self._check_lower_back(poses)
        
        all_risks = [knee_risk, hip_risk, ankle_risk, shoulder_risk, back_risk]
        
        # Adjust risk priorities based on movement type
        if movement_type == 'punching':
            # For punching, ONLY report upper body risks (shoulder, back)
            # Lower body risks (knee, hip, ankle) are NOT relevant for punching
            # unless there's a critical issue
            
            # Check upper body risks first
            upper_body_risks = []
            if shoulder_risk['level'] != 'Low':
                upper_body_risks.append(shoulder_risk)
            if back_risk['level'] != 'Low':
                upper_body_risks.append(back_risk)
            
            # If we have upper body risks, return the highest
            if upper_body_risks:
                return max(upper_body_risks, key=lambda x: self._risk_level_to_score(x['level']))
            
            # If all upper body risks are low, return a generic low-risk result
            # that mentions upper body, NOT knee
            return {
                'level': 'Low',
                'area': 'Shoulder',
                'reason': 'Good upper body mechanics maintained',
                'risk_type': 'N/A'
            }
        
        elif movement_type == 'kicking':
            # For kicking, prioritize lower body risks
            prioritized_risks = []
            
            # Lower body risks first (knee, hip, ankle)
            for risk in [knee_risk, hip_risk, ankle_risk]:
                if risk['level'] != 'Low':
                    prioritized_risks.append(risk)
            
            # Upper body risks only if high
            for risk in [shoulder_risk, back_risk]:
                if risk['level'] == 'High':
                    prioritized_risks.append(risk)
            
            # If no prioritized risks, check all
            if not prioritized_risks:
                lower_body_risks = [knee_risk, hip_risk, ankle_risk]
                highest = max(lower_body_risks, key=lambda x: self._risk_level_to_score(x['level']))
                if highest['level'] != 'Low':
                    return highest
                return self._get_lowest_risk_result(all_risks)
            
            return max(prioritized_risks, key=lambda x: self._risk_level_to_score(x['level']))
        
        else:
            # For general movement, return highest risk among all
            highest_risk = max(all_risks, key=lambda x: self._risk_level_to_score(x['level']))
            return highest_risk
    
    def _check_knee_alignment(self, poses: List[dict]) -> Dict:
        """
        Check for knee valgus (inward collapse) or hyperextension.
        Common injury risk in kicks and landing.
        """
        max_misalignment = 0
        problematic_frames = 0
        valid_frames = 0
        
        for pose in poses:
            landmarks = pose['landmarks']
            
            # Check right leg (can be extended for both legs)
            hip = get_landmark_by_id(landmarks, 'RIGHT_HIP')
            knee = get_landmark_by_id(landmarks, 'RIGHT_KNEE')
            ankle = get_landmark_by_id(landmarks, 'RIGHT_ANKLE')
            
            if not hip or not knee or not ankle:
                continue
            
            valid_frames += 1
            
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
        
        if valid_frames == 0:
            return {
                'level': 'Low',
                'area': 'Knee',
                'reason': 'Insufficient data for knee analysis',
                'risk_type': 'N/A'
            }
        
        # Calculate risk level
        risk_percentage = (problematic_frames / valid_frames) * 100
        
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
        valid_frames = 0
        
        for pose in poses:
            landmarks = pose['landmarks']
            
            shoulder = get_landmark_by_id(landmarks, 'RIGHT_SHOULDER')
            hip = get_landmark_by_id(landmarks, 'RIGHT_HIP')
            knee = get_landmark_by_id(landmarks, 'RIGHT_KNEE')
            
            if not shoulder or not hip or not knee:
                continue
            
            valid_frames += 1
            
            # Calculate hip angle
            angle = calculate_angle(
                (shoulder['x'], shoulder['y']),
                (hip['x'], hip['y']),
                (knee['x'], knee['y'])
            )
            
            # Check for excessive extension
            if angle > 200:  # Hyperextension threshold
                overextension_count += 1
        
        if valid_frames == 0:
            return {
                'level': 'Low',
                'area': 'Hip',
                'reason': 'Insufficient data for hip analysis',
                'risk_type': 'N/A'
            }
        
        risk_percentage = (overextension_count / valid_frames) * 100
        
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
        valid_frames = 0
        
        for pose in poses:
            landmarks = pose['landmarks']
            
            left_ankle = get_landmark_by_id(landmarks, 'LEFT_ANKLE')
            right_ankle = get_landmark_by_id(landmarks, 'RIGHT_ANKLE')
            
            if not left_ankle or not right_ankle:
                continue
            
            valid_frames += 1
            
            # Check ankle visibility (low visibility = unstable/twisted position)
            if left_ankle.get('visibility', 0) < 0.5 or right_ankle.get('visibility', 0) < 0.5:
                unstable_frames += 1
                continue
            
            # Check for excessive lateral movement
            ankle_separation = abs(left_ankle['x'] - right_ankle['x'])
            if ankle_separation < 0.05:  # Too narrow
                unstable_frames += 1
        
        if valid_frames == 0:
            return {
                'level': 'Low',
                'area': 'Ankle',
                'reason': 'Insufficient data for ankle analysis',
                'risk_type': 'N/A'
            }
        
        risk_percentage = (unstable_frames / valid_frames) * 100
        
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
        valid_frames = 0
        
        for pose in poses:
            landmarks = pose['landmarks']
            
            left_shoulder = get_landmark_by_id(landmarks, 'LEFT_SHOULDER')
            right_shoulder = get_landmark_by_id(landmarks, 'RIGHT_SHOULDER')
            
            if not left_shoulder or not right_shoulder:
                continue
            
            valid_frames += 1
            
            # Check shoulder rotation (z-axis difference)
            rotation = abs(left_shoulder.get('z', 0) - right_shoulder.get('z', 0))
            
            if rotation > 0.15:  # Excessive rotation threshold
                overrotation_count += 1
        
        if valid_frames == 0:
            return {
                'level': 'Low',
                'area': 'Shoulder',
                'reason': 'Insufficient data for shoulder analysis',
                'risk_type': 'N/A'
            }
        
        risk_percentage = (overrotation_count / valid_frames) * 100
        
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
        valid_frames = 0
        
        for pose in poses:
            landmarks = pose['landmarks']
            
            shoulder = get_landmark_by_id(landmarks, 'LEFT_SHOULDER')
            hip = get_landmark_by_id(landmarks, 'LEFT_HIP')
            
            if not shoulder or not hip:
                continue
            
            valid_frames += 1
            
            # Calculate forward lean (shoulder ahead of hip)
            forward_distance = shoulder.get('z', 0) - hip.get('z', 0)
            
            if forward_distance > 0.1:  # Excessive forward lean
                forward_lean_count += 1
        
        if valid_frames == 0:
            return {
                'level': 'Low',
                'area': 'Lower Back',
                'reason': 'Insufficient data for lower back analysis',
                'risk_type': 'N/A'
            }
        
        risk_percentage = (forward_lean_count / valid_frames) * 100
        
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
    
    def _get_lowest_risk_result(self, all_risks: List[Dict]) -> Dict:
        """
        Return the most relevant low-risk result when all risks are low.
        Prioritizes based on movement context.
        """
        # For punching context, prefer upper body even for low risks
        for risk in all_risks:
            if risk['area'] in ['Shoulder', 'Lower Back']:
                return risk
        # Otherwise return the first non-default risk
        for risk in all_risks:
            if risk['area'] != 'Overall':
                return risk
        return self._default_result()
    
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
