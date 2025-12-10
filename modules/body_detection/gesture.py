import numpy as np
from typing import Dict, List, Optional, Callable
from dataclasses import dataclass
from enum import Enum
import time
import math

from pose import PoseResult, Keypoint


class GestureType(Enum):

    # Hand gestures
    PEACE = "peace"  # ✌️
    STOP = "stop"  # ✋
    THUMBS_UP = "thumbs_up"  # 👍
    THUMBS_DOWN = "thumbs_down"  # 👎
    POINT = "point"  # 👉
    FIST = "fist"  # ✊
    OPEN_PALM = "open_palm"  # 🖐️
    
    # Body poses
    ARMS_CROSSED = "arms_crossed"
    ARMS_UP = "arms_up"
    LEANING_LEFT = "leaning_left"
    LEANING_RIGHT = "leaning_right"
    
    # Combined gestures
    SWIPE_LEFT = "swipe_left"
    SWIPE_RIGHT = "swipe_right"
    SWIPE_UP = "swipe_up"
    SWIPE_DOWN = "swipe_down"
    ZOOM_IN = "zoom_in"
    ZOOM_OUT = "zoom_out"
    
    # Special
    PAUSE = "pause"
    RESUME = "resume"
    UNKNOWN = "unknown"


@dataclass
class Gesture:
    
    type: GestureType
    confidence: float
    hand: Optional[str] = None  # "left", "right", or None for body
    timestamp: float = 0.0
    duration: float = 0.0
    parameters: Dict = None  # Additional data (e.g., swipe distance)
    
    def __post_init__(self):
        if self.timestamp == 0.0:
            self.timestamp = time.time()
        if self.parameters is None:
            self.parameters = {}


class GestureRecognizer:
    
    
    def __init__(self, 
                 min_confidence: float = 0.6,
                 gesture_hold_time: float = 0.3,
                 swipe_threshold: float = 0.15):
        
        self.min_confidence = min_confidence
        self.gesture_hold_time = gesture_hold_time
        self.swipe_threshold = swipe_threshold
        
        self.last_gestures: Dict[str, Gesture] = {}
        self.gesture_start_times: Dict[str, float] = {}
        self.previous_positions: Dict[str, Keypoint] = {}
        
    def recognize(self, pose_result: PoseResult) -> List[Gesture]:
        
        gestures = []
        
        if not pose_result.has_pose():
            return gestures
            
        # Recognize hand gestures
        if pose_result.left_hand_landmarks:
            left_gesture = self._recognize_hand_gesture(
                pose_result.left_hand_landmarks,
                "left"
            )
            if left_gesture:
                gestures.append(left_gesture)
                
        if pose_result.right_hand_landmarks:
            right_gesture = self._recognize_hand_gesture(
                pose_result.right_hand_landmarks,
                "right"
            )
            if right_gesture:
                gestures.append(right_gesture)
                
        # Recognize body poses
        body_gesture = self._recognize_body_pose(pose_result.pose_landmarks)
        if body_gesture:
            gestures.append(body_gesture)
            
        # Recognize motion gestures (swipes)
        motion_gestures = self._recognize_motion_gestures(pose_result)
        gestures.extend(motion_gestures)
        
        # Filter and validate gestures
        gestures = self._filter_gestures(gestures)
        
        return gestures
        
    def _recognize_hand_gesture(self,
                               hand_landmarks: Dict[str, Keypoint],
                               hand: str) -> Optional[Gesture]:
        
        if len(hand_landmarks) < 21:
            return None
            
        # Get finger states (extended or folded)
        fingers = self._get_finger_states(hand_landmarks)
        
        # Peace sign: index and middle extended, others folded
        if fingers == [False, True, True, False, False]:
            return Gesture(
                type=GestureType.PEACE,
                confidence=0.9,
                hand=hand
            )
            
        # Stop/Open palm: all fingers extended
        if fingers == [True, True, True, True, True]:
            return Gesture(
                type=GestureType.STOP,
                confidence=0.9,
                hand=hand
            )
            
        # Fist: all fingers folded
        if fingers == [False, False, False, False, False]:
            return Gesture(
                type=GestureType.FIST,
                confidence=0.85,
                hand=hand
            )
            
        # Thumbs up: thumb extended, others folded
        if fingers == [True, False, False, False, False]:
            # Check if thumb is pointing up
            if self._is_thumb_up(hand_landmarks):
                return Gesture(
                    type=GestureType.THUMBS_UP,
                    confidence=0.85,
                    hand=hand
                )
            else:
                return Gesture(
                    type=GestureType.THUMBS_DOWN,
                    confidence=0.85,
                    hand=hand
                )
                
        # Point: index extended, others folded
        if fingers == [False, True, False, False, False]:
            return Gesture(
                type=GestureType.POINT,
                confidence=0.85,
                hand=hand
            )
            
        return None
        
    def _get_finger_states(self, hand_landmarks: Dict[str, Keypoint]) -> List[bool]:
      
        fingers = []
        
        # Finger tip and pip landmark indices (MediaPipe)
        finger_tips = ['thumb_tip', 'index_finger_tip', 'middle_finger_tip', 
                      'ring_finger_tip', 'pinky_tip']
        finger_pips = ['thumb_ip', 'index_finger_pip', 'middle_finger_pip',
                      'ring_finger_pip', 'pinky_pip']
        
        for tip_name, pip_name in zip(finger_tips, finger_pips):
            if tip_name in hand_landmarks and pip_name in hand_landmarks:
                tip = hand_landmarks[tip_name]
                pip = hand_landmarks[pip_name]
                
                # Finger is extended if tip is above pip (smaller y value)
                # Add threshold for robustness
                extended = tip.y < (pip.y - 0.02)
                fingers.append(extended)
            else:
                fingers.append(False)
                
        return fingers
        
    def _is_thumb_up(self, hand_landmarks: Dict[str, Keypoint]) -> bool:
        
        if 'thumb_tip' not in hand_landmarks or 'wrist' not in hand_landmarks:
            return False
            
        thumb_tip = hand_landmarks['thumb_tip']
        wrist = hand_landmarks['wrist']
        
        # Thumb up if tip is significantly above wrist
        return thumb_tip.y < (wrist.y - 0.15)
        
    def _recognize_body_pose(self, 
                            pose_landmarks: Dict[str, Keypoint]) -> Optional[Gesture]:
       
        
        if not pose_landmarks:
            return None
            
        # Arms crossed
        if self._are_arms_crossed(pose_landmarks):
            return Gesture(
                type=GestureType.ARMS_CROSSED,
                confidence=0.8
            )
            
        # Arms up (pause gesture)
        if self._are_arms_up(pose_landmarks):
            return Gesture(
                type=GestureType.PAUSE,
                confidence=0.85
            )
            
        # Leaning detection
        lean = self._detect_lean(pose_landmarks)
        if lean:
            return lean
            
        return None
        
    def _are_arms_crossed(self, pose_landmarks: Dict[str, Keypoint]) -> bool:
        
        required = ['left_wrist', 'right_wrist', 'left_shoulder', 'right_shoulder']
        if not all(k in pose_landmarks for k in required):
            return False
            
        left_wrist = pose_landmarks['left_wrist']
        right_wrist = pose_landmarks['right_wrist']
        left_shoulder = pose_landmarks['left_shoulder']
        right_shoulder = pose_landmarks['right_shoulder']
        
        # Left wrist is right of center and right wrist is left of center
        center_x = (left_shoulder.x + right_shoulder.x) / 2
        
        return left_wrist.x > center_x and right_wrist.x < center_x
        
    def _are_arms_up(self, pose_landmarks: Dict[str, Keypoint]) -> bool:
        
        required = ['left_wrist', 'right_wrist', 'left_shoulder', 'right_shoulder']
        if not all(k in pose_landmarks for k in required):
            return False
            
        left_wrist = pose_landmarks['left_wrist']
        right_wrist = pose_landmarks['right_wrist']
        left_shoulder = pose_landmarks['left_shoulder']
        right_shoulder = pose_landmarks['right_shoulder']
        
        # Both wrists above shoulders
        return (left_wrist.y < left_shoulder.y - 0.1 and
                right_wrist.y < right_shoulder.y - 0.1)
                
    def _detect_lean(self, pose_landmarks: Dict[str, Keypoint]) -> Optional[Gesture]:
        
        if 'left_shoulder' not in pose_landmarks or 'right_shoulder' not in pose_landmarks:
            return None
            
        left = pose_landmarks['left_shoulder']
        right = pose_landmarks['right_shoulder']
        
        # Calculate shoulder angle
        angle = math.atan2(right.y - left.y, right.x - left.x)
        angle_deg = math.degrees(angle)
        
        # Leaning threshold
        if angle_deg > 15:
            return Gesture(
                type=GestureType.LEANING_RIGHT,
                confidence=0.7,
                parameters={'angle': angle_deg}
            )
        elif angle_deg < -15:
            return Gesture(
                type=GestureType.LEANING_LEFT,
                confidence=0.7,
                parameters={'angle': angle_deg}
            )
            
        return None
        
    def _recognize_motion_gestures(self, 
                                  pose_result: PoseResult) -> List[Gesture]:
       
        gestures = []
        
        # Track wrist positions for swipe detection
        for hand, landmarks in [('left', pose_result.left_hand_landmarks),
                               ('right', pose_result.right_hand_landmarks)]:
            if not landmarks or 'wrist' not in landmarks:
                continue
                
            wrist = landmarks['wrist']
            key = f"{hand}_wrist"
            
            if key in self.previous_positions:
                prev = self.previous_positions[key]
                
                # Calculate movement
                dx = wrist.x - prev.x
                dy = wrist.y - prev.y
                distance = math.sqrt(dx**2 + dy**2)
                
                if distance > self.swipe_threshold:
                    # Determine swipe direction
                    if abs(dx) > abs(dy):
                        if dx > 0:
                            gesture_type = GestureType.SWIPE_RIGHT
                        else:
                            gesture_type = GestureType.SWIPE_LEFT
                    else:
                        if dy > 0:
                            gesture_type = GestureType.SWIPE_DOWN
                        else:
                            gesture_type = GestureType.SWIPE_UP
                            
                    gestures.append(Gesture(
                        type=gesture_type,
                        confidence=0.8,
                        hand=hand,
                        parameters={'distance': distance, 'dx': dx, 'dy': dy}
                    ))
                    
            self.previous_positions[key] = wrist
            
        return gestures
        
    def _filter_gestures(self, gestures: List[Gesture]) -> List[Gesture]:
        
        filtered = []
        current_time = time.time()
        
        for gesture in gestures:
            if gesture.confidence < self.min_confidence:
                continue
                
            # Create unique key for gesture
            key = f"{gesture.type.value}_{gesture.hand or 'body'}"
            
            # Check if this is a new gesture or continuation
            if key in self.last_gestures:
                last = self.last_gestures[key]
                if last.type == gesture.type:
                    # Same gesture continuing
                    start_time = self.gesture_start_times.get(key, current_time)
                    duration = current_time - start_time
                    
                    # Only emit if held long enough
                    if duration >= self.gesture_hold_time:
                        gesture.duration = duration
                        filtered.append(gesture)
                else:
                    # Different gesture, reset
                    self.gesture_start_times[key] = current_time
            else:
                # New gesture
                self.gesture_start_times[key] = current_time
                
            self.last_gestures[key] = gesture
            
        return filtered