"""
NETRAX AI - Gesture Recognition Engine
Real-time hand and body gesture classification with temporal smoothing
"""

import numpy as np
from typing import Dict, Any, List, Optional
from collections import deque
import logging
from config import settings

logger = logging.getLogger("NETRAX.GestureEngine")

class GestureEngine:
    """
    Advanced gesture recognition using body and hand landmarks
    Supports static and dynamic gestures with temporal smoothing
    """
    
    def __init__(self):
        logger.info("👋 Initializing Gesture Engine...")
        
        # Gesture buffer for temporal smoothing
        self.gesture_buffer = deque(maxlen=settings.GESTURE_BUFFER_SIZE)
        self.confidence_buffer = deque(maxlen=settings.GESTURE_BUFFER_SIZE)
        
        # Cooldown to prevent rapid repeated detections
        self.cooldown_counter = 0
        self.last_gesture = None
        
        # Gesture history
        self.gesture_history = []
        
        # Define gesture rules
        self.gesture_rules = self._define_gesture_rules()
        
        logger.info(f"✅ Gesture Engine initialized with {len(self.gesture_rules)} gestures")
    
    def _define_gesture_rules(self) -> Dict[str, callable]:
        """Define gesture detection rules"""
        return {
            "peace": self._detect_peace,
            "stop": self._detect_stop,
            "thumbs_up": self._detect_thumbs_up,
            "thumbs_down": self._detect_thumbs_down,
            "fist": self._detect_fist,
            "point": self._detect_point,
            "swipe_left": self._detect_swipe_left,
            "swipe_right": self._detect_swipe_right,
            "arms_crossed": self._detect_arms_crossed
        }
    
    def process(self, body_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Process body data and recognize gestures
        
        Args:
            body_data: Body tracking data from BodyTracker
            
        Returns:
            Gesture recognition result
        """
        try:
            if not body_data.get("detected"):
                return {
                    "gesture": None,
                    "confidence": 0.0
                }
            
            # Decrement cooldown
            if self.cooldown_counter > 0:
                self.cooldown_counter -= 1
            
            # Check each gesture
            detected_gesture = None
            max_confidence = 0.0
            
            for gesture_name, detection_func in self.gesture_rules.items():
                confidence = detection_func(body_data)
                
                if confidence > max_confidence and confidence > settings.GESTURE_CONFIDENCE_THRESHOLD:
                    max_confidence = confidence
                    detected_gesture = gesture_name
            
            # Apply temporal smoothing
            self.gesture_buffer.append(detected_gesture)
            self.confidence_buffer.append(max_confidence)
            
            # Get most common gesture in buffer
            smoothed_gesture = self._smooth_gesture()
            smoothed_confidence = np.mean(list(self.confidence_buffer)) if self.confidence_buffer else 0.0
            
            # Check cooldown before returning new gesture
            if smoothed_gesture and smoothed_gesture != self.last_gesture:
                if self.cooldown_counter == 0:
                    self.last_gesture = smoothed_gesture
                    self.cooldown_counter = settings.GESTURE_COOLDOWN_FRAMES
                    
                    # Log gesture
                    if settings.LOG_GESTURES:
                        logger.info(f"👋 Gesture: {smoothed_gesture} ({smoothed_confidence:.2f})")
                    
                    # Store in history
                    self.gesture_history.append({
                        "gesture": smoothed_gesture,
                        "confidence": smoothed_confidence
                    })
                    
                    return {
                        "gesture": smoothed_gesture,
                        "confidence": float(smoothed_confidence)
                    }
            
            return {
                "gesture": None,
                "confidence": float(smoothed_confidence)
            }
            
        except Exception as e:
            logger.error(f"Gesture recognition error: {e}")
            return {
                "gesture": None,
                "confidence": 0.0,
                "error": str(e)
            }
    
    def _smooth_gesture(self) -> Optional[str]:
        """Apply temporal smoothing to gesture buffer"""
        if not self.gesture_buffer:
            return None
        
        # Count occurrences
        gesture_counts = {}
        for gesture in self.gesture_buffer:
            if gesture:
                gesture_counts[gesture] = gesture_counts.get(gesture, 0) + 1
        
        if not gesture_counts:
            return None
        
        # Return most common
        return max(gesture_counts, key=gesture_counts.get)
    
    # ========== Gesture Detection Functions ==========
    
    def _detect_peace(self, body_data: Dict) -> float:
        """Detect peace sign (✌️) - index and middle fingers up"""
        hands = body_data.get("hands", {})
        left = hands.get("left", {})
        right = hands.get("right", {})
        
        confidence = 0.0
        
        for hand in [left, right]:
            if not hand.get("detected"):
                continue
            
            fingertips = hand.get("fingertips", {})
            if not fingertips:
                continue
            
            landmarks = hand.get("landmarks", [])
            if len(landmarks) < 21:
                continue
            
            # Check if index and middle fingers are extended
            # Index finger landmarks: 5,6,7,8
            # Middle finger landmarks: 9,10,11,12
            
            index_tip = landmarks[8]
            index_base = landmarks[5]
            middle_tip = landmarks[12]
            middle_base = landmarks[9]
            
            # Fingers are up if tip is higher than base
            index_up = index_tip["y"] < index_base["y"]
            middle_up = middle_tip["y"] < middle_base["y"]
            
            # Check other fingers are down
            ring_tip = landmarks[16]
            ring_base = landmarks[13]
            pinky_tip = landmarks[20]
            pinky_base = landmarks[17]
            
            ring_down = ring_tip["y"] > ring_base["y"]
            pinky_down = pinky_tip["y"] > pinky_base["y"]
            
            if index_up and middle_up and ring_down and pinky_down:
                confidence = max(confidence, 0.9)
        
        return confidence
    
    def _detect_stop(self, body_data: Dict) -> float:
        """Detect stop sign (✋) - open palm"""
        hands = body_data.get("hands", {})
        confidence = 0.0
        
        for hand in [hands.get("left", {}), hands.get("right", {})]:
            if not hand.get("detected"):
                continue
            
            landmarks = hand.get("landmarks", [])
            if len(landmarks) < 21:
                continue
            
            # Check all fingers extended
            fingers_up = 0
            finger_pairs = [(8, 5), (12, 9), (16, 13), (20, 17)]  # tip, base
            
            for tip_idx, base_idx in finger_pairs:
                if landmarks[tip_idx]["y"] < landmarks[base_idx]["y"]:
                    fingers_up += 1
            
            if fingers_up >= 3:
                confidence = max(confidence, 0.85)
        
        return confidence
    
    def _detect_thumbs_up(self, body_data: Dict) -> float:
        """Detect thumbs up (👍)"""
        hands = body_data.get("hands", {})
        confidence = 0.0
        
        for hand in [hands.get("left", {}), hands.get("right", {})]:
            if not hand.get("detected"):
                continue
            
            landmarks = hand.get("landmarks", [])
            if len(landmarks) < 21:
                continue
            
            # Thumb extended upward
            thumb_tip = landmarks[4]
            thumb_base = landmarks[2]
            
            thumb_up = thumb_tip["y"] < thumb_base["y"] - 0.05
            
            # Other fingers curled
            fingers_down = 0
            for tip_idx in [8, 12, 16, 20]:
                if landmarks[tip_idx]["y"] > landmarks[tip_idx - 3]["y"]:
                    fingers_down += 1
            
            if thumb_up and fingers_down >= 3:
                confidence = max(confidence, 0.9)
        
        return confidence
    
    def _detect_thumbs_down(self, body_data: Dict) -> float:
        """Detect thumbs down (👎)"""
        hands = body_data.get("hands", {})
        confidence = 0.0
        
        for hand in [hands.get("left", {}), hands.get("right", {})]:
            if not hand.get("detected"):
                continue
            
            landmarks = hand.get("landmarks", [])
            if len(landmarks) < 21:
                continue
            
            # Thumb extended downward
            thumb_tip = landmarks[4]
            thumb_base = landmarks[2]
            
            thumb_down = thumb_tip["y"] > thumb_base["y"] + 0.05
            
            # Other fingers curled
            fingers_down = 0
            for tip_idx in [8, 12, 16, 20]:
                if landmarks[tip_idx]["y"] > landmarks[tip_idx - 3]["y"]:
                    fingers_down += 1
            
            if thumb_down and fingers_down >= 3:
                confidence = max(confidence, 0.9)
        
        return confidence
    
    def _detect_fist(self, body_data: Dict) -> float:
        """Detect closed fist (✊)"""
        hands = body_data.get("hands", {})
        confidence = 0.0
        
        for hand in [hands.get("left", {}), hands.get("right", {})]:
            if not hand.get("detected"):
                continue
            
            landmarks = hand.get("landmarks", [])
            if len(landmarks) < 21:
                continue
            
            # Check all fingers curled
            fingers_curled = 0
            for tip_idx in [8, 12, 16, 20]:
                if landmarks[tip_idx]["y"] > landmarks[tip_idx - 3]["y"]:
                    fingers_curled += 1
            
            if fingers_curled >= 3:
                confidence = max(confidence, 0.85)
        
        return confidence
    
    def _detect_point(self, body_data: Dict) -> float:
        """Detect pointing gesture (👉)"""
        hands = body_data.get("hands", {})
        confidence = 0.0
        
        for hand in [hands.get("left", {}), hands.get("right", {})]:
            if not hand.get("detected"):
                continue
            
            landmarks = hand.get("landmarks", [])
            if len(landmarks) < 21:
                continue
            
            # Index finger extended
            index_tip = landmarks[8]
            index_base = landmarks[5]
            index_extended = index_tip["y"] < index_base["y"]
            
            # Other fingers curled
            others_curled = (
                landmarks[12]["y"] > landmarks[9]["y"] and
                landmarks[16]["y"] > landmarks[13]["y"] and
                landmarks[20]["y"] > landmarks[17]["y"]
            )
            
            if index_extended and others_curled:
                confidence = max(confidence, 0.9)
        
        return confidence
    
    def _detect_swipe_left(self, body_data: Dict) -> float:
        """Detect left swipe motion"""
        # This requires motion tracking across frames
        # Simplified version: check hand moving left
        return 0.0  # Implement with motion history
    
    def _detect_swipe_right(self, body_data: Dict) -> float:
        """Detect right swipe motion"""
        return 0.0  # Implement with motion history
    
    def _detect_arms_crossed(self, body_data: Dict) -> float:
        """Detect arms crossed gesture"""
        pose = body_data.get("pose", {})
        keypoints = pose.get("keypoints", {})
        
        if not keypoints:
            return 0.0
        
        left_wrist = keypoints.get("left_wrist")
        right_wrist = keypoints.get("right_wrist")
        left_shoulder = keypoints.get("left_shoulder")
        right_shoulder = keypoints.get("right_shoulder")
        
        if not all([left_wrist, right_wrist, left_shoulder, right_shoulder]):
            return 0.0
        
        # Check if wrists are crossed
        wrists_crossed = (
            left_wrist["x"] > right_wrist["x"] or
            right_wrist["x"] > left_shoulder["x"]
        )
        
        return 0.8 if wrists_crossed else 0.0
    
    def reset(self):
        """Reset gesture engine state"""
        logger.info("🔄 Resetting gesture engine...")
        self.gesture_buffer.clear()
        self.confidence_buffer.clear()
        self.cooldown_counter = 0
        self.last_gesture = None
        self.gesture_history = []
    
    def cleanup(self):
        """Cleanup resources"""
        logger.info("🧹 Cleaning up gesture engine...")
        self.reset()