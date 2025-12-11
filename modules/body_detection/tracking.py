import numpy as np
from typing import Dict, Optional, Deque
from collections import deque
import math

from pose import PoseResult, Keypoint


class OneEuroFilter:
    """
    One Euro Filter for smoothing noisy signals
    https://cristal.univ-lille.fr/~casiez/1euro/
    """
    
    def __init__(self,
                 min_cutoff: float = 1.0,
                 beta: float = 0.007,
                 d_cutoff: float = 1.0):
        """
        Args:
            min_cutoff: Minimum cutoff frequency
            beta: Speed coefficient
            d_cutoff: Cutoff for derivative
        """
        self.min_cutoff = min_cutoff
        self.beta = beta
        self.d_cutoff = d_cutoff
        
        self.x_prev = None
        self.dx_prev = 0.0
        self.t_prev = None
        
    def __call__(self, x: float, t: float) -> float:
        """Filter value x at time t"""
        if self.x_prev is None:
            self.x_prev = x
            self.t_prev = t
            return x
            
        # Calculate time delta
        dt = t - self.t_prev
        if dt <= 0:
            dt = 0.001
            
        # Calculate derivative
        dx = (x - self.x_prev) / dt
        
        # Smooth derivative
        edx = self._smoothing_factor(dt, self.d_cutoff)
        dx_smooth = self._exponential_smoothing(edx, dx, self.dx_prev)
        
        # Calculate cutoff frequency
        cutoff = self.min_cutoff + self.beta * abs(dx_smooth)
        
        # Smooth value
        alpha = self._smoothing_factor(dt, cutoff)
        x_smooth = self._exponential_smoothing(alpha, x, self.x_prev)
        
        # Store for next iteration
        self.x_prev = x_smooth
        self.dx_prev = dx_smooth
        self.t_prev = t
        
        return x_smooth
        
    def _smoothing_factor(self, dt: float, cutoff: float) -> float:
        """Calculate smoothing factor"""
        r = 2 * math.pi * cutoff * dt
        return r / (r + 1)
        
    def _exponential_smoothing(self, alpha: float, x: float, x_prev: float) -> float:
        """Exponential smoothing"""
        return alpha * x + (1 - alpha) * x_prev
        
    def reset(self):
        """Reset filter state"""
        self.x_prev = None
        self.dx_prev = 0.0
        self.t_prev = None


class KeypointTracker:
    """Track and smooth individual keypoints"""
    
    def __init__(self,
                 smoothing_type: str = "one_euro",
                 window_size: int = 5):
        """
        Args:
            smoothing_type: "one_euro", "moving_average", or "exponential"
            window_size: Window size for moving average
        """
        self.smoothing_type = smoothing_type
        self.window_size = window_size
        
        self.filters: Dict[str, Dict[str, OneEuroFilter]] = {}
        self.history: Dict[str, Deque[Keypoint]] = {}
        
    def track(self, 
              keypoint_dict: Dict[str, Keypoint],
              timestamp: float,
              namespace: str = "default") -> Dict[str, Keypoint]:
        """
        Track and smooth keypoints
        
        Args:
            keypoint_dict: Dictionary of keypoints
            timestamp: Current timestamp
            namespace: Namespace for tracking (e.g., "left_hand", "pose")
            
        Returns:
            Smoothed keypoints
        """
        if not keypoint_dict:
            return {}
            
        smoothed = {}
        
        for name, kp in keypoint_dict.items():
            key = f"{namespace}_{name}"
            
            if self.smoothing_type == "one_euro":
                smoothed[name] = self._smooth_one_euro(key, kp, timestamp)
            elif self.smoothing_type == "moving_average":
                smoothed[name] = self._smooth_moving_average(key, kp)
            elif self.smoothing_type == "exponential":
                smoothed[name] = self._smooth_exponential(key, kp)
            else:
                smoothed[name] = kp
                
        return smoothed
        
    def _smooth_one_euro(self,
                        key: str,
                        kp: Keypoint,
                        timestamp: float) -> Keypoint:
        """Smooth using One Euro Filter"""
        if key not in self.filters:
            self.filters[key] = {
                'x': OneEuroFilter(),
                'y': OneEuroFilter(),
                'z': OneEuroFilter()
            }
            
        filters = self.filters[key]
        
        return Keypoint(
            x=filters['x'](kp.x, timestamp),
            y=filters['y'](kp.y, timestamp),
            z=filters['z'](kp.z, timestamp),
            confidence=kp.confidence,
            visibility=kp.visibility
        )
        
    def _smooth_moving_average(self, key: str, kp: Keypoint) -> Keypoint:
        """Smooth using moving average"""
        if key not in self.history:
            self.history[key] = deque(maxlen=self.window_size)
            
        history = self.history[key]
        history.append(kp)
        
        if len(history) < 2:
            return kp
            
        # Calculate average
        avg_x = sum(k.x for k in history) / len(history)
        avg_y = sum(k.y for k in history) / len(history)
        avg_z = sum(k.z for k in history) / len(history)
        
        return Keypoint(
            x=avg_x,
            y=avg_y,
            z=avg_z,
            confidence=kp.confidence,
            visibility=kp.visibility
        )
        
    def _smooth_exponential(self, key: str, kp: Keypoint, alpha: float = 0.3) -> Keypoint:
        """Smooth using exponential moving average"""
        if key not in self.history or not self.history[key]:
            self.history[key] = deque(maxlen=1)
            self.history[key].append(kp)
            return kp
            
        prev = self.history[key][0]
        
        smoothed = Keypoint(
            x=alpha * kp.x + (1 - alpha) * prev.x,
            y=alpha * kp.y + (1 - alpha) * prev.y,
            z=alpha * kp.z + (1 - alpha) * prev.z,
            confidence=kp.confidence,
            visibility=kp.visibility
        )
        
        self.history[key][0] = smoothed
        return smoothed
        
    def reset(self):
        """Reset all trackers"""
        self.filters.clear()
        self.history.clear()


class PoseTracker:
    """Track complete pose across frames"""
    
    def __init__(self,
                 smoothing_type: str = "one_euro",
                 confidence_threshold: float = 0.5,
                 interpolate_missing: bool = True):
        """
        Args:
            smoothing_type: Type of smoothing to apply
            confidence_threshold: Minimum confidence for tracking
            interpolate_missing: Interpolate missing keypoints
        """
        self.smoothing_type = smoothing_type
        self.confidence_threshold = confidence_threshold
        self.interpolate_missing = interpolate_missing
        
        self.pose_tracker = KeypointTracker(smoothing_type=smoothing_type)
        self.left_hand_tracker = KeypointTracker(smoothing_type=smoothing_type)
        self.right_hand_tracker = KeypointTracker(smoothing_type=smoothing_type)
        
        self.last_pose: Optional[PoseResult] = None
        
    def track(self, pose_result: PoseResult) -> PoseResult:
        """
        Track and smooth entire pose
        
        Args:
            pose_result: Raw pose detection result
            
        Returns:
            Smoothed and tracked pose
        """
        timestamp = pose_result.timestamp
        
        # Track each component
        smoothed_pose = PoseResult(
            timestamp=timestamp,
            confidence=pose_result.confidence,
            framework=pose_result.framework
        )
        
        # Track pose landmarks
        if pose_result.pose_landmarks:
            smoothed_pose.pose_landmarks = self.pose_tracker.track(
                pose_result.pose_landmarks,
                timestamp,
                namespace="pose"
            )
            
        # Track hands
        if pose_result.left_hand_landmarks:
            smoothed_pose.left_hand_landmarks = self.left_hand_tracker.track(
                pose_result.left_hand_landmarks,
                timestamp,
                namespace="left_hand"
            )
            
        if pose_result.right_hand_landmarks:
            smoothed_pose.right_hand_landmarks = self.right_hand_tracker.track(
                pose_result.right_hand_landmarks,
                timestamp,
                namespace="right_hand"
            )
            
        # Interpolate missing keypoints if enabled
        if self.interpolate_missing and self.last_pose:
            smoothed_pose = self._interpolate_missing(smoothed_pose, self.last_pose)
            
        self.last_pose = smoothed_pose
        
        return smoothed_pose
        
    def _interpolate_missing(self,
                            current: PoseResult,
                            previous: PoseResult) -> PoseResult:
        """Interpolate missing keypoints from previous frame"""
        # Interpolate pose landmarks
        if not current.pose_landmarks and previous.pose_landmarks:
            # If completely missing, use previous with reduced confidence
            current.pose_landmarks = {
                k: Keypoint(v.x, v.y, v.z, v.confidence * 0.5, v.visibility * 0.5)
                for k, v in previous.pose_landmarks.items()
            }
            
        # Similar for hands
        if not current.left_hand_landmarks and previous.left_hand_landmarks:
            current.left_hand_landmarks = {
                k: Keypoint(v.x, v.y, v.z, v.confidence * 0.5, v.visibility * 0.5)
                for k, v in previous.left_hand_landmarks.items()
            }
            
        if not current.right_hand_landmarks and previous.right_hand_landmarks:
            current.right_hand_landmarks = {
                k: Keypoint(v.x, v.y, v.z, v.confidence * 0.5, v.visibility * 0.5)
                for k, v in previous.right_hand_landmarks.items()
            }
            
        return current
        
    def reset(self):
        """Reset all trackers"""
        self.pose_tracker.reset()
        self.left_hand_tracker.reset()
        self.right_hand_tracker.reset()
        self.last_pose = None