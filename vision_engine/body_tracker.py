"""
NETRAX AI - Body Tracker
Full-body pose estimation and skeleton tracking using MediaPipe
"""

import mediapipe as mp
import numpy as np
from typing import Dict, Any, List, Optional
import logging
from config import settings

logger = logging.getLogger("NETRAX.BodyTracker")

class BodyTracker:
    """
    Full-body pose and skeleton tracking
    Uses MediaPipe Holistic for comprehensive body landmark detection
    """
    
    def __init__(self):
        logger.info("🏃 Initializing Body Tracker...")
        
        # Initialize MediaPipe Holistic
        self.mp_holistic = mp.solutions.holistic
        self.mp_pose = mp.solutions.pose
        
        self.holistic = self.mp_holistic.Holistic(
            model_complexity=settings.BODY_MODEL_COMPLEXITY,
            min_detection_confidence=settings.BODY_MIN_DETECTION_CONFIDENCE,
            min_tracking_confidence=settings.BODY_MIN_TRACKING_CONFIDENCE,
            smooth_landmarks=settings.BODY_SMOOTH_LANDMARKS
        )
        
        # Landmark indices for key body parts
        self.POSE_LANDMARKS = self.mp_pose.PoseLandmark
        
        # Tracking state
        self.last_landmarks = None
        self.landmark_history = []
        self.max_history = 10
        
        # Kalman filters for smoothing (optional)
        self.filters = {}
        if settings.ENABLE_KALMAN_FILTER:
            self._initialize_filters()
        
        logger.info("✅ Body Tracker initialized")
    
    def _initialize_filters(self):
        """Initialize Kalman filters for landmark smoothing"""
        from vision_engine.filters import KalmanFilter
        
        # Create filters for key landmarks
        for i in range(33):  # 33 pose landmarks
            self.filters[f"pose_{i}"] = KalmanFilter(
                process_noise=settings.KALMAN_PROCESS_NOISE,
                measurement_noise=settings.KALMAN_MEASUREMENT_NOISE
            )
    
    def process(self, rgb_frame: np.ndarray) -> Dict[str, Any]:
        """
        Process frame and extract body landmarks
        
        Args:
            rgb_frame: RGB image
            
        Returns:
            Dict containing body tracking data
        """
        try:
            # Process frame with MediaPipe
            results = self.holistic.process(rgb_frame)
            
            if not results.pose_landmarks:
                return {
                    "detected": False,
                    "confidence": 0.0
                }
            
            # Extract pose landmarks
            pose_landmarks = self._extract_landmarks(
                results.pose_landmarks,
                rgb_frame.shape
            )
            
            # Extract hand landmarks
            left_hand = self._extract_hand_landmarks(
                results.left_hand_landmarks,
                rgb_frame.shape,
                "left"
            )
            
            right_hand = self._extract_hand_landmarks(
                results.right_hand_landmarks,
                rgb_frame.shape,
                "right"
            )
            
            # Calculate body metrics
            metrics = self._calculate_metrics(pose_landmarks)
            
            # Build tracking data
            tracking_data = {
                "detected": True,
                "confidence": self._calculate_confidence(results),
                "pose": {
                    "landmarks": pose_landmarks,
                    "keypoints": self._get_keypoints(pose_landmarks),
                    "skeleton_connections": self._get_skeleton_connections()
                },
                "hands": {
                    "left": left_hand,
                    "right": right_hand
                },
                "metrics": metrics
            }
            
            # Store in history
            self.last_landmarks = pose_landmarks
            self.landmark_history.append(pose_landmarks)
            if len(self.landmark_history) > self.max_history:
                self.landmark_history.pop(0)
            
            return tracking_data
            
        except Exception as e:
            logger.error(f"Body tracking error: {e}")
            return {
                "detected": False,
                "confidence": 0.0,
                "error": str(e)
            }
    
    def _extract_landmarks(self, landmarks, frame_shape) -> List[Dict[str, float]]:
        """Extract and normalize landmarks"""
        h, w = frame_shape[:2]
        landmark_list = []
        
        for idx, landmark in enumerate(landmarks.landmark):
            # Apply Kalman filter if enabled
            if settings.ENABLE_KALMAN_FILTER and f"pose_{idx}" in self.filters:
                x, y = self.filters[f"pose_{idx}"].update(
                    landmark.x * w, landmark.y * h
                )
                x, y = x / w, y / h
            else:
                x, y = landmark.x, landmark.y
            
            landmark_list.append({
                "x": float(x),
                "y": float(y),
                "z": float(landmark.z),
                "visibility": float(landmark.visibility)
            })
        
        return landmark_list
    
    def _extract_hand_landmarks(self, landmarks, frame_shape, side: str) -> Dict:
        """Extract hand landmarks"""
        if not landmarks:
            return {"detected": False}
        
        h, w = frame_shape[:2]
        hand_landmarks = []
        
        for landmark in landmarks.landmark:
            hand_landmarks.append({
                "x": float(landmark.x),
                "y": float(landmark.y),
                "z": float(landmark.z)
            })
        
        return {
            "detected": True,
            "side": side,
            "landmarks": hand_landmarks,
            "fingertips": self._get_fingertips(hand_landmarks)
        }
    
    def _get_fingertips(self, hand_landmarks: List[Dict]) -> Dict:
        """Extract fingertip positions"""
        if len(hand_landmarks) < 21:
            return {}
        
        # MediaPipe hand landmark indices
        return {
            "thumb": hand_landmarks[4],
            "index": hand_landmarks[8],
            "middle": hand_landmarks[12],
            "ring": hand_landmarks[16],
            "pinky": hand_landmarks[20]
        }
    
    def _get_keypoints(self, landmarks: List[Dict]) -> Dict:
        """Extract key body points"""
        if len(landmarks) < 33:
            return {}
        
        return {
            "nose": landmarks[0],
            "left_eye": landmarks[2],
            "right_eye": landmarks[5],
            "left_shoulder": landmarks[11],
            "right_shoulder": landmarks[12],
            "left_elbow": landmarks[13],
            "right_elbow": landmarks[14],
            "left_wrist": landmarks[15],
            "right_wrist": landmarks[16],
            "left_hip": landmarks[23],
            "right_hip": landmarks[24],
            "left_knee": landmarks[25],
            "right_knee": landmarks[26],
            "left_ankle": landmarks[27],
            "right_ankle": landmarks[28]
        }
    
    def _get_skeleton_connections(self) -> List[tuple]:
        """Get skeleton connection pairs for visualization"""
        return list(self.mp_pose.POSE_CONNECTIONS)
    
    def _calculate_metrics(self, landmarks: List[Dict]) -> Dict:
        """Calculate body metrics (angles, distances, etc.)"""
        if len(landmarks) < 33:
            return {}
        
        return {
            "shoulder_width": self._calculate_distance(
                landmarks[11], landmarks[12]
            ),
            "torso_height": self._calculate_distance(
                landmarks[11], landmarks[23]
            ),
            "pose_height": self._calculate_distance(
                landmarks[0], landmarks[28]
            )
        }
    
    def _calculate_distance(self, point1: Dict, point2: Dict) -> float:
        """Calculate Euclidean distance between two points"""
        dx = point1["x"] - point2["x"]
        dy = point1["y"] - point2["y"]
        dz = point1["z"] - point2["z"]
        return float(np.sqrt(dx*dx + dy*dy + dz*dz))
    
    def _calculate_confidence(self, results) -> float:
        """Calculate overall tracking confidence"""
        if not results.pose_landmarks:
            return 0.0
        
        # Average visibility of all landmarks
        visibilities = [lm.visibility for lm in results.pose_landmarks.landmark]
        return float(np.mean(visibilities))
    
    def reset(self):
        """Reset tracking state"""
        logger.info("🔄 Resetting body tracker...")
        self.last_landmarks = None
        self.landmark_history = []
        
        if settings.ENABLE_KALMAN_FILTER:
            for filter_obj in self.filters.values():
                filter_obj.reset()
    
    def cleanup(self):
        """Release resources"""
        logger.info("🧹 Cleaning up body tracker...")
        if self.holistic:
            self.holistic.close()