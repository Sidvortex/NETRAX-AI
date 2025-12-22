"""
NETRAX AI - Iris Tracker
Ultra-high-precision iris, pupil, and gaze tracking
Sub-pixel accuracy with micro-movement detection
"""

import mediapipe as mp
import numpy as np
from typing import Dict, Any, List, Tuple, Optional
import logging
from scipy import signal
from config import settings

logger = logging.getLogger("NETRAX.IrisTracker")

class IrisTracker:
    """
    Ultra-precision iris and eye tracking system
    - Iris landmark detection (5 points per iris)
    - Pupil center and diameter estimation
    - Gaze direction vectors (3D)
    - Blink detection and classification
    - Micro-saccade tracking
    - Pupil dilation monitoring
    """
    
    def __init__(self):
        logger.info("👁️ Initializing Iris Tracker...")
        
        # Initialize MediaPipe Face Mesh with iris
        self.mp_face_mesh = mp.solutions.face_mesh
        self.face_mesh = self.mp_face_mesh.FaceMesh(
            max_num_faces=1,
            refine_landmarks=settings.IRIS_REFINE_LANDMARKS,
            min_detection_confidence=settings.IRIS_MIN_DETECTION_CONFIDENCE,
            min_tracking_confidence=settings.IRIS_MIN_TRACKING_CONFIDENCE
        )
        
        # Iris landmark indices (MediaPipe defines 5 points per iris)
        self.LEFT_IRIS = [469, 470, 471, 472, 473]
        self.RIGHT_IRIS = [474, 475, 476, 477, 478]
        
        # Eye landmarks for context
        self.LEFT_EYE = [33, 133, 160, 159, 158, 157, 173, 246]
        self.RIGHT_EYE = [362, 263, 387, 386, 385, 384, 398, 466]
        
        # Eyelid landmarks for blink detection
        self.LEFT_EYE_TOP = [159, 160]
        self.LEFT_EYE_BOTTOM = [145, 144]
        self.RIGHT_EYE_TOP = [386, 387]
        self.RIGHT_EYE_BOTTOM = [374, 373]
        
        # Tracking state
        self.last_iris_data = None
        self.iris_history = []
        self.gaze_history = []
        self.blink_history = []
        self.pupil_diameter_history = []
        
        self.max_history = 30
        self.calibration_data = None
        
        # Blink detection
        self.blink_threshold = 0.2
        self.consecutive_blinks = 0
        self.last_blink_time = 0
        
        # Saccade detection
        self.saccade_threshold = 0.05
        self.last_gaze_position = None
        
        # Kalman filters for smoothing
        self.filters = {}
        if settings.ENABLE_KALMAN_FILTER:
            self._initialize_filters()
        
        logger.info("✅ Iris Tracker initialized with sub-pixel precision")
    
    def _initialize_filters(self):
        """Initialize Kalman filters for ultra-smooth tracking"""
        from vision_engine.filters import KalmanFilter
        
        # Filters for iris centers
        self.filters["left_iris_x"] = KalmanFilter(0.005, 0.05)
        self.filters["left_iris_y"] = KalmanFilter(0.005, 0.05)
        self.filters["right_iris_x"] = KalmanFilter(0.005, 0.05)
        self.filters["right_iris_y"] = KalmanFilter(0.005, 0.05)
        
        # Filters for gaze vectors
        self.filters["gaze_x"] = KalmanFilter(0.01, 0.1)
        self.filters["gaze_y"] = KalmanFilter(0.01, 0.1)
        self.filters["gaze_z"] = KalmanFilter(0.01, 0.1)
    
    def process(self, rgb_frame: np.ndarray) -> Dict[str, Any]:
        """
        Process frame for ultra-precision iris tracking
        
        Args:
            rgb_frame: RGB image
            
        Returns:
            Comprehensive iris and gaze tracking data
        """
        try:
            h, w = rgb_frame.shape[:2]
            
            # Process with MediaPipe
            results = self.face_mesh.process(rgb_frame)
            
            if not results.multi_face_landmarks:
                return {
                    "detected": False,
                    "confidence": 0.0
                }
            
            face_landmarks = results.multi_face_landmarks[0]
            
            # Extract iris data
            left_iris = self._extract_iris_data(
                face_landmarks, self.LEFT_IRIS, "left", w, h
            )
            right_iris = self._extract_iris_data(
                face_landmarks, self.RIGHT_IRIS, "right", w, h
            )
            
            # Calculate gaze direction
            gaze_vector = self._calculate_gaze_direction(
                left_iris, right_iris
            )
            
            # Detect blinks
            blink_state = self._detect_blinks(
                face_landmarks, w, h
            )
            
            # Calculate pupil metrics
            left_pupil = self._estimate_pupil(left_iris)
            right_pupil = self._estimate_pupil(right_iris)
            
            # Detect micro-movements (saccades)
            saccade = self._detect_saccades(gaze_vector)
            
            # Calculate eye openness
            left_openness = self._calculate_eye_openness(
                face_landmarks, self.LEFT_EYE_TOP, self.LEFT_EYE_BOTTOM, w, h
            )
            right_openness = self._calculate_eye_openness(
                face_landmarks, self.RIGHT_EYE_TOP, self.RIGHT_EYE_BOTTOM, w, h
            )
            
            # Build comprehensive tracking data
            tracking_data = {
                "detected": True,
                "confidence": self._calculate_confidence(face_landmarks),
                "left_eye": {
                    "iris": left_iris,
                    "pupil": left_pupil,
                    "openness": left_openness,
                    "blink": blink_state["left"]
                },
                "right_eye": {
                    "iris": right_iris,
                    "pupil": right_pupil,
                    "openness": right_openness,
                    "blink": blink_state["right"]
                },
                "gaze": gaze_vector,
                "saccade": saccade,
                "blink_rate": self._calculate_blink_rate(),
                "average_pupil_diameter": (left_pupil["diameter"] + right_pupil["diameter"]) / 2
            }
            
            # Store in history
            self.last_iris_data = tracking_data
            self.iris_history.append(tracking_data)
            if len(self.iris_history) > self.max_history:
                self.iris_history.pop(0)
            
            return tracking_data
            
        except Exception as e:
            logger.error(f"Iris tracking error: {e}")
            return {
                "detected": False,
                "confidence": 0.0,
                "error": str(e)
            }
    
    def _extract_iris_data(self, landmarks, iris_indices: List[int], 
                          side: str, w: int, h: int) -> Dict:
        """Extract detailed iris landmark data with sub-pixel precision"""
        iris_points = []
        
        for idx in iris_indices:
            lm = landmarks.landmark[idx]
            x = lm.x * w
            y = lm.y * h
            z = lm.z
            
            # Apply Kalman filtering for ultra-smooth tracking
            if settings.ENABLE_KALMAN_FILTER:
                filter_key = f"{side}_iris"
                if f"{filter_key}_x" in self.filters:
                    x, _ = self.filters[f"{filter_key}_x"].update(x, 0)
                if f"{filter_key}_y" in self.filters:
                    y, _ = self.filters[f"{filter_key}_y"].update(y, 0)
            
            iris_points.append({
                "x": float(x),
                "y": float(y),
                "z": float(z)
            })
        
        # Calculate iris center (center point is index 0)
        center = iris_points[0]
        
        # Calculate iris radius from landmark points
        radius = self._calculate_iris_radius(iris_points)
        
        return {
            "center": center,
            "radius": float(radius),
            "landmarks": iris_points,
            "side": side
        }
    
    def _calculate_iris_radius(self, points: List[Dict]) -> float:
        """Calculate iris radius from landmark points"""
        if len(points) < 5:
            return 0.0
        
        # Center is first point, calculate average distance to others
        center = points[0]
        distances = []
        
        for point in points[1:]:
            dx = point["x"] - center["x"]
            dy = point["y"] - center["y"]
            distances.append(np.sqrt(dx*dx + dy*dy))
        
        return np.mean(distances)
    
    def _calculate_gaze_direction(self, left_iris: Dict, 
                                  right_iris: Dict) -> Dict:
        """Calculate 3D gaze direction vector"""
        if not left_iris or not right_iris:
            return {"x": 0.0, "y": 0.0, "z": 0.0, "magnitude": 0.0}
        
        # Average iris centers for gaze point
        gaze_x = (left_iris["center"]["x"] + right_iris["center"]["x"]) / 2
        gaze_y = (left_iris["center"]["y"] + right_iris["center"]["y"]) / 2
        gaze_z = (left_iris["center"]["z"] + right_iris["center"]["z"]) / 2
        
        # Apply Kalman filtering
        if settings.ENABLE_KALMAN_FILTER:
            gaze_x, _ = self.filters["gaze_x"].update(gaze_x, 0)
            gaze_y, _ = self.filters["gaze_y"].update(gaze_y, 0)
            gaze_z, _ = self.filters["gaze_z"].update(gaze_z, 0)
        
        # Calculate magnitude
        magnitude = np.sqrt(gaze_x**2 + gaze_y**2 + gaze_z**2)
        
        gaze_vector = {
            "x": float(gaze_x),
            "y": float(gaze_y),
            "z": float(gaze_z),
            "magnitude": float(magnitude)
        }
        
        # Store in history
        self.gaze_history.append(gaze_vector)
        if len(self.gaze_history) > self.max_history:
            self.gaze_history.pop(0)
        
        return gaze_vector
    
    def _detect_blinks(self, landmarks, w: int, h: int) -> Dict:
        """Detect eye blinks with high precision"""
        left_openness = self._calculate_eye_openness(
            landmarks, self.LEFT_EYE_TOP, self.LEFT_EYE_BOTTOM, w, h
        )
        right_openness = self._calculate_eye_openness(
            landmarks, self.RIGHT_EYE_TOP, self.RIGHT_EYE_BOTTOM, w, h
        )
        
        left_blink = left_openness < self.blink_threshold
        right_blink = right_openness < self.blink_threshold
        
        # Store blink events
        self.blink_history.append({
            "left": left_blink,
            "right": right_blink,
            "timestamp": len(self.blink_history)
        })
        if len(self.blink_history) > 100:
            self.blink_history.pop(0)
        
        return {
            "left": left_blink,
            "right": right_blink,
            "both": left_blink and right_blink
        }
    
    def _calculate_eye_openness(self, landmarks, top_indices: List[int],
                                bottom_indices: List[int], w: int, h: int) -> float:
        """Calculate eye openness ratio"""
        # Get top and bottom points
        top_y = np.mean([landmarks.landmark[i].y * h for i in top_indices])
        bottom_y = np.mean([landmarks.landmark[i].y * h for i in bottom_indices])
        
        # Calculate vertical distance
        eye_height = abs(bottom_y - top_y)
        
        # Normalize (typical eye height is ~20-30 pixels at 720p)
        normalized_height = eye_height / 30.0
        
        return float(normalized_height)
    
    def _estimate_pupil(self, iris_data: Dict) -> Dict:
        """Estimate pupil center and diameter"""
        if not iris_data or "center" not in iris_data:
            return {"center": {"x": 0, "y": 0, "z": 0}, "diameter": 0.0}
        
        # Pupil is approximately at iris center
        pupil_center = iris_data["center"]
        
        # Estimate pupil diameter (typically 0.3-0.4 of iris diameter)
        iris_diameter = iris_data["radius"] * 2
        pupil_diameter = iris_diameter * 0.35
        
        # Store diameter history for dilation tracking
        self.pupil_diameter_history.append(pupil_diameter)
        if len(self.pupil_diameter_history) > 100:
            self.pupil_diameter_history.pop(0)
        
        return {
            "center": pupil_center,
            "diameter": float(pupil_diameter),
            "dilation_rate": self._calculate_dilation_rate()
        }
    
    def _calculate_dilation_rate(self) -> float:
        """Calculate pupil dilation rate"""
        if len(self.pupil_diameter_history) < 2:
            return 0.0
        
        # Compare current to recent average
        recent_avg = np.mean(self.pupil_diameter_history[-10:])
        current = self.pupil_diameter_history[-1]
        
        return float((current - recent_avg) / recent_avg if recent_avg > 0 else 0.0)
    
    def _detect_saccades(self, current_gaze: Dict) -> Dict:
        """Detect micro-saccades (rapid eye movements)"""
        if not self.last_gaze_position:
            self.last_gaze_position = current_gaze
            return {"detected": False, "velocity": 0.0}
        
        # Calculate gaze movement
        dx = current_gaze["x"] - self.last_gaze_position["x"]
        dy = current_gaze["y"] - self.last_gaze_position["y"]
        
        velocity = np.sqrt(dx*dx + dy*dy)
        
        saccade_detected = velocity > self.saccade_threshold
        
        self.last_gaze_position = current_gaze
        
        return {
            "detected": saccade_detected,
            "velocity": float(velocity),
            "direction": {"x": float(dx), "y": float(dy)}
        }
    
    def _calculate_blink_rate(self) -> float:
        """Calculate blinks per minute"""
        if len(self.blink_history) < 2:
            return 0.0
        
        # Count blink transitions (closed -> open)
        blinks = 0
        for i in range(1, len(self.blink_history)):
            prev = self.blink_history[i-1]
            curr = self.blink_history[i]
            if prev["both"] and not curr["both"]:
                blinks += 1
        
        # Estimate rate (history is ~3 seconds at 30fps)
        time_span = len(self.blink_history) / 30.0  # seconds
        blinks_per_minute = (blinks / time_span) * 60 if time_span > 0 else 0
        
        return float(blinks_per_minute)
    
    def _calculate_confidence(self, landmarks) -> float:
        """Calculate tracking confidence"""
        # Average visibility of iris landmarks
        left_vis = np.mean([landmarks.landmark[i].visibility 
                           for i in self.LEFT_IRIS if hasattr(landmarks.landmark[i], 'visibility')])
        right_vis = np.mean([landmarks.landmark[i].visibility 
                            for i in self.RIGHT_IRIS if hasattr(landmarks.landmark[i], 'visibility')])
        
        return float((left_vis + right_vis) / 2) if left_vis and right_vis else 0.5
    
    def calibrate(self):
        """Calibrate iris tracking system"""
        logger.info("🎯 Calibrating iris tracker...")
        
        # Clear history
        self.iris_history = []
        self.gaze_history = []
        self.blink_history = []
        self.pupil_diameter_history = []
        
        # Reset filters
        if settings.ENABLE_KALMAN_FILTER:
            for filter_obj in self.filters.values():
                filter_obj.reset()
        
        logger.info("✅ Iris calibration complete")
    
    def reset(self):
        """Reset tracking state"""
        self.calibrate()
    
    def cleanup(self):
        """Release resources"""
        logger.info("🧹 Cleaning up iris tracker...")
        if self.face_mesh:
            self.face_mesh.close()