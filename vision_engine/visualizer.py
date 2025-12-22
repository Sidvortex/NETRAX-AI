"""
NETRAX AI - Cyberpunk Visualizer
Dystopian visual effects and HUD overlays
"""

import cv2
import numpy as np
from typing import Dict, Any, List, Tuple
import logging
from config import settings

logger = logging.getLogger("NETRAX.Visualizer")

class CyberpunkVisualizer:
    """
    Cyberpunk-themed visualization system
    Neon overlays, glitch effects, and dystopian aesthetics
    """
    
    def __init__(self):
        logger.info("🎨 Initializing Cyberpunk Visualizer...")
        
        # Colors (BGR format)
        self.COLOR_RED = (0, 0, 255)
        self.COLOR_CYAN = (255, 255, 0)
        self.COLOR_MAGENTA = (255, 0, 255)
        self.COLOR_GREEN = (0, 255, 0)
        self.COLOR_PURPLE = (255, 0, 128)
        
        # Primary colors from settings
        self.PRIMARY = settings.OVERLAY_COLOR_PRIMARY
        self.SECONDARY = settings.OVERLAY_COLOR_SECONDARY
        
        # Fonts
        self.font = cv2.FONT_HERSHEY_SIMPLEX
        self.font_mono = cv2.FONT_HERSHEY_DUPLEX
        
        logger.info("✅ Visualizer initialized")
    
    def draw_body(self, frame: np.ndarray, body_data: Dict) -> np.ndarray:
        """Draw body skeleton with cyberpunk style"""
        if not body_data.get("detected"):
            return frame
        
        pose = body_data.get("pose", {})
        landmarks = pose.get("landmarks", [])
        connections = pose.get("skeleton_connections", [])
        
        if not landmarks:
            return frame
        
        h, w = frame.shape[:2]
        
        # Draw connections (skeleton)
        for connection in connections:
            start_idx, end_idx = connection
            
            if start_idx < len(landmarks) and end_idx < len(landmarks):
                start = landmarks[start_idx]
                end = landmarks[end_idx]
                
                # Skip if low visibility
                if start["visibility"] < 0.5 or end["visibility"] < 0.5:
                    continue
                
                start_point = (int(start["x"] * w), int(start["y"] * h))
                end_point = (int(end["x"] * w), int(end["y"] * h))
                
                # Draw glowing line
                cv2.line(frame, start_point, end_point, 
                        self.PRIMARY, settings.OVERLAY_THICKNESS + 2)
                cv2.line(frame, start_point, end_point, 
                        self.COLOR_CYAN, settings.OVERLAY_THICKNESS)
        
        # Draw key points with glow
        keypoints = pose.get("keypoints", {})
        for name, point in keypoints.items():
            if point["visibility"] < 0.5:
                continue
            
            center = (int(point["x"] * w), int(point["y"] * h))
            
            # Outer glow
            cv2.circle(frame, center, 8, self.PRIMARY, -1)
            # Inner bright spot
            cv2.circle(frame, center, 4, self.COLOR_CYAN, -1)
            # Highlight
            cv2.circle(frame, center, 2, (255, 255, 255), -1)
        
        return frame
    
    def draw_iris(self, frame: np.ndarray, iris_data: Dict) -> np.ndarray:
        """Draw iris tracking with ultra-precision visualization"""
        if not iris_data.get("detected"):
            return frame
        
        h, w = frame.shape[:2]
        
        # Draw left eye
        left_eye = iris_data.get("left_eye", {})
        if left_eye.get("iris"):
            frame = self._draw_single_iris(frame, left_eye, w, h)
        
        # Draw right eye
        right_eye = iris_data.get("right_eye", {})
        if right_eye.get("iris"):
            frame = self._draw_single_iris(frame, right_eye, w, h)
        
        # Draw gaze vector
        gaze = iris_data.get("gaze", {})
        if gaze and gaze.get("magnitude", 0) > 0:
            frame = self._draw_gaze_vector(frame, gaze, w, h)
        
        return frame
    
    def _draw_single_iris(self, frame: np.ndarray, eye_data: Dict, 
                         w: int, h: int) -> np.ndarray:
        """Draw single iris with detailed visualization"""
        iris = eye_data.get("iris", {})
        center = iris.get("center", {})
        radius = iris.get("radius", 0)
        
        if not center or radius == 0:
            return frame
        
        center_point = (int(center["x"]), int(center["y"]))
        radius_px = int(radius)
        
        # Iris outer circle (red glow)
        cv2.circle(frame, center_point, radius_px + 4, self.COLOR_RED, 2)
        cv2.circle(frame, center_point, radius_px + 2, self.COLOR_RED, 1)
        
        # Iris main circle (cyan)
        cv2.circle(frame, center_point, radius_px, self.COLOR_CYAN, 2)
        
        # Pupil
        pupil = eye_data.get("pupil", {})
        pupil_center = pupil.get("center", center)
        pupil_radius = int(pupil.get("diameter", radius * 0.35) / 2)
        
        pupil_point = (int(pupil_center["x"]), int(pupil_center["y"]))
        cv2.circle(frame, pupil_point, pupil_radius, self.COLOR_MAGENTA, -1)
        cv2.circle(frame, pupil_point, pupil_radius, self.PRIMARY, 2)
        
        # Iris landmarks
        iris_landmarks = iris.get("landmarks", [])
        for landmark in iris_landmarks:
            lm_point = (int(landmark["x"]), int(landmark["y"]))
            cv2.circle(frame, lm_point, 2, self.COLOR_GREEN, -1)
        
        # Draw crosshair at center
        crosshair_size = 10
        cv2.line(frame, 
                (center_point[0] - crosshair_size, center_point[1]),
                (center_point[0] + crosshair_size, center_point[1]),
                self.COLOR_CYAN, 1)
        cv2.line(frame, 
                (center_point[0], center_point[1] - crosshair_size),
                (center_point[0], center_point[1] + crosshair_size),
                self.COLOR_CYAN, 1)
        
        return frame
    
    def _draw_gaze_vector(self, frame: np.ndarray, gaze: Dict, 
                         w: int, h: int) -> np.ndarray:
        """Draw gaze direction vector"""
        # Calculate start point (center of frame)
        start_x = int(gaze["x"])
        start_y = int(gaze["y"])
        
        # Draw vector line
        vector_length = 100
        end_x = int(start_x + gaze["x"] * vector_length)
        end_y = int(start_y + gaze["y"] * vector_length)
        
        cv2.arrowedLine(frame, 
                       (start_x, start_y),
                       (end_x, end_y),
                       self.COLOR_PURPLE, 2, tipLength=0.3)
        
        return frame
    
    def draw_gesture(self, frame: np.ndarray, gesture_data: Dict) -> np.ndarray:
        """Draw gesture indicator"""
        gesture = gesture_data.get("gesture")
        confidence = gesture_data.get("confidence", 0.0)
        
        if not gesture:
            return frame
        
        # Draw gesture name at top center
        text = f"{gesture.upper()} ({confidence:.0%})"
        text_size = cv2.getTextSize(text, self.font, 1.2, 2)[0]
        
        x = (frame.shape[1] - text_size[0]) // 2
        y = 50
        
        # Background
        cv2.rectangle(frame, 
                     (x - 10, y - text_size[1] - 10),
                     (x + text_size[0] + 10, y + 10),
                     (0, 0, 0), -1)
        
        # Glow
        cv2.putText(frame, text, (x, y), self.font, 1.2, self.PRIMARY, 4)
        
        # Text
        cv2.putText(frame, text, (x, y), self.font, 1.2, self.COLOR_CYAN, 2)
        
        return frame
    
    def draw_objects(self, frame: np.ndarray, object_data: Dict) -> np.ndarray:
        """Draw detected objects"""
        detections = object_data.get("detections", [])
        
        for detection in detections:
            bbox = detection["bbox"]
            label = detection["label"]
            confidence = detection["confidence"]
            
            # Draw bounding box
            x1, y1 = int(bbox["x1"]), int(bbox["y1"])
            x2, y2 = int(bbox["x2"]), int(bbox["y2"])
            
            cv2.rectangle(frame, (x1, y1), (x2, y2), self.COLOR_GREEN, 2)
            
            # Draw label
            text = f"{label} {confidence:.2f}"
            cv2.putText(frame, text, (x1, y1 - 10), 
                       self.font, 0.6, self.COLOR_GREEN, 2)
        
        return frame
    
    def draw_hud(self, frame: np.ndarray, tracking_data: Dict, fps: float) -> np.ndarray:
        """Draw heads-up display overlay"""
        h, w = frame.shape[:2]
        
        # FPS counter (top right)
        fps_text = f"FPS: {fps:.1f}"
        cv2.putText(frame, fps_text, (w - 150, 30), 
                   self.font_mono, 0.7, self.COLOR_CYAN, 2)
        
        # System status (top left)
        status_text = "NETRAX ACTIVE"
        cv2.putText(frame, status_text, (20, 30), 
                   self.font_mono, 0.7, self.COLOR_RED, 2)
        
        # Confidence indicator (bottom right)
        confidence = tracking_data.get("confidence", 0.0)
        conf_text = f"CONF: {confidence:.0%}"
        cv2.putText(frame, conf_text, (w - 150, h - 20), 
                   self.font_mono, 0.6, self.COLOR_CYAN, 2)
        
        return frame
    
    def apply_effects(self, frame: np.ndarray) -> np.ndarray:
        """Apply cyberpunk visual effects"""
        if settings.ENABLE_GLOW_EFFECT:
            frame = self._apply_glow(frame)
        
        if settings.ENABLE_SCANLINES:
            frame = self._apply_scanlines(frame)
        
        return frame
    
    def _apply_glow(self, frame: np.ndarray) -> np.ndarray:
        """Apply subtle glow effect"""
        blurred = cv2.GaussianBlur(frame, 
                                   (settings.GLOW_KERNEL_SIZE, settings.GLOW_KERNEL_SIZE), 
                                   0)
        
        glowing = cv2.addWeighted(frame, 1 - settings.GLOW_INTENSITY, 
                                 blurred, settings.GLOW_INTENSITY, 0)
        
        return glowing
    
    def _apply_scanlines(self, frame: np.ndarray) -> np.ndarray:
        """Apply CRT scanline effect"""
        h, w = frame.shape[:2]
        
        # Create scanline mask
        for y in range(0, h, settings.SCANLINE_SPACING):
            frame[y:y+1, :] = (frame[y:y+1, :] * 
                              (1 - settings.SCANLINE_INTENSITY)).astype(np.uint8)
        
        return frame