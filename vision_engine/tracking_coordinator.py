"""
NETRAX AI - Tracking Coordinator
Orchestrates all vision subsystems and data fusion
"""

import cv2
import numpy as np
from typing import Dict, Any, Tuple, Optional
import logging
from datetime import datetime
import time

from vision_engine.body_tracker import BodyTracker
from vision_engine.iris_tracker import IrisTracker
from vision_engine.gesture_engine import GestureEngine
from vision_engine.object_detector import ObjectDetector
from vision_engine.visualizer import CyberpunkVisualizer
from config import settings

logger = logging.getLogger("NETRAX.Coordinator")

class TrackingCoordinator:
    """
    Master coordinator for all vision subsystems
    Handles data fusion, synchronization, and visualization
    """
    
    def __init__(self, 
                 enable_body: bool = True,
                 enable_iris: bool = True,
                 enable_gestures: bool = True,
                 enable_objects: bool = False):
        
        logger.info("🧠 Initializing Tracking Coordinator...")
        
        self.enable_body = enable_body
        self.enable_iris = enable_iris
        self.enable_gestures = enable_gestures
        self.enable_objects = enable_objects
        
        # Initialize subsystems
        self.body_tracker = BodyTracker() if enable_body else None
        self.iris_tracker = IrisTracker() if enable_iris else None
        self.gesture_engine = GestureEngine() if enable_gestures else None
        self.object_detector = ObjectDetector() if enable_objects else None
        
        # Initialize visualizer
        self.visualizer = CyberpunkVisualizer() if settings.ENABLE_CYBERPUNK_OVERLAY else None
        
        # Performance tracking
        self.frame_count = 0
        self.start_time = time.time()
        self.processing_times = []
        self.max_processing_history = 100
        
        # Calibration state
        self.calibrated = False
        self.calibration_frames = []
        
        # Tracking state
        self.last_valid_data = None
        self.confidence_history = []
        
        logger.info(f"✅ Coordinator initialized - Body: {enable_body}, Iris: {enable_iris}, "
                   f"Gestures: {enable_gestures}, Objects: {enable_objects}")
    
    def process_frame(self, frame: np.ndarray) -> Tuple[np.ndarray, Dict[str, Any]]:
        """
        Process a single frame through all vision subsystems
        Returns: (processed_frame, tracking_data)
        """
        start_time = time.time()
        
        # Prepare output frame
        output_frame = frame.copy()
        
        # Initialize tracking data
        tracking_data = {
            "timestamp": datetime.now().isoformat(),
            "frame_number": self.frame_count,
            "confidence": 0.0
        }
        
        try:
            # Convert to RGB for MediaPipe
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            
            # Body tracking
            if self.body_tracker and self.enable_body:
                body_results = self.body_tracker.process(rgb_frame)
                tracking_data["body"] = body_results
                
                if body_results.get("detected"):
                    if self.visualizer:
                        output_frame = self.visualizer.draw_body(
                            output_frame, body_results
                        )
            
            # Iris tracking
            if self.iris_tracker and self.enable_iris:
                iris_results = self.iris_tracker.process(rgb_frame)
                tracking_data["eyes"] = iris_results
                
                if iris_results.get("detected"):
                    if self.visualizer:
                        output_frame = self.visualizer.draw_iris(
                            output_frame, iris_results
                        )
            
            # Gesture recognition
            if self.gesture_engine and self.enable_gestures:
                gesture_results = self.gesture_engine.process(
                    tracking_data.get("body", {})
                )
                tracking_data["gesture"] = gesture_results.get("gesture")
                tracking_data["gesture_confidence"] = gesture_results.get("confidence", 0.0)
                
                if gesture_results.get("gesture"):
                    if self.visualizer:
                        output_frame = self.visualizer.draw_gesture(
                            output_frame, gesture_results
                        )
            
            # Object detection
            if self.object_detector and self.enable_objects:
                object_results = self.object_detector.process(frame)
                tracking_data["objects"] = object_results
                
                if object_results.get("detections"):
                    if self.visualizer:
                        output_frame = self.visualizer.draw_objects(
                            output_frame, object_results
                        )
            
            # Calculate overall confidence
            tracking_data["confidence"] = self._calculate_confidence(tracking_data)
            
            # Add cyberpunk effects
            if self.visualizer:
                output_frame = self.visualizer.apply_effects(output_frame)
            
            # Draw HUD overlay
            if self.visualizer:
                output_frame = self.visualizer.draw_hud(
                    output_frame, tracking_data, self._get_fps()
                )
            
        except Exception as e:
            logger.error(f"Frame processing error: {e}")
            tracking_data["error"] = str(e)
        
        # Performance tracking
        processing_time = time.time() - start_time
        self.processing_times.append(processing_time)
        if len(self.processing_times) > self.max_processing_history:
            self.processing_times.pop(0)
        
        self.frame_count += 1
        
        # Store last valid data
        if tracking_data.get("confidence", 0) > 0.5:
            self.last_valid_data = tracking_data
        
        return output_frame, tracking_data
    
    def _calculate_confidence(self, tracking_data: Dict[str, Any]) -> float:
        """Calculate overall tracking confidence"""
        confidences = []
        
        if tracking_data.get("body", {}).get("confidence"):
            confidences.append(tracking_data["body"]["confidence"])
        
        if tracking_data.get("eyes", {}).get("confidence"):
            confidences.append(tracking_data["eyes"]["confidence"])
        
        if tracking_data.get("gesture_confidence"):
            confidences.append(tracking_data["gesture_confidence"])
        
        if tracking_data.get("objects", {}).get("confidence"):
            confidences.append(tracking_data["objects"]["confidence"])
        
        return sum(confidences) / len(confidences) if confidences else 0.0
    
    def _get_fps(self) -> float:
        """Calculate current FPS"""
        if self.frame_count == 0:
            return 0.0
        elapsed = time.time() - self.start_time
        return self.frame_count / elapsed if elapsed > 0 else 0.0
    
    def _get_avg_processing_time(self) -> float:
        """Get average processing time in ms"""
        if not self.processing_times:
            return 0.0
        return (sum(self.processing_times) / len(self.processing_times)) * 1000
    
    def calibrate(self):
        """Calibrate all tracking systems"""
        logger.info("🎯 Starting calibration...")
        self.calibration_frames = []
        self.calibrated = False
        
        if self.body_tracker:
            self.body_tracker.reset()
        if self.iris_tracker:
            self.iris_tracker.calibrate()
        if self.gesture_engine:
            self.gesture_engine.reset()
        
        self.calibrated = True
        logger.info("✅ Calibration complete")
    
    def reset(self):
        """Reset all tracking systems"""
        logger.info("🔄 Resetting tracking systems...")
        
        self.frame_count = 0
        self.start_time = time.time()
        self.processing_times = []
        self.confidence_history = []
        self.last_valid_data = None
        
        if self.body_tracker:
            self.body_tracker.reset()
        if self.iris_tracker:
            self.iris_tracker.reset()
        if self.gesture_engine:
            self.gesture_engine.reset()
        if self.object_detector:
            self.object_detector.reset()
        
        logger.info("✅ Reset complete")
    
    def cleanup(self):
        """Cleanup resources"""
        logger.info("🧹 Cleaning up coordinator...")
        
        if self.body_tracker:
            self.body_tracker.cleanup()
        if self.iris_tracker:
            self.iris_tracker.cleanup()
        if self.gesture_engine:
            self.gesture_engine.cleanup()
        if self.object_detector:
            self.object_detector.cleanup()
        
        logger.info("✅ Cleanup complete")
    
    def get_stats(self) -> Dict[str, Any]:
        """Get performance statistics"""
        return {
            "fps": self._get_fps(),
            "frame_count": self.frame_count,
            "avg_processing_time_ms": self._get_avg_processing_time(),
            "calibrated": self.calibrated,
            "active_modules": {
                "body": self.enable_body,
                "iris": self.enable_iris,
                "gestures": self.enable_gestures,
                "objects": self.enable_objects
            }
        }