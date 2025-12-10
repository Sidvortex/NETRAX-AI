"""
JARVIS Body Detection System - Main Manager
Orchestrates camera, pose detection, gesture recognition, and JARVIS integration
"""

import cv2
import threading
import time
from typing import Optional, Callable, Dict, Any
from pathlib import Path
import sys

from camera import CameraStream
from pose import PoseDetectorFactory, PoseFramework, PoseResult, BasePoseDetector
from gesture import GestureRecognizer, Gesture
from tracking import PoseTracker
from adapter import JARVISAdapter, GestureCommandMapper, JARVISCommand


class BodyDetectionConfig:
    """Configuration for body detection system"""
    
    def __init__(self):
        # Camera settings
        self.camera_id = 0
        self.camera_width = 1280
        self.camera_height = 720
        self.camera_fps = 30
        self.mirror_camera = True
        
        # Pose detection
        self.pose_framework = PoseFramework.MEDIAPIPE
        self.pose_confidence = 0.5
        self.pose_tracking_confidence = 0.5
        self.pose_model_complexity = 1
        
        # Gesture recognition
        self.gesture_min_confidence = 0.6
        self.gesture_hold_time = 0.3
        self.swipe_threshold = 0.15
        
        # Tracking/smoothing
        self.enable_smoothing = True
        self.smoothing_type = "one_euro"  # "one_euro", "moving_average", "exponential"
        
        # Integration
        self.integration_mode = "callback"  # "callback", "queue", "event_bus"
        self.gesture_mapping_file = None
        
        # Display
        self.show_visualization = True
        self.show_fps = True
        self.visualization_scale = 0.5
        
        # Performance
        self.target_fps = 30
        self.skip_frames = 0  # Process every Nth frame (0 = process all)
        
    @classmethod
    def from_file(cls, config_path: Path) -> 'BodyDetectionConfig':
        """Load configuration from JSON file"""
        import json
        config = cls()
        
        try:
            with open(config_path, 'r') as f:
                data = json.load(f)
                
            # Update config from dict
            for key, value in data.items():
                if hasattr(config, key):
                    # Handle enums
                    if key == "pose_framework":
                        value = PoseFramework[value.upper()]
                    setattr(config, key, value)
                    
            print(f"[Config] Loaded from {config_path}")
            
        except Exception as e:
            print(f"[Config] Error loading: {e}, using defaults")
            
        return config
        
    def save(self, config_path: Path):
        """Save configuration to JSON file"""
        import json
        
        data = {}
        for key, value in self.__dict__.items():
            # Handle enums
            if isinstance(value, PoseFramework):
                value = value.value
            elif isinstance(value, Path):
                value = str(value)
            data[key] = value
            
        with open(config_path, 'w') as f:
            json.dump(data, f, indent=2)
            
        print(f"[Config] Saved to {config_path}")


class BodyDetectionSystem:
    """Main body detection system manager"""
    
    def __init__(self, config: Optional[BodyDetectionConfig] = None):
        """
        Initialize body detection system
        
        Args:
            config: System configuration
        """
        self.config = config or BodyDetectionConfig()
        
        # Components
        self.camera: Optional[CameraStream] = None
        self.pose_detector: Optional[BasePoseDetector] = None
        self.gesture_recognizer: Optional[GestureRecognizer] = None
        self.pose_tracker: Optional[PoseTracker] = None
        self.adapter: Optional[JARVISAdapter] = None
        
        # State
        self.running = False
        self.paused = False
        self.thread: Optional[threading.Thread] = None
        
        # Statistics
        self.frame_count = 0
        self.detection_count = 0
        self.gesture_count = 0
        self.start_time = 0.0
        self.fps = 0.0
        
        # Visualization
        self.current_frame = None
        self.current_pose = None
        self.current_gestures = []
        
        print("[BodyDetection] System initialized")
        
    def initialize(self) -> bool:
        """Initialize all components"""
        print("[BodyDetection] Initializing components...")
        
        try:
            # Initialize camera
            self.camera = CameraStream(
                camera_id=self.config.camera_id,
                width=self.config.camera_width,
                height=self.config.camera_height,
                fps=self.config.camera_fps
            )
            
            if not self.camera.start():
                print("[BodyDetection] Failed to start camera")
                return False
                
            # Initialize pose detector
            self.pose_detector = PoseDetectorFactory.create(
                self.config.pose_framework,
                min_detection_confidence=self.config.pose_confidence,
                min_tracking_confidence=self.config.pose_tracking_confidence,
                model_complexity=self.config.pose_model_complexity
            )
            
            if not self.pose_detector.initialized:
                print("[BodyDetection] Failed to initialize pose detector")
                return False
                
            # Initialize gesture recognizer
            self.gesture_recognizer = GestureRecognizer(
                min_confidence=self.config.gesture_min_confidence,
                gesture_hold_time=self.config.gesture_hold_time,
                swipe_threshold=self.config.swipe_threshold
            )
            
            # Initialize pose tracker
            if self.config.enable_smoothing:
                self.pose_tracker = PoseTracker(
                    smoothing_type=self.config.smoothing_type
                )
                
            # Initialize JARVIS adapter
            mapper = GestureCommandMapper(config_path=self.config.gesture_mapping_file)
            self.adapter = JARVISAdapter(
                mapper=mapper,
                mode=self.config.integration_mode
            )
            
            print("[BodyDetection] All components initialized successfully")
            return True
            
        except Exception as e:
            print(f"[BodyDetection] Initialization error: {e}")
            return False
            
    def start(self) -> bool:
        """Start detection system"""
        if self.running:
            print("[BodyDetection] Already running")
            return False
            
        if not self.initialize():
            return False
            
        self.running = True
        self.start_time = time.time()
        
        # Start processing thread
        self.thread = threading.Thread(target=self._detection_loop, daemon=True)
        self.thread.start()
        
        print("[BodyDetection] System started")
        return True
        
    def stop(self):
        """Stop detection system"""
        if not self.running:
            return
            
        print("[BodyDetection] Stopping...")
        self.running = False
        
        if self.thread:
            self.thread.join(timeout=2.0)
            
        # Cleanup components
        if self.camera:
            self.camera.stop()
            
        if self.pose_detector:
            self.pose_detector.close()
            
        # Print statistics
        elapsed = time.time() - self.start_time
        print(f"[BodyDetection] Stopped. Stats:")
        print(f"  Frames processed: {self.frame_count}")
        print(f"  Poses detected: {self.detection_count}")
        print(f"  Gestures recognized: {self.gesture_count}")
        print(f"  Average FPS: {self.frame_count / elapsed:.1f}")
        
    def pause(self):
        """Pause detection"""
        self.paused = True
        print("[BodyDetection] Paused")
        
    def resume(self):
        """Resume detection"""
        self.paused = False
        print("[BodyDetection] Resumed")
        
    def _detection_loop(self):
        """Main detection loop"""
        frame_skip_counter = 0
        last_fps_time = time.time()
        fps_frame_count = 0
        
        while self.running:
            if self.paused:
                time.sleep(0.1)
                continue
                
            # Read frame from camera
            frame = self.camera.read()
            if frame is None:
                continue
                
            # Mirror frame if configured
            if self.config.mirror_camera:
                frame = self.camera.flip_frame(frame)
                
            self.frame_count += 1
            fps_frame_count += 1
            
            # Skip frames if configured
            frame_skip_counter += 1
            if self.config.skip_frames > 0 and frame_skip_counter <= self.config.skip_frames:
                continue
            frame_skip_counter = 0
            
            # Detect pose
            pose_result = self.pose_detector.detect(frame)
            gestures = []
            
            if pose_result.has_pose():
                self.detection_count += 1
                
                # Apply tracking/smoothing
                if self.pose_tracker:
                    pose_result = self.pose_tracker.track(pose_result)
                    
                # Recognize gestures
                gestures = self.gesture_recognizer.recognize(pose_result)
                
                if gestures:
                    self.gesture_count += len(gestures)
                    
                    # Process gestures through adapter
                    self.adapter.process_gestures(gestures)
                    
                # Store for visualization
                self.current_pose = pose_result
                self.current_gestures = gestures
                
            # Update visualization
            if self.config.show_visualization:
                self.current_frame = self._create_visualization(frame, pose_result, gestures)
                
            # Calculate FPS
            now = time.time()
            if now - last_fps_time >= 1.0:
                self.fps = fps_frame_count / (now - last_fps_time)
                fps_frame_count = 0
                last_fps_time = now
                
            # Maintain target FPS
            time.sleep(max(0, 1.0 / self.config.target_fps - 0.001))
            
    def _create_visualization(self, 
                             frame, 
                             pose_result: PoseResult,
                             gestures: list) -> Any:
        """Create visualization frame"""
        vis_frame = frame.copy()
        
        # Draw pose landmarks
        if hasattr(self.pose_detector, 'draw_landmarks'):
            vis_frame = self.pose_detector.draw_landmarks(vis_frame, pose_result)
            
        # Draw gesture labels
        if gestures:
            y_offset = 30
            for gesture in gestures:
                label = f"{gesture.type.value}"
                if gesture.hand:
                    label += f" ({gesture.hand})"
                label += f" {gesture.confidence:.2f}"
                
                cv2.putText(vis_frame, label, (10, y_offset),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                y_offset += 30
                
        # Draw FPS
        if self.config.show_fps:
            fps_text = f"FPS: {self.fps:.1f}"
            cv2.putText(vis_frame, fps_text, (10, vis_frame.shape[0] - 10),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
                       
        # Scale if configured
        if self.config.visualization_scale != 1.0:
            new_width = int(vis_frame.shape[1] * self.config.visualization_scale)
            new_height = int(vis_frame.shape[0] * self.config.visualization_scale)
            vis_frame = cv2.resize(vis_frame, (new_width, new_height))
            
        return vis_frame
        
    def show_visualization_window(self):
        """Show visualization in OpenCV window (blocking)"""
        if not self.config.show_visualization:
            print("[BodyDetection] Visualization disabled in config")
            return
            
        print("[BodyDetection] Showing visualization window (press 'q' to quit, 'p' to pause)")
        
        while self.running:
            if self.current_frame is not None:
                cv2.imshow("JARVIS Body Detection", self.current_frame)
                
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                break
            elif key == ord('p'):
                self.paused = not self.paused
                
        cv2.destroyAllWindows()
        
    def register_command_callback(self, callback: Callable[[JARVISCommand], None]):
        """Register callback for JARVIS commands"""
        if self.adapter:
            self.adapter.register_callback(callback)
            
    def get_stats(self) -> Dict[str, Any]:
        """Get system statistics"""
        stats = {
            "running": self.running,
            "paused": self.paused,
            "frames_processed": self.frame_count,
            "poses_detected": self.detection_count,
            "gestures_recognized": self.gesture_count,
            "current_fps": self.fps,
            "camera_fps": self.camera.get_fps() if self.camera else 0.0
        }
        
        if self.adapter:
            stats.update(self.adapter.get_stats())
            
        return stats