"""
JARVIS Body Detection System - Camera Layer
Handles webcam capture with frame buffering and preprocessing
"""

import cv2
import threading
import queue
import time
from typing import Optional, Tuple, Callable
import numpy as np


class CameraStream:
    """Thread-safe camera stream with frame buffering"""
    
    def __init__(self, 
                 camera_id: int = 0,
                 width: int = 1280,
                 height: int = 720,
                 fps: int = 30,
                 buffer_size: int = 2):
        """
        Initialize camera stream
        
        Args:
            camera_id: Camera device ID (0 for default)
            width: Frame width
            height: Frame height
            fps: Target frames per second
            buffer_size: Frame buffer size (smaller = lower latency)
        """
        self.camera_id = camera_id
        self.width = width
        self.height = height
        self.fps = fps
        
        self.cap: Optional[cv2.VideoCapture] = None
        self.frame_queue = queue.Queue(maxsize=buffer_size)
        self.running = False
        self.thread: Optional[threading.Thread] = None
        
        self.frame_count = 0
        self.dropped_frames = 0
        self.last_fps_check = time.time()
        self.actual_fps = 0.0
        
    def start(self) -> bool:
        """Start camera capture thread"""
        if self.running:
            return True
            
        self.cap = cv2.VideoCapture(self.camera_id)
        if not self.cap.isOpened():
            print(f"[Camera] Failed to open camera {self.camera_id}")
            return False
            
        # Configure camera
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.width)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.height)
        self.cap.set(cv2.CAP_PROP_FPS, self.fps)
        self.cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)  # Minimize latency
        
        # Get actual camera settings
        actual_w = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        actual_h = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        actual_fps = int(self.cap.get(cv2.CAP_PROP_FPS))
        
        print(f"[Camera] Opened camera {self.camera_id}: {actual_w}x{actual_h} @ {actual_fps}fps")
        
        self.running = True
        self.thread = threading.Thread(target=self._capture_loop, daemon=True)
        self.thread.start()
        
        return True
        
    def stop(self):
        """Stop camera capture thread"""
        self.running = False
        if self.thread:
            self.thread.join(timeout=2.0)
        if self.cap:
            self.cap.release()
            self.cap = None
        print(f"[Camera] Stopped. Dropped {self.dropped_frames} frames")
        
    def _capture_loop(self):
        """Main capture loop running in background thread"""
        while self.running and self.cap:
            ret, frame = self.cap.read()
            if not ret:
                print("[Camera] Failed to read frame")
                time.sleep(0.01)
                continue
                
            self.frame_count += 1
            
            # Update FPS counter
            now = time.time()
            if now - self.last_fps_check >= 1.0:
                self.actual_fps = self.frame_count / (now - self.last_fps_check)
                self.frame_count = 0
                self.last_fps_check = now
            
            # Try to add frame to queue (non-blocking)
            try:
                self.frame_queue.put_nowait(frame)
            except queue.Full:
                self.dropped_frames += 1
                # Remove old frame and add new one
                try:
                    self.frame_queue.get_nowait()
                    self.frame_queue.put_nowait(frame)
                except:
                    pass
                    
    def read(self) -> Optional[np.ndarray]:
        """
        Read latest frame from camera
        
        Returns:
            Frame as numpy array (BGR) or None if not available
        """
        try:
            return self.frame_queue.get(timeout=0.1)
        except queue.Empty:
            return None
            
    def get_fps(self) -> float:
        """Get actual FPS"""
        return self.actual_fps
        
    def preprocess_frame(self, 
                        frame: np.ndarray,
                        target_size: Optional[Tuple[int, int]] = None,
                        normalize: bool = False) -> np.ndarray:
        """
        Preprocess frame for model input
        
        Args:
            frame: Input frame (BGR)
            target_size: Resize to (width, height), None to keep original
            normalize: Normalize to [0, 1] range
            
        Returns:
            Preprocessed frame
        """
        if frame is None:
            return None
            
        # Resize if needed
        if target_size:
            frame = cv2.resize(frame, target_size, interpolation=cv2.INTER_LINEAR)
            
        # Normalize if requested
        if normalize:
            frame = frame.astype(np.float32) / 255.0
            
        return frame
        
    def flip_frame(self, frame: np.ndarray, horizontal: bool = True) -> np.ndarray:
        """Flip frame (useful for mirror effect)"""
        if horizontal:
            return cv2.flip(frame, 1)
        return frame


class MultiCameraStream:
    """Manage multiple camera streams"""
    
    def __init__(self):
        self.cameras = {}
        
    def add_camera(self, name: str, camera_id: int = 0, **kwargs) -> CameraStream:
        """Add and start a camera stream"""
        if name in self.cameras:
            return self.cameras[name]
            
        cam = CameraStream(camera_id=camera_id, **kwargs)
        if cam.start():
            self.cameras[name] = cam
            return cam
        return None
        
    def get_camera(self, name: str) -> Optional[CameraStream]:
        """Get camera stream by name"""
        return self.cameras.get(name)
        
    def stop_all(self):
        """Stop all camera streams"""
        for cam in self.cameras.values():
            cam.stop()
        self.cameras.clear()
        
    def __del__(self):
        self.stop_all()