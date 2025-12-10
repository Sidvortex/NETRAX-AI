"""
JARVIS Body Detection Module

Real-time gesture control system for JARVIS using computer vision.
"""

from .body_detection import BodyDetectionSystem, BodyDetectionConfig
from .adapter import JARVISAdapter, GestureCommandMapper, JARVISCommand
from .gesture import GestureType, Gesture, GestureRecognizer
from .pose import PoseFramework, PoseResult, Keypoint
from .camera import CameraStream
from .tracking import PoseTracker, KeypointTracker

__version__ = "1.0.0"
__author__ = "JARVIS Team"

__all__ = [
    # Main system
    'BodyDetectionSystem',
    'BodyDetectionConfig',
    
    # Integration
    'JARVISAdapter',
    'GestureCommandMapper',
    'JARVISCommand',
    
    # Gestures
    'GestureType',
    'Gesture',
    'GestureRecognizer',
    
    # Pose detection
    'PoseFramework',
    'PoseResult',
    'Keypoint',
    
    # Components
    'CameraStream',
    'PoseTracker',
    'KeypointTracker',
]