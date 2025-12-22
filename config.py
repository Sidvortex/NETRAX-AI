"""
NETRAX AI - System Configuration
Environment-based configuration management
"""

from pydantic_settings import BaseSettings
from typing import Optional
import os

class Settings(BaseSettings):
    """Application settings"""
    
    # Server Configuration
    HOST: str = "0.0.0.0"
    PORT: int = 8000
    DEBUG: bool = False
    
    # Camera Settings
    CAMERA_INDEX: int = 0
    FRAME_WIDTH: int = 1280
    FRAME_HEIGHT: int = 720
    TARGET_FPS: int = 30
    JPEG_QUALITY: int = 85
    
    # Vision Module Toggles
    ENABLE_BODY_TRACKING: bool = True
    ENABLE_IRIS_TRACKING: bool = True
    ENABLE_GESTURE_RECOGNITION: bool = True
    ENABLE_OBJECT_DETECTION: bool = False  # Disabled by default (heavy)
    
    # Body Tracking Settings
    BODY_MODEL_COMPLEXITY: int = 1  # 0=Lite, 1=Full, 2=Heavy
    BODY_MIN_DETECTION_CONFIDENCE: float = 0.5
    BODY_MIN_TRACKING_CONFIDENCE: float = 0.5
    BODY_SMOOTH_LANDMARKS: bool = True
    
    # Iris Tracking Settings
    IRIS_MIN_DETECTION_CONFIDENCE: float = 0.5
    IRIS_MIN_TRACKING_CONFIDENCE: float = 0.5
    IRIS_ENABLE_REFINEMENT: bool = True
    IRIS_SMOOTH_LANDMARKS: bool = True
    IRIS_REFINE_LANDMARKS: bool = True
    
    # Gesture Recognition Settings
    GESTURE_CONFIDENCE_THRESHOLD: float = 0.75
    GESTURE_COOLDOWN_FRAMES: int = 20
    GESTURE_TEMPORAL_SMOOTHING: int = 5
    GESTURE_BUFFER_SIZE: int = 10
    
    # Object Detection Settings
    YOLO_MODEL: str = "yolov8n.pt"  # n=nano, s=small, m=medium
    YOLO_CONFIDENCE: float = 0.5
    YOLO_IOU_THRESHOLD: float = 0.45
    YOLO_MAX_DETECTIONS: int = 100
    YOLO_DEVICE: str = "cuda"  # cuda or cpu
    
    # Performance Settings
    USE_GPU: bool = True
    GPU_DEVICE_ID: int = 0
    ENABLE_FRAME_SKIP: bool = False
    FRAME_SKIP_INTERVAL: int = 2
    MAX_PROCESSING_TIME: float = 0.033  # 30ms max processing
    
    # Kalman Filter Settings
    ENABLE_KALMAN_FILTER: bool = True
    KALMAN_PROCESS_NOISE: float = 0.01
    KALMAN_MEASUREMENT_NOISE: float = 0.1
    
    # Data Streaming Settings
    SEND_DETAILED_DATA: bool = False
    SEND_RAW_LANDMARKS: bool = False
    COMPRESS_DATA: bool = True
    
    # Cyberpunk Visual Settings
    ENABLE_CYBERPUNK_OVERLAY: bool = True
    OVERLAY_COLOR_PRIMARY: tuple = (0, 0, 255)  # Red (BGR)
    OVERLAY_COLOR_SECONDARY: tuple = (255, 0, 255)  # Cyan (BGR)
    OVERLAY_THICKNESS: int = 2
    OVERLAY_ALPHA: float = 0.7
    
    # Glow effect settings
    ENABLE_GLOW_EFFECT: bool = True
    GLOW_KERNEL_SIZE: int = 15
    GLOW_INTENSITY: float = 0.3
    
    # Scanline effect
    ENABLE_SCANLINES: bool = False
    SCANLINE_INTENSITY: float = 0.1
    SCANLINE_SPACING: int = 3
    
    # Logging
    LOG_LEVEL: str = "INFO"
    LOG_GESTURES: bool = True
    LOG_PERFORMANCE: bool = True
    
    # Model Paths
    MODEL_DIR: str = "models"
    GESTURE_MODEL_PATH: Optional[str] = None
    
    # Calibration
    AUTO_CALIBRATE_ON_START: bool = True
    CALIBRATION_FRAMES: int = 30
    
    class Config:
        env_file = ".env"
        env_file_encoding = "utf-8"
        case_sensitive = True

# Global settings instance
settings = Settings()

# GPU device configuration
def get_device():
    """Get optimal processing device"""
    if settings.USE_GPU:
        try:
            import torch
            if torch.cuda.is_available():
                return f"cuda:{settings.GPU_DEVICE_ID}"
        except ImportError:
            pass
    return "cpu"

# Validate settings on import
def validate_settings():
    """Validate configuration"""
    errors = []
    
    if settings.TARGET_FPS < 1 or settings.TARGET_FPS > 120:
        errors.append("TARGET_FPS must be between 1 and 120")
    
    if settings.BODY_MODEL_COMPLEXITY not in [0, 1, 2]:
        errors.append("BODY_MODEL_COMPLEXITY must be 0, 1, or 2")
    
    if not 0 <= settings.GESTURE_CONFIDENCE_THRESHOLD <= 1:
        errors.append("GESTURE_CONFIDENCE_THRESHOLD must be between 0 and 1")
    
    if errors:
        raise ValueError(f"Configuration errors: {', '.join(errors)}")

# Run validation
validate_settings()

# Export commonly used settings
__all__ = ['settings', 'get_device', 'Settings']