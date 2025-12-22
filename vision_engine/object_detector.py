"""
NETRAX AI - Object Detector
Real-time object detection using YOLOv8 with tracking
"""

import numpy as np
from typing import Dict, Any, List
import logging
from config import settings

logger = logging.getLogger("NETRAX.ObjectDetector")

class ObjectDetector:
    """
    Real-time object detection and tracking
    Uses YOLOv8 for high-performance detection
    """
    
    def __init__(self):
        logger.info("🎯 Initializing Object Detector...")
        
        try:
            from ultralytics import YOLO
            
            # Load YOLO model
            self.model = YOLO(settings.YOLO_MODEL)
            
            # Move to device
            if settings.USE_GPU:
                try:
                    self.model.to(settings.YOLO_DEVICE)
                    logger.info(f"✅ Using GPU: {settings.YOLO_DEVICE}")
                except Exception as e:
                    logger.warning(f"GPU not available, using CPU: {e}")
                    self.model.to("cpu")
            else:
                self.model.to("cpu")
            
            # Tracking state
            self.tracked_objects = {}
            self.next_track_id = 0
            
            logger.info(f"✅ Object Detector initialized with {settings.YOLO_MODEL}")
            
        except ImportError:
            logger.error("❌ ultralytics not installed. Object detection disabled.")
            logger.info("Install with: pip install ultralytics")
            self.model = None
        except Exception as e:
            logger.error(f"Failed to initialize object detector: {e}")
            self.model = None
    
    def process(self, frame: np.ndarray) -> Dict[str, Any]:
        """
        Detect objects in frame
        
        Args:
            frame: BGR image
            
        Returns:
            Detection results with bounding boxes and labels
        """
        if self.model is None:
            return {
                "detected": False,
                "detections": [],
                "confidence": 0.0
            }
        
        try:
            # Run inference
            results = self.model(
                frame,
                conf=settings.YOLO_CONFIDENCE,
                iou=settings.YOLO_IOU_THRESHOLD,
                max_det=settings.YOLO_MAX_DETECTIONS,
                verbose=False
            )
            
            # Parse results
            detections = []
            
            for result in results:
                boxes = result.boxes
                
                for box in boxes:
                    # Extract box data
                    x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                    confidence = float(box.conf[0].cpu().numpy())
                    class_id = int(box.cls[0].cpu().numpy())
                    label = self.model.names[class_id]
                    
                    detection = {
                        "bbox": {
                            "x1": float(x1),
                            "y1": float(y1),
                            "x2": float(x2),
                            "y2": float(y2)
                        },
                        "label": label,
                        "confidence": confidence,
                        "class_id": class_id
                    }
                    
                    detections.append(detection)
            
            # Calculate average confidence
            avg_confidence = np.mean([d["confidence"] for d in detections]) if detections else 0.0
            
            return {
                "detected": len(detections) > 0,
                "detections": detections,
                "count": len(detections),
                "confidence": float(avg_confidence)
            }
            
        except Exception as e:
            logger.error(f"Object detection error: {e}")
            return {
                "detected": False,
                "detections": [],
                "confidence": 0.0,
                "error": str(e)
            }
    
    def reset(self):
        """Reset tracking state"""
        logger.info("🔄 Resetting object detector...")
        self.tracked_objects = {}
        self.next_track_id = 0
    
    def cleanup(self):
        """Cleanup resources"""
        logger.info("🧹 Cleaning up object detector...")
        self.model = None