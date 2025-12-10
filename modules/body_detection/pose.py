import cv2
import numpy as np
from typing import Dict, List, Tuple, Optional, Any
from dataclasses import dataclass, field
from enum import Enum
import time


class PoseFramework(Enum):
    
    MEDIAPIPE = "mediapipe"
    MOVENET = "movenet"
    OPENPOSE = "openpose"


@dataclass
class Keypoint:
   
    x: float  # Normalized 0-1
    y: float  # Normalized 0-1
    z: float = 0.0  # Depth (if available)
    confidence: float = 1.0
    visibility: float = 1.0  # MediaPipe specific
    
    def to_pixel(self, width: int, height: int) -> Tuple[int, int]:
       
        return (int(self.x * width), int(self.y * height))


@dataclass
class PoseResult:
    
    pose_landmarks: Dict[str, Keypoint] = field(default_factory=dict)
    left_hand_landmarks: Dict[str, Keypoint] = field(default_factory=dict)
    right_hand_landmarks: Dict[str, Keypoint] = field(default_factory=dict)
    face_landmarks: Dict[str, Keypoint] = field(default_factory=dict)
    
    timestamp: float = field(default_factory=time.time)
    confidence: float = 1.0
    framework: str = "unknown"
    
    def has_pose(self) -> bool:
        
        return len(self.pose_landmarks) > 0
        
    def has_hands(self) -> bool:
       
        return len(self.left_hand_landmarks) > 0 or len(self.right_hand_landmarks) > 0


class BasePoseDetector:
    
    
    def __init__(self, **kwargs):
        self.initialized = False
        
    def detect(self, frame: np.ndarray) -> PoseResult:
       
        raise NotImplementedError
        
    def close(self):
        
        pass


class MediaPipeDetector(BasePoseDetector):
    
    
    def __init__(self,
                 min_detection_confidence: float = 0.5,
                 min_tracking_confidence: float = 0.5,
                 model_complexity: int = 1,
                 enable_segmentation: bool = False,
                 smooth_landmarks: bool = True):
       
        super().__init__()
        
        try:
            import mediapipe as mp
            self.mp = mp
            self.mp_holistic = mp.solutions.holistic
            self.mp_drawing = mp.solutions.drawing_utils
            self.mp_drawing_styles = mp.solutions.drawing_styles
            
            self.holistic = self.mp_holistic.Holistic(
                min_detection_confidence=min_detection_confidence,
                min_tracking_confidence=min_tracking_confidence,
                model_complexity=model_complexity,
                enable_segmentation=enable_segmentation,
                smooth_landmarks=smooth_landmarks
            )
            
            self.initialized = True
            print(f"[Pose] MediaPipe Holistic initialized (complexity={model_complexity})")
            
        except ImportError:
            print("[Pose] MediaPipe not installed. Install: pip install mediapipe")
            self.initialized = False
            
    def detect(self, frame: np.ndarray) -> PoseResult:
        
        if not self.initialized:
            return PoseResult()
            
        # Convert BGR to RGB
        image_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        image_rgb.flags.writeable = False
        
        # Process
        results = self.holistic.process(image_rgb)
        
        # Convert to PoseResult
        pose_result = PoseResult(framework="mediapipe")
        
        # Extract pose landmarks
        if results.pose_landmarks:
            pose_result.pose_landmarks = self._extract_landmarks(
                results.pose_landmarks,
                self.mp_holistic.PoseLandmark
            )
            
        # Extract hand landmarks
        if results.left_hand_landmarks:
            pose_result.left_hand_landmarks = self._extract_landmarks(
                results.left_hand_landmarks,
                self.mp.solutions.hands.HandLandmark
            )
            
        if results.right_hand_landmarks:
            pose_result.right_hand_landmarks = self._extract_landmarks(
                results.right_hand_landmarks,
                self.mp.solutions.hands.HandLandmark
            )
            
        # Extract face landmarks
        if results.face_landmarks:
            # Only keep key face points (eyes, nose, mouth corners)
            key_indices = [0, 4, 13, 14, 33, 133, 152, 263, 362, 468]
            all_landmarks = self._extract_landmarks(
                results.face_landmarks,
                range(468)
            )
            pose_result.face_landmarks = {
                f"face_{i}": all_landmarks.get(str(i))
                for i in key_indices
                if str(i) in all_landmarks
            }
            
        return pose_result
        
    def _extract_landmarks(self, 
                          landmarks,
                          landmark_enum) -> Dict[str, Keypoint]:
        
        result = {}
        
        if hasattr(landmark_enum, '__members__'):
            # Enum with names
            for name, idx in landmark_enum.__members__.items():
                if hasattr(idx, 'value'):
                    idx = idx.value
                if idx < len(landmarks.landmark):
                    lm = landmarks.landmark[idx]
                    result[name.lower()] = Keypoint(
                        x=lm.x,
                        y=lm.y,
                        z=lm.z if hasattr(lm, 'z') else 0.0,
                        visibility=lm.visibility if hasattr(lm, 'visibility') else 1.0
                    )
        else:
            # Numeric indices
            for idx in landmark_enum:
                if idx < len(landmarks.landmark):
                    lm = landmarks.landmark[idx]
                    result[str(idx)] = Keypoint(
                        x=lm.x,
                        y=lm.y,
                        z=lm.z if hasattr(lm, 'z') else 0.0,
                        visibility=lm.visibility if hasattr(lm, 'visibility') else 1.0
                    )
                    
        return result
        
    def draw_landmarks(self, 
                      frame: np.ndarray,
                      pose_result: PoseResult) -> np.ndarray:
        
        if not self.initialized:
            return frame
            
       
        annotated = frame.copy()
        
        # Draw pose
        if pose_result.pose_landmarks:
            self._draw_pose_connections(annotated, pose_result.pose_landmarks, frame.shape)
            
        # Draw hands
        if pose_result.left_hand_landmarks:
            self._draw_hand_connections(annotated, pose_result.left_hand_landmarks, frame.shape)
            
        if pose_result.right_hand_landmarks:
            self._draw_hand_connections(annotated, pose_result.right_hand_landmarks, frame.shape)
            
        return annotated
        
    def _draw_pose_connections(self, frame, landmarks, shape):
       
        h, w = shape[:2]
        
        # Define connections
        connections = [
            ('left_shoulder', 'right_shoulder'),
            ('left_shoulder', 'left_elbow'),
            ('left_elbow', 'left_wrist'),
            ('right_shoulder', 'right_elbow'),
            ('right_elbow', 'right_wrist'),
            ('left_shoulder', 'left_hip'),
            ('right_shoulder', 'right_hip'),
            ('left_hip', 'right_hip'),
            ('left_hip', 'left_knee'),
            ('left_knee', 'left_ankle'),
            ('right_hip', 'right_knee'),
            ('right_knee', 'right_ankle'),
        ]
        
        for start, end in connections:
            if start in landmarks and end in landmarks:
                pt1 = landmarks[start].to_pixel(w, h)
                pt2 = landmarks[end].to_pixel(w, h)
                cv2.line(frame, pt1, pt2, (0, 255, 0), 2)
                
        # Draw keypoints
        for kp in landmarks.values():
            pt = kp.to_pixel(w, h)
            cv2.circle(frame, pt, 4, (0, 0, 255), -1)
            
    def _draw_hand_connections(self, frame, landmarks, shape):
        
        h, w = shape[:2]
        
        # Draw all hand keypoints
        for kp in landmarks.values():
            pt = kp.to_pixel(w, h)
            cv2.circle(frame, pt, 3, (255, 0, 0), -1)
            
    def close(self):
        
        if self.initialized and hasattr(self, 'holistic'):
            self.holistic.close()


class MoveNetDetector(BasePoseDetector):
  
    
    def __init__(self, model_type: str = "thunder"):
       
        super().__init__()
        print(f"[Pose] MoveNet not yet implemented (placeholder)")
        # TODO: Implement MoveNet
        # https://www.tensorflow.org/hub/tutorials/movenet
        
    def detect(self, frame: np.ndarray) -> PoseResult:
        return PoseResult(framework="movenet")


class PoseDetectorFactory:
   
    
    @staticmethod
    def create(framework: PoseFramework, **kwargs) -> BasePoseDetector:
        
        if framework == PoseFramework.MEDIAPIPE:
            return MediaPipeDetector(**kwargs)
        elif framework == PoseFramework.MOVENET:
            return MoveNetDetector(**kwargs)
        elif framework == PoseFramework.OPENPOSE:
            raise NotImplementedError("OpenPose not yet implemented")
        else:
            raise ValueError(f"Unknown framework: {framework}")