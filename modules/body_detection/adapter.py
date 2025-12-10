import json
import threading
import queue
from typing import Dict, Callable, Optional, Any, List
from dataclasses import dataclass, asdict
from pathlib import Path

from gesture import Gesture, GestureType


@dataclass
class JARVISCommand:
   
    action: str
    parameters: Dict[str, Any]
    source: str = "body_detection"
    priority: int = 1
    timestamp: float = 0.0
    
    def to_dict(self) -> dict:
        return asdict(self)
        
    def to_json(self) -> str:
        return json.dumps(self.to_dict())


class GestureCommandMapper:
    
    
    DEFAULT_MAPPINGS = {
        # Hand gestures
        GestureType.PEACE: {
            "action": "screenshot",
            "parameters": {}
        },
        GestureType.STOP: {
            "action": "pause_media",
            "parameters": {}
        },
        GestureType.THUMBS_UP: {
            "action": "volume_up",
            "parameters": {"amount": 10}
        },
        GestureType.THUMBS_DOWN: {
            "action": "volume_down",
            "parameters": {"amount": 10}
        },
        GestureType.FIST: {
            "action": "mute",
            "parameters": {}
        },
        GestureType.POINT: {
            "action": "select",
            "parameters": {}
        },
        
        
        # Swipe
        GestureType.SWIPE_LEFT: {
            "action": "previous_track",
            "parameters": {}
        },
        GestureType.SWIPE_RIGHT: {
            "action": "next_track",
            "parameters": {}
        },
        GestureType.SWIPE_UP: {
            "action": "scroll_up",
            "parameters": {"amount": 3}
        },
        GestureType.SWIPE_DOWN: {
            "action": "scroll_down",
            "parameters": {"amount": 3}
        },
        
        
        # Body-poses
        GestureType.PAUSE: {
            "action": "pause_detection",
            "parameters": {"duration": 5.0}
        },
        GestureType.ARMS_CROSSED: {
            "action": "lock_screen",
            "parameters": {}
        },
        GestureType.LEANING_LEFT: {
            "action": "switch_workspace",
            "parameters": {"direction": "left"}
        },
        GestureType.LEANING_RIGHT: {
            "action": "switch_workspace",
            "parameters": {"direction": "right"}
        },
        
        
        #Zoom
        GestureType.ZOOM_IN: {
            "action": "zoom_in",
            "parameters": {}
        },
        GestureType.ZOOM_OUT: {
            "action": "zoom_out",
            "parameters": {}
        },
    }
    
    def __init__(self, config_path: Optional[Path] = None):
     
        self.mappings = self.DEFAULT_MAPPINGS.copy()
        
        # Load custom mappings if provided
        if config_path and config_path.exists():
            self.load_mappings(config_path)
            
    def map(self, gesture: Gesture) -> Optional[JARVISCommand]:
        
        if gesture.type not in self.mappings:
            return None
            
        mapping = self.mappings[gesture.type]
        
        # Merge gesture parameters with default parameters
        parameters = mapping["parameters"].copy()
        if gesture.parameters:
            parameters.update(gesture.parameters)
            
        # Add gesture metadata
        parameters["gesture_confidence"] = gesture.confidence
        parameters["gesture_hand"] = gesture.hand
        parameters["gesture_duration"] = gesture.duration
        
        return JARVISCommand(
            action=mapping["action"],
            parameters=parameters,
            timestamp=gesture.timestamp
        )
        
    def load_mappings(self, config_path: Path):
        
        try:
            with open(config_path, 'r') as f:
                custom = json.load(f)
                
            # Update mappings
            for gesture_name, mapping in custom.items():
                try:
                    gesture_type = GestureType[gesture_name.upper()]
                    self.mappings[gesture_type] = mapping
                except KeyError:
                    print(f"[Mapper] Unknown gesture type: {gesture_name}")
                    
            print(f"[Mapper] Loaded custom mappings from {config_path}")
            
        except Exception as e:
            print(f"[Mapper] Error loading mappings: {e}")
            
    def save_mappings(self, config_path: Path):
        
        try:
            # Convert to serializable format
            serializable = {
                gesture_type.name.lower(): mapping
                for gesture_type, mapping in self.mappings.items()
            }
            
            with open(config_path, 'w') as f:
                json.dump(serializable, f, indent=2)
                
            print(f"[Mapper] Saved mappings to {config_path}")
            
        except Exception as e:
            print(f"[Mapper] Error saving mappings: {e}")
            
    def add_mapping(self, 
                   gesture_type: GestureType,
                   action: str,
                   parameters: Dict[str, Any]):
        
        self.mappings[gesture_type] = {
            "action": action,
            "parameters": parameters
        }
        
    def remove_mapping(self, gesture_type: GestureType):
        """Remove a gesture mapping"""
        if gesture_type in self.mappings:
            del self.mappings[gesture_type]


class JARVISAdapter:
    
    
    def __init__(self,
                 mapper: Optional[GestureCommandMapper] = None,
                 mode: str = "callback"):
        
        self.mapper = mapper or GestureCommandMapper()
        self.mode = mode
        
        
        self.command_callbacks: List[Callable[[JARVISCommand], None]] = []
        
        
        self.command_queue = queue.Queue()
        
        
        self.event_bus = None
        
        
        self.gestures_processed = 0
        self.commands_sent = 0
        
        print(f"[Adapter] Initialized in {mode} mode")
        
    def process_gesture(self, gesture: Gesture):
       
        self.gestures_processed += 1
        
        
        command = self.mapper.map(gesture)
        if not command:
            return
            
        
        self._send_command(command)
        self.commands_sent += 1
        
    def process_gestures(self, gestures: List[Gesture]):
        
        for gesture in gestures:
            self.process_gesture(gesture)
            
    def _send_command(self, command: JARVISCommand):
        
        if self.mode == "callback":
            self._send_via_callbacks(command)
        elif self.mode == "queue":
            self._send_via_queue(command)
        elif self.mode == "event_bus":
            self._send_via_event_bus(command)
        else:
            print(f"[Adapter] Unknown mode: {self.mode}")
            
    def _send_via_callbacks(self, command: JARVISCommand):
        
        for callback in self.command_callbacks:
            try:
                callback(command)
            except Exception as e:
                print(f"[Adapter] Callback error: {e}")
                
    def _send_via_queue(self, command: JARVISCommand):
        
        self.command_queue.put_nowait(command)
        
    def _send_via_event_bus(self, command: JARVISCommand):
       
        if self.event_bus:
            self.event_bus.emit("body_detection_command", command.to_dict())
        else:
            print("[Adapter] Event bus not configured")
            
    
    def register_callback(self, callback: Callable[[JARVISCommand], None]):
      
        self.command_callbacks.append(callback)
        print(f"[Adapter] Registered callback: {callback.__name__}")
        
    def unregister_callback(self, callback: Callable[[JARVISCommand], None]):
        
        if callback in self.command_callbacks:
            self.command_callbacks.remove(callback)
            
    
    def get_command(self, timeout: Optional[float] = None) -> Optional[JARVISCommand]:
        
        try:
            return self.command_queue.get(timeout=timeout)
        except queue.Empty:
            return None
            
    def has_commands(self) -> bool:
        
        return not self.command_queue.empty()
        
    
    def set_event_bus(self, event_bus):
        
        self.event_bus = event_bus
        self.mode = "event_bus"
        print("[Adapter] Event bus configured")
        
    
    def get_stats(self) -> Dict[str, int]:
        
        return {
            "gestures_processed": self.gestures_processed,
            "commands_sent": self.commands_sent,
            "queue_size": self.command_queue.qsize() if self.mode == "queue" else 0
        }
        
    def reset_stats(self):
       
        self.gestures_processed = 0
        self.commands_sent = 0



def example_jarvis_callback(command: JARVISCommand):
   
    print(f"[JARVIS] Executing: {command.action} with {command.parameters}")
    
    
    if command.action == "volume_up":
        
        print(f"  → Volume up by {command.parameters['amount']}")
        
    elif command.action == "screenshot":
        
        print(f"  → Taking screenshot")
        
    elif command.action == "scroll_up":
        
        print(f"  → Scrolling up by {command.parameters['amount']}")
        
    