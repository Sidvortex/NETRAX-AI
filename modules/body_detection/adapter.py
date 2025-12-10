"""
JARVIS Body Detection System - Integration Adapter
Converts gestures to JARVIS commands and integrates with existing system
"""

import json
import threading
import queue
from typing import Dict, Callable, Optional, Any, List
from dataclasses import dataclass, asdict
from pathlib import Path

from gesture import Gesture, GestureType


@dataclass
class JARVISCommand:
    """Command to be executed by JARVIS"""
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
    """Maps gestures to JARVIS commands"""
    
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
        
        # Swipe gestures
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
        
        # Body poses
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
        
        # Zoom gestures
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
        """
        Initialize mapper
        
        Args:
            config_path: Path to custom gesture mapping config (JSON)
        """
        self.mappings = self.DEFAULT_MAPPINGS.copy()
        
        # Load custom mappings if provided
        if config_path and config_path.exists():
            self.load_mappings(config_path)
            
    def map(self, gesture: Gesture) -> Optional[JARVISCommand]:
        """
        Map gesture to JARVIS command
        
        Args:
            gesture: Detected gesture
            
        Returns:
            JARVIS command or None if no mapping exists
        """
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
        """Load custom gesture mappings from JSON file"""
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
        """Save current mappings to JSON file"""
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
        """Add or update a gesture mapping"""
        self.mappings[gesture_type] = {
            "action": action,
            "parameters": parameters
        }
        
    def remove_mapping(self, gesture_type: GestureType):
        """Remove a gesture mapping"""
        if gesture_type in self.mappings:
            del self.mappings[gesture_type]


class JARVISAdapter:
    """
    Adapter for integrating body detection with JARVIS core
    Provides multiple integration methods: callbacks, queue, or event bus
    """
    
    def __init__(self,
                 mapper: Optional[GestureCommandMapper] = None,
                 mode: str = "callback"):
        """
        Initialize adapter
        
        Args:
            mapper: Gesture to command mapper
            mode: Integration mode - "callback", "queue", or "event_bus"
        """
        self.mapper = mapper or GestureCommandMapper()
        self.mode = mode
        
        # Callback handlers
        self.command_callbacks: List[Callable[[JARVISCommand], None]] = []
        
        # Command queue for queue mode
        self.command_queue = queue.Queue()
        
        # Event bus integration (if JARVIS has one)
        self.event_bus = None
        
        # Statistics
        self.gestures_processed = 0
        self.commands_sent = 0
        
        print(f"[Adapter] Initialized in {mode} mode")
        
    def process_gesture(self, gesture: Gesture):
        """
        Process a detected gesture and send to JARVIS
        
        Args:
            gesture: Detected gesture
        """
        self.gestures_processed += 1
        
        # Map to command
        command = self.mapper.map(gesture)
        if not command:
            return
            
        # Send command based on mode
        self._send_command(command)
        self.commands_sent += 1
        
    def process_gestures(self, gestures: List[Gesture]):
        """Process multiple gestures"""
        for gesture in gestures:
            self.process_gesture(gesture)
            
    def _send_command(self, command: JARVISCommand):
        """Send command to JARVIS based on integration mode"""
        if self.mode == "callback":
            self._send_via_callbacks(command)
        elif self.mode == "queue":
            self._send_via_queue(command)
        elif self.mode == "event_bus":
            self._send_via_event_bus(command)
        else:
            print(f"[Adapter] Unknown mode: {self.mode}")
            
    def _send_via_callbacks(self, command: JARVISCommand):
        """Send command via registered callbacks"""
        for callback in self.command_callbacks:
            try:
                callback(command)
            except Exception as e:
                print(f"[Adapter] Callback error: {e}")
                
    def _send_via_queue(self, command: JARVISCommand):
        """Send command via queue"""
        self.command_queue.put_nowait(command)
        
    def _send_via_event_bus(self, command: JARVISCommand):
        """Send command via event bus"""
        if self.event_bus:
            self.event_bus.emit("body_detection_command", command.to_dict())
        else:
            print("[Adapter] Event bus not configured")
            
    # Callback registration
    def register_callback(self, callback: Callable[[JARVISCommand], None]):
        """Register a callback for commands"""
        self.command_callbacks.append(callback)
        print(f"[Adapter] Registered callback: {callback.__name__}")
        
    def unregister_callback(self, callback: Callable[[JARVISCommand], None]):
        """Unregister a callback"""
        if callback in self.command_callbacks:
            self.command_callbacks.remove(callback)
            
    # Queue mode interface
    def get_command(self, timeout: Optional[float] = None) -> Optional[JARVISCommand]:
        """Get command from queue (for queue mode)"""
        try:
            return self.command_queue.get(timeout=timeout)
        except queue.Empty:
            return None
            
    def has_commands(self) -> bool:
        """Check if commands are available in queue"""
        return not self.command_queue.empty()
        
    # Event bus integration
    def set_event_bus(self, event_bus):
        """Set event bus for event_bus mode"""
        self.event_bus = event_bus
        self.mode = "event_bus"
        print("[Adapter] Event bus configured")
        
    # Statistics
    def get_stats(self) -> Dict[str, int]:
        """Get processing statistics"""
        return {
            "gestures_processed": self.gestures_processed,
            "commands_sent": self.commands_sent,
            "queue_size": self.command_queue.qsize() if self.mode == "queue" else 0
        }
        
    def reset_stats(self):
        """Reset statistics"""
        self.gestures_processed = 0
        self.commands_sent = 0


# Example callback implementation for JARVIS integration
def example_jarvis_callback(command: JARVISCommand):
    """
    Example callback that shows how JARVIS would handle commands
    Replace this with actual JARVIS command execution
    """
    print(f"[JARVIS] Executing: {command.action} with {command.parameters}")
    
    # Example command routing
    if command.action == "volume_up":
        # jarvis.audio.volume_up(command.parameters["amount"])
        print(f"  → Volume up by {command.parameters['amount']}")
        
    elif command.action == "screenshot":
        # jarvis.system.screenshot()
        print(f"  → Taking screenshot")
        
    elif command.action == "scroll_up":
        # jarvis.input.scroll(command.parameters["amount"])
        print(f"  → Scrolling up by {command.parameters['amount']}")
        
    # Add more command handlers here