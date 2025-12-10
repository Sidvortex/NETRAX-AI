# JARVIS Body Detection System

Complete gesture-based control system for JARVIS using computer vision and pose detection.

## 📋 Features

- **Real-time Pose Detection**: MediaPipe Holistic, MoveNet, OpenPose support
- **Gesture Recognition**: Hand gestures (✌️ peace, ✋ stop, 👍 thumbs up, etc.) and body poses
- **Motion Tracking**: Smooth tracking with jitter reduction (One Euro Filter)
- **JARVIS Integration**: Multiple integration methods (callbacks, queue, event bus)
- **User Configurable**: Editable gesture→command mappings via JSON
- **High Performance**: 30+ FPS with optimized processing pipeline

## 🏗️ Architecture

```
┌─────────────┐     ┌──────────────┐     ┌─────────────────┐
│   Camera    │────▶│ Pose Detector│────▶│ Gesture         │
│   Layer     │     │ (MediaPipe)  │     │ Recognizer      │
└─────────────┘     └──────────────┘     └─────────────────┘
                                                   │
                                                   ▼
┌─────────────┐     ┌──────────────┐     ┌─────────────────┐
│   JARVIS    │◀────│   Adapter    │◀────│ Motion Tracker  │
│    Core     │     │   (Queue)    │     │ (Smoothing)     │
└─────────────┘     └──────────────┘     └─────────────────┘
```

## 📁 Project Structure

```
jarvis/
├── modules/
│   └── body_detection/          # NEW: Body detection module
│       ├── __init__.py
│       ├── camera.py            # Camera capture & streaming
│       ├── pose.py              # Pose detection models
│       ├── gesture.py           # Gesture recognition
│       ├── tracking.py          # Motion smoothing & tracking
│       ├── adapter.py           # JARVIS integration adapter
│       ├── body_detection.py   # Main system manager
│       └── test_body_detection.py  # Unit tests
│
├── config/
│   ├── body_detection_config.json    # System configuration
│   └── gesture_mappings.json         # User-editable gesture→command mappings
│
├── jarvis.py                    # MODIFIED: Main JARVIS class
└── requirements.txt             # UPDATED: Added body detection dependencies
```

## 🚀 Installation

### Step 1: Install Dependencies

```bash
cd /path/to/jarvis
pip install -r requirements.txt
```

Required packages:
- `opencv-python>=4.8.0`
- `numpy>=1.24.0`
- `mediapipe>=0.10.0`

### Step 2: Create Module Directory

```bash
mkdir -p modules/body_detection
mkdir -p config
```

### Step 3: Add Module Files

Copy the following files to `modules/body_detection/`:
- `camera.py`
- `pose.py`
- `gesture.py`
- `tracking.py`
- `adapter.py`
- `body_detection.py`
- `test_body_detection.py`

Create `modules/body_detection/__init__.py`:
```python
"""JARVIS Body Detection Module"""
from .body_detection import BodyDetectionSystem, BodyDetectionConfig
from .adapter import JARVISAdapter, GestureCommandMapper, JARVISCommand

__all__ = [
    'BodyDetectionSystem',
    'BodyDetectionConfig',
    'JARVISAdapter',
    'GestureCommandMapper',
    'JARVISCommand'
]
```

### Step 4: Add Configuration Files

Copy to `config/`:
- `body_detection_config.json`
- `gesture_mappings.json`

### Step 5: Integrate with JARVIS Core

#### Option A: Direct Integration (Recommended)

Modify your main `jarvis.py`:

```python
# Add imports at the top
from modules.body_detection import BodyDetectionSystem, BodyDetectionConfig

class JARVIS:
    def __init__(self):
        # ... existing initialization ...
        
        # Add body detection
        self.body_detection = None
        self._init_body_detection()
    
    def _init_body_detection(self):
        """Initialize body detection system"""
        try:
            from pathlib import Path
            
            # Load config
            config_path = Path("config/body_detection_config.json")
            if config_path.exists():
                config = BodyDetectionConfig.from_file(config_path)
            else:
                config = BodyDetectionConfig()
                config.save(config_path)
            
            # Create system
            self.body_detection = BodyDetectionSystem(config)
            
            # Register command callback
            self.body_detection.register_command_callback(self._handle_body_command)
            
            # Start system
            if self.body_detection.start():
                print("[JARVIS] Body detection enabled")
            else:
                print("[JARVIS] Body detection failed to start")
                self.body_detection = None
                
        except Exception as e:
            print(f"[JARVIS] Body detection initialization error: {e}")
            self.body_detection = None
    
    def _handle_body_command(self, command):
        """Handle commands from body detection system"""
        print(f"[JARVIS] Body command: {command.action}")
        
        # Route to appropriate handler
        if hasattr(self, command.action):
            handler = getattr(self, command.action)
            handler(**command.parameters)
        else:
            print(f"[JARVIS] Unknown body command: {command.action}")
    
    # Add command handlers (examples)
    def volume_up(self, amount=10, **kwargs):
        """Increase volume"""
        # Your implementation
        pass
    
    def volume_down(self, amount=10, **kwargs):
        """Decrease volume"""
        # Your implementation
        pass
    
    def screenshot(self, **kwargs):
        """Take screenshot"""
        # Your implementation
        pass
    
    def pause_media(self, **kwargs):
        """Pause media playback"""
        # Your implementation
        pass
    
    def scroll_up(self, amount=3, **kwargs):
        """Scroll up"""
        # Your implementation
        pass
    
    def scroll_down(self, amount=3, **kwargs):
        """Scroll down"""
        # Your implementation
        pass
    
    def next_track(self, **kwargs):
        """Next track"""
        # Your implementation
        pass
    
    def previous_track(self, **kwargs):
        """Previous track"""
        # Your implementation
        pass
    
    def mute(self, **kwargs):
        """Mute audio"""
        # Your implementation
        pass
    
    def shutdown(self):
        """Shutdown JARVIS"""
        # Stop body detection
        if self.body_detection:
            self.body_detection.stop()
        
        # ... existing shutdown code ...
```

#### Option B: Plugin-Style Integration

Create `plugins/body_detection_plugin.py`:

```python
from modules.body_detection import BodyDetectionSystem, BodyDetectionConfig

class BodyDetectionPlugin:
    def __init__(self, jarvis_instance):
        self.jarvis = jarvis_instance
        self.system = None
    
    def load(self):
        config = BodyDetectionConfig.from_file("config/body_detection_config.json")
        self.system = BodyDetectionSystem(config)
        self.system.register_command_callback(self._execute_command)
        return self.system.start()
    
    def unload(self):
        if self.system:
            self.system.stop()
    
    def _execute_command(self, command):
        if hasattr(self.jarvis, command.action):
            handler = getattr(self.jarvis, command.action)
            handler(**command.parameters)
```

Then in `jarvis.py`:
```python
from plugins.body_detection_plugin import BodyDetectionPlugin

class JARVIS:
    def __init__(self):
        # ... existing code ...
        self.plugins['body_detection'] = BodyDetectionPlugin(self)
        self.plugins['body_detection'].load()
```

## ⚙️ Configuration

### System Configuration (`body_detection_config.json`)

```json
{
  "camera_id": 0,              // Camera device ID
  "camera_width": 1280,        // Frame width
  "camera_height": 720,        // Frame height
  "camera_fps": 30,            // Target FPS
  "mirror_camera": true,       // Mirror horizontally
  
  "pose_framework": "mediapipe",  // Detection framework
  "pose_confidence": 0.5,         // Detection confidence threshold
  "pose_tracking_confidence": 0.5,
  "pose_model_complexity": 1,     // 0=Lite, 1=Full, 2=Heavy
  
  "gesture_min_confidence": 0.6,  // Gesture confidence threshold
  "gesture_hold_time": 0.3,       // Time to hold gesture (seconds)
  "swipe_threshold": 0.15,        // Minimum swipe distance
  
  "enable_smoothing": true,       // Enable motion smoothing
  "smoothing_type": "one_euro",   // Smoothing algorithm
  
  "integration_mode": "callback", // Integration method
  "show_visualization": true,     // Show debug window
  "target_fps": 30                // Processing FPS target
}
```

### Gesture Mappings (`gesture_mappings.json`)

Customize gesture→command mappings:

```json
{
  "peace": {
    "action": "screenshot",
    "parameters": {}
  },
  "thumbs_up": {
    "action": "volume_up",
    "parameters": {"amount": 10}
  },
  "swipe_right": {
    "action": "next_track",
    "parameters": {}
  }
}
```

Available gestures:
- **Hand**: `peace`, `stop`, `thumbs_up`, `thumbs_down`, `fist`, `point`, `open_palm`
- **Swipe**: `swipe_left`, `swipe_right`, `swipe_up`, `swipe_down`
- **Body**: `pause`, `arms_crossed`, `arms_up`, `leaning_left`, `leaning_right`
- **Zoom**: `zoom_in`, `zoom_out`

## 🎮 Usage

### Basic Usage

```python
from modules.body_detection import BodyDetectionSystem, BodyDetectionConfig

# Create system
config = BodyDetectionConfig()
system = BodyDetectionSystem(config)

# Register callback
def handle_command(command):
    print(f"Command: {command.action}")

system.register_command_callback(handle_command)

# Start
system.start()

# Show visualization (blocking)
system.show_visualization_window()

# Stop
system.stop()
```

### Integration Modes

#### 1. Callback Mode (Recommended)
```python
config.integration_mode = "callback"
system.register_command_callback(your_callback)
```

#### 2. Queue Mode
```python
config.integration_mode = "queue"

# In your processing loop:
command = system.adapter.get_command(timeout=0.1)
if command:
    execute(command)
```

#### 3. Event Bus Mode
```python
config.integration_mode = "event_bus"
system.adapter.set_event_bus(your_event_bus)
```

## 🧪 Testing

Run unit tests:

```bash
cd modules/body_detection
python test_body_detection.py
```

Run standalone demo:

```bash
cd modules/body_detection
python jarvis_integration.py
```

## 📊 Performance Tuning

### High Performance Mode
```json
{
  "pose_model_complexity": 0,
  "skip_frames": 1,
  "enable_smoothing": false,
  "target_fps": 60
}
```

### High Accuracy Mode
```json
{
  "pose_model_complexity": 2,
  "skip_frames": 0,
  "enable_smoothing": true,
  "smoothing_type": "one_euro",
  "gesture_hold_time": 0.5
}
```

### Battery Saver Mode
```json
{
  "camera_width": 640,
  "camera_height": 480,
  "pose_model_complexity": 0,
  "skip_frames": 2,
  "target_fps": 15
}
```

## 🔧 Troubleshooting

### Camera Not Found
- Check `camera_id` in config (try 0, 1, 2)
- Verify camera permissions
- Test with: `python -c "import cv2; print(cv2.VideoCapture(0).isOpened())"`

### Low FPS
- Reduce `camera_width` and `camera_height`
- Set `pose_model_complexity` to 0
- Increase `skip_frames`
- Disable `enable_smoothing`

### Gestures Not Detected
- Lower `gesture_min_confidence`
- Reduce `gesture_hold_time`
- Check camera is not mirrored incorrectly
- Ensure good lighting
- Move closer to camera

### Import Errors
```bash
pip install --upgrade mediapipe opencv-python numpy
```

## 🎯 Gesture Recognition Tips

1. **Position**: Stand 1-2 meters from camera
2. **Lighting**: Ensure face and hands are well-lit
3. **Background**: Plain background works best
4. **Gestures**: Hold for 0.3-0.5 seconds for recognition
5. **Speed**: Move deliberately, not too fast

## 🔌 Extending the System

### Add Custom Gesture

1. Add gesture type to `gesture.py`:
```python
class GestureType(Enum):
    CUSTOM_GESTURE = "custom_gesture"
```

2. Implement recognition logic in `GestureRecognizer.recognize()`

3. Add mapping to `gesture_mappings.json`:
```json
{
  "custom_gesture": {
    "action": "custom_action",
    "parameters": {}
  }
}
```

### Add New Pose Framework

1. Create detector class in `pose.py`:
```python
class CustomDetector(BasePoseDetector):
    def detect(self, frame):
        # Your implementation
        return PoseResult()
```

2. Register in factory:
```python
class PoseFramework(Enum):
    CUSTOM = "custom"

@staticmethod
def create(framework, **kwargs):
    if framework == PoseFramework.CUSTOM:
        return CustomDetector(**kwargs)
```

## 📝 API Reference

### BodyDetectionSystem

```python
system = BodyDetectionSystem(config)
system.start()                      # Start detection
system.stop()                       # Stop detection
system.pause()                      # Pause temporarily
system.resume()                     # Resume after pause
system.get_stats()                  # Get performance stats
system.register_command_callback(fn)  # Register callback
```

### JARVISCommand

```python
command.action        # Command name (string)
command.parameters    # Command parameters (dict)
command.timestamp     # When command was created
command.to_dict()     # Convert to dictionary
```

## 📄 License

Integrate this module into your JARVIS project. Modify freely.

## 🤝 Contributing

To add features:
1. Add new gesture types in `gesture.py`
2. Update mappings in `gesture_mappings.json`
3. Add command handlers in your JARVIS core
4. Test with `test_body_detection.py`

## 📧 Support

For issues:
1. Check configuration files
2. Run tests
3. Enable visualization to debug
4. Check camera and lighting
5. Verify MediaPipe installation

---

**Ready to use!** Start JARVIS with body detection enabled and control with gestures! ✨