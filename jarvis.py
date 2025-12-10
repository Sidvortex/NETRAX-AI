from modules.body_detection import BodyDetectionSystem, BodyDetectionConfig

class JARVIS:
    def __init__(self):
        # ... existing code ...
        self.body_detection = None
        self._init_body_detection()
    
    def _init_body_detection(self):
        config = BodyDetectionConfig.from_file("config/body_detection_config.json")
        self.body_detection = BodyDetectionSystem(config)
        self.body_detection.register_command_callback(self._handle_body_command)
        self.body_detection.start()
    
    def _handle_body_command(self, command):
        if hasattr(self, command.action):
            getattr(self, command.action)(**command.parameters)
            
# ```

# ### **Option B: Plugin-Style**

# See `jarvis_integration.py` for complete plugin implementation.

# ---

# ## 📋 **DIRECTORY STRUCTURE**
# ```
# your_jarvis_project/
# ├── jarvis.py                          # MODIFIED: Add body detection init
# ├── modules/
# │   └── body_detection/                # NEW MODULE
# │       ├── __init__.py                # Module initialization
# │       ├── camera.py                  # Camera layer
# │       ├── pose.py                    # Pose detection
# │       ├── gesture.py                 # Gesture recognition
# │       ├── tracking.py                # Motion smoothing
# │       ├── adapter.py                 # JARVIS adapter
# │       ├── body_detection.py          # Main system
# │       ├── test_body_detection.py     # Tests
# │       └── jarvis_integration.py      # Integration examples
# ├── config/
# │   ├── body_detection_config.json     # System config
# │   └── gesture_mappings.json          # Gesture mappings
# ├── requirements.txt                   # UPDATED: New dependencies
# └── README.md                          # Your existing README