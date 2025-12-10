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
            
