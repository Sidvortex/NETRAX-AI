"""
JARVIS Integration Example
Shows how to integrate body detection into existing JARVIS core
"""

# Method 1: Direct Integration (add to your main JARVIS class)
class JARVIS:
    def __init__(self):
        # ... existing JARVIS initialization ...
        
        # Add body detection system
        from body_detection import BodyDetectionSystem, BodyDetectionConfig
        
        self.body_detection = None
        self._init_body_detection()
        
    def _init_body_detection(self):
        """Initialize body detection system"""
        try:
            # Load config
            config = BodyDetectionConfig.from_file("config/body_detection_config.json")
            
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
            
    # Add command handlers
    def volume_up(self, amount=10, **kwargs):
        """Increase volume"""
        print(f"[JARVIS] Volume up by {amount}")
        # Your volume control code here
        
    def volume_down(self, amount=10, **kwargs):
        """Decrease volume"""
        print(f"[JARVIS] Volume down by {amount}")
        # Your volume control code here
        
    def screenshot(self, **kwargs):
        """Take screenshot"""
        print(f"[JARVIS] Taking screenshot")
        # Your screenshot code here
        
    def pause_media(self, **kwargs):
        """Pause media playback"""
        print(f"[JARVIS] Pausing media")
        # Your media control code here
        
    def scroll_up(self, amount=3, **kwargs):
        """Scroll up"""
        print(f"[JARVIS] Scrolling up by {amount}")
        # Your scroll code here
        
    def scroll_down(self, amount=3, **kwargs):
        """Scroll down"""
        print(f"[JARVIS] Scrolling down by {amount}")
        # Your scroll code here
        
    def next_track(self, **kwargs):
        """Next track"""
        print(f"[JARVIS] Next track")
        # Your media control code here
        
    def previous_track(self, **kwargs):
        """Previous track"""
        print(f"[JARVIS] Previous track")
        # Your media control code here
        
    def mute(self, **kwargs):
        """Mute audio"""
        print(f"[JARVIS] Muting")
        # Your audio control code here
        
    def pause_detection(self, duration=5.0, **kwargs):
        """Pause body detection temporarily"""
        if self.body_detection:
            self.body_detection.pause()
            # Set timer to resume
            import threading
            threading.Timer(duration, self.body_detection.resume).start()
            
    def shutdown(self):
        """Shutdown JARVIS"""
        # ... existing shutdown code ...
        
        # Stop body detection
        if self.body_detection:
            self.body_detection.stop()


# Method 2: Plugin-style Integration
class BodyDetectionPlugin:
    """Body detection as a JARVIS plugin"""
    
    def __init__(self, jarvis_instance):
        self.jarvis = jarvis_instance
        self.system = None
        
    def load(self, config_path=None):
        """Load plugin"""
        from body_detection import BodyDetectionSystem, BodyDetectionConfig
        
        if config_path:
            config = BodyDetectionConfig.from_file(config_path)
        else:
            config = BodyDetectionConfig()
            
        self.system = BodyDetectionSystem(config)
        self.system.register_command_callback(self._execute_command)
        
        if self.system.start():
            print("[Plugin] Body detection loaded")
            return True
        return False
        
    def unload(self):
        """Unload plugin"""
        if self.system:
            self.system.stop()
            self.system = None
            
    def _execute_command(self, command):
        """Execute command on JARVIS"""
        # Forward to JARVIS command handler
        if hasattr(self.jarvis, 'execute_command'):
            self.jarvis.execute_command(command.action, command.parameters)
        else:
            # Direct method call
            if hasattr(self.jarvis, command.action):
                handler = getattr(self.jarvis, command.action)
                handler(**command.parameters)


# Method 3: Standalone with Queue
def run_standalone_with_queue():
    """Run body detection standalone and consume commands via queue"""
    from body_detection import BodyDetectionSystem, BodyDetectionConfig
    
    # Create system with queue mode
    config = BodyDetectionConfig()
    config.integration_mode = "queue"
    
    system = BodyDetectionSystem(config)
    system.start()
    
    # Process commands from queue
    import threading
    
    def command_processor():
        while system.running:
            command = system.adapter.get_command(timeout=0.1)
            if command:
                print(f"[Queue] Processing: {command.action}")
                # Execute command
                # jarvis.execute(command)
                
    thread = threading.Thread(target=command_processor, daemon=True)
    thread.start()
    
    # Show visualization
    system.show_visualization_window()
    
    # Cleanup
    system.stop()


# Method 4: Event Bus Integration (if JARVIS has event bus)
class EventBusAdapter:
    """Integrate with JARVIS event bus"""
    
    def __init__(self, event_bus):
        self.event_bus = event_bus
        self.system = None
        
        # Subscribe to control events
        self.event_bus.subscribe("body_detection.enable", self.enable)
        self.event_bus.subscribe("body_detection.disable", self.disable)
        
    def enable(self):
        """Enable body detection"""
        if self.system:
            return
            
        from body_detection import BodyDetectionSystem, BodyDetectionConfig
        
        config = BodyDetectionConfig()
        config.integration_mode = "event_bus"
        
        self.system = BodyDetectionSystem(config)
        self.system.adapter.set_event_bus(self.event_bus)
        self.system.start()
        
        print("[EventBus] Body detection enabled")
        
    def disable(self):
        """Disable body detection"""
        if self.system:
            self.system.stop()
            self.system = None
            print("[EventBus] Body detection disabled")


# Example usage
if __name__ == "__main__":
    # Simple test
    from body_detection import BodyDetectionSystem, BodyDetectionConfig
    from adapter import example_jarvis_callback
    
    # Create and configure system
    config = BodyDetectionConfig()
    config.show_visualization = True
    
    system = BodyDetectionSystem(config)
    
    # Register callback
    system.register_command_callback(example_jarvis_callback)
    
    # Start
    if system.start():
        print("Body detection running. Press 'q' in window to quit.")
        system.show_visualization_window()
        system.stop()
    else:
        print("Failed to start body detection")