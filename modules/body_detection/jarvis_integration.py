#1
class JARVIS:
    def __init__(self):
        
        from body_detection import BodyDetectionSystem, BodyDetectionConfig
        
        self.body_detection = None
        self._init_body_detection()
        
    def _init_body_detection(self):
       
        try:
            
            config = BodyDetectionConfig.from_file("config/body_detection_config.json")
            
           
            self.body_detection = BodyDetectionSystem(config)
            
            
            self.body_detection.register_command_callback(self._handle_body_command)
            
            
            if self.body_detection.start():
                print("[JARVIS] Body detection enabled")
            else:
                print("[JARVIS] Body detection failed to start")
                self.body_detection = None
                
        except Exception as e:
            print(f"[JARVIS] Body detection initialization error: {e}")
            self.body_detection = None
            
    def _handle_body_command(self, command):
        
        print(f"[JARVIS] Body command: {command.action}")
        
        
        if hasattr(self, command.action):
            handler = getattr(self, command.action)
            handler(**command.parameters)
        else:
            print(f"[JARVIS] Unknown body command: {command.action}")
        
    def volume_up(self, amount=10, **kwargs):
        
        print(f"[JARVIS] Volume up by {amount}")
        
        
    def volume_down(self, amount=10, **kwargs):
       
        print(f"[JARVIS] Volume down by {amount}")
        
        
    def screenshot(self, **kwargs):
        
        print(f"[JARVIS] Taking screenshot")
        
        
    def pause_media(self, **kwargs):
        
        print(f"[JARVIS] Pausing media")
    
        
    def scroll_up(self, amount=3, **kwargs):
        
        print(f"[JARVIS] Scrolling up by {amount}")

        
    def scroll_down(self, amount=3, **kwargs):
        
        print(f"[JARVIS] Scrolling down by {amount}")

        
    def next_track(self, **kwargs):
       
        print(f"[JARVIS] Next track")
    
        
    def previous_track(self, **kwargs):
        
        print(f"[JARVIS] Previous track")
    
        
    def mute(self, **kwargs):
      
        print(f"[JARVIS] Muting")
    
        
    def pause_detection(self, duration=5.0, **kwargs):
       
        if self.body_detection:
            self.body_detection.pause()
           
            import threading
            threading.Timer(duration, self.body_detection.resume).start()
            
    def shutdown(self):
        """Shutdown JARVIS"""
       
        if self.body_detection:
            self.body_detection.stop()


#2
class BodyDetectionPlugin:
   
    def __init__(self, jarvis_instance):
        self.jarvis = jarvis_instance
        self.system = None
        
    def load(self, config_path=None):
        
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
       
        if self.system:
            self.system.stop()
            self.system = None
            
    def _execute_command(self, command):
       
        #
        if hasattr(self.jarvis, 'execute_command'):
            self.jarvis.execute_command(command.action, command.parameters)
        else:
           
            if hasattr(self.jarvis, command.action):
                handler = getattr(self.jarvis, command.action)
                handler(**command.parameters)


#3
def run_standalone_with_queue():
  
    from body_detection import BodyDetectionSystem, BodyDetectionConfig
    
    
    config = BodyDetectionConfig()
    config.integration_mode = "queue"
    
    system = BodyDetectionSystem(config)
    system.start()
    
    
    import threading
    
    def command_processor():
        while system.running:
            command = system.adapter.get_command(timeout=0.1)
            if command:
                print(f"[Queue] Processing: {command.action}")
               
                
    thread = threading.Thread(target=command_processor, daemon=True)
    thread.start()
    
   
    system.show_visualization_window()
 
    system.stop()


#4
class EventBusAdapter:
  
    
    def __init__(self, event_bus):
        self.event_bus = event_bus
        self.system = None
        
       
        self.event_bus.subscribe("body_detection.enable", self.enable)
        self.event_bus.subscribe("body_detection.disable", self.disable)
        
    def enable(self):
        
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
        
        if self.system:
            self.system.stop()
            self.system = None
            print("[EventBus] Body detection disabled")


#xtra
if __name__ == "__main__":

    from body_detection import BodyDetectionSystem, BodyDetectionConfig
    from adapter import example_jarvis_callback
    
    
    config = BodyDetectionConfig()
    config.show_visualization = True
    
    system = BodyDetectionSystem(config)
    
 
    system.register_command_callback(example_jarvis_callback)
    

    if system.start():
        print("Body detection running. Press 'q' in window to quit.")
        system.show_visualization_window()
        system.stop()
    else:
        print("Failed to start body detection")