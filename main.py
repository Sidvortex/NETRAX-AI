"""
NETRAX AI - Vision System Core Server
Production-grade real-time computer vision engine
"""

from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.responses import StreamingResponse, HTMLResponse
from fastapi.middleware.cors import CORSMiddleware
from contextlib import asynccontextmanager
import cv2
import json
import asyncio
import logging
from typing import List, Dict, Any
import numpy as np
from datetime import datetime

from vision_engine.body_tracker import BodyTracker
from vision_engine.iris_tracker import IrisTracker
from vision_engine.gesture_engine import GestureEngine
from vision_engine.object_detector import ObjectDetector
from vision_engine.tracking_coordinator import TrackingCoordinator
from config import settings

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("NETRAX")

# Global state
active_connections: List[WebSocket] = []
camera = None
vision_coordinator = None

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Initialize and cleanup vision systems"""
    global camera, vision_coordinator
    
    logger.info("🔴 NETRAX AI - Vision System Initializing...")
    
    # Initialize camera
    camera = cv2.VideoCapture(settings.CAMERA_INDEX)
    camera.set(cv2.CAP_PROP_FRAME_WIDTH, settings.FRAME_WIDTH)
    camera.set(cv2.CAP_PROP_FRAME_HEIGHT, settings.FRAME_HEIGHT)
    camera.set(cv2.CAP_PROP_FPS, settings.TARGET_FPS)
    
    if not camera.isOpened():
        logger.error("❌ Failed to open camera")
        raise RuntimeError("Camera initialization failed")
    
    # Initialize vision coordinator
    vision_coordinator = TrackingCoordinator(
        enable_body=settings.ENABLE_BODY_TRACKING,
        enable_iris=settings.ENABLE_IRIS_TRACKING,
        enable_gestures=settings.ENABLE_GESTURE_RECOGNITION,
        enable_objects=settings.ENABLE_OBJECT_DETECTION
    )
    
    logger.info("✅ Vision system initialized")
    logger.info(f"📹 Camera: {settings.FRAME_WIDTH}x{settings.FRAME_HEIGHT} @ {settings.TARGET_FPS}fps")
    logger.info(f"🧠 Body: {settings.ENABLE_BODY_TRACKING} | Iris: {settings.ENABLE_IRIS_TRACKING}")
    logger.info(f"👋 Gestures: {settings.ENABLE_GESTURE_RECOGNITION} | 🎯 Objects: {settings.ENABLE_OBJECT_DETECTION}")
    
    yield
    
    # Cleanup
    logger.info("🔴 Shutting down vision system...")
    if camera:
        camera.release()
    if vision_coordinator:
        vision_coordinator.cleanup()
    logger.info("✅ Cleanup complete")

app = FastAPI(
    title="NETRAX AI Vision System",
    description="Production-grade real-time computer vision engine",
    version="1.0.0",
    lifespan=lifespan
)

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# WebSocket connection manager
class ConnectionManager:
    def __init__(self):
        self.active_connections: List[WebSocket] = []
    
    async def connect(self, websocket: WebSocket):
        await websocket.accept()
        self.active_connections.append(websocket)
        logger.info(f"🔗 Client connected. Total: {len(self.active_connections)}")
    
    def disconnect(self, websocket: WebSocket):
        self.active_connections.remove(websocket)
        logger.info(f"🔌 Client disconnected. Total: {len(self.active_connections)}")
    
    async def broadcast(self, message: dict):
        """Broadcast to all connected clients"""
        for connection in self.active_connections:
            try:
                await connection.send_json(message)
            except Exception as e:
                logger.error(f"Failed to send to client: {e}")

manager = ConnectionManager()

@app.get("/")
async def root():
    """Health check endpoint"""
    return {
        "status": "operational",
        "system": "NETRAX AI Vision System",
        "version": "1.0.0",
        "timestamp": datetime.now().isoformat()
    }

@app.get("/status")
async def get_status():
    """Get system status"""
    return {
        "camera_active": camera is not None and camera.isOpened(),
        "active_connections": len(manager.active_connections),
        "vision_coordinator": vision_coordinator is not None,
        "settings": {
            "body_tracking": settings.ENABLE_BODY_TRACKING,
            "iris_tracking": settings.ENABLE_IRIS_TRACKING,
            "gesture_recognition": settings.ENABLE_GESTURE_RECOGNITION,
            "object_detection": settings.ENABLE_OBJECT_DETECTION,
            "fps_target": settings.TARGET_FPS
        }
    }

@app.get("/video_feed")
async def video_feed():
    """Stream video feed with vision overlays"""
    def generate():
        while True:
            if camera is None or not camera.isOpened():
                break
            
            success, frame = camera.read()
            if not success:
                logger.warning("Failed to read frame")
                break
            
            # Process frame through vision coordinator
            if vision_coordinator:
                processed_frame, tracking_data = vision_coordinator.process_frame(frame)
            else:
                processed_frame = frame
            
            # Encode frame
            ret, buffer = cv2.imencode('.jpg', processed_frame, 
                                      [cv2.IMWRITE_JPEG_QUALITY, settings.JPEG_QUALITY])
            if not ret:
                continue
            
            frame_bytes = buffer.tobytes()
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')
    
    return StreamingResponse(
        generate(),
        media_type="multipart/x-mixed-replace; boundary=frame"
    )

@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    """WebSocket endpoint for real-time vision data"""
    await manager.connect(websocket)
    
    try:
        # Start vision processing loop
        asyncio.create_task(vision_processing_loop(websocket))
        
        # Keep connection alive and handle incoming messages
        while True:
            data = await websocket.receive_text()
            message = json.loads(data)
            
            # Handle client commands
            if message.get("type") == "command":
                handle_client_command(message)
            
    except WebSocketDisconnect:
        manager.disconnect(websocket)
    except Exception as e:
        logger.error(f"WebSocket error: {e}")
        manager.disconnect(websocket)

async def vision_processing_loop(websocket: WebSocket):
    """Main vision processing loop"""
    frame_count = 0
    gesture_count = 0
    start_time = datetime.now()
    last_gesture = None
    
    while True:
        try:
            if camera is None or not camera.isOpened():
                await asyncio.sleep(0.1)
                continue
            
            success, frame = camera.read()
            if not success:
                logger.warning("Failed to read frame")
                await asyncio.sleep(0.1)
                continue
            
            # Process frame
            if vision_coordinator:
                processed_frame, tracking_data = vision_coordinator.process_frame(frame)
                
                # Calculate FPS
                frame_count += 1
                elapsed = (datetime.now() - start_time).total_seconds()
                fps = frame_count / elapsed if elapsed > 0 else 0
                
                # Send stats update
                stats_message = {
                    "type": "stats",
                    "stats": {
                        "fps": fps,
                        "gesture_count": gesture_count,
                        "confidence": tracking_data.get("confidence", 0.0),
                        "timestamp": datetime.now().isoformat()
                    }
                }
                await websocket.send_json(stats_message)
                
                # Send gesture commands
                gesture = tracking_data.get("gesture")
                if gesture and gesture != last_gesture:
                    gesture_count += 1
                    last_gesture = gesture
                    
                    gesture_message = {
                        "type": "gesture_command",
                        "command": gesture,
                        "confidence": tracking_data.get("gesture_confidence", 0.0),
                        "timestamp": datetime.now().isoformat()
                    }
                    await websocket.send_json(gesture_message)
                    logger.info(f"👋 Gesture detected: {gesture}")
                
                # Send detailed tracking data
                if settings.SEND_DETAILED_DATA:
                    detail_message = {
                        "type": "tracking_detail",
                        "data": tracking_data,
                        "timestamp": datetime.now().isoformat()
                    }
                    await websocket.send_json(detail_message)
            
            # Control frame rate
            await asyncio.sleep(1.0 / settings.TARGET_FPS)
            
        except WebSocketDisconnect:
            break
        except Exception as e:
            logger.error(f"Vision processing error: {e}")
            await asyncio.sleep(0.1)

def handle_client_command(message: Dict[str, Any]):
    """Handle commands from client"""
    command = message.get("command")
    
    if command == "calibrate":
        logger.info("🎯 Calibration requested")
        if vision_coordinator:
            vision_coordinator.calibrate()
    
    elif command == "reset_tracking":
        logger.info("🔄 Tracking reset requested")
        if vision_coordinator:
            vision_coordinator.reset()
    
    elif command == "toggle_module":
        module = message.get("module")
        logger.info(f"⚙️ Toggling module: {module}")
        # Implement module toggling logic
    
    else:
        logger.warning(f"Unknown command: {command}")

@app.get("/api/gestures")
async def get_available_gestures():
    """Get list of available gesture commands"""
    return {
        "gestures": [
            {"name": "peace", "emoji": "✌️", "description": "Peace sign / Screenshot"},
            {"name": "stop", "emoji": "✋", "description": "Stop / Pause media"},
            {"name": "thumbs_up", "emoji": "👍", "description": "Thumbs up / Volume up"},
            {"name": "thumbs_down", "emoji": "👎", "description": "Thumbs down / Volume down"},
            {"name": "fist", "emoji": "✊", "description": "Fist / Mute"},
            {"name": "point", "emoji": "👉", "description": "Point / Select"},
            {"name": "swipe", "emoji": "➡️", "description": "Swipe / Next/Previous"},
            {"name": "arms_crossed", "emoji": "🙆", "description": "Arms crossed / Pause detection"}
        ]
    }

@app.post("/api/calibrate")
async def calibrate_system():
    """Trigger system calibration"""
    if vision_coordinator:
        vision_coordinator.calibrate()
        return {"status": "success", "message": "Calibration started"}
    return {"status": "error", "message": "Vision coordinator not initialized"}

@app.post("/api/reset")
async def reset_tracking():
    """Reset all tracking systems"""
    if vision_coordinator:
        vision_coordinator.reset()
        return {"status": "success", "message": "Tracking reset"}
    return {"status": "error", "message": "Vision coordinator not initialized"}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        "main:app",
        host=settings.HOST,
        port=settings.PORT,
        reload=settings.DEBUG,
        log_level="info"
    )