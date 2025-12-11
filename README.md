🌀 NETRAX AI - नेत्रः
Human-Motion Driven Personal Assistant

Gesture-controlled • Body-aware • JARVIS-inspired

NETRAX AI (pronounced as, "नेत्रः") is an advanced, real-time body detection + gesture recognition system designed to integrate directly with a personal AI assistant.
Using MediaPipe Holistic, custom gesture recognition logic, and a modular adapter, NETRAX AI transforms human motion into commands.

Whether you raise a hand, make a peace sign, swipe your arm, or lean your body — NETRAX detects it and routes it as a precise command to your JARVIS-style system.

📌 Features
🔥 60+ Keypoint Body Tracking

Powered by MediaPipe Holistic (pose + hands + face landmarks).

✋ Hand Gesture Recognition

Supports:

Peace ✌️

Stop/Open-Palm ✋

Fist ✊

Thumbs Up 👍

Thumbs Down 👎

Point 👉

Open Palm

Combined/multi-hand gestures

🧍‍♂️ Full-Body Pose Detection

Arms crossed

Arms up

Lean left/right

Pause pose

Zoom in/out (two-hand)

🧭 Motion Tracking (Swipe Gestures)

Swipe Left

Swipe Right

Swipe Up

Swipe Down

🔄 Headless + Modular Integration

NETRAX AI integrates with any AI assistant via:

Callback mode

Queue mode

Event bus mode

⚡ High-Performance Tracking

One-Euro filter smoothing

Moving average & exponential filters

Frame skipping for high FPS

Visualization mode with real-time FPS overlay

🧱 Project Structure
NETRAX_AI/
│── jarvis.py
│── run_jarvis.py
│── requirements.txt
│── README.md
│
├── config/
│   ├── body_detection_config.json
│   └── gesture_mappings.json
│
└── modules/
    └── body_detection/
        ├── __init__.py
        ├── body_detection.py
        ├── camera.py
        ├── gesture.py
        ├── pose.py
        ├── tracking.py
        ├── adapter.py
        ├── jarvis_integration.py
        ├── test_body_detection.py
        └── preview_camera.py

⚙️ Installation
1️⃣ Install Python 3.11 (recommended)

MediaPipe does NOT support Python 3.13.

2️⃣ Create a virtual environment
py -3.11 -m venv venv
venv\Scripts\activate

3️⃣ Install dependencies
pip install --upgrade pip
pip install -r requirements.txt

4️⃣ (Optional) Use DroidCam as main camera

If you want to use your phone camera:

Install DroidCam PC + mobile app

Start the feed

Run preview_camera.py to find the correct camera index

Set "camera_id": X in config/body_detection_config.json

🎮 Usage
▶️ Run the Body Detection Standalone
cd modules/body_detection
python jarvis_integration.py

▶️ Run NETRAX AI fully integrated with your JARVIS core
python run_jarvis.py


This opens the visualization window and routes recognized gestures directly into your JARVIS assistant.

Press:

Q → Quit

P → Pause body detection

🖐️ Available Gestures & Actions
Gesture	Action
Peace ✌️	screenshot
Stop ✋	pause_media
Fist ✊	mute
Thumbs Up 👍	volume_up
Thumbs Down 👎	volume_down
Point 👉	select
Swipe Left →	previous_track
Swipe Right ←	next_track
Swipe Up ↑	scroll_up
Swipe Down ↓	scroll_down
Arms Crossed	lock_screen
Arms Up	pause_detection
Lean Left	switch_workspace(left)
Lean Right	switch_workspace(right)
Zoom In	zoom_in
Zoom Out	zoom_out
🧠 How It Works

NETRAX AI consists of four main layers:

1️⃣ Camera Layer

Captures frames with minimal latency using a threaded OpenCV stream.

2️⃣ Pose Detection Layer

Uses MediaPipe Holistic to extract:

33 body landmarks

21 left-hand landmarks

21 right-hand landmarks

Key face landmarks

3️⃣ Gesture Recognition Layer

Processes pose data into gestures using:

Finger-state analysis

Body-angle analysis

Motion vector tracking

Swipe direction logic

Gesture-hold timing filters

4️⃣ Integration Adapter

Converts gestures → high-level commands like:

volume_up
pause_media
scroll_down
screenshot


These are then sent to your JARVIS system.

🌐 Why "NETRAX AI"?

NETRAX comes from:
NETRA (Sanskrit: “eye / vision”) + X (hyper-extension, unknown, futuristic).

Meaning:

"The AI that sees."

Perfect for a vision-driven, gesture-aware personal assistant.

🚀 Roadmap

 Face expression recognition

 Hand pose refinement

 Natural language + gesture fusion

 Multi-user gesture support

 AR/Hologram UI integration

📜 License

This project is for personal educational & experimental use.
Modify freely.
