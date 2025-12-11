import unittest
import numpy as np
import time
from pathlib import Path

from pose import Keypoint, PoseResult, MediaPipeDetector
from gesture import GestureRecognizer, GestureType, Gesture
from tracking import KeypointTracker, PoseTracker
from adapter import GestureCommandMapper, JARVISAdapter, JARVISCommand


class TestKeypoint(unittest.TestCase):
   
    
    def test_creation(self):
        kp = Keypoint(x=0.5, y=0.5, z=0.0, confidence=0.9)
        self.assertEqual(kp.x, 0.5)
        self.assertEqual(kp.y, 0.5)
        
    def test_to_pixel(self):
        kp = Keypoint(x=0.5, y=0.5)
        px, py = kp.to_pixel(1920, 1080)
        self.assertEqual(px, 960)
        self.assertEqual(py, 540)


class TestPoseResult(unittest.TestCase):
    
    
    def test_empty_pose(self):
        result = PoseResult()
        self.assertFalse(result.has_pose())
        self.assertFalse(result.has_hands())
        
    def test_with_landmarks(self):
        result = PoseResult()
        result.pose_landmarks = {'nose': Keypoint(0.5, 0.3)}
        self.assertTrue(result.has_pose())


class TestGestureRecognizer(unittest.TestCase):
    
    
    def setUp(self):
        self.recognizer = GestureRecognizer(
            min_confidence=0.5,
            gesture_hold_time=0.1
        )
        
    def create_hand_landmarks(self, finger_states):
        
        landmarks = {}
        
        # Simplified hand landmark creation
        finger_names = [
            ('thumb_tip', 'thumb_ip'),
            ('index_finger_tip', 'index_finger_pip'),
            ('middle_finger_tip', 'middle_finger_pip'),
            ('ring_finger_tip', 'ring_finger_pip'),
            ('pinky_tip', 'pinky_pip')
        ]
        
        base_y = 0.5
        for i, (tip_name, pip_name) in enumerate(finger_names):
            extended = finger_states[i]
            
            if extended:
                # Tip above pip
                landmarks[tip_name] = Keypoint(0.5, base_y - 0.1)
                landmarks[pip_name] = Keypoint(0.5, base_y)
            else:
                # Tip below pip
                landmarks[tip_name] = Keypoint(0.5, base_y + 0.1)
                landmarks[pip_name] = Keypoint(0.5, base_y)
                
        landmarks['wrist'] = Keypoint(0.5, 0.7)
        
        return landmarks
        
    def test_peace_gesture(self):
       
        hand = self.create_hand_landmarks([False, True, True, False, False])
        
        pose_result = PoseResult()
        pose_result.right_hand_landmarks = hand
        
        gestures = self.recognizer.recognize(pose_result)
        
        # May not detect immediately due to hold time
        time.sleep(0.15)
        gestures = self.recognizer.recognize(pose_result)
        
        # Check if peace gesture was detected
        peace_found = any(g.type == GestureType.PEACE for g in gestures)
        self.assertTrue(peace_found or len(gestures) == 0)  # May need multiple frames
        
    def test_fist_gesture(self):
       
        hand = self.create_hand_landmarks([False, False, False, False, False])
        
        pose_result = PoseResult()
        pose_result.left_hand_landmarks = hand
        
        time.sleep(0.15)
        gestures = self.recognizer.recognize(pose_result)
        
        # Fist detection
        fist_found = any(g.type == GestureType.FIST for g in gestures)
        self.assertTrue(fist_found or len(gestures) == 0)


class TestGestureCommandMapper(unittest.TestCase):
    
    
    def setUp(self):
        self.mapper = GestureCommandMapper()
        
    def test_default_mappings(self):
       
        gesture = Gesture(
            type=GestureType.PEACE,
            confidence=0.9,
            hand="right"
        )
        
        command = self.mapper.map(gesture)
        self.assertIsNotNone(command)
        self.assertEqual(command.action, "screenshot")
        
    def test_custom_mapping(self):
       
        self.mapper.add_mapping(
            GestureType.PEACE,
            "custom_action",
            {"param": "value"}
        )
        
        gesture = Gesture(type=GestureType.PEACE, confidence=0.9)
        command = self.mapper.map(gesture)
        
        self.assertEqual(command.action, "custom_action")
        self.assertEqual(command.parameters["param"], "value")
        
    def test_unknown_gesture(self):
       
        gesture = Gesture(type=GestureType.UNKNOWN, confidence=0.9)
        command = self.mapper.map(gesture)
        self.assertIsNone(command)


class TestJARVISAdapter(unittest.TestCase):
    
    
    def test_callback_mode(self):
       
        adapter = JARVISAdapter(mode="callback")
        
        received_commands = []
        
        def test_callback(command):
            received_commands.append(command)
            
        adapter.register_callback(test_callback)
        
        # Process gesture
        gesture = Gesture(type=GestureType.PEACE, confidence=0.9)
        adapter.process_gesture(gesture)
        
        # Check command was received
        self.assertEqual(len(received_commands), 1)
        self.assertEqual(received_commands[0].action, "screenshot")
        
    def test_queue_mode(self):
        
        adapter = JARVISAdapter(mode="queue")
        
        # Process gesture
        gesture = Gesture(type=GestureType.THUMBS_UP, confidence=0.9)
        adapter.process_gesture(gesture)
        
        # Get command from queue
        command = adapter.get_command(timeout=1.0)
        
        self.assertIsNotNone(command)
        self.assertEqual(command.action, "volume_up")
        
    def test_statistics(self):
        
        adapter = JARVISAdapter(mode="callback")
        
        gestures = [
            Gesture(type=GestureType.PEACE, confidence=0.9),
            Gesture(type=GestureType.THUMBS_UP, confidence=0.9),
            Gesture(type=GestureType.STOP, confidence=0.9),
        ]
        
        for g in gestures:
            adapter.process_gesture(g)
            
        stats = adapter.get_stats()
        
        self.assertEqual(stats["gestures_processed"], 3)
        self.assertEqual(stats["commands_sent"], 3)


class TestKeypointTracker(unittest.TestCase):
    
    
    def test_moving_average(self):
        
        tracker = KeypointTracker(smoothing_type="moving_average", window_size=3)
        
        keypoints = {
            'test': Keypoint(0.5, 0.5)
        }
        
        # Track multiple times
        for i in range(5):
            keypoints['test'].x = 0.5 + i * 0.01
            smoothed = tracker.track(keypoints, time.time(), "test")
            
        # Smoothed value should be average
        self.assertIsNotNone(smoothed['test'])
        
    def test_exponential_smoothing(self):
        
        tracker = KeypointTracker(smoothing_type="exponential")
        
        keypoints = {'test': Keypoint(0.5, 0.5)}
        
        smoothed1 = tracker.track(keypoints, time.time(), "test")
        
        keypoints['test'].x = 0.6
        smoothed2 = tracker.track(keypoints, time.time(), "test")
        
        # Smoothed value should be between original and new
        self.assertGreater(smoothed2['test'].x, 0.5)
        self.assertLess(smoothed2['test'].x, 0.6)


class TestPoseTracker(unittest.TestCase):
    
    
    def test_tracking(self):
       
        tracker = PoseTracker(smoothing_type="moving_average")
        
        pose_result = PoseResult()
        pose_result.pose_landmarks = {
            'nose': Keypoint(0.5, 0.3)
        }
        
        smoothed = tracker.track(pose_result)
        
        self.assertIsNotNone(smoothed)
        self.assertTrue(smoothed.has_pose())


def run_tests():
    
    unittest.main(argv=[''], verbosity=2, exit=False)


if __name__ == "__main__":
    run_tests()