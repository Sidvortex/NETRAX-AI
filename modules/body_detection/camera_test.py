import cv2

print("Testing camera IDs...")

for i in range(6):
    cap = cv2.VideoCapture(i)
    if cap.isOpened():
        print(f"Camera {i}: AVAILABLE")
        cap.release()
    else:
        print(f"Camera {i}: NOT available")
