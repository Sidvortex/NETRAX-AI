# preview_camera.py
import sys, cv2

if len(sys.argv) < 2:
    print("Usage: python preview_camera.py <camera_index>")
    sys.exit(1)

idx = int(sys.argv[1])
print(f"Opening camera {idx} — press 'q' in the window to quit")

cap = cv2.VideoCapture(idx)
if not cap.isOpened():
    print(f"Camera {idx} cannot be opened")
    sys.exit(1)

while True:
    ret, frame = cap.read()
    if not ret:
        print("Frame read failed")
        break
    cv2.imshow(f"Camera {idx}", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
