import cv2
from ultralytics import YOLO

# Load pose model
model = YOLO("yolo26n_hand.pt")   # or yolov8n-pose.pt for testing

# Open webcam
cap = cv2.VideoCapture(0)

# Optional HD resolution
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

while cap.isOpened():
    success, frame = cap.read()
    if not success:
        break

    # Run YOLO inference directly on frame
    results = model(frame, conf=0.5)

    # 🔥 This uses YOLO built-in drawing:
    # - Bounding boxes
    # - Keypoints skeleton
    # - Labels + confidence
    annotated_frame = results[0].plot()

    # Display output
    cv2.imshow("YOLO26 Pose (Official Rendering)", annotated_frame)

    # Press Q to quit
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()