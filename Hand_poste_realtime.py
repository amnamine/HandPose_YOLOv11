import cv2
from ultralytics import YOLO

# Load YOLO model (replace with your model path)
model = YOLO("hand.pt")

# Open webcam (0 = default camera)
cap = cv2.VideoCapture(0)

if not cap.isOpened():
    print("❌ Error: Could not open webcam.")
    exit()

# Set full-screen window (optional)
cv2.namedWindow("YOLO Hand Detection", cv2.WINDOW_NORMAL)
cv2.setWindowProperty("YOLO Hand Detection", cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)

print("🎥 Press 'q' to quit...")

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # Run YOLO detection
    results = model(frame, stream=True)

    # Draw bounding boxes directly
    for r in results:
        boxes = r.boxes
        for box in boxes:
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            conf = float(box.conf[0])
            cls = int(box.cls[0])
            label = model.names[cls] if hasattr(model, 'names') else "hand"
            
            # Draw box
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(frame, f"{label} {conf:.2f}", (x1, y1 - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)

    # Show the frame
    cv2.imshow("YOLO Hand Detection", frame)

    # Press 'q' to exit
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
