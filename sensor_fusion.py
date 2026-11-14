from ultralytics import YOLO
import cv2

# Load the object detection model
model = YOLO("yolo11n.pt")

# Open camera
cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()

    if not ret:
        print("Camera could not be opened.")
        break

    # Detect objects
    results = model(frame)

    # Check detected objects
    for box in results[0].boxes:
        class_id = int(box.cls[0])
        object_name = model.names[class_id]

        # Simulated sensor distance
        distance = 20

        # Sensor fusion information
        fusion_text = f"{object_name} | Distance: {distance} m"

        # Get box coordinates
        x1, y1, x2, y2 = map(int, box.xyxy[0])

        # Draw box
        cv2.rectangle(
            frame,
            (x1, y1),
            (x2, y2),
            (0, 255, 0),
            2
        )

        # Display fused information
        cv2.putText(
            frame,
            fusion_text,
            (x1, max(y1 - 10, 20)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (0, 255, 0),
            2
        )

        print(f"Sensor Fusion: {object_name} detected at approximately {distance} m")

    # Show final result
    cv2.imshow("Autonomous Vehicle - Sensor Fusion", frame)

    # Press Q to stop
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()