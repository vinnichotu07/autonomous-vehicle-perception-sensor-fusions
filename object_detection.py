from ultralytics import YOLO
import cv2

# Load the object detection model
model = YOLO("yolo11n.pt")

# Open the camera
cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()

    if not ret:
        print("Camera could not be opened.")
        break

    # Detect objects
    results = model(frame)

    # Draw detection results
    annotated_frame = results[0].plot()

    # Show the result
    cv2.imshow("Autonomous Vehicle - Object Detection", annotated_frame)

    # Press Q to quit
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()