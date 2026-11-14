from ultralytics import YOLO
import cv2

# Load object detection model
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

    # Get detected objects
    for box in results[0].boxes:
        class_id = int(box.cls[0])
        object_name = model.names[class_id]

        # Simple simulated distance
        distance = 20

        print(f"{object_name} detected - Approx. distance: {distance} m")

    # Display detection boxes
    annotated_frame = results[0].plot()

    cv2.imshow("Distance Estimation", annotated_frame)

    # Press Q to stop
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()