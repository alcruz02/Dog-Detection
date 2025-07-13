import cv2
from ultralytics import YOLO

# Load your trained model
model = YOLO("runs/detect/train2/weights/best.pt")

# Open webcam (0 = default camera)
cap = cv2.VideoCapture(0)

# Check if camera opened successfully
if not cap.isOpened():
    print("Error: Could not open webcam.")
    exit()

print("Press 'q' to quit.")

while True:
    ret, frame = cap.read()
    if not ret:
        print("Error: Failed to grab frame.")
        break

    # Run inference on the current frame
    results = model(frame)

    # Plot the results on the frame
    for r in results:
        annotated_frame = r.plot()

    # Display the frame
    cv2.imshow("YOLOv8 - Dog Detection", annotated_frame)

    # Exit loop on 'q' key press
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

# Release resources
cap.release()
cv2.destroyAllWindows()
