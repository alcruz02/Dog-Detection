import cv2
from ultralytics import YOLO


model = YOLO("runs/detect/train2/weights/best.pt")


cap = cv2.VideoCapture(0)


if not cap.isOpened():
    print("Error: Could not open webcam.")
    exit()

print("Press 'q' to quit.")

while True:
    ret, frame = cap.read()
    if not ret:
        print("Error: Failed to grab frame.")
        break


    results = model(frame)


    for r in results:
        annotated_frame = r.plot()


    cv2.imshow("YOLOv8 - Dog Detection", annotated_frame)


    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()
