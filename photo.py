import os
from ultralytics import YOLO
import cv2

# Load trained model
model = YOLO("runs/detect/train2/weights/best.pt")

# Path to folder with dog images
folder_path = "dog"
output_folder = "predicted"

os.makedirs(output_folder, exist_ok=True)

# Loop over all image files in folder
for filename in os.listdir(folder_path):
    if filename.lower().endswith((".jpg", ".jpeg", ".png", ".jp2")):
        img_path = os.path.join(folder_path, filename)

        results = model(img_path)

        for r in results:
            annotated_frame = r.plot()
            save_path = os.path.join(output_folder, f"pred_{filename}")
            cv2.imwrite(save_path, annotated_frame)
            print(f"Saved prediction: {save_path}")
