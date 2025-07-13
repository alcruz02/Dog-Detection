import os
from ultralytics import YOLO
import cv2

# Load your trained model
model = YOLO("runs/detect/train2/weights/best.pt")

# Path to the folder with dog images
folder_path = "dog"
output_folder = "predicted"

# Create output folder if it doesn't exist
os.makedirs(output_folder, exist_ok=True)

# Loop over all image files in the folder
for filename in os.listdir(folder_path):
    if filename.lower().endswith((".jpg", ".jpeg", ".png", ".jp2")):
        img_path = os.path.join(folder_path, filename)

        # Run inference
        results = model(img_path)

        # Save prediction image with boxes
        for r in results:
            # Save the image with predictions drawn
            annotated_frame = r.plot()
            save_path = os.path.join(output_folder, f"pred_{filename}")
            cv2.imwrite(save_path, annotated_frame)
            print(f"Saved prediction: {save_path}")
