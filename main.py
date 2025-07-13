# main.py
import os
import xml.etree.ElementTree as ET
from tqdm import tqdm
from ultralytics import YOLO
import shutil
import torch

DATA_DIR = "./"
ANNOTATION_DIR = os.path.join(DATA_DIR, "Annotation")
IMAGE_DIR = os.path.join(DATA_DIR, "Images")
YOLO_DATASET_DIR = os.path.join(DATA_DIR, "dog_yolo_dataset")

# Check GPU availability
print("CUDA available:", torch.cuda.is_available())
if torch.cuda.is_available():
    print("GPU:", torch.cuda.get_device_name(0))

# 1. Get breed names and map to class ids
breeds = sorted(os.listdir(ANNOTATION_DIR))
breed_to_id = {breed: idx for idx, breed in enumerate(breeds)}

# 2. Create YOLO format dataset folders
def create_yolo_dataset():
    for split in ['train', 'valid']:
        os.makedirs(os.path.join(YOLO_DATASET_DIR, split, "images"), exist_ok=True)
        os.makedirs(os.path.join(YOLO_DATASET_DIR, split, "labels"), exist_ok=True)

    for breed in tqdm(breeds, desc="Processing Annotations"):
        annot_files = os.listdir(os.path.join(ANNOTATION_DIR, breed))
        split_idx = int(0.8 * len(annot_files))  # 80% train, 20% valid
        for i, annot_file in enumerate(annot_files):
            annot_path = os.path.join(ANNOTATION_DIR, breed, annot_file)
            image_filename = annot_file + ".jpg"
            image_path = os.path.join(IMAGE_DIR, breed, image_filename)

            if not os.path.exists(image_path):
                continue

            try:
                tree = ET.parse(annot_path)
                root = tree.getroot()
            except ET.ParseError:
                continue

            img_width = int(root.find("size/width").text)
            img_height = int(root.find("size/height").text)

            yolo_label = ""
            for obj in root.findall("object"):
                bbox = obj.find("bndbox")
                xmin = int(bbox.find("xmin").text)
                ymin = int(bbox.find("ymin").text)
                xmax = int(bbox.find("xmax").text)
                ymax = int(bbox.find("ymax").text)

                x_center = (xmin + xmax) / 2 / img_width
                y_center = (ymin + ymax) / 2 / img_height
                w = (xmax - xmin) / img_width
                h = (ymax - ymin) / img_height

                class_id = breed_to_id[breed]
                yolo_label += f"{class_id} {x_center:.6f} {y_center:.6f} {w:.6f} {h:.6f}\n"

            split = "train" if i < split_idx else "valid"
            base_name = os.path.splitext(annot_file)[0]
            shutil.copy(image_path, os.path.join(YOLO_DATASET_DIR, split, "images", f"{base_name}.jpg"))
            with open(os.path.join(YOLO_DATASET_DIR, split, "labels", f"{base_name}.txt"), "w") as f:
                f.write(yolo_label)

# 3. Create YOLO data config YAML
def create_yaml():
    with open("dog_config.yaml", "w") as f:
        f.write(f"path: {YOLO_DATASET_DIR}\n")
        f.write("train: train/images\n")
        f.write("val: valid/images\n")
        f.write(f"nc: {len(breeds)}\n")
        f.write("names: [\n")
        for breed in breeds:
            readable_name = breed.split('-')[-1]  # Strip prefix
            f.write(f"  '{readable_name}',\n")
        f.write("]\n")

# 4. Train YOLOv8 model using GPU
def train_yolov8():
    model = YOLO("yolov8n.pt")
    model.train(
        data="dog_config.yaml",
        epochs=50,
        imgsz=640,
        batch=8,
        device="cuda"
    )

if __name__ == "__main__":
    create_yolo_dataset()
    create_yaml()
    train_yolov8()
