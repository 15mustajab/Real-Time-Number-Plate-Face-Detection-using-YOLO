import os
import xml.etree.ElementTree as ET
from pathlib import Path
import shutil
from sklearn.model_selection import train_test_split
IMAGE_DIR = r"C:\Users\HP\Desktop\numberPlateProject\YOLO dataset\train\images"         
XML_DIR   = r"C:\Users\HP\Desktop\numberPlateProject\YOLO dataset\train\labels"     
YOLO_ROOT = r"C:\Users\HP\Desktop\numberPlateProject\yolo_dataset"

CLASSES = ["licence", "license", "License", "Licence", "plate", "number_plate", "carplate"]         
def convert_bbox(size, box):
    """Convert VOC (xmin,ymin,xmax,ymax) → YOLO normalized (x_center, y_center, w, h)"""
    dw = 1. / size[0]
    dh = 1. / size[1]
    x = (box[0] + box[1]) / 2.0 - 1
    y = (box[2] + box[3]) / 2.0 - 1
    w = box[1] - box[0]
    h = box[3] - box[2]
    x = x * dw
    w = w * dw
    y = y * dh
    h = h * dh
    return (x, y, w, h)

def convert_xml_to_txt(xml_path, img_width, img_height):
    tree = ET.parse(xml_path)
    root = tree.getroot()

    labels = []
    for obj in root.iter("object"):
        difficult = obj.find("difficult").text
        if int(difficult) == 1:
            continue

        cls_name = obj.find("name").text.strip()
        if cls_name not in CLASSES:
            continue

        cls_id = CLASSES.index(cls_name)

        xmlbox = obj.find("bndbox")
        b = (
            float(xmlbox.find("xmin").text),
            float(xmlbox.find("xmax").text),
            float(xmlbox.find("ymin").text),
            float(xmlbox.find("ymax").text)
        )
        bb = convert_bbox((img_width, img_height), b)
        labels.append(f"{cls_id} {' '.join([f'{a:.6f}' for a in bb])}")

    return labels


for split in ["train", "val"]:
    for sub in ["images", "labels"]:
        os.makedirs(os.path.join(YOLO_ROOT, split, sub), exist_ok=True)

image_files = [f for f in os.listdir(IMAGE_DIR) if f.lower().endswith((".png", ".jpg", ".jpeg"))]
train_files, val_files = train_test_split(image_files, test_size=0.2, random_state=42)

for split_name, split_list in [("train", train_files), ("val", val_files)]:
    for img_name in split_list:
        img_path = os.path.join(IMAGE_DIR, img_name)
        xml_name = Path(img_name).stem + ".xml"
        xml_path = os.path.join(XML_DIR, xml_name)
        
        if not os.path.exists(xml_path):
            print(f"Warning: No XML for {img_name} → skipping")
            continue
        from PIL import Image
        with Image.open(img_path) as im:
            width, height = im.size
        yolo_labels = convert_xml_to_txt(xml_path, width, height)
        
        if not yolo_labels:
            print(f"Warning: No valid objects in {xml_name}")
            continue  
        dest_img = os.path.join(YOLO_ROOT, split_name, "images", img_name)
        shutil.copy(img_path, dest_img)
        txt_name = Path(img_name).stem + ".txt"
        txt_path = os.path.join(YOLO_ROOT, split_name, "labels", txt_name)
        with open(txt_path, "w") as f:
            f.write("\n".join(yolo_labels))
        
        print(f"Converted: {img_name}")

print("\nConversion finished!")
print(f"Check: {YOLO_ROOT}\\train\\labels and {YOLO_ROOT}\\val\\labels")