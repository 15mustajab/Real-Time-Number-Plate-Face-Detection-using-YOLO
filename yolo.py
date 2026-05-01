import os
import xml.etree.ElementTree as ET
from pathlib import Path
import shutil
from sklearn.model_selection import train_test_split
from PIL import Image

IMAGE_DIR = r"C:\Users\HP\Desktop\numberPlateProject\archive\images"
XML_DIR   = r"C:\Users\HP\Desktop\numberPlateProject\archive\annotations"  # ← confirm this path!
YOLO_ROOT = r"C:\Users\HP\Desktop\numberPlateProject\yolo_dataset"

CLASSES = ["licence"]

print(f"Checking input folders:")
print(f"  Images: {IMAGE_DIR} → exists? {os.path.exists(IMAGE_DIR)}")
print(f"  XMLs:   {XML_DIR}   → exists? {os.path.exists(XML_DIR)}")

if not os.path.exists(IMAGE_DIR) or not os.path.exists(XML_DIR):
    print("ERROR: One or both input folders missing. Fix paths above.")
    exit(1)

image_files = [f for f in os.listdir(IMAGE_DIR) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
print(f"Found {len(image_files)} images")

if len(image_files) == 0:
    print("ERROR: No images found. Check IMAGE_DIR path.")
    exit(1)

# Create output structure
for split in ['train', 'val']:
    for sub in ['images', 'labels']:
        p = os.path.join(YOLO_ROOT, split, sub)
        os.makedirs(p, exist_ok=True)
        print(f"Created/checked: {p}")

train_files, val_files = train_test_split(image_files, test_size=0.2, random_state=42)
print(f"Train: {len(train_files)} images | Val: {len(val_files)} images")

def xml_to_yolo(xml_path, w, h):
    tree = ET.parse(xml_path)
    root = tree.getroot()
    labels = []
    for obj in root.iter('object'):
        name = obj.find('name').text.strip()
        if name not in CLASSES:
            continue
        cls_id = CLASSES.index(name)
        bndbox = obj.find('bndbox')
        xmin = float(bndbox.find('xmin').text)
        xmax = float(bndbox.find('xmax').text)
        ymin = float(bndbox.find('ymin').text)
        ymax = float(bndbox.find('ymax').text)
        x = (xmin + xmax) / 2 / w
        y = (ymin + ymax) / 2 / h
        bw = (xmax - xmin) / w
        bh = (ymax - ymin) / h
        if 0 < x < 1 and 0 < y < 1 and 0 < bw < 1 and 0 < bh < 1:
            labels.append(f"{cls_id} {x:.6f} {y:.6f} {bw:.6f} {bh:.6f}")
    return labels

for split, files in [('train', train_files), ('val', val_files)]:
    print(f"\n--- Processing {split} ({len(files)} files) ---")
    count_ok = 0
    for fname in files:
        img_path = os.path.join(IMAGE_DIR, fname)
        xml_path = os.path.join(XML_DIR, Path(fname).stem + ".xml")

        if not os.path.exists(xml_path):
            print(f"  Skip {fname} → no XML")
            continue

        try:
            img = Image.open(img_path)
            w, h = img.size
            img.close()
        except Exception as e:
            print(f"  Skip {fname} → image error: {e}")
            continue

        lines = xml_to_yolo(xml_path, w, h)
        if not lines:
            print(f"  Skip {fname} → no valid 'licence' objects")
            continue

        # Copy image
        shutil.copy(img_path, os.path.join(YOLO_ROOT, split, 'images', fname))

        # Write label
        txt_path = os.path.join(YOLO_ROOT, split, 'labels', Path(fname).stem + ".txt")
        with open(txt_path, 'w') as f:
            f.write('\n'.join(lines))

        count_ok += 1
        if count_ok % 20 == 0:
            print(f"  OK so far: {count_ok}")

    print(f"{split} finished: {count_ok} valid image/label pairs created")

print("\nFinal folder counts:")
for split in ['train', 'val']:
    for sub in ['images', 'labels']:
        p = os.path.join(YOLO_ROOT, split, sub)
        cnt = len(os.listdir(p)) if os.path.exists(p) else 0
        print(f"  {split}/{sub}: {cnt} files")