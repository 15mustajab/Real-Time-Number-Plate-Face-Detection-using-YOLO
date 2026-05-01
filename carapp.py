from flask import Flask, render_template, request
import cv2
import pytesseract
import os
from werkzeug.utils import secure_filename
from PIL import Image
import numpy as np
from flask import url_for  
import os
import xml.etree.ElementTree as ET
from sklearn.model_selection import train_test_split

def convert_voc_to_yolo(xml_path, img_width, img_height):
    tree = ET.parse(xml_path)
    root = tree.getroot()
    
    labels = []
    for obj in root.iter('object'):
        cls = obj.find('name').text
        if cls != 'license_plate': 
            continue
        cls_id = 0  
        
        bndbox = obj.find('bndbox')
        xmin = float(bndbox.find('xmin').text)
        ymin = float(bndbox.find('ymin').text)
        xmax = float(bndbox.find('xmax').text)
        ymax = float(bndbox.find('ymax').text)
        
        x_center = (xmin + xmax) / 2 / img_width
        y_center = (ymin + ymax) / 2 / img_height
        width = (xmax - xmin) / img_width
        height = (ymax - ymin) / img_height
        
        labels.append(f"{cls_id} {x_center} {y_center} {width} {height}")
    
    return labels
#os.makedirs(r'C:\Users\HP\Desktop\numberPlateProject\train_data', exist_ok=True)
#os.makedirs(r'C:\Users\HP\Desktop\numberPlateProject\train_labels', exist_ok=True)
#os.makedirs(r'C:\Users\HP\Desktop\numberPlateProject\test_data', exist_ok=True)
#os.makedirs(r'C:\Users\HP\Desktop\numberPlateProject\test_labels', exist_ok=True)

image_dir = r'C:\Users\HP\Desktop\numberPlateProject\yolo_dataset\train\images'
annot_dir = r'C:\Users\HP\Desktop\numberPlateProject\yolo_dataset\train\labels'
images = [f for f in os.listdir(image_dir) if f.endswith('.png')]

train_images, val_images = train_test_split(images, test_size=0.2, random_state=42)

for split, split_dir in zip([train_images, val_images], ['train', 'val']):
    for img_file in split:
        img_path = os.path.join(image_dir, img_file)
        xml_file = img_file.replace('.png', '.xml')
        xml_path = os.path.join(annot_dir, xml_file)
        
        if not os.path.exists(xml_path):
            continue
        
        from PIL import Image
        img = Image.open(img_path)
        img_width, img_height = img.size
        
        labels = convert_voc_to_yolo(xml_path, img_width, img_height)
        
        if labels:
            os.system(f"cp {img_path} yolo_dataset/{split_dir}/images/")
            label_path = os.path.join(f"yolo_dataset/{split_dir}/labels", img_file.replace('.png', '.txt'))
            with open(label_path, 'w') as f:
                f.write('\n'.join(labels))

with open('data.yaml', 'w') as f:
    f.write("""
train: yolo_dataset/train/images
val: yolo_dataset/val/images

nc: 1
names: ['license_plate']
""")
from flask import send_from_directory
from flask import Flask, render_template, request
import cv2
from ultralytics import YOLO
import easyocr
import os
from werkzeug.utils import secure_filename
import numpy as np
import face
import face_recognition
app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'uploads'
# ← Add these two lines right here
if not os.path.exists(app.config['UPLOAD_FOLDER']):
    os.makedirs(app.config['UPLOAD_FOLDER'])

@app.route('/uploads/<filename>')
def uploaded_file(filename):
    return send_from_directory(app.config['UPLOAD_FOLDER'], filename)

model = YOLO(r'C:\Users\HP\Desktop\numberPlateProject\runs\detect\license_plate_run7\weights\best.pt')  # relative path — safest if you're running from project root

reader = easyocr.Reader(['en'], gpu=False)  
if not os.path.exists(app.config['UPLOAD_FOLDER']):
    os.makedirs(app.config['UPLOAD_FOLDER'])
face.init_face_recognition("face_encodings.pkl")
face.register_face_routes(app, app.config['UPLOAD_FOLDER'])
def detect_and_recognize(image_path):
   
    img = cv2.imread(image_path)
    results = model(img)
    plate_text = None
    for result in results:
        boxes = result.boxes
        for box in boxes:
            
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            conf = box.conf[0]
            if conf > 0.5:  
                
                plate_img = img[y1:y2, x1:x2]
            
                plate_rgb = cv2.cvtColor(plate_img, cv2.COLOR_BGR2RGB)
                
                # Recognize text
                ocr_results = reader.readtext(plate_rgb)
                text = ' '.join([res[1] for res in ocr_results if res[2] > 0.2])  # Filter low conf
                plate_text = text.strip().upper()  # Clean up
                
                # For simplicity, assume one plate per image
                return plate_text
    
    return plate_text

@app.route('/', methods=['GET', 'POST'])
def index():
    plate_text = None
    uploaded_image = None
    
    if request.method == 'POST':
        if 'image' not in request.files:
            return render_template('index.html')
        
        file = request.files['image']
        if file.filename == '':
            return render_template('index.html', plate_text=plate_text, uploaded_image=uploaded_image)
        
        filename = secure_filename(file.filename)
        file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(file_path)
        
        # Detect and recognize
        plate_text = detect_and_recognize(file_path)
        
        uploaded_image = f'/{app.config["UPLOAD_FOLDER"]}/{filename}'
    
    return render_template('index.html', plate_text=plate_text, uploaded_image=uploaded_image)
@app.route('/car', methods=['GET', 'POST'])
def car_recognition():
    plate_text = None
    uploaded_image = None

    if request.method == 'POST':
        if 'image' not in request.files:
            return render_template('car.html', error="No file selected")

        file = request.files['image']
        if file.filename == '':
            return render_template('car.html', error="No file selected")

        filename = secure_filename(file.filename)
        file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(file_path)

        plate_text = detect_and_recognize(file_path)
        uploaded_image = f'/uploads/{filename}'

    return render_template('car.html', plate_text=plate_text, uploaded_image=uploaded_image)

if __name__ == '__main__':
    app.run(debug=True)