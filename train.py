from ultralytics import YOLO
import torch
model = YOLO("yolov8n.pt")  
results = model.train(
    data="data.yaml",       
    epochs=50,              
    imgsz=640,              
    batch=16,              
    name="license_plate_run",  
    patience=20,            
    device="0" if torch.cuda.is_available() else "cpu", 
)