# Real-Time-Number-Plate-Face-Detection-using-YOLO
This project implements a real-time object detection system capable of detecting vehicle number plates and human faces using deep learning techniques. The system is built in Python and leverages the power of the YOLO (You Only Look Once) algorithm for fast and accurate detection.

<img width="1911" height="922" alt="image" src="https://github.com/user-attachments/assets/2aa2b663-2ac0-428b-a90b-3e628896eb49" />

Dataset will be provided on request.

Core Functionalities:
Modern, responsive UI with tabs to switch between Number Plate and Face Recognition
Upload image → process → show result (plate text or person name) on the same page
Loading animation during processing
Shared upload folder for both features
Clean separation of concerns (plate logic in main file, face logic in separate module)
Technologies Summary:
•	Backend: Flask (lightweight Python web framework)
•	Frontend: HTML + Tailwind CSS (via CDN) + minimal JavaScript (tabs & loading)
Computer Vision:
•	Plate detection → YOLOv8 (Ultralytics)
•	Plate OCR → EasyOCR
•	Face recognition → face_recognition library (dlib-based 128D embeddings)
•	Image handling: OpenCV, Pillow
•	Deployment Readiness: Local development (run with python carapp.py)

To run this project 
Install required dependencies 
•	Python 3.12.1
•	Flask == 3.0.3
•	Opencv-python==4.10.0.84
•	Ultralytics==8.4.12
•	Easyocr==1.7.1
•	Werkzeug==3.0.3
•	Pillow==10.4.0
•	Scikit-learn==1.5.1
•	Face_recognition==1.3.0
•	Dlib==19.24.6
•	Torch==2.10.0

run carapp.py 
