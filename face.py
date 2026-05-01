import pickle
import face_recognition
import numpy as np
from flask import Flask, render_template, request
import cv2
import pytesseract
import os
from werkzeug.utils import secure_filename
from PIL import Image
import numpy as np
import xml.etree.ElementTree as ET
from sklearn.model_selection import train_test_split
import pickle
import face_recognition
import numpy as np
from flask import render_template, request, send_from_directory

# Global variables (will be set from main app)
known_encodings = []
known_names = []

def init_face_recognition(encodings_file="face_encodings.pkl"):
    """
    Load pre-computed face encodings once at startup.
    Call this from the main app.
    """
    global known_encodings, known_names
    
    try:
        with open(encodings_file, "rb") as f:
            face_data = pickle.load(f)
        known_encodings = face_data["encodings"]
        known_names = face_data["names"]
        print(f"[FACE] Loaded {len(known_encodings)} face encodings for {len(set(known_names))} people")
    except FileNotFoundError:
        print("[FACE] face_encodings.pkl not found. Run prepare_faces.py first!")
        known_encodings = []
        known_names = []
    except Exception as e:
        print(f"[FACE] Error loading encodings: {e}")
        known_encodings = []
        known_names = []

def register_face_routes(app, upload_folder):
    """
    Registers the /face route and any related helpers to the Flask app.
    Pass the app instance and upload folder path.
    """
    
    @app.route('/face', methods=['GET', 'POST'])
    def face_recognition_page():
        recognized_name = None
        uploaded_face_img = None

        if request.method == 'POST':
            if 'face_image' not in request.files:
                return render_template('face.html')

            file = request.files['face_image']
            if file.filename == '':
                return render_template('face.html')

            filename = secure_filename(file.filename)
            file_path = os.path.join(upload_folder, filename)
            file.save(file_path)

            # Process face recognition
            try:
                image = face_recognition.load_image_file(file_path)
                face_locations = face_recognition.face_locations(image)

                if not face_locations:
                    recognized_name = "No face detected"
                else:
                    face_enc = face_recognition.face_encodings(image, face_locations)
                    if not face_enc:
                        recognized_name = "Could not encode face"
                    else:
                        # Compare with known faces
                        matches = face_recognition.compare_faces(
                            known_encodings, 
                            face_enc[0], 
                            tolerance=0.5
                        )
                        name = "Unknown"

                        if True in matches:
                            first_match_index = matches.index(True)
                            name = known_names[first_match_index]

                        # Alternative: best match by distance
                        # face_distances = face_recognition.face_distance(known_encodings, face_enc[0])
                        # best_match_index = np.argmin(face_distances)
                        # if matches[best_match_index]:
                        #     name = known_names[best_match_index]

                        recognized_name = name

                uploaded_face_img = f'/uploads/{filename}'

            except Exception as e:
                recognized_name = f"Error processing image: {str(e)}"

        return render_template('face.html',
                             recognized_name=recognized_name,
                             uploaded_face_img=uploaded_face_img)

   