# prepare_faces_from_csv.py
# Run this once to generate face_encodings.pkl

import os
import pickle
import face_recognition
import pandas as pd
from pathlib import Path


CSV_PATH       = r"C:\Users\HP\Desktop\numberPlateProject\faceData\Dataset.csv"
FACES_DIR      = r"C:\Users\HP\Desktop\numberPlateProject\faceData\faces"           # change if images are in Original Images/
OUTPUT_PKL     = "face_encodings.pkl"
# ────────────────────────────────────────────────

print("Reading CSV...")
df = pd.read_csv(CSV_PATH)

print(f"CSV shape: {df.shape}")
print(f"Columns: {df.columns.tolist()}")
print(f"Unique people: {df['label'].nunique()}")
print(f"Total images in CSV: {len(df)}")

known_encodings = []
known_names = []

missing_files = 0

for idx, row in df.iterrows():
    filename = row['id'].strip()
    name = row['label'].strip().replace("_", " ")  # clean name (optional)

    # Build image path - assume all in Faces/ folder
    img_path = os.path.join(FACES_DIR, filename)

    if not os.path.exists(img_path):
        # Try alternative: maybe images are flat or in subfolder named after person
        # Uncomment/adjust if needed:
        # person_folder = name.replace(" ", "_")
        # img_path = os.path.join(FACES_DIR, person_folder, filename)
        
        print(f"Missing file: {img_path}")
        missing_files += 1
        continue

    try:
        image = face_recognition.load_image_file(img_path)
        face_locations = face_recognition.face_locations(image)

        if not face_locations:
            print(f"No face detected in: {filename}")
            continue

        # Take the first detected face (most datasets have one main face per image)
        encoding = face_recognition.face_encodings(image, face_locations)[0]

        known_encodings.append(encoding)
        known_names.append(name)

        if len(known_encodings) % 50 == 0:
            print(f"Processed {len(known_encodings)} images...")

    except Exception as e:
        print(f"Error processing {filename}: {str(e)}")
        continue

# Save everything
if known_encodings:
    data = {"encodings": known_encodings, "names": known_names}
    with open(OUTPUT_PKL, "wb") as f:
        pickle.dump(data, f)
    print(f"\nSuccess! Saved {len(known_encodings)} encodings for {len(set(known_names))} people")
else:
    print("\nNo valid encodings generated.")

if missing_files > 0:
    print(f"Warning: {missing_files} images from CSV were not found on disk.")