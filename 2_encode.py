import face_recognition
import os
import pickle
import cv2

# ==================== CONFIGURATION ====================
DATASET_DIR = "Students"
ENCODINGS_FILE = "encodings.pkl"
VALID_EXTENSIONS = ('.jpg', '.jpeg', '.png')

known_encodings = []
known_ids = []

print(f"\n{'='*40}")
print("STARTING ENCODING PROCESS")
print(f"{'='*40}")

if not os.path.exists(DATASET_DIR):
    raise FileNotFoundError(f"Directory '{DATASET_DIR}' not found. Please run capture.py first.")

students = os.listdir(DATASET_DIR)
total_students = len(students)
processed_count = 0

for student_folder in students:
    student_path = os.path.join(DATASET_DIR, student_folder)

    # Skip files that are not directories
    if not os.path.isdir(student_path):
        continue

    print(f"Processing: {student_folder}...")

    # Iterate through images in the student's folder
    for img_name in os.listdir(student_path):
        # Filter for valid image files only
        if not img_name.lower().endswith(VALID_EXTENSIONS):
            continue

        img_path = os.path.join(student_path, img_name)

        # Read and convert image
        image = cv2.imread(img_path)
        if image is None:
            print(f"Warning: Could not read {img_name}, skipping.")
            continue
            
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        # Detect and encode faces
        encs = face_recognition.face_encodings(image_rgb)
        
        if encs:
            known_encodings.append(encs[0])
            known_ids.append(student_folder)
        else:
            print(f"No face found in {img_name}")

    processed_count += 1

# ==================== SAVE DATA ====================
print(f"\nSerializing and saving {len(known_encodings)} encodings...")

with open(ENCODINGS_FILE, "wb") as f:
    pickle.dump((known_encodings, known_ids), f)

print(f"Success! Saved to '{ENCODINGS_FILE}'")
