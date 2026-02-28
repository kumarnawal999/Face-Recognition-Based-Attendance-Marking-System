import cv2
import face_recognition
import pickle
import time
import os
import csv
import numpy as np
import tensorflow as tf
from datetime import datetime, timedelta

# ==================== CONFIGURATION ====================
CONFIG = {
    "PRESENT_TIME": 20,          # Seconds required to mark present
    "TOLERANCE": 0.45,           # Face recognition strictness (Lower is stricter)
    "MODEL_PATH": "detect.tflite",
    "CONFIDENCE_THRESHOLD": 0.5, # TFLite Person detection threshold
    "ENCODINGS_FILE": "encodings.pkl",
    "CSV_FILE": "attendance_register.csv",
    "CAMERA_ID": 0,
    "FRAME_WIDTH": 640,
    "FRAME_HEIGHT": 480
}

# ==================== UTILITY FUNCTIONS ====================
def save_attendance(attendance_dict, unique_ids):
    """Saves attendance data to CSV in column-wise format."""
    csv_file = CONFIG["CSV_FILE"]
    session_col = datetime.now().strftime("%Y-%m-%d_%H-%M")
    data = {}

    # Read existing data to preserve history
    if os.path.exists(csv_file):
        with open(csv_file, newline="") as f:
            reader = csv.DictReader(f)
            fieldnames = list(reader.fieldnames) if reader.fieldnames else ["Name"]
            for row in reader:
                data[row["Name"]] = row
    else:
        fieldnames = ["Name"]

    # Add new column if it doesn't exist
    if session_col not in fieldnames:
        fieldnames.append(session_col)

    # Update data for current session
    for student in unique_ids:
        # Extract Name from "Roll_Name"
        _, name = student.split("_", 1)
        
        if name not in data:
            data[name] = {"Name": name}
            
        status = "PRESENT" if attendance_dict.get(student, False) else "ABSENT"
        data[name][session_col] = status

    # Write back to CSV
    with open(csv_file, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in data.values():
            writer.writerow(row)
            
    print(f"Attendance saved to: {csv_file}")

# ==================== INITIALIZATION ====================
print(f"\n{'='*70}")
print(f"SMART ATTENDANCE SYSTEM (TFLite + FaceRec)")
print(f"{'='*70}\n")

# 1. Load Encodings
if not os.path.exists(CONFIG["ENCODINGS_FILE"]):
    raise FileNotFoundError(f"Error: {CONFIG['ENCODINGS_FILE']} not found. Run encode.py first.")

with open(CONFIG["ENCODINGS_FILE"], "rb") as f:
    # lookup_ids contains one ID per image (can have duplicates)
    known_encodings, lookup_ids = pickle.load(f)

# Create a UNIQUE list for tracking timers (separate from lookup list)
unique_ids = list(dict.fromkeys(lookup_ids))

print(f"Loaded {len(lookup_ids)} encodings for {len(unique_ids)} students.")
print(f"Students: {', '.join([x.split('_')[1] for x in unique_ids])}")

# 2. Load TFLite Model
if not os.path.exists(CONFIG["MODEL_PATH"]):
    raise FileNotFoundError(f"Error: {CONFIG['MODEL_PATH']} not found.")

print("Loading TFLite model...")
interpreter = tf.lite.Interpreter(model_path=CONFIG["MODEL_PATH"])
interpreter.allocate_tensors()

input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()
input_shape = input_details[0]['shape']
print("TFLite Model Active")

# 3. Setup Camera
cap = cv2.VideoCapture(CONFIG["CAMERA_ID"])
cap.set(cv2.CAP_PROP_FRAME_WIDTH, CONFIG["FRAME_WIDTH"])
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, CONFIG["FRAME_HEIGHT"])
cap.set(cv2.CAP_PROP_FPS, 30)

if not cap.isOpened():
    raise RuntimeError("Cannot open camera!")
print("Camera Initialized\n")

# ==================== RUNTIME VARIABLES ====================
# Use unique_ids for the state dictionaries
timers = {s: 0.0 for s in unique_ids}
last_seen = {s: None for s in unique_ids}
marked = {s: False for s in unique_ids}

class_start_time = time.time()
fps_tracker = {'count': 0, 'time': time.time(), 'fps': 0}

print(f"{'='*70}")
print(f"SYSTEM LIVE | Required Time: {CONFIG['PRESENT_TIME']}s")
print(f"Press 'Q' to quit and save.")
print(f"{'='*70}\n")

# ==================== MAIN LOOP ====================
try:
    while True:
        ret, frame = cap.read()
        if not ret:
            print("Frame capture failed")
            break

        frame = cv2.flip(frame, 1)
        now = time.time()

        # Update FPS
        fps_tracker['count'] += 1
        if now - fps_tracker['time'] >= 1.0:
            fps_tracker['fps'] = fps_tracker['count'] / (now - fps_tracker['time'])
            fps_tracker['count'] = 0
            fps_tracker['time'] = now

        # ---------------------------------------------------------
        # STAGE 1: PERSON DETECTION (TFLite)
        # ---------------------------------------------------------
        person_detected = False
        
        # Preprocess
        resized = cv2.resize(frame, (input_shape[1], input_shape[2]))
        input_data = np.expand_dims(resized, axis=0)
        
        if input_details[0]['dtype'] == np.float32:
            input_data = (np.float32(input_data) - 127.5) / 127.5
        
        # Inference
        interpreter.set_tensor(input_details[0]['index'], input_data)
        interpreter.invoke()

        # Parse Results
        # Indices: 0=Boxes, 1=Classes, 2=Scores (Typical for SSD MobileNet)
        classes = interpreter.get_tensor(output_details[1]['index'])[0]
        scores = interpreter.get_tensor(output_details[2]['index'])[0]
        
        # Check for 'Person' (Class 0)
        for i in range(len(scores)):
            if scores[i] > CONFIG["CONFIDENCE_THRESHOLD"] and int(classes[i]) == 0:
                person_detected = True
                break

        # ---------------------------------------------------------
        # STAGE 2: FACE RECOGNITION (Conditional)
        # ---------------------------------------------------------
        face_info_to_draw = []
        currently_visible = {s: False for s in unique_ids}

        if person_detected:
            # Convert for Face Recognition Lib (BGR -> RGB)
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            
            face_locs = face_recognition.face_locations(rgb, model="hog")
            face_encs = face_recognition.face_encodings(rgb, face_locs)
            
            for encoding, loc in zip(face_encs, face_locs):
                matches = face_recognition.compare_faces(
                    known_encodings, encoding, tolerance=CONFIG["TOLERANCE"]
                )
                
                name = "Unknown"
                top, right, bottom, left = loc
                
                if True in matches:
                    match_index = matches.index(True)
                    
                    # FIX: Use lookup_ids (full list) to find the ID
                    student_id = lookup_ids[match_index] 
                    _, name = student_id.split("_", 1) # Display Name
                    
                    currently_visible[student_id] = True
                    
                    # Update Timer
                    if last_seen[student_id] is not None:
                        timers[student_id] += (now - last_seen[student_id])
                    last_seen[student_id] = now
                    
                    # Mark Present
                    if timers[student_id] >= CONFIG["PRESENT_TIME"] and not marked[student_id]:
                        marked[student_id] = True
                        print(f"{name} marked PRESENT")

                face_info_to_draw.append((left, top, right-left, bottom-top, name))

        # Reset timers for students who left the frame
        for s in unique_ids:
            if not currently_visible[s]:
                last_seen[s] = None

        # ---------------------------------------------------------
        # STAGE 3: UI DRAWING
        # ---------------------------------------------------------
        display = frame.copy()
        
        # Header
        cv2.rectangle(display, (0, 0), (640, 90), (30, 30, 30), -1)
        
        elapsed_str = str(timedelta(seconds=int(now - class_start_time)))
        status_text = "SCANNING..." if person_detected else "IDLE"
        status_color = (0, 255, 0) if person_detected else (100, 100, 100)
        
        cv2.putText(display, f"Time: {elapsed_str} | FPS: {fps_tracker['fps']:.1f}", (10, 25), 
                    cv2.FONT_HERSHEY_DUPLEX, 0.5, (255, 255, 0), 1)
        cv2.putText(display, f"Status: {status_text}", (10, 50), 
                    cv2.FONT_HERSHEY_DUPLEX, 0.6, status_color, 1)
        
        present_count = sum(marked.values())
        cv2.putText(display, f"Attendance: {present_count}/{len(unique_ids)}", (10, 75), 
                    cv2.FONT_HERSHEY_DUPLEX, 0.5, (255, 255, 255), 1)

        # Draw Faces
        for (x, y, w, h, name) in face_info_to_draw:
            color = (0, 0, 255) # Red for unknown
            display_status = ""
            
            if name != "Unknown":
                # Find ID from Name (Reverse lookup for display logic)
                s_id = next((s for s in unique_ids if s.endswith(f"_{name}")), None)
                if s_id:
                    is_marked = marked[s_id]
                    color = (0, 255, 0) if is_marked else (0, 165, 255) # Green vs Orange
                    display_status = "PRESENT" if is_marked else f"{int(timers[s_id])}s"

            cv2.rectangle(display, (x, y), (x+w, y+h), color, 2)
            cv2.putText(display, f"{name} {display_status}", (x, y-10), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

        # Side Panel (Status)
        panel_x = 480
        start_y = 110
        cv2.putText(display, "STATUS:", (panel_x, start_y), cv2.FONT_HERSHEY_DUPLEX, 0.5, (255, 255, 0), 1)
        
        for i, s_id in enumerate(unique_ids):
            _, s_name = s_id.split("_", 1)
            text_color = (0, 255, 0) if marked[s_id] else (150, 150, 150)
            status_symbol = "P" if marked[s_id] else f"{int(timers[s_id])}s"
            
            cv2.putText(display, f"{s_name[:8]}: {status_symbol}", 
                        (panel_x, start_y + 25 * (i + 1)), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.45, text_color, 1)

        cv2.imshow("Real Time Attendance", display)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            print("\nStop signal received.")
            break

except KeyboardInterrupt:
    print("\nKeyboard Interrupt.")

finally:
    cap.release()
    cv2.destroyAllWindows()
    
    # ---------------------------------------------------------
    # STAGE 4: SAVE & SUMMARY
    # ---------------------------------------------------------
    print(f"\n{'='*70}")
    print(f"SESSION COMPLETE")
    print(f"{'='*70}")
    
    # Check if marked dictionary is populated
    if 'marked' in locals() and 'unique_ids' in locals():
        save_attendance(marked, unique_ids)
        
        print(f"\nSUMMARY:")
        print(f"Total Duration: {str(timedelta(seconds=int(time.time() - class_start_time)))}")
        print(f"Present: {sum(marked.values())}")
        print(f"Absent: {len(unique_ids) - sum(marked.values())}")
    else:
        print("System stopped before initialization was complete.")
        
    print(f"{'='*70}\n")
