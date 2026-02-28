import cv2
import os

# ==================== CONFIGURATION ====================
MAX_IMAGES = 10
SAVE_DIR = "Students"

# ==================== USER INPUT ====================
print(f"\n{'='*40}")
print("NEW STUDENT REGISTRATION")
print(f"{'='*40}")
roll = input("Enter Roll Number: ").strip()
name = input("Enter Name: ").strip()

# Create standard folder path: Students/Roll_Name
folder_name = f"{roll}_{name}"
save_path = os.path.join(SAVE_DIR, folder_name)
os.makedirs(save_path, exist_ok=True)

print(f"\nSaving images to: {save_path}")
print("Press [SPACE] to capture")
print("Press [ESC] to exit\n")

# ==================== CAMERA SETUP ====================
cap = cv2.VideoCapture(0)
if not cap.isOpened():
    raise RuntimeError("Error: Could not open webcam.")

count = 0

# ==================== MAIN LOOP ====================
while True:
    ret, frame = cap.read()
    if not ret:
        print("Error: Failed to grab frame.")
        break
    
    # Optional: Flip for mirror effect
    display_frame = cv2.flip(frame, 1)

    # UI Overlay
    cv2.putText(display_frame, f"Captured: {count}/{MAX_IMAGES}", (20, 40), 
                cv2.FONT_HERSHEY_DUPLEX, 0.8, (0, 255, 0), 2)
    cv2.putText(display_frame, "Press SPACE to Save", (20, 430), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)

    cv2.imshow("Register Student", display_frame)

    key = cv2.waitKey(1) & 0xFF

    # [SPACE] Capture
    if key == 32:  
        img_name = f"{count}.jpg"
        full_path = os.path.join(save_path, img_name)
        
        # Save the original frame (not the one with text)
        cv2.imwrite(full_path, frame)
        print(f"Saved: {img_name}")
        count += 1

    # [ESC] Exit or Auto-exit
    elif key == 27 or count >= MAX_IMAGES:
        if count >= MAX_IMAGES:
            print("\nCollection complete!")
        else:
            print("\nCollection stopped by user.")
        break

cap.release()
cv2.destroyAllWindows()
