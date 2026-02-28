FACE RECOGNITION BASED ATTENDANCE MARKING SYSTEM

An automated, real-time face recognition-based attendance system designed to eliminate proxy attendance while maximizing classroom teaching time.

--------------------------------------------------

PROBLEM STATEMENT

In traditional classroom environments, teachers face a difficult choice:

- Circulated attendance sheets: Students mark proxy attendance for absent friends.
- Manual roll calls: Consumes 5-10 minutes of valuable class time (longer for larger classes).

Both approaches reduce effective teaching time and impact classroom productivity.

Our Solution: A touchless, automated attendance system that runs silently in the background, allowing teachers to focus entirely on teaching.

--------------------------------------------------

KEY FEATURES

- Real-time Face Recognition: Automatically detects and identifies students via webcam.
- Smart Timer System: Tracks cumulative presence time per student.
- Customizable Threshold: Set minimum required presence duration (e.g., 45 mins out of 60).
- Anti-Proxy: Only physically present students are marked.
- Flexible Breaks: Students can leave for washroom without losing attendance.
- Excel Export: Auto-generates/modifies CSV attendance records with timestamps.

--------------------------------------------------

HOW IT WORKS

┌─────────────────┐     ┌─────────────────┐     ┌─────────────────┐
│  Camera Feed    │────▶│  Face Detection │────▶│  Recognition    │
│  (Real-time)    │     │  (TFLite Model) │     │  (Encoding Match)│
└─────────────────┘     └─────────────────┘     └─────────────────┘
│
┌─────────────────────────┘
▼
┌─────────────────┐
│  Student Timer  │◄── Threshold: 45 min
│  (Cumulative)   │
└─────────────────┘
│
┌───────────┴───────────┐
▼                       ▼
[Timer >= Threshold]    [Timer < Threshold]
│                       │
▼                       ▼
Marked: PRESENT         Marked: ABSENT

--------------------------------------------------

PREREQUISITES

- Python 3.10
- Windows 10/11 (64-bit)
- Webcam (built-in or external)
- Good lighting conditions in classroom

--------------------------------------------------

INSTALLATION

It is highly recommended to create a virtual environment before installing the dependencies to keep your project isolated and prevent conflicts.

Step 1: Create and Activate a Virtual Environment
Command to create: python -m venv venv
Command to activate (Windows): venv\Scripts\activate

Step 2: Install dlib (Pre-built Wheel)
Important: The included dlib wheel is compiled for Python 3.10 on Windows 64-bit.

Command: pip install dlib-19.22.99-cp310-cp310-win_amd64.whl
(For other Python versions, download the appropriate wheel from dlib-wheels releases.)

Step 3: Install Dependencies
Command: pip install -r requirements.txt

--------------------------------------------------

USAGE GUIDE

Step 1: Student Registration (1_capture.py)
Capture face images for each student. One-time setup.
Command: python 1_capture.py

Instructions:
- Enter Roll Number and Name when prompted.
- Press SPACE to capture each photo (10 photos required).
- Press ESC to finish early.
- Images saved to: Students/RollNumber_Name/

Alternative: Students can submit photos in advance:
- Folder format: RollNumber_Name (e.g., 101_Rahul)
- Image names: 0.jpg, 1.jpg, 2.jpg ... 9.jpg
- Supported formats: .jpg, .jpeg, .png

Step 2: Model Training (2_encode.py)
Generate face encodings from registered images.
Command: python 2_encode.py

When to run:
- First-time setup
- New student added
- Student photos updated
- Output: encodings.pkl (used by attendance script)

Step 3: Real-time Attendance (3_attendance_timer.py)
Start automated attendance tracking.
Command: python 3_attendance_timer.py

Configuration:
Edit these values in 3_attendance_timer.py:
CONFIG = {
    "PRESENT_TIME": 45 * 60,    # Minimum seconds for attendance (default: 45 min)
}

During Class:
System runs automatically in background.
- Green box = Recognized student (marked present)
- Orange box = Recognized, timer running
- Red box = Unknown person

Controls:
Press Q to end class and save attendance.
Output: attendance_register.csv with date-wise columns.

--------------------------------------------------

ATTENDANCE OUTPUT FORMAT

Name      2024-03-01_09-00    2024-03-01_14-00    2024-03-02_09-00
Rahul     PRESENT             ABSENT              PRESENT
Priya     PRESENT             PRESENT             PRESENT
Amit      ABSENT              PRESENT             ABSENT

- New column auto-added for each session.
- Historical data preserved across sessions.

--------------------------------------------------

MODEL VALIDATION

The system was validated using public figures to ensure recognition accuracy:

Test Subject          Recognition Accuracy
Ajay Devgn            High
Akshay Kumar          High
Aamir Khan            High
Shraddha Kapoor       High

--------------------------------------------------

SYSTEM ARCHITECTURE

Component             Technology
Face Detection        TensorFlow Lite (detect.tflite)
Face Recognition      face_recognition library (dlib)
Image Processing      OpenCV
Model Format          128-d face encoding
Data Storage          CSV (Excel-compatible)

--------------------------------------------------

IMPORTANT NOTES

- Lighting: Ensure consistent, bright lighting for best accuracy.
- Camera Position: Place webcam to capture faces frontally.
- Privacy: All student data stored locally; no cloud upload.
- Backup: Regularly backup Students/ folder and encodings.pkl.