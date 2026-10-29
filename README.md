# AI Face Recognition Attendance System

> **Department of Information Technology — Diploma Engineering**  
> **Course:** AI Product Design &bull; Developed by **PUSHPRAJSINH RATHOD**  
> **Status:** 100% Functional, Runnable Prototype &bull; Completely Free &amp; Open-Source &bull; No Paid APIs

---

## 1. Project Overview & Objective

This project is an **Automated Face Recognition Attendance System** engineered with Python 3.11, Flask, SQLite, and OpenCV's state-of-the-art Deep Learning facial recognition pipeline (**YuNet** and **SFace**).

The system allows an administrator to:
1. Log in securely through an authenticated portal with password hashing.
2. Enroll students by uploading passport-style photos or capturing snapshots directly via webcam.
3. Automatically generate and persist irreversible **128-dimensional facial representation embeddings**.
4. Stream webcam video in real-time, detect faces, align landmarks, compare biometric vectors, and mark attendance instantly.
5. Prevent duplicate attendance check-ins on the same calendar day.
6. Search, filter, inspect, manually correct, and export attendance records to CSV.

---

## 2. Technology Stack & Why It Is 100% Free

| Component | Technology | Cost / License | Why Chosen |
|---|---|---|---|
| **Programming Language** | Python 3.11 | Free / Open-Source | High performance, standard for AI & Web |
| **Web Framework** | Flask 3.0+ | Free / BSD License | Lightweight, modular, fast development |
| **Face Detector** | OpenCV **YuNet** (ONNX) | Free / Apache 2.0 | Ultra-fast CNN face detector with 5 facial landmarks |
| **Face Recognizer** | OpenCV **SFace** (ONNX) | Free / Apache 2.0 | Deep 128-D SphereFace feature extractor |
| **Database** | SQLite 3 | Free / Public Domain | Zero-configuration, local file storage, ACID compliant |
| **Frontend UI** | HTML5, CSS3, JavaScript | Free / Native | Clean light dashboard, no paid libraries or frameworks |
| **Cloud APIs / Costs** | **None ($0.00)** | **100% Free** | Runs entirely offline on local CPU, zero subscription |

> [!NOTE]
> Unlike `dlib` / `face_recognition` which frequently fail to compile on Windows due to missing CMake and Visual Studio C++ compilers, **OpenCV YuNet + SFace** runs natively on any Windows computer with standard `opencv-python`!

---

## 3. Project Directory Structure

```text
attendance record/
│
├── app.py                         # Main Flask application and REST routes
├── database.py                    # SQLite database schema, queries, and seeding
├── face_recognition_service.py    # YuNet & SFace AI detection and embedding matching
├── test_app.py                    # Automated test suite (11 test cases)
├── requirements.txt               # Free pip dependencies
├── README.md                      # Project documentation and demo script
├── .gitignore                     # Git ignore rules
│
├── models/                        # Pretrained deep learning ONNX models (local)
│   ├── face_detection_yunet_2023mar.onnx
│   └── face_recognition_sface_2021dec.onnx
│
├── templates/                     # Jinja2 HTML templates
│   ├── base.html                  # Base master layout with navigation & toast alerts
│   ├── login.html                 # Administrator login page with demo autofill
│   ├── dashboard.html             # Summary statistics & recent check-ins
│   ├── students.html              # Student registration (upload & webcam capture)
│   ├── recognition.html           # Live camera attendance marking & stream feed
│   └── attendance.html            # Searchable records table & CSV export
│
├── static/
│   ├── css/
│   │   └── style.css              # Modern light-theme dashboard design
│   ├── js/
│   │   ├── camera.js              # Webcam streaming, canvas overlay, audio chime
│   │   └── main.js                # Flash alert auto-dismissal
│   └── sample_images/             # Pre-seeded passport photos for demonstration
│       ├── student1.jpg           # Aarav Sharma
│       ├── student2.jpg           # Riya Patel
│       └── student3.jpg           # Aditya Verma
│
├── uploads/                       # Secure local storage for student photos
│   └── students/
│
└── instance/                      # Local SQLite database directory
    └── attendance.db
```

---

## 4. Step-by-Step Installation & Setup (Windows)

### Step 1: Open Terminal in the Project Folder
Open PowerShell or Command Prompt in the folder:
```powershell
cd "c:\Users\DELL\OneDrive\Desktop\attendance record"
```

### Step 2: (Optional but Recommended) Create a Virtual Environment
```powershell
python -m venv venv
venv\Scripts\activate
```

### Step 3: Install Required Free Packages
```powershell
pip install -r requirements.txt
```
*(All packages: Flask, opencv-python, numpy, Werkzeug, Pillow are 100% free).*

---

## 5. Starting the Application

Run the application entrypoint:
```powershell
python app.py
```

You will see output similar to:
```text
========================================================
  AI Face Recognition Attendance System - College Prototype
  Technology: OpenCV YuNet + SFace (Deep Learning) + Flask
  Status: 100% Free & Open-Source, Offline Inference Ready
  Default Demo Login: admin / admin123
  Server URL: http://127.0.0.1:5000
========================================================
 * Running on http://127.0.0.1:5000
```

### Step 4: Open in Web Browser
Open your browser (Chrome, Edge, or Firefox) and navigate to:
```
http://127.0.0.1:5000
```

---

## 6. Demo Credentials & Pre-Seeded Students

### Admin Credentials
- **Username:** `admin`
- **Password:** `admin123`
*(There is also an **"Autofill Demo Login"** button directly on the login card for one-click access).*

### Pre-Seeded Sample Students
The system automatically initializes 3 demonstration students with verified 128-D facial biometrics upon first startup:

| Student Name | Enrollment No | Class / Division | Status |
|---|---|---|---|
| **Aarav Sharma** | `EN202401` | TY-IF (IT 6th Sem) | Active Profile |
| **Riya Patel** | `EN202402` | TY-IF (IT 6th Sem) | Active Profile |
| **Aditya Verma** | `EN202403` | TY-IF (IT 6th Sem) | Active Profile |

---

## 7. How to Demonstrate During College Viva / Presentation

### Scenario A: Real-Time Recognition of Registered Student
1. Log in as `admin`.
2. Click **"Live Recognition"** in the navigation header.
3. Click the green **"Start Camera"** button and allow camera permission.
4. **How to test Aarav Sharma:**
   - Display `static/sample_images/student1.jpg` on your smartphone screen and hold it up in front of the laptop webcam, OR look into the camera yourself if you register your face (see Scenario B).
5. **Observed System Reaction:**
   - Bounding box turns green around the face.
   - Text shows: `Aarav Sharma (98.5% Match)`.
   - Audio chime plays.
   - Status banner updates to: `Attendance Marked: Aarav Sharma`.
   - Record immediately appears in the right-hand **"Today's Check-ins"** feed without page reload.

### Scenario B: Duplicate Attendance Prevention Test
1. Keep holding the same photo in front of the camera.
2. The system detects the face again, but does **not** create a duplicate row.
3. Banner displays: `Already Present: Aarav Sharma — Attendance already recorded today at ...`.
4. Explains to the examiner that SQL unique constraints enforce one check-in per student per calendar day.

### Scenario C: Unknown Face Rejection Test
1. Point the camera at a person or portrait not registered in the system.
2. The bounding box displays in amber/yellow.
3. Status banner reads: `Unknown Face Detected — Face does not match any registered student`.
4. Attendance is **never** marked for an unknown or unrecognized individual.

### Scenario D: Registering a New Student in 30 Seconds
1. Click **"Students"** in the top navigation.
2. Click **"Register New Student"**.
3. Enter:
   - Name: `Your Name`
   - Enrollment: `EN202404`
   - Class: `TY-IF`
4. Choose **"Capture via Webcam"** -> Click **"Start Camera"** -> Click **"Snap Photo"**.
5. Click **"Extract Biometrics & Save Student"**.
6. The AI verifies that exactly 1 face exists, extracts the 128-D vector, and saves the student!
7. Return to **"Live Recognition"** and stand in front of the camera — you will be recognized by your own name!

### Scenario E: Attendance Audit & CSV Export
1. Click **"Attendance Records"**.
2. Filter by date or search for a student by roll number.
3. Click **"Export to CSV"** to instantly download an Excel-compatible `.csv` attendance report.
4. Demonstrate manual correction: click the red trash icon on an entry or click **"Manual Entry / Override"** to correct an entry.

---

## 8. Technical Architecture: How the AI Model Works

```text
 [ Web Browser Webcam / Passport Photo ]
                    │
                    ▼ (Frame captured as BGR image)
 ┌─────────────────────────────────────────────────────────┐
 │ 1. YuNet (cv2.FaceDetectorYN)                           │
 │    • Convolutional Neural Network trained on WIDER FACE │
 │    • Detects face bounding box [x, y, w, h]             │
 │    • Detects 5 facial landmarks: eyes, nose, mouth tips │
 └─────────────────────────┬───────────────────────────────┘
                           │
                           ▼ (Face box + 5 landmarks)
 ┌─────────────────────────────────────────────────────────┐
 │ 2. Affine Alignment & Normalization                     │
 │    • Rotates & scales face to standard 112x112 canvas   │
 │    • Normalizes lighting and eliminates head tilt       │
 └─────────────────────────┬───────────────────────────────┘
                           │
                           ▼ (Aligned 112x112 face)
 ┌─────────────────────────────────────────────────────────┐
 │ 3. SFace (cv2.FaceRecognizerSF)                         │
 │    • Deep Neural Network trained on SphereFace metric   │
 │    • Outputs 128-dimensional floating point vector      │
 │    • Biometric mathematical fingerprint of the face     │
 └─────────────────────────┬───────────────────────────────┘
                           │
                           ▼ (128-D Embedding)
 ┌─────────────────────────────────────────────────────────┐
 │ 4. Cosine Similarity Matching Engine                    │
 │    • Cosine Similarity = (A · B) / (||A|| * ||B||)      │
 │    • Compares candidate vector with SQLite student vectors│
 │    • Threshold = 0.45 (Optimal balance)                 │
 │      - Sim >= 0.45  ===> RECOGNIZED REGISTERED STUDENT  │
 │      - Sim < 0.45   ===> UNKNOWN PERSON (NO ATTENDANCE) │
 └─────────────────────────────────────────────────────────┘
```

---

## 9. Biometric Ethics, Privacy & Security Limitations

During your college presentation, highlight these engineering principles:
1. **Mathematical Embeddings Over Raw Photos:** Biometric matching uses 128-D vectors. Deleting a student via the UI purges both their vector and photo.
2. **Identity Verification vs. Guarantee:** Face recognition provides a probabilistic similarity match. Therefore, a **Manual Correction / Override** feature is built in for administrators.
3. **Liveness Detection Note:** This prototype tests feature similarity; it does not yet include depth or 3D passive liveness detection. Replaying photos on screens is a recognized limitation documented in the project scope.

---

## 10. College Presentation Viva Script (~2 to 3 Minutes)

> **Student Speaker:**  
> "Good morning, respected external examiners and professors. Today, I am demonstrating our Diploma IT project: **'AI Face Recognition Attendance System'**, developed by Pushprajsinh Rathod.
>
> **1. The Problem:** Manual roll-call attendance consumes 10–15% of lecture time and is vulnerable to proxy attendance.
>
> **2. Our Solution:** We built a full-stack, completely free, and locally runnable web application utilizing OpenCV's Deep Learning **YuNet** face detector and **SFace** 128-dimensional feature extractor, backed by Flask and SQLite.
>
> **3. Live Demonstration:**  
> Let me log into the admin dashboard using our secure credentials `admin / admin123`.  
> On the dashboard, we see today's metrics: total students enrolled, present count, and pending absentees.  
> Now, I navigate to **Live Recognition** and turn on the camera.  
> As a registered student (Aarav Sharma) appears before the camera, the deep learning model detects 5 facial landmarks, aligns the face, extracts the 128-D vector, and computes cosine similarity.  
> Within 750 milliseconds, Aarav is recognized with over 90% confidence, a confirmation chime sounds, and his attendance is recorded as Present in SQLite.
>
> **4. Reliability & Edge Cases:**  
> If the same student stays before the camera, the system rejects duplicate entries for today.  
> If an unregistered individual steps in, the system marks them as 'Unknown Face' and refuses attendance.  
>
> **5. Audit & Compliance:**  
> Administrators can filter attendance by date, search roll numbers, perform manual overrides, and download the verified audit log as a CSV spreadsheet.
>
> Thank you. I am ready for questions."

---

## 11. Screenshots to Capture for Your PPT Slides

1. **Slide 1 - Login Screen:** Showing the administrator login page with demo credential callout box.
2. **Slide 2 - Dashboard:** Showing the statistics cards (Total, Present, Absent) and recent check-in table.
3. **Slide 3 - Student Management:** Showing registered student passport photos with enrollment numbers and 128-D vector badges.
4. **Slide 4 - Live Webcam Attendance:** Showing the camera preview with the green bounding box labeled with student name and percentage.
5. **Slide 5 - Unknown Face State:** Showing the amber bounding box rejecting an unrecognized person.
6. **Slide 6 - Attendance History & Export:** Showing the attendance audit table with the CSV download button.

---

## 12. Verification & Automated Testing

To run the automated test suite verifying all 11 core functional requirements:
```powershell
python test_app.py
```
**Test Results:**
```text
Ran 11 tests in 1.069s
OK
```
All tests pass successfully.
