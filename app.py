"""
AI Face Recognition Attendance System - Flask Application
---------------------------------------------------------
Author: PUSHPRAJSINH RATHOD
Diploma Engineering IT Project - AI Product Design
100% Free & Open-Source (OpenCV YuNet + SFace, Flask, SQLite)
"""

import os
import re
import csv
import io
import base64
import datetime
from functools import wraps
from flask import (
    Flask, render_template, request, redirect,
    url_for, session, flash, jsonify, Response, send_from_directory
)
from werkzeug.utils import secure_filename
import cv2
import numpy as np

import database
from face_recognition_service import get_face_service, DEFAULT_COSINE_THRESHOLD

# App initialization
app = Flask(__name__)
app.config["SECRET_KEY"] = os.environ.get("SECRET_KEY", "super-secret-college-prototype-key-2026")
app.config["UPLOAD_FOLDER"] = database.UPLOADS_DIR
app.config["MAX_CONTENT_LENGTH"] = 16 * 1024 * 1024  # 16 MB max upload
ALLOWED_EXTENSIONS = {"png", "jpg", "jpeg", "webp"}

# Ensure database and models exist upon application launch
database.init_db()


def allowed_file(filename: str) -> bool:
    return "." in filename and filename.rsplit(".", 1)[1].lower() in ALLOWED_EXTENSIONS


def login_required(f):
    @wraps(f)
    def decorated_function(*args, **kwargs):
        if "admin_id" not in session:
            flash("Please log in to access this page.", "warning")
            return redirect(url_for("login", next=request.url))
        return f(*args, **kwargs)
    return decorated_function


@app.context_processor
def inject_global_data():
    """Inject active session date and simulation status into all templates."""
    active_date = database.get_active_session_date()
    real_today = datetime.datetime.now().strftime("%Y-%m-%d")
    return {
        "active_session_date": active_date,
        "is_simulation_date": (active_date != real_today),
        "real_today": real_today
    }


# -------------------------------------------------------------
# Session Date & Everyday Attendance Simulation Routes
# -------------------------------------------------------------
@app.route("/session/advance-day", methods=["POST"])
@login_required
def advance_day():
    old_date, new_date = database.advance_session_date(1)
    flash(f"Advanced session date to {new_date}! You can now record Day 2/Next Day attendance.", "success")
    return redirect(request.referrer or url_for("dashboard"))


@app.route("/session/reset-date", methods=["POST"])
@login_required
def reset_day():
    today = database.reset_session_date_to_today()
    flash(f"Session date reset to current real date ({today}).", "info")
    return redirect(request.referrer or url_for("dashboard"))


@app.route("/session/set-date", methods=["POST"])
@login_required
def set_date():
    target = request.form.get("session_date", "").strip()
    if target:
        database.set_active_session_date(target)
        flash(f"Active session date changed to {target}.", "success")
    return redirect(request.referrer or url_for("dashboard"))


# -------------------------------------------------------------
# Authentication Routes
# -------------------------------------------------------------
@app.route("/login", methods=["GET", "POST"])
def login():
    if "admin_id" in session:
        return redirect(url_for("dashboard"))

    if request.method == "POST":
        username = request.form.get("username", "").strip()
        password = request.form.get("password", "").strip()

        if not username or not password:
            flash("Please enter both username and password.", "danger")
            return render_template("login.html")

        ok, admin, message = database.verify_admin_login(username, password)
        if ok and admin:
            session.clear()
            session["admin_id"] = admin["id"]
            session["admin_username"] = admin["username"]
            session["admin_name"] = admin["full_name"]
            flash(f"Welcome back, {admin['full_name']}!", "success")
            next_url = request.args.get("next")
            return redirect(next_url or url_for("dashboard"))
        else:
            flash(message, "danger")

    return render_template("login.html")


@app.route("/logout")
def logout():
    session.clear()
    flash("You have been successfully logged out.", "info")
    return redirect(url_for("login"))


# -------------------------------------------------------------
# Dashboard Route
# -------------------------------------------------------------
@app.route("/")
@app.route("/dashboard")
@login_required
def dashboard():
    stats = database.get_dashboard_stats()
    return render_template("dashboard.html", stats=stats)


# -------------------------------------------------------------
# Students Management Routes
# -------------------------------------------------------------
@app.route("/students", methods=["GET", "POST"])
@login_required
def students():
    if request.method == "POST":
        name = request.form.get("name", "").strip()
        enrollment_no = request.form.get("enrollment_no", "").strip().upper()
        student_class = request.form.get("student_class", "").strip()
        webcam_data = request.form.get("webcam_image", "").strip()
        file = request.files.get("photo_file")

        # Field validation
        if not name or not enrollment_no or not student_class:
            flash("All student fields (Name, Enrollment No, Class) are required.", "danger")
            return redirect(url_for("students"))

        if not re.match(r"^[A-Z0-9\-_]{3,20}$", enrollment_no):
            flash("Enrollment number must be 3-20 alphanumeric characters (e.g. EN202401).", "danger")
            return redirect(url_for("students"))

        # Check existing enrollment
        existing = database.get_student_by_enrollment(enrollment_no)
        if existing:
            flash(f"A student with Enrollment Number '{enrollment_no}' already exists.", "danger")
            return redirect(url_for("students"))

        face_service = get_face_service()
        photo_filename = None
        embedding = None

        # 1. Option A: Image uploaded via webcam snapshot
        if webcam_data and webcam_data.startswith("data:image"):
            try:
                header, encoded = webcam_data.split(",", 1)
                img_bytes = base64.b64decode(encoded)
                ok, msg, emb, bbox = face_service.process_image_bytes(img_bytes)
                if not ok or emb is None:
                    flash(f"Face validation failed: {msg}", "danger")
                    return redirect(url_for("students"))

                safe_enroll = secure_filename(enrollment_no).lower()
                photo_filename = f"student_{safe_enroll}_{int(datetime.datetime.now().timestamp())}.jpg"
                save_path = os.path.join(app.config["UPLOAD_FOLDER"], photo_filename)
                with open(save_path, "wb") as f:
                    f.write(img_bytes)
                embedding = emb
            except Exception as e:
                flash(f"Error processing webcam snapshot: {str(e)}", "danger")
                return redirect(url_for("students"))

        # 2. Option B: Image uploaded via file input
        elif file and file.filename != "":
            if not allowed_file(file.filename):
                flash("Invalid file format. Allowed formats: PNG, JPG, JPEG, WEBP.", "danger")
                return redirect(url_for("students"))

            try:
                img_bytes = file.read()
                ok, msg, emb, bbox = face_service.process_image_bytes(img_bytes)
                if not ok or emb is None:
                    flash(f"Face validation failed: {msg}", "danger")
                    return redirect(url_for("students"))

                ext = file.filename.rsplit(".", 1)[1].lower()
                safe_enroll = secure_filename(enrollment_no).lower()
                photo_filename = f"student_{safe_enroll}_{int(datetime.datetime.now().timestamp())}.{ext}"
                save_path = os.path.join(app.config["UPLOAD_FOLDER"], photo_filename)
                with open(save_path, "wb") as f:
                    f.write(img_bytes)
                embedding = emb
            except Exception as e:
                flash(f"Error uploading file: {str(e)}", "danger")
                return redirect(url_for("students"))
        else:
            flash("Please either upload a passport photo or take a photo using the webcam.", "danger")
            return redirect(url_for("students"))

        # Save to database
        saved, save_msg, new_id = database.add_student(
            name=name,
            enrollment_no=enrollment_no,
            student_class=student_class,
            photo_filename=photo_filename,
            face_encoding=embedding
        )

        if saved:
            flash(f"Student '{name}' registered successfully with facial biometrics!", "success")
        else:
            # Clean up uploaded file if DB insert fails
            if photo_filename:
                try:
                    os.remove(os.path.join(app.config["UPLOAD_FOLDER"], photo_filename))
                except Exception:
                    pass
            flash(f"Registration failed: {save_msg}", "danger")

        return redirect(url_for("students"))

    all_students = database.get_students_with_attendance_stats()
    return render_template("students.html", students=all_students)


@app.route("/students/delete/<int:student_id>", methods=["POST"])
@login_required
def delete_student(student_id):
    ok, msg, photo_file = database.delete_student(student_id)
    if ok:
        if photo_file:
            path = os.path.join(app.config["UPLOAD_FOLDER"], photo_file)
            if os.path.exists(path) and not photo_file.startswith("sample_"):
                try:
                    os.remove(path)
                except Exception:
                    pass
        flash("Student registration and associated biometric representations deleted.", "info")
    else:
        flash(f"Could not delete student: {msg}", "danger")
    return redirect(url_for("students"))


# -------------------------------------------------------------
# Live Face Recognition & Automatic Attendance Routes
# -------------------------------------------------------------
@app.route("/recognition")
@login_required
def recognition():
    students_count = len(database.get_all_students())
    today_records = database.get_today_attendance()
    return render_template(
        "recognition.html",
        students_count=students_count,
        today_records=today_records,
        threshold=DEFAULT_COSINE_THRESHOLD
    )


@app.route("/api/recognize", methods=["POST"])
@login_required
def api_recognize():
    """
    Receives base64 video frame from client webcam.
    Performs face detection, extraction, cosine matching against database,
    and automatic attendance recording.
    """
    data = request.get_json(silent=True)
    if not data or "image" not in data:
        return jsonify({"status": "error", "message": "No image payload received."}), 400

    img_data = data["image"]
    if "," in img_data:
        img_data = img_data.split(",", 1)[1]

    try:
        raw_bytes = base64.b64decode(img_data)
        np_arr = np.frombuffer(raw_bytes, np.uint8)
        frame = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)
        if frame is None:
            return jsonify({"status": "error", "message": "Failed to decode frame."}), 400
    except Exception as e:
        return jsonify({"status": "error", "message": f"Decode error: {str(e)}"}), 400

    # Retrieve registered students with encodings
    known_students = database.get_all_student_encodings()
    face_service = get_face_service()

    # Allow custom threshold from client if provided, else use default
    threshold = float(data.get("threshold", DEFAULT_COSINE_THRESHOLD))
    result = face_service.identify_face_in_frame(frame, known_students, threshold=threshold)

    # If successfully recognized a registered student, process attendance automatically
    if result.get("status") == "recognized" and result.get("student"):
        student = result["student"]
        confidence = result.get("confidence", 0.0)

        marked_new, att_msg, att_record = database.mark_attendance(
            student_id=student["id"],
            confidence=confidence,
            marked_by="AI Face Recognition"
        )

        response_payload = {
            "status": "recognized",
            "message": att_msg,
            "newly_marked": marked_new,
            "student": {
                "id": student["id"],
                "name": student["name"],
                "enrollment_no": student["enrollment_no"],
                "student_class": student["student_class"],
                "photo_filename": student["photo_filename"]
            },
            "confidence": confidence,
            "raw_score": result.get("raw_score", 0.0),
            "threshold": threshold,
            "box": result.get("box"),
            "attendance_record": att_record
        }
        return jsonify(response_payload)

    # Return unknown, no_face, or multiple_faces status
    return jsonify(result)


# -------------------------------------------------------------
# Attendance History & Manual Correction Routes
# -------------------------------------------------------------
@app.route("/attendance")
@login_required
def attendance():
    date_filter = request.args.get("date", "").strip()
    search_query = request.args.get("search", "").strip()

    records = database.get_attendance_history(date_filter=date_filter, search_query=search_query)
    all_students = database.get_all_students()

    return render_template(
        "attendance.html",
        records=records,
        all_students=all_students,
        date_filter=date_filter,
        search_query=search_query,
        today_date=datetime.datetime.now().strftime("%Y-%m-%d")
    )


@app.route("/attendance/export")
@login_required
def export_attendance_csv():
    """Generates and downloads a CSV export of filtered or full attendance records."""
    date_filter = request.args.get("date", "").strip()
    search_query = request.args.get("search", "").strip()

    records = database.get_attendance_history(date_filter=date_filter, search_query=search_query)

    output = io.StringIO()
    writer = csv.writer(output)

    # CSV Header
    writer.writerow([
        "Record ID", "Student Name", "Enrollment No", "Class",
        "Date", "Time", "Status", "Confidence (%)", "Marked Method"
    ])

    for r in records:
        writer.writerow([
            r["id"],
            r["name"],
            r["enrollment_no"],
            r["student_class"],
            r["date"],
            r["time"],
            r["status"],
            f"{r['confidence']}%",
            r["marked_by"]
        ])

    csv_data = output.getvalue()
    filename = f"attendance_report_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"

    return Response(
        csv_data,
        mimetype="text/csv",
        headers={"Content-Disposition": f"attachment; filename={filename}"}
    )


@app.route("/attendance/manual", methods=["POST"])
@login_required
def manual_attendance():
    """Manual correction: Allows administrator to record or override an attendance status."""
    student_id = request.form.get("student_id", type=int)
    date_str = request.form.get("date", "").strip()
    time_str = request.form.get("time", "").strip()
    status = request.form.get("status", "Present").strip()

    if not student_id or not date_str:
        flash("Student and Date are required for manual attendance.", "danger")
        return redirect(url_for("attendance"))

    if not time_str:
        time_str = datetime.datetime.now().strftime("%I:%M:%S %p")

    ok, msg = database.manual_mark_attendance(
        student_id=student_id,
        date_str=date_str,
        time_str=time_str,
        status=status,
        marked_by="Admin Manual Entry"
    )

    if ok:
        flash(msg, "success")
    else:
        flash(msg, "danger")

    return redirect(url_for("attendance", date=date_str))


@app.route("/attendance/delete/<int:record_id>", methods=["POST"])
@login_required
def delete_attendance(record_id):
    """Manual correction: Delete an erroneous attendance entry."""
    ok, msg = database.delete_attendance_record(record_id)
    if ok:
        flash("Attendance record deleted successfully.", "info")
    else:
        flash(f"Failed to delete record: {msg}", "danger")
    return redirect(request.referrer or url_for("attendance"))


# -------------------------------------------------------------
# Static Media File Serving
# -------------------------------------------------------------
@app.route("/uploads/students/<filename>")
@login_required
def student_photo(filename):
    """Serve student passport photos securely with access control."""
    safe_name = secure_filename(filename)
    return send_from_directory(app.config["UPLOAD_FOLDER"], safe_name)


# -------------------------------------------------------------
# Application Entry Point
# -------------------------------------------------------------
if __name__ == "__main__":
    print("\n========================================================")
    print("  AI Face Recognition Attendance System - College Prototype")
    print("  Technology: OpenCV YuNet + SFace (Deep Learning) + Flask")
    print("  Status: 100% Free & Open-Source, Offline Inference Ready")
    print("  Default Demo Login: admin / admin123")
    print("  Server URL: http://127.0.0.1:5000")
    print("========================================================\n")
    app.run(host="127.0.0.1", port=5000, debug=True)
