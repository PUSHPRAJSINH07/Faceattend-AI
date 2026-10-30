"""
AI Face Recognition Attendance System - Flask Application
---------------------------------------------------------
Author: PUSHPRAJSINH RATHOD
Diploma Engineering IT Project - AI Product Design
Features:
- Multi-Batch Management (e.g. Batch 2024, Batch 2025 trackrecords)
- Common Teacher Authentication & Sign-Up
- Persistent Biometrics & Automatic Photo Recovery
- Real-time Deep Learning Recognition (OpenCV YuNet + SFace 128-D)
- Multi-Day Everyday Attendance Progression & Simulation
- Mobile-First Clean UI/UX
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

# Initialize database schema and verify models
database.init_db()


def allowed_file(filename: str) -> bool:
    return "." in filename and filename.rsplit(".", 1)[1].lower() in ALLOWED_EXTENSIONS


def login_required(f):
    @wraps(f)
    def decorated_function(*args, **kwargs):
        if "teacher_id" not in session and "admin_id" not in session:
            flash("Please sign in to access your attendance portal.", "warning")
            return redirect(url_for("login", next=request.url))
        return f(*args, **kwargs)
    return decorated_function


@app.context_processor
def inject_global_data():
    """Inject active session date, active batch, teacher profile, and simulation status into all templates."""
    active_date = database.get_active_session_date()
    real_today = datetime.datetime.now().strftime("%Y-%m-%d")

    teacher_id = session.get("teacher_id") or session.get("admin_id")
    teacher = database.get_teacher_by_id(teacher_id) if teacher_id else None

    teacher_batches = database.get_batches_by_teacher(teacher_id) if teacher_id else []

    active_batch_id = session.get("active_batch_id")
    active_batch = None
    if active_batch_id:
        active_batch = database.get_batch_by_id(active_batch_id)
    elif teacher_batches:
        active_batch = teacher_batches[0]
        session["active_batch_id"] = active_batch["id"]

    return {
        "active_session_date": active_date,
        "is_simulation_date": (active_date != real_today),
        "real_today": real_today,
        "current_teacher": teacher,
        "teacher_batches": teacher_batches,
        "active_batch": active_batch,
        "active_batch_id": session.get("active_batch_id")
    }


# -------------------------------------------------------------
# Common Teacher Authentication Routes (Sign Up & Login)
# -------------------------------------------------------------
@app.route("/register", methods=["GET", "POST"])
def register():
    if "teacher_id" in session:
        return redirect(url_for("dashboard"))

    if request.method == "POST":
        name = request.form.get("name", "").strip()
        email = request.form.get("email", "").strip().lower()
        password = request.form.get("password", "").strip()
        institution_name = request.form.get("institution_name", "").strip()
        department = request.form.get("department", "").strip()
        initial_batch = request.form.get("initial_batch", "").strip() or "Batch 2024"

        ok, msg, teacher = database.register_teacher(
            name=name,
            email=email,
            password=password,
            institution_name=institution_name,
            department=department,
            initial_batch_name=initial_batch
        )

        if ok and teacher:
            session.clear()
            session["teacher_id"] = teacher["id"]
            session["teacher_name"] = teacher["name"]
            session["teacher_email"] = teacher["email"]
            session["institution_name"] = teacher["institution_name"]
            session["department"] = teacher["department"]
            session["active_batch_id"] = teacher["active_batch_id"]
            session["admin_id"] = teacher["id"]  # compatibility
            flash(f"Welcome, {teacher['name']}! Your account and '{initial_batch}' have been created.", "success")
            return redirect(url_for("dashboard"))
        else:
            flash(msg, "danger")

    return render_template("register.html")


@app.route("/login", methods=["GET", "POST"])
def login():
    if "teacher_id" in session:
        return redirect(url_for("dashboard"))

    if request.method == "POST":
        identifier = request.form.get("username", "").strip()
        password = request.form.get("password", "").strip()

        if not identifier or not password:
            flash("Please enter both your email/username and password.", "danger")
            return render_template("login.html")

        ok, user, message = database.verify_login(identifier, password)
        if ok and user:
            session.clear()
            session["teacher_id"] = user["id"]
            session["teacher_name"] = user["name"]
            session["teacher_email"] = user["email"]
            session["institution_name"] = user.get("institution_name", "")
            session["department"] = user.get("department", "")
            session["active_batch_id"] = user.get("active_batch_id")
            session["admin_id"] = user["id"]  # compatibility
            session["admin_name"] = user["name"]
            flash(f"Welcome back, {user['name']}!", "success")
            next_url = request.args.get("next")
            return redirect(next_url or url_for("dashboard"))
        else:
            flash(message, "danger")

    return render_template("login.html")


@app.route("/logout")
def logout():
    session.clear()
    flash("You have been signed out safely.", "info")
    return redirect(url_for("login"))


# -------------------------------------------------------------
# Batch Management Routes (Multi-Batch Trackrecords)
# -------------------------------------------------------------
@app.route("/batches/switch", methods=["POST"])
@login_required
def switch_batch():
    batch_id = request.form.get("batch_id", type=int)
    if batch_id:
        batch = database.get_batch_by_id(batch_id)
        if batch:
            session["active_batch_id"] = batch_id
            flash(f"Switched active workspace to '{batch['batch_name']}'.", "success")
    return redirect(request.referrer or url_for("dashboard"))


@app.route("/batches/create", methods=["POST"])
@login_required
def create_batch():
    teacher_id = session.get("teacher_id") or session.get("admin_id")
    batch_name = request.form.get("batch_name", "").strip()
    academic_year = request.form.get("academic_year", "").strip()
    course_name = request.form.get("course_name", "").strip()
    division = request.form.get("division", "").strip()

    ok, msg, new_batch_id = database.create_batch(
        teacher_id=teacher_id,
        batch_name=batch_name,
        academic_year=academic_year,
        course_name=course_name,
        division=division
    )
    if ok and new_batch_id:
        session["active_batch_id"] = new_batch_id
        flash(f"Batch '{batch_name}' created successfully!", "success")
    else:
        flash(msg, "danger")
    return redirect(request.referrer or url_for("dashboard"))


@app.route("/batches/delete/<int:batch_id>", methods=["POST"])
@login_required
def delete_batch(batch_id):
    ok, msg = database.delete_batch(batch_id)
    if ok:
        flash("Batch removed.", "info")
        teacher_id = session.get("teacher_id") or session.get("admin_id")
        batches = database.get_batches_by_teacher(teacher_id)
        session["active_batch_id"] = batches[0]["id"] if batches else None
    else:
        flash(msg, "danger")
    return redirect(request.referrer or url_for("dashboard"))


# -------------------------------------------------------------
# Session Date & Simulation Routes
# -------------------------------------------------------------
@app.route("/session/advance-day", methods=["POST"])
@login_required
def advance_day():
    old_date, new_date = database.advance_session_date(1)
    flash(f"Calendar moved to next day ({new_date}). You can now take attendance for this day.", "success")
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
        flash(f"Active attendance date changed to {target}.", "success")
    return redirect(request.referrer or url_for("dashboard"))


# -------------------------------------------------------------
# Dashboard Route
# -------------------------------------------------------------
@app.route("/")
@app.route("/dashboard")
@login_required
def dashboard():
    teacher_id = session.get("teacher_id") or session.get("admin_id")
    active_batch_id = session.get("active_batch_id")

    # Allow query parameter override
    req_batch = request.args.get("batch_id", type=int)
    if req_batch:
        active_batch_id = req_batch
        session["active_batch_id"] = req_batch

    stats = database.get_dashboard_stats(batch_id=active_batch_id, teacher_id=teacher_id)
    return render_template("dashboard.html", stats=stats)


# -------------------------------------------------------------
# Student Management Routes
# -------------------------------------------------------------
@app.route("/students", methods=["GET", "POST"])
@login_required
def students():
    teacher_id = session.get("teacher_id") or session.get("admin_id")
    active_batch_id = session.get("active_batch_id")

    if request.method == "POST":
        name = request.form.get("name", "").strip()
        enrollment_no = request.form.get("enrollment_no", "").strip().upper()
        student_class = request.form.get("student_class", "").strip()
        target_batch_id = request.form.get("batch_id", type=int) or active_batch_id
        webcam_data = request.form.get("webcam_image", "").strip()
        file = request.files.get("photo_file")

        if not name or not enrollment_no or not student_class:
            flash("All student fields (Name, Enrollment No, Class) are required.", "danger")
            return redirect(url_for("students"))

        if not re.match(r"^[A-Z0-9\-_]{3,20}$", enrollment_no):
            flash("Enrollment number must be 3-20 alphanumeric characters (e.g. EN202401).", "danger")
            return redirect(url_for("students"))

        existing = database.get_student_by_enrollment(enrollment_no)
        if existing:
            flash(f"A student with Enrollment Number '{enrollment_no}' already exists.", "danger")
            return redirect(url_for("students"))

        face_service = get_face_service()
        photo_filename = None
        embedding = None
        photo_data_uri = None

        # Option A: Webcam Snapshot
        if webcam_data and webcam_data.startswith("data:image"):
            try:
                header, encoded = webcam_data.split(",", 1)
                img_bytes = base64.b64decode(encoded)
                ok, msg, emb, bbox = face_service.process_image_bytes(img_bytes)
                if not ok or emb is None:
                    flash(f"Face verification failed: {msg}", "danger")
                    return redirect(url_for("students"))

                safe_enroll = secure_filename(enrollment_no).lower()
                photo_filename = f"student_{safe_enroll}_{int(datetime.datetime.now().timestamp())}.jpg"
                save_path = os.path.join(app.config["UPLOAD_FOLDER"], photo_filename)
                with open(save_path, "wb") as f:
                    f.write(img_bytes)

                embedding = emb
                photo_data_uri = webcam_data
            except Exception as e:
                flash(f"Error processing webcam image: {str(e)}", "danger")
                return redirect(url_for("students"))

        # Option B: File Upload
        elif file and file.filename != "":
            if not allowed_file(file.filename):
                flash("Invalid format. Supported formats: JPG, PNG, WEBP.", "danger")
                return redirect(url_for("students"))

            try:
                img_bytes = file.read()
                ok, msg, emb, bbox = face_service.process_image_bytes(img_bytes)
                if not ok or emb is None:
                    flash(f"Face verification failed: {msg}", "danger")
                    return redirect(url_for("students"))

                ext = file.filename.rsplit(".", 1)[1].lower()
                safe_enroll = secure_filename(enrollment_no).lower()
                photo_filename = f"student_{safe_enroll}_{int(datetime.datetime.now().timestamp())}.{ext}"
                save_path = os.path.join(app.config["UPLOAD_FOLDER"], photo_filename)
                with open(save_path, "wb") as f:
                    f.write(img_bytes)

                embedding = emb
                photo_data_uri = f"data:image/{ext};base64,{base64.b64encode(img_bytes).decode('utf-8')}"
            except Exception as e:
                flash(f"Error processing photo file: {str(e)}", "danger")
                return redirect(url_for("students"))
        else:
            flash("Please either snap a photo via webcam or upload a clear passport photo.", "danger")
            return redirect(url_for("students"))

        # Save student with persistent photo data in database
        saved, save_msg, new_id = database.add_student(
            name=name,
            enrollment_no=enrollment_no,
            student_class=student_class,
            photo_filename=photo_filename,
            face_encoding=embedding,
            batch_id=target_batch_id,
            photo_data=photo_data_uri
        )

        if saved:
            flash(f"Student '{name}' registered successfully with biometrics!", "success")
        else:
            if photo_filename:
                try:
                    os.remove(os.path.join(app.config["UPLOAD_FOLDER"], photo_filename))
                except Exception:
                    pass
            flash(f"Registration error: {save_msg}", "danger")

        return redirect(url_for("students"))

    # Filter students for active batch (or all)
    all_students = database.get_students_with_attendance_stats(batch_id=active_batch_id)
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
        flash("Student registration and biometrics deleted permanently.", "info")
    else:
        flash(f"Could not delete student: {msg}", "danger")
    return redirect(url_for("students"))


# -------------------------------------------------------------
# Live Recognition & Automatic Attendance Routes
# -------------------------------------------------------------
@app.route("/recognition")
@login_required
def recognition():
    active_batch_id = session.get("active_batch_id")
    students_count = len(database.get_all_students(batch_id=active_batch_id))
    today_records = database.get_today_attendance(batch_id=active_batch_id)
    return render_template(
        "recognition.html",
        students_count=students_count,
        today_records=today_records,
        threshold=DEFAULT_COSINE_THRESHOLD
    )


@app.route("/api/recognize", methods=["POST"])
@login_required
def api_recognize():
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
            return jsonify({"status": "error", "message": "Failed to decode camera frame."}), 400
    except Exception as e:
        return jsonify({"status": "error", "message": f"Decode error: {str(e)}"}), 400

    # Match across students in the active batch (or all teacher students)
    active_batch_id = session.get("active_batch_id")
    known_students = database.get_all_student_encodings(batch_id=active_batch_id)
    # If active batch has 0 students, fallback to all students to prevent false unknown on new batches
    if not known_students:
        known_students = database.get_all_student_encodings()

    face_service = get_face_service()
    threshold = float(data.get("threshold", DEFAULT_COSINE_THRESHOLD))
    result = face_service.identify_face_in_frame(frame, known_students, threshold=threshold)

    # Process automatic attendance if recognized
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

    return jsonify(result)


# -------------------------------------------------------------
# Attendance History, Manual Entry & CSV Export Routes
# -------------------------------------------------------------
@app.route("/attendance")
@login_required
def attendance():
    active_batch_id = session.get("active_batch_id")
    date_filter = request.args.get("date", "").strip()
    search_query = request.args.get("search", "").strip()

    records = database.get_attendance_history(
        date_filter=date_filter,
        search_query=search_query,
        batch_id=active_batch_id
    )
    all_students = database.get_all_students(batch_id=active_batch_id)

    return render_template(
        "attendance.html",
        records=records,
        all_students=all_students,
        date_filter=date_filter,
        search_query=search_query,
        today_date=database.get_active_session_date()
    )


@app.route("/attendance/export")
@login_required
def export_attendance_csv():
    active_batch_id = session.get("active_batch_id")
    date_filter = request.args.get("date", "").strip()
    search_query = request.args.get("search", "").strip()

    records = database.get_attendance_history(
        date_filter=date_filter,
        search_query=search_query,
        batch_id=active_batch_id
    )

    batch_info = database.get_batch_by_id(active_batch_id) if active_batch_id else None
    batch_name_str = batch_info["batch_name"] if batch_info else "All Batches"

    output = io.StringIO()
    writer = csv.writer(output)

    # Clean Header
    writer.writerow([
        "Record ID", "Student Name", "Enrollment No", "Class",
        "Batch", "Date", "Timestamp", "Status", "Confidence", "Marked Method"
    ])

    for r in records:
        writer.writerow([
            r["id"],
            r["name"],
            r["enrollment_no"],
            r["student_class"],
            r.get("batch_name") or batch_name_str,
            r["date"],
            r["time"],
            r["status"],
            f"{r['confidence']}%",
            r["marked_by"]
        ])

    csv_data = output.getvalue()
    safe_bname = secure_filename(batch_name_str).lower()
    filename = f"attendance_{safe_bname}_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"

    return Response(
        csv_data,
        mimetype="text/csv",
        headers={"Content-Disposition": f"attachment; filename={filename}"}
    )


@app.route("/attendance/manual", methods=["POST"])
@login_required
def manual_attendance():
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
        marked_by="Teacher Manual Override"
    )

    if ok:
        flash(msg, "success")
    else:
        flash(msg, "danger")

    return redirect(url_for("attendance", date=date_str))


@app.route("/attendance/delete/<int:record_id>", methods=["POST"])
@login_required
def delete_attendance(record_id):
    ok, msg = database.delete_attendance_record(record_id)
    if ok:
        flash("Attendance entry removed.", "info")
    else:
        flash(f"Could not remove: {msg}", "danger")
    return redirect(request.referrer or url_for("attendance"))


# -------------------------------------------------------------
# Permanent Photo Serving (with Self-Healing Recovery)
# -------------------------------------------------------------
@app.route("/uploads/students/<filename>")
def student_photo(filename):
    """
    Serves student photo. If the file is missing from local disk (e.g. after cloud restart),
    restores it directly from the database base64 BLOB on-the-fly!
    """
    safe_name = secure_filename(filename)
    # Ensure file is on disk (restores from DB if missing)
    database.ensure_photo_on_disk(safe_name)
    return send_from_directory(app.config["UPLOAD_FOLDER"], safe_name)


# -------------------------------------------------------------
# Application Entry Point
# -------------------------------------------------------------
if __name__ == "__main__":
    print("\n========================================================")
    print("  AI Face Recognition Attendance System")
    print("  Teacher Multi-Batch Portal | YuNet + SFace (128-D)")
    print("  Developed by PUSHPRAJSINH RATHOD")
    print("  Local & Cloud Ready: Dynamic PORT, Auto Photo Recovery")
    print("========================================================\n")
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port, debug=False)
