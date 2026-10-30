"""
Database Management Module
--------------------------
Handles SQLite connection, schema definition, initialization, demo seeding,
multi-tenant Teacher Authentication, Multi-Batch tracking, student biometrics,
and persistent photo-recovery mechanism.
"""

import os
import shutil
import sqlite3
import datetime
import base64
import numpy as np
from werkzeug.security import generate_password_hash, check_password_hash
from typing import List, Dict, Any, Optional, Tuple

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
INSTANCE_DIR = os.path.join(BASE_DIR, "instance")
DB_PATH = os.path.join(INSTANCE_DIR, "attendance.db")
UPLOADS_DIR = os.path.join(BASE_DIR, "uploads", "students")
SAMPLE_IMAGES_DIR = os.path.join(BASE_DIR, "static", "sample_images")


def get_db_connection() -> sqlite3.Connection:
    """Create and return a row-mapped SQLite connection."""
    os.makedirs(INSTANCE_DIR, exist_ok=True)
    conn = sqlite3.connect(DB_PATH, detect_types=sqlite3.PARSE_DECLTYPES)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA foreign_keys = ON")
    return conn


def init_db():
    """Initialize database tables, apply schema migrations, and perform one-time seeding."""
    os.makedirs(INSTANCE_DIR, exist_ok=True)
    os.makedirs(UPLOADS_DIR, exist_ok=True)

    with get_db_connection() as conn:
        cursor = conn.cursor()

        # 1. Teachers / Users Table (Common Authentication)
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS teachers (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                name TEXT NOT NULL,
                email TEXT UNIQUE NOT NULL,
                password_hash TEXT NOT NULL,
                institution_name TEXT DEFAULT '',
                department TEXT DEFAULT '',
                role TEXT DEFAULT 'teacher',
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        """)

        # 2. Admins Table (Preserved for backward compatibility)
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS admins (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                username TEXT UNIQUE NOT NULL,
                password_hash TEXT NOT NULL,
                full_name TEXT NOT NULL,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        """)

        # 3. Batches Table (Supports Batch 2024, Batch 2025, etc.)
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS batches (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                teacher_id INTEGER NOT NULL,
                batch_name TEXT NOT NULL,
                academic_year TEXT DEFAULT '',
                course_name TEXT DEFAULT '',
                division TEXT DEFAULT '',
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                FOREIGN KEY (teacher_id) REFERENCES teachers(id) ON DELETE CASCADE
            )
        """)

        # 4. Students Table
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS students (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                batch_id INTEGER,
                name TEXT NOT NULL,
                enrollment_no TEXT UNIQUE NOT NULL,
                student_class TEXT NOT NULL,
                photo_filename TEXT NOT NULL,
                photo_data TEXT,
                face_encoding BLOB NOT NULL,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                FOREIGN KEY (batch_id) REFERENCES batches(id) ON DELETE CASCADE
            )
        """)

        # 5. Attendance Table
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS attendance (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                student_id INTEGER NOT NULL,
                batch_id INTEGER,
                date TEXT NOT NULL,
                time TEXT NOT NULL,
                status TEXT NOT NULL DEFAULT 'Present',
                confidence REAL DEFAULT 0.0,
                marked_by TEXT DEFAULT 'AI Face Recognition',
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                FOREIGN KEY (student_id) REFERENCES students(id) ON DELETE CASCADE,
                FOREIGN KEY (batch_id) REFERENCES batches(id) ON DELETE SET NULL,
                UNIQUE(student_id, date)
            )
        """)

        # 6. System Settings Table (Simulation date & persistence flags)
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS system_settings (
                key TEXT PRIMARY KEY,
                value TEXT NOT NULL
            )
        """)

        # Safe Column Migrations for existing databases
        cursor.execute("PRAGMA table_info(students)")
        student_cols = [c[1] for c in cursor.fetchall()]
        if "batch_id" not in student_cols:
            cursor.execute("ALTER TABLE students ADD COLUMN batch_id INTEGER")
        if "photo_data" not in student_cols:
            cursor.execute("ALTER TABLE students ADD COLUMN photo_data TEXT")

        cursor.execute("PRAGMA table_info(attendance)")
        att_cols = [c[1] for c in cursor.fetchall()]
        if "batch_id" not in att_cols:
            cursor.execute("ALTER TABLE attendance ADD COLUMN batch_id INTEGER")

        # Seed default Admin and default Teacher account if none exists
        cursor.execute("SELECT COUNT(*) FROM admins")
        if cursor.fetchone()[0] == 0:
            default_admin_hash = generate_password_hash("admin123")
            cursor.execute(
                "INSERT INTO admins (username, password_hash, full_name) VALUES (?, ?, ?)",
                ("admin", default_admin_hash, "College Administrator")
            )

        cursor.execute("SELECT COUNT(*) FROM teachers")
        if cursor.fetchone()[0] == 0:
            default_teacher_hash = generate_password_hash("admin123")
            cursor.execute("""
                INSERT INTO teachers (name, email, password_hash, institution_name, department, role)
                VALUES (?, ?, ?, ?, ?, ?)
            """, (
                "Prof. Pushprajsinh Rathod",
                "admin",
                default_teacher_hash,
                "Government Polytechnic / Engineering College",
                "Department of Information Technology",
                "admin"
            ))
            default_teacher_id = cursor.lastrowid

            # Create default batches for this teacher
            cursor.execute("""
                INSERT INTO batches (teacher_id, batch_name, academic_year, course_name, division)
                VALUES (?, ?, ?, ?, ?)
            """, (default_teacher_id, "Batch 2024 - TY-IF", "2024-2025", "Information Technology", "TY-IF"))
            b2024_id = cursor.lastrowid

            cursor.execute("""
                INSERT INTO batches (teacher_id, batch_name, academic_year, course_name, division)
                VALUES (?, ?, ?, ?, ?)
            """, (default_teacher_id, "Batch 2025 - SY-CO", "2025-2026", "Computer Engineering", "SY-CO"))

            # Update existing students without a batch_id
            cursor.execute("UPDATE students SET batch_id = ? WHERE batch_id IS NULL", (b2024_id,))

        conn.commit()

    # One-time sample seeding (only if never seeded before)
    _seed_sample_students_if_needed()


def _seed_sample_students_if_needed():
    """Seed initial demonstration students only once. If user deletes them, they NEVER reappear."""
    with get_db_connection() as conn:
        cursor = conn.cursor()
        cursor.execute("SELECT value FROM system_settings WHERE key = 'sample_seeded'")
        row = cursor.fetchone()
        if row and row["value"] == "1":
            return  # Already seeded in the past; respect teacher's choice!

        # Check if any batch exists
        cursor.execute("SELECT id FROM batches ORDER BY id ASC LIMIT 1")
        batch_row = cursor.fetchone()
        batch_id = batch_row[0] if batch_row else 1

    # Import face service
    from face_recognition_service import get_face_service
    face_service = get_face_service()

    sample_students = [
        {
            "name": "Aarav Sharma",
            "enrollment_no": "EN202401",
            "student_class": "TY-IF (IT 6th Sem)",
            "sample_file": "student1.jpg",
            "target_filename": "sample_aarav_sharma.jpg"
        },
        {
            "name": "Riya Patel",
            "enrollment_no": "EN202402",
            "student_class": "TY-IF (IT 6th Sem)",
            "sample_file": "student2.jpg",
            "target_filename": "sample_riya_patel.jpg"
        },
        {
            "name": "Aditya Verma",
            "enrollment_no": "EN202403",
            "student_class": "TY-IF (IT 6th Sem)",
            "sample_file": "student3.jpg",
            "target_filename": "sample_aditya_verma.jpg"
        }
    ]

    for item in sample_students:
        src = os.path.join(SAMPLE_IMAGES_DIR, item["sample_file"])
        dst = os.path.join(UPLOADS_DIR, item["target_filename"])

        if os.path.exists(src):
            shutil.copy2(src, dst)
            ok, msg, embedding, _ = face_service.process_image_file(dst)
            if ok and embedding is not None:
                # Read base64
                with open(dst, "rb") as img_f:
                    b64_data = base64.b64encode(img_f.read()).decode("utf-8")
                photo_uri = f"data:image/jpeg;base64,{b64_data}"

                add_student(
                    name=item["name"],
                    enrollment_no=item["enrollment_no"],
                    student_class=item["student_class"],
                    photo_filename=item["target_filename"],
                    face_encoding=embedding,
                    batch_id=batch_id,
                    photo_data=photo_uri
                )

    # Mark as seeded forever
    with get_db_connection() as conn:
        cursor = conn.cursor()
        cursor.execute("INSERT OR REPLACE INTO system_settings (key, value) VALUES ('sample_seeded', '1')")
        conn.commit()


# -------------------------------------------------------------
# Photo Persistence & Recovery
# -------------------------------------------------------------
def ensure_photo_on_disk(photo_filename: str) -> Optional[str]:
    """
    Ensures student photo exists on disk.
    If cloud container restarted and wiped local disk, retrieves base64 from database
    and restores it to disk immediately so no photo is ever lost.
    """
    if not photo_filename:
        return None
    file_path = os.path.join(UPLOADS_DIR, photo_filename)
    if os.path.exists(file_path) and os.path.getsize(file_path) > 0:
        return file_path

    # Restore from database
    with get_db_connection() as conn:
        cursor = conn.cursor()
        cursor.execute("SELECT photo_data FROM students WHERE photo_filename = ?", (photo_filename,))
        row = cursor.fetchone()
        if row and row["photo_data"]:
            try:
                data_uri = row["photo_data"]
                if "," in data_uri:
                    data_uri = data_uri.split(",", 1)[1]
                raw_bytes = base64.b64decode(data_uri)
                os.makedirs(UPLOADS_DIR, exist_ok=True)
                with open(file_path, "wb") as f:
                    f.write(raw_bytes)
                return file_path
            except Exception as e:
                print(f"[PhotoRecovery] Could not recover photo {photo_filename}: {e}")
    return None


# -------------------------------------------------------------
# Common Authentication: Teachers & Admins
# -------------------------------------------------------------
def register_teacher(
    name: str,
    email: str,
    password: str,
    institution_name: str = "",
    department: str = "",
    initial_batch_name: str = "Batch 2024"
) -> Tuple[bool, str, Optional[Dict[str, Any]]]:
    """Register a new teacher and create their first batch."""
    email_clean = email.strip().lower()
    name_clean = name.strip()

    if not email_clean or not name_clean or not password:
        return False, "Name, Email, and Password are required.", None

    with get_db_connection() as conn:
        cursor = conn.cursor()
        cursor.execute("SELECT id FROM teachers WHERE email = ?", (email_clean,))
        if cursor.fetchone():
            return False, "An account with this email already exists.", None

        pw_hash = generate_password_hash(password)
        cursor.execute("""
            INSERT INTO teachers (name, email, password_hash, institution_name, department, role)
            VALUES (?, ?, ?, ?, ?, 'teacher')
        """, (name_clean, email_clean, pw_hash, institution_name.strip(), department.strip()))
        teacher_id = cursor.lastrowid

        # Create teacher's first batch
        b_name = initial_batch_name.strip() or "Batch 2024"
        cursor.execute("""
            INSERT INTO batches (teacher_id, batch_name, academic_year, course_name, division)
            VALUES (?, ?, ?, ?, ?)
        """, (teacher_id, b_name, "2024-2025", department.strip() or "General", "Div A"))
        first_batch_id = cursor.lastrowid

        conn.commit()

        cursor.execute("SELECT * FROM teachers WHERE id = ?", (teacher_id,))
        teacher_dict = dict(cursor.fetchone())
        teacher_dict["active_batch_id"] = first_batch_id
        return True, "Teacher registered successfully.", teacher_dict


def verify_login(login_identifier: str, password: str) -> Tuple[bool, Optional[Dict[str, Any]], str]:
    """Unified login for teachers and admins (supports email or username)."""
    ident = login_identifier.strip().lower()

    with get_db_connection() as conn:
        cursor = conn.cursor()
        # 1. Check in teachers table
        cursor.execute("SELECT * FROM teachers WHERE LOWER(email) = ?", (ident,))
        row = cursor.fetchone()
        if row:
            t = dict(row)
            if check_password_hash(t["password_hash"], password):
                # Fetch their default / first batch
                cursor.execute("SELECT id FROM batches WHERE teacher_id = ? ORDER BY id ASC LIMIT 1", (t["id"],))
                batch_row = cursor.fetchone()
                t["active_batch_id"] = batch_row[0] if batch_row else None
                return True, t, "Login successful."

        # 2. Check in admins table (fallback)
        cursor.execute("SELECT * FROM admins WHERE LOWER(username) = ?", (ident,))
        admin_row = cursor.fetchone()
        if admin_row:
            a = dict(admin_row)
            if check_password_hash(a["password_hash"], password):
                # Map admin to a teacher context
                cursor.execute("SELECT * FROM teachers WHERE role = 'admin' LIMIT 1")
                t_match = cursor.fetchone()
                if t_match:
                    t = dict(t_match)
                    cursor.execute("SELECT id FROM batches WHERE teacher_id = ? ORDER BY id ASC LIMIT 1", (t["id"],))
                    b_row = cursor.fetchone()
                    t["active_batch_id"] = b_row[0] if b_row else None
                    return True, t, "Login successful."

    return False, None, "Invalid email/username or password."


def get_teacher_by_id(teacher_id: int) -> Optional[Dict[str, Any]]:
    with get_db_connection() as conn:
        cursor = conn.cursor()
        cursor.execute("SELECT * FROM teachers WHERE id = ?", (teacher_id,))
        row = cursor.fetchone()
        return dict(row) if row else None


def update_teacher_profile(
    teacher_id: int,
    name: str,
    institution_name: str,
    department: str
) -> Tuple[bool, str]:
    with get_db_connection() as conn:
        cursor = conn.cursor()
        cursor.execute("""
            UPDATE teachers
            SET name = ?, institution_name = ?, department = ?
            WHERE id = ?
        """, (name.strip(), institution_name.strip(), department.strip(), teacher_id))
        conn.commit()
        return True, "Profile updated successfully."


# -------------------------------------------------------------
# Batch Management (Multi-Batch System)
# -------------------------------------------------------------
def create_batch(
    teacher_id: int,
    batch_name: str,
    academic_year: str = "",
    course_name: str = "",
    division: str = ""
) -> Tuple[bool, str, Optional[int]]:
    """Create a new batch for a teacher (e.g. Batch 2024, Batch 2025)."""
    b_name = batch_name.strip()
    if not b_name:
        return False, "Batch name is required.", None

    with get_db_connection() as conn:
        cursor = conn.cursor()
        # Check duplicate batch name for same teacher
        cursor.execute("SELECT id FROM batches WHERE teacher_id = ? AND LOWER(batch_name) = ?", (teacher_id, b_name.lower()))
        if cursor.fetchone():
            return False, f"Batch '{b_name}' already exists in your account.", None

        cursor.execute("""
            INSERT INTO batches (teacher_id, batch_name, academic_year, course_name, division)
            VALUES (?, ?, ?, ?, ?)
        """, (teacher_id, b_name, academic_year.strip(), course_name.strip(), division.strip()))
        conn.commit()
        return True, f"Batch '{b_name}' created successfully.", cursor.lastrowid


def get_batches_by_teacher(teacher_id: int) -> List[Dict[str, Any]]:
    """Retrieve all batches belonging to a teacher with student counts."""
    with get_db_connection() as conn:
        cursor = conn.cursor()
        cursor.execute("""
            SELECT b.*, COUNT(s.id) as student_count
            FROM batches b
            LEFT JOIN students s ON b.id = s.batch_id
            WHERE b.teacher_id = ?
            GROUP BY b.id
            ORDER BY b.created_at DESC
        """, (teacher_id,))
        return [dict(row) for row in cursor.fetchall()]


def get_batch_by_id(batch_id: int) -> Optional[Dict[str, Any]]:
    with get_db_connection() as conn:
        cursor = conn.cursor()
        cursor.execute("""
            SELECT b.*, t.name as teacher_name, t.institution_name, t.department
            FROM batches b
            JOIN teachers t ON b.teacher_id = t.id
            WHERE b.id = ?
        """, (batch_id,))
        row = cursor.fetchone()
        return dict(row) if row else None


def delete_batch(batch_id: int) -> Tuple[bool, str]:
    with get_db_connection() as conn:
        cursor = conn.cursor()
        cursor.execute("DELETE FROM batches WHERE id = ?", (batch_id,))
        conn.commit()
        return True, "Batch and its associated records deleted."


# -------------------------------------------------------------
# Student Operations (Batch-Linked)
# -------------------------------------------------------------
def add_student(
    name: str,
    enrollment_no: str,
    student_class: str,
    photo_filename: str,
    face_encoding: np.ndarray,
    batch_id: Optional[int] = None,
    photo_data: Optional[str] = None
) -> Tuple[bool, str, Optional[int]]:
    """Save student details, 128-D facial vector, and persistent base64 photo data."""
    try:
        encoding_bytes = face_encoding.astype(np.float32).tobytes()
        with get_db_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("""
                INSERT INTO students (batch_id, name, enrollment_no, student_class, photo_filename, photo_data, face_encoding)
                VALUES (?, ?, ?, ?, ?, ?, ?)
            """, (
                batch_id,
                name.strip(),
                enrollment_no.strip().upper(),
                student_class.strip(),
                photo_filename,
                photo_data,
                encoding_bytes
            ))
            conn.commit()
            return True, "Student registered successfully.", cursor.lastrowid
    except sqlite3.IntegrityError:
        return False, f"Enrollment number '{enrollment_no}' already exists.", None
    except Exception as e:
        return False, f"Database error: {str(e)}", None


def get_all_students(batch_id: Optional[int] = None) -> List[Dict[str, Any]]:
    query = """
        SELECT s.id, s.batch_id, s.name, s.enrollment_no, s.student_class, s.photo_filename, s.created_at,
               b.batch_name
        FROM students s
        LEFT JOIN batches b ON s.batch_id = b.id
        WHERE 1=1
    """
    params: List[Any] = []
    if batch_id is not None:
        query += " AND s.batch_id = ?"
        params.append(batch_id)
    query += " ORDER BY s.name ASC"

    with get_db_connection() as conn:
        cursor = conn.cursor()
        cursor.execute(query, params)
        return [dict(row) for row in cursor.fetchall()]


def get_student_by_id(student_id: int) -> Optional[Dict[str, Any]]:
    with get_db_connection() as conn:
        cursor = conn.cursor()
        cursor.execute("""
            SELECT s.*, b.batch_name
            FROM students s
            LEFT JOIN batches b ON s.batch_id = b.id
            WHERE s.id = ?
        """, (student_id,))
        row = cursor.fetchone()
        return dict(row) if row else None


def get_student_by_enrollment(enrollment_no: str) -> Optional[Dict[str, Any]]:
    with get_db_connection() as conn:
        cursor = conn.cursor()
        cursor.execute("SELECT * FROM students WHERE enrollment_no = ?", (enrollment_no.strip().upper(),))
        row = cursor.fetchone()
        return dict(row) if row else None


def delete_student(student_id: int) -> Tuple[bool, str, Optional[str]]:
    """Delete student and their biometric representations permanently."""
    student = get_student_by_id(student_id)
    if not student:
        return False, "Student not found.", None
    photo_file = student.get("photo_filename")
    with get_db_connection() as conn:
        cursor = conn.cursor()
        cursor.execute("DELETE FROM students WHERE id = ?", (student_id,))
        conn.commit()
    return True, "Student and associated biometric representations deleted.", photo_file


def get_all_student_encodings(batch_id: Optional[int] = None) -> List[Dict[str, Any]]:
    """Retrieve students with deserialized 128-D embeddings, filtered by batch if specified."""
    query = "SELECT id, batch_id, name, enrollment_no, student_class, photo_filename, face_encoding FROM students WHERE 1=1"
    params: List[Any] = []
    if batch_id is not None:
        query += " AND batch_id = ?"
        params.append(batch_id)

    with get_db_connection() as conn:
        cursor = conn.cursor()
        cursor.execute(query, params)
        rows = cursor.fetchall()

    result = []
    for row in rows:
        raw_bytes = row["face_encoding"]
        embedding = np.frombuffer(raw_bytes, dtype=np.float32)
        result.append({
            "id": row["id"],
            "batch_id": row["batch_id"],
            "name": row["name"],
            "enrollment_no": row["enrollment_no"],
            "student_class": row["student_class"],
            "photo_filename": row["photo_filename"],
            "embedding": embedding
        })
    return result


# -------------------------------------------------------------
# Active Session Date Operations
# -------------------------------------------------------------
def get_active_session_date() -> str:
    with get_db_connection() as conn:
        cursor = conn.cursor()
        cursor.execute("SELECT value FROM system_settings WHERE key = 'active_session_date'")
        row = cursor.fetchone()
        if row and row["value"]:
            return row["value"]
    return datetime.datetime.now().strftime("%Y-%m-%d")


def set_active_session_date(date_str: str) -> str:
    clean_date = date_str.strip()
    with get_db_connection() as conn:
        cursor = conn.cursor()
        cursor.execute("""
            INSERT INTO system_settings (key, value) VALUES ('active_session_date', ?)
            ON CONFLICT(key) DO UPDATE SET value = excluded.value
        """, (clean_date,))
        conn.commit()
    return clean_date


def advance_session_date(days: int = 1) -> Tuple[str, str]:
    current = get_active_session_date()
    dt = datetime.datetime.strptime(current, "%Y-%m-%d")
    next_dt = dt + datetime.timedelta(days=days)
    next_str = next_dt.strftime("%Y-%m-%d")
    set_active_session_date(next_str)
    return current, next_str


def reset_session_date_to_today() -> str:
    today_str = datetime.datetime.now().strftime("%Y-%m-%d")
    return set_active_session_date(today_str)


# -------------------------------------------------------------
# Attendance Tracking & Cumulative Multi-Day Stats
# -------------------------------------------------------------
def get_students_with_attendance_stats(batch_id: Optional[int] = None) -> List[Dict[str, Any]]:
    """Retrieve students with cumulative attendance tracking per batch."""
    with get_db_connection() as conn:
        cursor = conn.cursor()

        # Count total recorded dates for this batch
        if batch_id:
            cursor.execute("""
                SELECT COUNT(DISTINCT a.date) 
                FROM attendance a 
                JOIN students s ON a.student_id = s.id 
                WHERE s.batch_id = ?
            """, (batch_id,))
        else:
            cursor.execute("SELECT COUNT(DISTINCT date) FROM attendance")
        total_working_days = cursor.fetchone()[0]

        query = """
            SELECT 
                s.id, s.batch_id, s.name, s.enrollment_no, s.student_class, s.photo_filename, s.created_at,
                b.batch_name,
                COUNT(DISTINCT a.date) as days_present,
                MAX(a.date) as last_seen_date,
                MAX(a.time) as last_seen_time
            FROM students s
            LEFT JOIN batches b ON s.batch_id = b.id
            LEFT JOIN attendance a ON s.id = a.student_id AND a.status = 'Present'
            WHERE 1=1
        """
        params: List[Any] = []
        if batch_id is not None:
            query += " AND s.batch_id = ?"
            params.append(batch_id)

        query += " GROUP BY s.id ORDER BY s.name ASC"
        cursor.execute(query, params)
        rows = cursor.fetchall()

    result = []
    for row in rows:
        d = dict(row)
        days = d["days_present"] or 0
        eff_total = max(total_working_days, 1) if total_working_days > 0 else (1 if days > 0 else 0)
        rate = round((days / eff_total) * 100, 1) if eff_total > 0 else 0.0
        d["total_working_days"] = total_working_days
        d["attendance_rate"] = rate
        result.append(d)
    return result


def mark_attendance(
    student_id: int,
    confidence: float,
    marked_by: str = "AI Face Recognition",
    target_date: Optional[str] = None
) -> Tuple[bool, str, Optional[Dict[str, Any]]]:
    """
    Mark attendance for a student for the active session date.
    Enforces uniqueness: cannot mark duplicate attendance on the same date.
    """
    session_date = target_date or get_active_session_date()
    current_time = datetime.datetime.now().strftime("%I:%M:%S %p")

    student = get_student_by_id(student_id)
    if not student:
        return False, "Student not found.", None

    batch_id = student.get("batch_id")

    with get_db_connection() as conn:
        cursor = conn.cursor()
        cursor.execute("""
            SELECT a.id, a.date, a.time, a.status, a.confidence, s.name, s.enrollment_no, s.student_class, s.photo_filename
            FROM attendance a
            JOIN students s ON a.student_id = s.id
            WHERE a.student_id = ? AND a.date = ?
        """, (student_id, session_date))
        existing = cursor.fetchone()

        if existing:
            rec = dict(existing)
            return False, f"Attendance already recorded for {session_date} at {rec['time']}.", rec

        try:
            cursor.execute("""
                INSERT INTO attendance (student_id, batch_id, date, time, status, confidence, marked_by)
                VALUES (?, ?, ?, ?, 'Present', ?, ?)
            """, (student_id, batch_id, session_date, current_time, round(confidence, 1), marked_by))
            conn.commit()
            new_id = cursor.lastrowid

            cursor.execute("""
                SELECT a.id, a.date, a.time, a.status, a.confidence, s.name, s.enrollment_no, s.student_class, s.photo_filename,
                       b.batch_name
                FROM attendance a
                JOIN students s ON a.student_id = s.id
                LEFT JOIN batches b ON a.batch_id = b.id
                WHERE a.id = ?
            """, (new_id,))
            record = dict(cursor.fetchone())

            cursor.execute("SELECT COUNT(DISTINCT date) FROM attendance WHERE student_id = ? AND status = 'Present'", (student_id,))
            record["total_days_attended"] = cursor.fetchone()[0]

            return True, f"Attendance marked Present for {record['name']} on {session_date}.", record
        except sqlite3.IntegrityError:
            return False, f"Attendance already recorded for {session_date}.", None
        except Exception as e:
            return False, f"Error saving attendance: {str(e)}", None


def manual_mark_attendance(
    student_id: int,
    date_str: str,
    time_str: str,
    status: str = "Present",
    marked_by: str = "Teacher Manual Entry"
) -> Tuple[bool, str]:
    student = get_student_by_id(student_id)
    batch_id = student.get("batch_id") if student else None

    with get_db_connection() as conn:
        cursor = conn.cursor()
        try:
            cursor.execute("""
                INSERT INTO attendance (student_id, batch_id, date, time, status, confidence, marked_by)
                VALUES (?, ?, ?, ?, ?, 100.0, ?)
            """, (student_id, batch_id, date_str, time_str, status, marked_by))
            conn.commit()
            return True, "Manual attendance recorded successfully."
        except sqlite3.IntegrityError:
            cursor.execute("""
                UPDATE attendance
                SET status = ?, time = ?, marked_by = ?, batch_id = ?
                WHERE student_id = ? AND date = ?
            """, (status, time_str, marked_by, batch_id, student_id, date_str))
            conn.commit()
            return True, "Existing attendance record updated."
        except Exception as e:
            return False, f"Failed: {str(e)}"


def delete_attendance_record(record_id: int) -> Tuple[bool, str]:
    with get_db_connection() as conn:
        cursor = conn.cursor()
        cursor.execute("DELETE FROM attendance WHERE id = ?", (record_id,))
        conn.commit()
        if cursor.rowcount > 0:
            return True, "Attendance record removed successfully."
        return False, "Record not found."


def get_today_attendance(date_str: Optional[str] = None, batch_id: Optional[int] = None) -> List[Dict[str, Any]]:
    session_date = date_str or get_active_session_date()
    query = """
        SELECT a.id, a.student_id, a.batch_id, a.date, a.time, a.status, a.confidence, a.marked_by,
               s.name, s.enrollment_no, s.student_class, s.photo_filename,
               b.batch_name
        FROM attendance a
        JOIN students s ON a.student_id = s.id
        LEFT JOIN batches b ON s.batch_id = b.id
        WHERE a.date = ?
    """
    params: List[Any] = [session_date]
    if batch_id is not None:
        query += " AND s.batch_id = ?"
        params.append(batch_id)

    query += " ORDER BY a.id DESC"

    with get_db_connection() as conn:
        cursor = conn.cursor()
        cursor.execute(query, params)
        return [dict(row) for row in cursor.fetchall()]


def get_attendance_history(
    date_filter: Optional[str] = None,
    search_query: Optional[str] = None,
    batch_id: Optional[int] = None
) -> List[Dict[str, Any]]:
    query = """
        SELECT a.id, a.student_id, a.batch_id, a.date, a.time, a.status, a.confidence, a.marked_by,
               s.name, s.enrollment_no, s.student_class, s.photo_filename,
               b.batch_name
        FROM attendance a
        JOIN students s ON a.student_id = s.id
        LEFT JOIN batches b ON s.batch_id = b.id
        WHERE 1=1
    """
    params: List[Any] = []

    if batch_id is not None:
        query += " AND s.batch_id = ?"
        params.append(batch_id)

    if date_filter and date_filter.strip():
        query += " AND a.date = ?"
        params.append(date_filter.strip())

    if search_query and search_query.strip():
        pattern = f"%{search_query.strip()}%"
        query += " AND (s.name LIKE ? OR s.enrollment_no LIKE ? OR s.student_class LIKE ? OR b.batch_name LIKE ?)"
        params.extend([pattern, pattern, pattern, pattern])

    query += " ORDER BY a.date DESC, a.time DESC"

    with get_db_connection() as conn:
        cursor = conn.cursor()
        cursor.execute(query, params)
        return [dict(row) for row in cursor.fetchall()]


def get_dashboard_stats(batch_id: Optional[int] = None, teacher_id: Optional[int] = None) -> Dict[str, Any]:
    session_date = get_active_session_date()
    dt = datetime.datetime.strptime(session_date, "%Y-%m-%d")
    today_real = datetime.datetime.now().strftime("%Y-%m-%d")
    is_simulation = (session_date != today_real)

    with get_db_connection() as conn:
        cursor = conn.cursor()

        # Batch info
        active_batch = None
        if batch_id:
            cursor.execute("SELECT * FROM batches WHERE id = ?", (batch_id,))
            b_row = cursor.fetchone()
            if b_row:
                active_batch = dict(b_row)

        # Total students in scope
        if batch_id:
            cursor.execute("SELECT COUNT(*) FROM students WHERE batch_id = ?", (batch_id,))
        else:
            cursor.execute("SELECT COUNT(*) FROM students")
        total_students = cursor.fetchone()[0]

        # Present on active session date
        if batch_id:
            cursor.execute("""
                SELECT COUNT(DISTINCT a.student_id) 
                FROM attendance a 
                JOIN students s ON a.student_id = s.id 
                WHERE a.date = ? AND a.status = 'Present' AND s.batch_id = ?
            """, (session_date, batch_id))
        else:
            cursor.execute("SELECT COUNT(DISTINCT student_id) FROM attendance WHERE date = ? AND status = 'Present'", (session_date,))
        present_today = cursor.fetchone()[0]

        absent_today = max(0, total_students - present_today)

        # Total working days recorded
        if batch_id:
            cursor.execute("""
                SELECT COUNT(DISTINCT a.date)
                FROM attendance a
                JOIN students s ON a.student_id = s.id
                WHERE s.batch_id = ?
            """, (batch_id,))
        else:
            cursor.execute("SELECT COUNT(DISTINCT date) FROM attendance")
        total_working_days = cursor.fetchone()[0]

        # Recent 5 check-ins
        rec_query = """
            SELECT a.id, a.date, a.time, a.status, a.confidence, a.marked_by,
                   s.name, s.enrollment_no, s.student_class, s.photo_filename,
                   b.batch_name
            FROM attendance a
            JOIN students s ON a.student_id = s.id
            LEFT JOIN batches b ON s.batch_id = b.id
            WHERE 1=1
        """
        rec_params: List[Any] = []
        if batch_id:
            rec_query += " AND s.batch_id = ?"
            rec_params.append(batch_id)
        rec_query += " ORDER BY a.id DESC LIMIT 5"
        cursor.execute(rec_query, rec_params)
        recent_records = [dict(row) for row in cursor.fetchall()]

        # Students pending today
        unm_query = """
            SELECT s.id, s.name, s.enrollment_no, s.student_class, s.photo_filename, b.batch_name
            FROM students s
            LEFT JOIN batches b ON s.batch_id = b.id
            WHERE s.id NOT IN (
                SELECT student_id FROM attendance WHERE date = ? AND status = 'Present'
            )
        """
        unm_params: List[Any] = [session_date]
        if batch_id:
            unm_query += " AND s.batch_id = ?"
            unm_params.append(batch_id)
        unm_query += " ORDER BY s.name ASC"
        cursor.execute(unm_query, unm_params)
        unmarked_students = [dict(row) for row in cursor.fetchall()]

        # Multi-batch comparative track records
        all_batches_stats = []
        if teacher_id:
            cursor.execute("SELECT * FROM batches WHERE teacher_id = ? ORDER BY batch_name ASC", (teacher_id,))
            b_rows = cursor.fetchall()
            for br in b_rows:
                b_dict = dict(br)
                bid = b_dict["id"]
                cursor.execute("SELECT COUNT(*) FROM students WHERE batch_id = ?", (bid,))
                b_dict["total_students"] = cursor.fetchone()[0]

                cursor.execute("""
                    SELECT COUNT(DISTINCT a.student_id) 
                    FROM attendance a 
                    JOIN students s ON a.student_id = s.id 
                    WHERE a.date = ? AND a.status = 'Present' AND s.batch_id = ?
                """, (session_date, bid))
                b_dict["present_today"] = cursor.fetchone()[0]

                cursor.execute("""
                    SELECT COUNT(DISTINCT a.date)
                    FROM attendance a
                    JOIN students s ON a.student_id = s.id
                    WHERE s.batch_id = ?
                """, (bid,))
                b_dict["total_sessions"] = cursor.fetchone()[0]

                all_batches_stats.append(b_dict)

    students_summary = get_students_with_attendance_stats(batch_id=batch_id)

    return {
        "today_date": session_date,
        "today_formatted": dt.strftime("%A, %d %B %Y"),
        "is_simulation": is_simulation,
        "real_today": today_real,
        "active_batch": active_batch,
        "total_students": total_students,
        "present_today": present_today,
        "absent_today": absent_today,
        "total_working_days": total_working_days,
        "recent_records": recent_records,
        "unmarked_students": unmarked_students,
        "students_summary": students_summary,
        "all_batches_stats": all_batches_stats
    }
