"""
Database Management Module
--------------------------
Handles SQLite connection, schema definition, initialization, demo seeding,
and parameterized CRUD operations for students, attendance, and admin users.
"""

import os
import shutil
import sqlite3
import datetime
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
    """Initialize database tables, ensure directories exist, and seed demo records."""
    os.makedirs(INSTANCE_DIR, exist_ok=True)
    os.makedirs(UPLOADS_DIR, exist_ok=True)

    with get_db_connection() as conn:
        cursor = conn.cursor()

        # 1. Admins Table
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS admins (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                username TEXT UNIQUE NOT NULL,
                password_hash TEXT NOT NULL,
                full_name TEXT NOT NULL,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        """)

        # 2. Students Table
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS students (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                name TEXT NOT NULL,
                enrollment_no TEXT UNIQUE NOT NULL,
                student_class TEXT NOT NULL,
                photo_filename TEXT NOT NULL,
                face_encoding BLOB NOT NULL,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        """)

        # 3. Attendance Table
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS attendance (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                student_id INTEGER NOT NULL,
                date TEXT NOT NULL,
                time TEXT NOT NULL,
                status TEXT NOT NULL DEFAULT 'Present',
                confidence REAL DEFAULT 0.0,
                marked_by TEXT DEFAULT 'AI Face Recognition',
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                FOREIGN KEY (student_id) REFERENCES students(id) ON DELETE CASCADE,
                UNIQUE(student_id, date)
            )
        """)

        # 4. System Settings Table (for multi-day simulation & active session date)
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS system_settings (
                key TEXT PRIMARY KEY,
                value TEXT NOT NULL
            )
        """)

        # Seed default Admin account if none exists
        cursor.execute("SELECT COUNT(*) FROM admins")
        if cursor.fetchone()[0] == 0:
            default_admin_hash = generate_password_hash("admin123")
            cursor.execute(
                "INSERT INTO admins (username, password_hash, full_name) VALUES (?, ?, ?)",
                ("admin", default_admin_hash, "College Project Administrator")
            )
            print("[Database] Default admin created: username='admin', password='admin123'")

        conn.commit()

    # Seed 2-3 sample students if none exist
    _seed_sample_students_if_needed()


def _seed_sample_students_if_needed():
    """Seed initial demonstration students with verified face encodings and photos."""
    with get_db_connection() as conn:
        cursor = conn.cursor()
        cursor.execute("SELECT COUNT(*) FROM students")
        if cursor.fetchone()[0] > 0:
            return  # Already seeded

    # Import face service here to avoid circular dependencies
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
                add_student(
                    name=item["name"],
                    enrollment_no=item["enrollment_no"],
                    student_class=item["student_class"],
                    photo_filename=item["target_filename"],
                    face_encoding=embedding
                )
                print(f"[Database] Seeded sample student: {item['name']} ({item['enrollment_no']})")


# Admin operations
def get_admin_by_username(username: str) -> Optional[Dict[str, Any]]:
    with get_db_connection() as conn:
        cursor = conn.cursor()
        cursor.execute("SELECT * FROM admins WHERE username = ?", (username.strip(),))
        row = cursor.fetchone()
        return dict(row) if row else None


def verify_admin_login(username: str, password: str) -> Tuple[bool, Optional[Dict[str, Any]], str]:
    admin = get_admin_by_username(username)
    if not admin:
        return False, None, "Invalid username or password."
    if not check_password_hash(admin["password_hash"], password):
        return False, None, "Invalid username or password."
    return True, admin, "Login successful."


# Student operations
def add_student(
    name: str,
    enrollment_no: str,
    student_class: str,
    photo_filename: str,
    face_encoding: np.ndarray
) -> Tuple[bool, str, Optional[int]]:
    """Save student details and 128-D facial vector."""
    try:
        encoding_bytes = face_encoding.astype(np.float32).tobytes()
        with get_db_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("""
                INSERT INTO students (name, enrollment_no, student_class, photo_filename, face_encoding)
                VALUES (?, ?, ?, ?, ?)
            """, (name.strip(), enrollment_no.strip().upper(), student_class.strip(), photo_filename, encoding_bytes))
            conn.commit()
            return True, "Student registered successfully.", cursor.lastrowid
    except sqlite3.IntegrityError:
        return False, f"Enrollment number '{enrollment_no}' already exists.", None
    except Exception as e:
        return False, f"Database error: {str(e)}", None


def get_all_students() -> List[Dict[str, Any]]:
    with get_db_connection() as conn:
        cursor = conn.cursor()
        cursor.execute("SELECT id, name, enrollment_no, student_class, photo_filename, created_at FROM students ORDER BY name ASC")
        return [dict(row) for row in cursor.fetchall()]


def get_student_by_id(student_id: int) -> Optional[Dict[str, Any]]:
    with get_db_connection() as conn:
        cursor = conn.cursor()
        cursor.execute("SELECT * FROM students WHERE id = ?", (student_id,))
        row = cursor.fetchone()
        return dict(row) if row else None


def get_student_by_enrollment(enrollment_no: str) -> Optional[Dict[str, Any]]:
    with get_db_connection() as conn:
        cursor = conn.cursor()
        cursor.execute("SELECT * FROM students WHERE enrollment_no = ?", (enrollment_no.strip().upper(),))
        row = cursor.fetchone()
        return dict(row) if row else None


def delete_student(student_id: int) -> Tuple[bool, str, Optional[str]]:
    """Delete student, cascading attendance, and return photo filename to clean up storage."""
    student = get_student_by_id(student_id)
    if not student:
        return False, "Student not found.", None
    photo_file = student.get("photo_filename")
    with get_db_connection() as conn:
        cursor = conn.cursor()
        cursor.execute("DELETE FROM students WHERE id = ?", (student_id,))
        conn.commit()
    return True, "Student and associated biometric data deleted.", photo_file


def get_all_student_encodings() -> List[Dict[str, Any]]:
    """Retrieve all student records with deserialized 128-D numpy embeddings."""
    with get_db_connection() as conn:
        cursor = conn.cursor()
        cursor.execute("SELECT id, name, enrollment_no, student_class, photo_filename, face_encoding FROM students")
        rows = cursor.fetchall()

    result = []
    for row in rows:
        raw_bytes = row["face_encoding"]
        embedding = np.frombuffer(raw_bytes, dtype=np.float32)
        result.append({
            "id": row["id"],
            "name": row["name"],
            "enrollment_no": row["enrollment_no"],
            "student_class": row["student_class"],
            "photo_filename": row["photo_filename"],
            "embedding": embedding
        })
    return result


# Session Date & Simulation operations
def get_active_session_date() -> str:
    """Retrieve the current active session date. Defaults to actual today if unset."""
    with get_db_connection() as conn:
        cursor = conn.cursor()
        cursor.execute("SELECT value FROM system_settings WHERE key = 'active_session_date'")
        row = cursor.fetchone()
        if row and row["value"]:
            return row["value"]
    return datetime.datetime.now().strftime("%Y-%m-%d")


def set_active_session_date(date_str: str) -> str:
    """Explicitly set the active attendance session date."""
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
    """Advance the active session date by N days (for demoing everyday attendance progression)."""
    current = get_active_session_date()
    dt = datetime.datetime.strptime(current, "%Y-%m-%d")
    next_dt = dt + datetime.timedelta(days=days)
    next_str = next_dt.strftime("%Y-%m-%d")
    set_active_session_date(next_str)
    return current, next_str


def reset_session_date_to_today() -> str:
    """Reset the active session date back to today's real calendar date."""
    today_str = datetime.datetime.now().strftime("%Y-%m-%d")
    return set_active_session_date(today_str)


def get_students_with_attendance_stats() -> List[Dict[str, Any]]:
    """
    Retrieve students with cumulative attendance tracking:
    - days_present: Total days the student attended
    - total_working_days: Total unique days attendance was recorded across all students
    - attendance_rate: Attendance percentage (days_present / total_working_days * 100)
    - last_seen_date, last_seen_time: Most recent check-in
    """
    with get_db_connection() as conn:
        cursor = conn.cursor()
        # Total distinct dates recorded in the system
        cursor.execute("SELECT COUNT(DISTINCT date) FROM attendance")
        total_working_days = cursor.fetchone()[0]

        cursor.execute("""
            SELECT 
                s.id, s.name, s.enrollment_no, s.student_class, s.photo_filename, s.created_at,
                COUNT(DISTINCT a.date) as days_present,
                MAX(a.date) as last_seen_date,
                MAX(a.time) as last_seen_time
            FROM students s
            LEFT JOIN attendance a ON s.id = a.student_id AND a.status = 'Present'
            GROUP BY s.id
            ORDER BY s.name ASC
        """)
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


# Attendance operations
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

    # Check if already marked on this session date
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

        # Insert new record
        try:
            cursor.execute("""
                INSERT INTO attendance (student_id, date, time, status, confidence, marked_by)
                VALUES (?, ?, ?, 'Present', ?, ?)
            """, (student_id, session_date, current_time, round(confidence, 1), marked_by))
            conn.commit()
            new_id = cursor.lastrowid

            cursor.execute("""
                SELECT a.id, a.date, a.time, a.status, a.confidence, s.name, s.enrollment_no, s.student_class, s.photo_filename
                FROM attendance a
                JOIN students s ON a.student_id = s.id
                WHERE a.id = ?
            """, (new_id,))
            record = dict(cursor.fetchone())

            # Also fetch total days attended by this student to display on screen
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
    marked_by: str = "Admin Manual Correction"
) -> Tuple[bool, str]:
    """Allows administrator to manually add or correct an attendance entry."""
    with get_db_connection() as conn:
        cursor = conn.cursor()
        try:
            cursor.execute("""
                INSERT INTO attendance (student_id, date, time, status, confidence, marked_by)
                VALUES (?, ?, ?, ?, 100.0, ?)
            """, (student_id, date_str, time_str, status, marked_by))
            conn.commit()
            return True, "Manual attendance recorded successfully."
        except sqlite3.IntegrityError:
            # Update existing record for that date
            cursor.execute("""
                UPDATE attendance
                SET status = ?, time = ?, marked_by = ?
                WHERE student_id = ? AND date = ?
            """, (status, time_str, marked_by, student_id, date_str))
            conn.commit()
            return True, "Existing attendance record updated."
        except Exception as e:
            return False, f"Failed: {str(e)}"


def delete_attendance_record(record_id: int) -> Tuple[bool, str]:
    """Delete an attendance record (Admin manual correction)."""
    with get_db_connection() as conn:
        cursor = conn.cursor()
        cursor.execute("DELETE FROM attendance WHERE id = ?", (record_id,))
        conn.commit()
        if cursor.rowcount > 0:
            return True, "Attendance record removed successfully."
        return False, "Record not found."


def get_today_attendance(date_str: Optional[str] = None) -> List[Dict[str, Any]]:
    session_date = date_str or get_active_session_date()
    with get_db_connection() as conn:
        cursor = conn.cursor()
        cursor.execute("""
            SELECT a.id, a.student_id, a.date, a.time, a.status, a.confidence, a.marked_by,
                   s.name, s.enrollment_no, s.student_class, s.photo_filename
            FROM attendance a
            JOIN students s ON a.student_id = s.id
            WHERE a.date = ?
            ORDER BY a.id DESC
        """, (session_date,))
        return [dict(row) for row in cursor.fetchall()]


def get_attendance_history(date_filter: Optional[str] = None, search_query: Optional[str] = None) -> List[Dict[str, Any]]:
    """Retrieve full or filtered attendance records with student information."""
    query = """
        SELECT a.id, a.student_id, a.date, a.time, a.status, a.confidence, a.marked_by,
               s.name, s.enrollment_no, s.student_class, s.photo_filename
        FROM attendance a
        JOIN students s ON a.student_id = s.id
        WHERE 1=1
    """
    params: List[Any] = []

    if date_filter and date_filter.strip():
        query += " AND a.date = ?"
        params.append(date_filter.strip())

    if search_query and search_query.strip():
        pattern = f"%{search_query.strip()}%"
        query += " AND (s.name LIKE ? OR s.enrollment_no LIKE ? OR s.student_class LIKE ?)"
        params.extend([pattern, pattern, pattern])

    query += " ORDER BY a.date DESC, a.time DESC"

    with get_db_connection() as conn:
        cursor = conn.cursor()
        cursor.execute(query, params)
        return [dict(row) for row in cursor.fetchall()]


def get_dashboard_stats() -> Dict[str, Any]:
    """Calculate dashboard metrics for the current active session date."""
    session_date = get_active_session_date()
    dt = datetime.datetime.strptime(session_date, "%Y-%m-%d")
    today_real = datetime.datetime.now().strftime("%Y-%m-%d")
    is_simulation = (session_date != today_real)

    with get_db_connection() as conn:
        cursor = conn.cursor()

        # Total students
        cursor.execute("SELECT COUNT(*) FROM students")
        total_students = cursor.fetchone()[0]

        # Present on active session date
        cursor.execute("SELECT COUNT(DISTINCT student_id) FROM attendance WHERE date = ? AND status = 'Present'", (session_date,))
        present_today = cursor.fetchone()[0]

        # Students not marked present on active session date
        absent_today = max(0, total_students - present_today)

        # Total working days recorded across all dates
        cursor.execute("SELECT COUNT(DISTINCT date) FROM attendance")
        total_working_days = cursor.fetchone()[0]

        # Recent 5 attendance records across all dates
        cursor.execute("""
            SELECT a.id, a.date, a.time, a.status, a.confidence, a.marked_by,
                   s.name, s.enrollment_no, s.student_class, s.photo_filename
            FROM attendance a
            JOIN students s ON a.student_id = s.id
            ORDER BY a.id DESC LIMIT 5
        """)
        recent_records = [dict(row) for row in cursor.fetchall()]

        # Students not yet marked present for this active session date
        cursor.execute("""
            SELECT s.id, s.name, s.enrollment_no, s.student_class, s.photo_filename
            FROM students s
            WHERE s.id NOT IN (
                SELECT student_id FROM attendance WHERE date = ? AND status = 'Present'
            )
            ORDER BY s.name ASC
        """, (session_date,))
        unmarked_students = [dict(row) for row in cursor.fetchall()]

    # Student leaderboard / attendance track
    students_summary = get_students_with_attendance_stats()

    return {
        "today_date": session_date,
        "today_formatted": dt.strftime("%A, %d %B %Y"),
        "is_simulation": is_simulation,
        "real_today": today_real,
        "total_students": total_students,
        "present_today": present_today,
        "absent_today": absent_today,
        "total_working_days": total_working_days,
        "recent_records": recent_records,
        "unmarked_students": unmarked_students,
        "students_summary": students_summary
    }
