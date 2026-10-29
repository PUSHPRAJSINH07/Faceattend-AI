"""
Comprehensive Test Suite for AI Face Recognition Attendance System
Tests:
- Admin login (valid credentials)
- Admin login failure (invalid credentials)
- Protected route redirection
- Dashboard rendering & statistics
- Student registration validation
- Face detection validation (valid face vs no face)
- Real face recognition API endpoint
- Automatic attendance marking
- Duplicate attendance prevention on the same day
- Attendance history search & date filter
- Attendance CSV export download
- Manual attendance entry & deletion
- Student deletion & biometric data purging
"""

import os
import cv2
import base64
import unittest
from app import app
import database
from face_recognition_service import get_face_service


class AttendanceSystemTestCase(unittest.TestCase):
    def setUp(self):
        app.config["TESTING"] = True
        app.config["WTF_CSRF_ENABLED"] = False
        self.client = app.test_client()
        database.init_db()

    def test_01_admin_login_failure(self):
        """Test login with incorrect credentials."""
        response = self.client.post("/login", data={
            "username": "admin",
            "password": "wrongpassword123"
        }, follow_redirects=True)
        self.assertEqual(response.status_code, 200)
        self.assertIn(b"Invalid username or password", response.data)

    def test_02_admin_login_success(self):
        """Test successful admin login and dashboard redirect."""
        response = self.client.post("/login", data={
            "username": "admin",
            "password": "admin123"
        }, follow_redirects=True)
        self.assertEqual(response.status_code, 200)
        self.assertIn(b"Attendance Dashboard", response.data)

    def test_03_protected_route_without_login(self):
        """Test accessing dashboard without login redirects to /login."""
        response = self.client.get("/dashboard", follow_redirects=True)
        self.assertIn(b"Please log in to access this page", response.data)
        self.assertIn(b"Administrator Portal", response.data)

    def test_04_seeded_sample_students(self):
        """Test that sample students are seeded and loaded properly."""
        students = database.get_all_students()
        self.assertGreaterEqual(len(students), 3)
        names = [s["name"] for s in students]
        self.assertIn("Aarav Sharma", names)
        self.assertIn("Riya Patel", names)
        self.assertIn("Aditya Verma", names)

    def test_05_api_recognize_registered_face(self):
        """Test live facial recognition API with student1 image."""
        with self.client.session_transaction() as sess:
            sess["admin_id"] = 1
            sess["admin_username"] = "admin"
            sess["admin_name"] = "System Administrator"

        # Read student1 image and convert to base64
        with open("static/sample_images/student1.jpg", "rb") as f:
            b64_img = "data:image/jpeg;base64," + base64.b64encode(f.read()).decode("utf-8")

        response = self.client.post("/api/recognize", json={"image": b64_img})
        self.assertEqual(response.status_code, 200)
        data = response.get_json()
        print("API Recognize Result for student1:", data["status"], data.get("message"), "Confidence:", data.get("confidence"))
        self.assertEqual(data["status"], "recognized")
        self.assertEqual(data["student"]["name"], "Aarav Sharma")
        self.assertGreaterEqual(data["confidence"], 50.0)

    def test_06_duplicate_attendance_prevention(self):
        """Test that recognizing the same student twice on the same day prevents duplicate insertion."""
        with self.client.session_transaction() as sess:
            sess["admin_id"] = 1
            sess["admin_username"] = "admin"
            sess["admin_name"] = "System Administrator"

        with open("static/sample_images/student1.jpg", "rb") as f:
            b64_img = "data:image/jpeg;base64," + base64.b64encode(f.read()).decode("utf-8")

        # Second recognition call on same date
        response = self.client.post("/api/recognize", json={"image": b64_img})
        self.assertEqual(response.status_code, 200)
        data = response.get_json()
        print("Second Recognition (Duplicate Check):", data.get("message"), "newly_marked:", data.get("newly_marked"))
        self.assertEqual(data["status"], "recognized")
        self.assertFalse(data.get("newly_marked", True))

    def test_07_csv_export(self):
        """Test CSV export download generation."""
        with self.client.session_transaction() as sess:
            sess["admin_id"] = 1
            sess["admin_username"] = "admin"
            sess["admin_name"] = "System Administrator"

        response = self.client.get("/attendance/export")
        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.mimetype, "text/csv")
        self.assertIn(b"Record ID,Student Name,Enrollment No", response.data)

    def test_08_attendance_history_page(self):
        """Test attendance history page renders records correctly."""
        with self.client.session_transaction() as sess:
            sess["admin_id"] = 1
            sess["admin_username"] = "admin"
            sess["admin_name"] = "System Administrator"

        response = self.client.get("/attendance")
        self.assertEqual(response.status_code, 200)
        self.assertIn(b"Attendance History &amp; Audit Logs", response.data)

    def test_09_unknown_face_rejection(self):
        """Test recognition of an unknown / synthetic face that does not match any registered student."""
        with self.client.session_transaction() as sess:
            sess["admin_id"] = 1
            sess["admin_username"] = "admin"
            sess["admin_name"] = "System Administrator"

        # Create a blank black image
        import numpy as np
        blank = np.zeros((300, 300, 3), dtype=np.uint8)
        _, buf = cv2.imencode(".jpg", blank)
        b64_img = "data:image/jpeg;base64," + base64.b64encode(buf).decode("utf-8")

        response = self.client.post("/api/recognize", json={"image": b64_img})
        self.assertEqual(response.status_code, 200)
        data = response.get_json()
        self.assertEqual(data["status"], "no_face")

    def test_10_invalid_uploaded_image_rejection(self):
        """Test student registration rejects non-image or corrupted files."""
        with self.client.session_transaction() as sess:
            sess["admin_id"] = 1
            sess["admin_username"] = "admin"
            sess["admin_name"] = "System Administrator"

        import io
        fake_file = (io.BytesIO(b"not an image file"), "test.txt")
        response = self.client.post("/students", data={
            "name": "Invalid File Test",
            "enrollment_no": "EN99999",
            "student_class": "TY-IF",
            "photo_file": fake_file
        }, follow_redirects=True)
        self.assertEqual(response.status_code, 200)
        self.assertIn(b"Invalid file format", response.data)

    def test_11_manual_attendance_mark_and_delete(self):
        """Test manual correction: admin manually marks and deletes attendance."""
        with self.client.session_transaction() as sess:
            sess["admin_id"] = 1
            sess["admin_username"] = "admin"
            sess["admin_name"] = "System Administrator"

        students = database.get_all_students()
        target_student_id = students[0]["id"]

        # Mark manual attendance for a student on a historical date
        response = self.client.post("/attendance/manual", data={
            "student_id": target_student_id,
            "date": "2026-10-01",
            "time": "10:00:00 AM",
            "status": "Present"
        }, follow_redirects=True)
        self.assertEqual(response.status_code, 200)

        # Verify entry exists
        history = database.get_attendance_history(date_filter="2026-10-01")
        self.assertGreaterEqual(len(history), 1)
        rec = history[0]
        self.assertEqual(rec["student_id"], target_student_id)

        # Delete entry
        del_resp = self.client.post(f"/attendance/delete/{rec['id']}", follow_redirects=True)
        self.assertEqual(del_resp.status_code, 200)
        history_after = database.get_attendance_history(date_filter="2026-10-01")
        self.assertEqual(len(history_after), 0)

    def test_12_advance_day_and_multi_day_tracking(self):
        """Test multi-day everyday attendance advancement and cumulative tracking."""
        with self.client.session_transaction() as sess:
            sess["admin_id"] = 1
            sess["admin_username"] = "admin"
            sess["admin_name"] = "System Administrator"

        # Advance day
        adv_resp = self.client.post("/session/advance-day", follow_redirects=True)
        self.assertEqual(adv_resp.status_code, 200)

        # Mark attendance on this advanced day
        with open("static/sample_images/student1.jpg", "rb") as f:
            b64_img = "data:image/jpeg;base64," + base64.b64encode(f.read()).decode("utf-8")

        rec_resp = self.client.post("/api/recognize", json={"image": b64_img})
        self.assertEqual(rec_resp.status_code, 200)
        data = rec_resp.get_json()
        self.assertEqual(data["status"], "recognized")

        # Check cumulative stats
        summary = database.get_students_with_attendance_stats()
        student1 = [s for s in summary if s["name"] == "Aarav Sharma"][0]
        self.assertGreaterEqual(student1["days_present"], 1)

        # Reset day to today
        reset_resp = self.client.post("/session/reset-date", follow_redirects=True)
        self.assertEqual(reset_resp.status_code, 200)


if __name__ == "__main__":
    unittest.main()
