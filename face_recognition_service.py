"""
Face Recognition Service Module
--------------------------------
100% Free & Open-Source face detection and recognition using OpenCV Deep Learning:
- YuNet (FaceDetectorYN): Lightweight CNN face detector with 5 facial landmarks.
- SFace (FaceRecognizerSF): Deep 128-D embedding extraction with Cosine similarity matching.

Runs entirely locally on CPU without paid APIs, cloud services, or C++ compilers.
"""

import os
import cv2
import numpy as np
import urllib.request
from typing import Tuple, Optional, List, Dict, Any

# Model paths and URLs
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODELS_DIR = os.path.join(BASE_DIR, "models")
YUNET_MODEL_PATH = os.path.join(MODELS_DIR, "face_detection_yunet_2023mar.onnx")
SFACE_MODEL_PATH = os.path.join(MODELS_DIR, "face_recognition_sface_2021dec.onnx")

YUNET_URL = "https://github.com/opencv/opencv_zoo/raw/main/models/face_detection_yunet/face_detection_yunet_2023mar.onnx"
SFACE_URL = "https://github.com/opencv/opencv_zoo/raw/main/models/face_recognition_sface/face_recognition_sface_2021dec.onnx"

# Default cosine similarity threshold for SFace (0.45 is recommended balance between precision & recall)
DEFAULT_COSINE_THRESHOLD = 0.45


def ensure_models_exist():
    """Ensure pretrained ONNX models are present in models/ directory."""
    os.makedirs(MODELS_DIR, exist_ok=True)
    if not os.path.exists(YUNET_MODEL_PATH) or os.path.getsize(YUNET_MODEL_PATH) < 100000:
        print("[FaceService] Downloading YuNet face detection model...")
        urllib.request.urlretrieve(YUNET_URL, YUNET_MODEL_PATH)
        print("[FaceService] YuNet downloaded successfully.")

    if not os.path.exists(SFACE_MODEL_PATH) or os.path.getsize(SFACE_MODEL_PATH) < 1000000:
        print("[FaceService] Downloading SFace face recognition model...")
        urllib.request.urlretrieve(SFACE_URL, SFACE_MODEL_PATH)
        print("[FaceService] SFace downloaded successfully.")


class FaceRecognitionService:
    def __init__(self, score_threshold: float = 0.6, nms_threshold: float = 0.3):
        ensure_models_exist()
        self.score_threshold = score_threshold
        self.nms_threshold = nms_threshold

        # Initialize YuNet detector with initial dummy dimensions
        self.detector = cv2.FaceDetectorYN.create(
            YUNET_MODEL_PATH,
            "",
            (320, 320),
            score_threshold=self.score_threshold,
            nms_threshold=self.nms_threshold,
            top_k=5000
        )
        # Initialize SFace recognizer
        self.recognizer = cv2.FaceRecognizerSF.create(SFACE_MODEL_PATH, "")
        print("[FaceService] YuNet and SFace models initialized successfully.")

    def detect_faces(self, image: np.ndarray) -> Optional[np.ndarray]:
        """Detect faces in a BGR image array. Returns array of faces or None."""
        if image is None or image.size == 0:
            return None
        h, w = image.shape[:2]
        self.detector.setInputSize((w, h))
        _, faces = self.detector.detect(image)
        return faces

    def extract_embedding(self, image: np.ndarray, face_data: np.ndarray) -> np.ndarray:
        """Align face using 5 landmarks and extract 128-dimensional embedding vector."""
        aligned_face = self.recognizer.alignCrop(image, face_data)
        feature = self.recognizer.feature(aligned_face)
        # Normalize feature vector to unit length
        feat_flat = feature.flatten().astype(np.float32)
        norm = np.linalg.norm(feat_flat)
        if norm > 0:
            feat_flat = feat_flat / norm
        return feat_flat

    def process_image_file(self, image_path: str) -> Tuple[bool, str, Optional[np.ndarray], Optional[List[int]]]:
        """
        Validate and extract face embedding from an image file.
        Returns: (success, message, embedding_vector, bounding_box [x, y, w, h])
        """
        if not os.path.exists(image_path):
            return False, "Image file not found.", None, None

        image = cv2.imread(image_path)
        if image is None:
            return False, "Could not decode image. Please upload a valid JPG or PNG file.", None, None

        faces = self.detect_faces(image)
        if faces is None or len(faces) == 0:
            return False, "No face detected in the photograph. Please upload a clear front-facing passport photo.", None, None

        if len(faces) > 1:
            return False, f"Multiple faces ({len(faces)}) detected. Please upload an image with only one student.", None, None

        face = faces[0]
        bbox = [int(face[0]), int(face[1]), int(face[2]), int(face[3])]
        embedding = self.extract_embedding(image, face)
        return True, "Face detected and biometric embedding generated.", embedding, bbox

    def process_image_bytes(self, image_bytes: bytes) -> Tuple[bool, str, Optional[np.ndarray], Optional[List[int]]]:
        """Validate and extract face embedding from raw image bytes."""
        np_arr = np.frombuffer(image_bytes, np.uint8)
        image = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)
        if image is None:
            return False, "Could not decode uploaded image bytes.", None, None

        faces = self.detect_faces(image)
        if faces is None or len(faces) == 0:
            return False, "No face detected. Please ensure good lighting and face the camera directly.", None, None

        if len(faces) > 1:
            return False, f"Multiple faces ({len(faces)}) detected. Only one student should be in frame.", None, None

        face = faces[0]
        bbox = [int(face[0]), int(face[1]), int(face[2]), int(face[3])]
        embedding = self.extract_embedding(image, face)
        return True, "Face detected successfully.", embedding, bbox

    def compute_similarity(self, emb1: np.ndarray, emb2: np.ndarray) -> float:
        """Compute cosine similarity between two 128-D normalized embeddings."""
        cand = emb1.flatten()
        known = emb2.flatten()
        dot = np.dot(cand, known)
        denom = (np.linalg.norm(cand) * np.linalg.norm(known))
        if denom == 0:
            return 0.0
        return float(dot / denom)

    def identify_face_in_frame(
        self,
        frame: np.ndarray,
        known_students: List[Dict[str, Any]],
        threshold: float = DEFAULT_COSINE_THRESHOLD
    ) -> Dict[str, Any]:
        """
        Analyze a webcam frame against registered students.
        Returns a dictionary with status, matches, bounding boxes, and student details.
        """
        if frame is None or frame.size == 0:
            return {
                "status": "error",
                "message": "Empty frame received from webcam.",
                "faces_count": 0
            }

        faces = self.detect_faces(frame)
        if faces is None or len(faces) == 0:
            return {
                "status": "no_face",
                "message": "No face detected in camera view.",
                "faces_count": 0,
                "box": None
            }

        if len(faces) > 1:
            # Ambiguous: multiple faces in view
            boxes = [[int(f[0]), int(f[1]), int(f[2]), int(f[3])] for f in faces]
            return {
                "status": "multiple_faces",
                "message": f"Multiple faces ({len(faces)}) detected. Please ensure only one person is in front of the camera.",
                "faces_count": len(faces),
                "boxes": boxes
            }

        # Exactly 1 face detected
        face = faces[0]
        box = [int(face[0]), int(face[1]), int(face[2]), int(face[3])]
        candidate_emb = self.extract_embedding(frame, face)

        if not known_students:
            return {
                "status": "unknown",
                "message": "No registered students in the system.",
                "faces_count": 1,
                "confidence": 0.0,
                "box": box
            }

        # Find closest match among registered students
        best_match = None
        best_score = -1.0

        for student in known_students:
            student_emb = student.get("embedding")
            if student_emb is None:
                continue
            sim = self.compute_similarity(candidate_emb, student_emb)
            if sim > best_score:
                best_score = sim
                best_match = student

        # Confidence percentage formatted for human display (normalized between threshold and 1.0)
        conf_percent = round(max(0.0, min(100.0, best_score * 100)), 1)

        if best_match is not None and best_score >= threshold:
            return {
                "status": "recognized",
                "message": f"Recognized: {best_match['name']}",
                "student": best_match,
                "confidence": conf_percent,
                "raw_score": round(best_score, 4),
                "threshold": threshold,
                "box": box,
                "faces_count": 1
            }
        else:
            return {
                "status": "unknown",
                "message": "Unknown Person — Face does not match any registered student.",
                "student": None,
                "confidence": conf_percent,
                "raw_score": round(best_score if best_score > 0 else 0.0, 4),
                "threshold": threshold,
                "box": box,
                "faces_count": 1
            }


# Singleton service instance
_service_instance: Optional[FaceRecognitionService] = None

def get_face_service() -> FaceRecognitionService:
    global _service_instance
    if _service_instance is None:
        _service_instance = FaceRecognitionService()
    return _service_instance
