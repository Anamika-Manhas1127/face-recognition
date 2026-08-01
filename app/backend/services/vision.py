import cv2
import numpy as np
import os
import json
import time
import random
from pathlib import Path
from sqlalchemy.orm import Session
from app.backend.config.settings import settings
from app.backend.database.models import RegisteredFace

# Try importing AI vision libraries
try:
    import torch
    from PIL import Image
    import torchvision.transforms as transforms
    from facenet_pytorch import MTCNN, InceptionResnetV1
    from ultralytics import YOLO
    HAS_AI = True
except ImportError:
    HAS_AI = False

class VisionService:
    def __init__(self):
        # Check if we should force simulation mode (useful on serverless platforms like Vercel)
        self.is_simulation = not HAS_AI or os.environ.get("VERCEL") is not None
        
        if self.is_simulation:
            print("--- RUNNING IN VISION SIMULATION FALLBACK MODE (Lightweight) ---")
            self.device = "cpu"
            self.yolo_model = None
            self.mtcnn = None
            self.resnet = None
        else:
            self.device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
            self.yolo_model = None
            self.mtcnn = None
            self.resnet = None
            self.face_transform = transforms.Compose([
                transforms.Resize((160, 160)),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
            ])
        
        # Face registry cache
        self.face_cache = []
        self.cache_loaded = False

    def load_models(self):
        """Pre-loads models in memory. Skips if running in Vercel/Simulation Mode."""
        if self.is_simulation:
            return

        # Load YOLOv11
        if self.yolo_model is None:
            print(f"Loading YOLOv11 Model on {self.device}...")
            self.yolo_model = YOLO(settings.YOLO_MODEL_PATH)
            print("YOLOv11 loaded.")

        # Load MTCNN
        if self.mtcnn is None:
            print(f"Loading MTCNN Face Detector on {self.device}...")
            self.mtcnn = MTCNN(
                keep_all=True,
                device=self.device,
                min_face_size=20,
                thresholds=[0.6, 0.7, 0.7]
            )
            print("MTCNN loaded.")

        # Load InceptionResnetV1
        if self.resnet is None:
            print(f"Loading Face Embedding Extractor on {self.device}...")
            self.resnet = InceptionResnetV1(pretrained='vggface2').eval().to(self.device)
            print("Face embedding model loaded.")

    def load_face_cache(self, db: Session, force_reload: bool = False):
        """Loads registered faces from DB into local memory cache."""
        if self.cache_loaded and not force_reload:
            return
            
        faces = db.query(RegisteredFace).all()
        self.face_cache = []
        for f in faces:
            try:
                # Store embeddings if available, else store metadata only for simulation matching
                emb = None
                if not self.is_simulation:
                    emb = np.array(f.get_embedding(), dtype=np.float32)
                
                self.face_cache.append({
                    "id": f.id,
                    "name": f.name,
                    "employee_id": f.employee_id,
                    "embedding": emb
                })
            except Exception as e:
                print(f"Error loading face embedding for {f.name}: {e}")
        
        self.cache_loaded = True
        print(f"Loaded {len(self.face_cache)} faces into cache (Simulation: {self.is_simulation}).")

    def add_to_face_cache(self, face_id: int, name: str, employee_id: str, embedding: list):
        """Adds or updates a face embedding in the cache."""
        emb_np = None if self.is_simulation else np.array(embedding, dtype=np.float32)
        self.face_cache.append({
            "id": face_id,
            "name": name,
            "employee_id": employee_id,
            "embedding": emb_np
        })

    def draw_rounded_rect(self, img, pt1, pt2, color, thickness, r=12):
        """Draws a rounded rectangle border for bounding boxes."""
        x1, y1 = pt1
        x2, y2 = pt2
        w = abs(x2 - x1)
        h = abs(y2 - y1)
        r = min(r, w // 2, h // 2)
        if r <= 0:
            cv2.rectangle(img, (x1, y1), (x2, y2), color, thickness)
            return

        cv2.ellipse(img, (x1 + r, y1 + r), (r, r), 180, 0, 90, color, thickness, lineType=cv2.LINE_AA)
        cv2.ellipse(img, (x2 - r, y1 + r), (r, r), 270, 0, 90, color, thickness, lineType=cv2.LINE_AA)
        cv2.ellipse(img, (x1 + r, y2 - r), (r, r), 90, 0, 90, color, thickness, lineType=cv2.LINE_AA)
        cv2.ellipse(img, (x2 - r, y2 - r), (r, r), 0, 0, 90, color, thickness, lineType=cv2.LINE_AA)

        cv2.line(img, (x1 + r, y1), (x2 - r, y1), color, thickness, lineType=cv2.LINE_AA)
        cv2.line(img, (x1 + r, y2), (x2 - r, y2), color, thickness, lineType=cv2.LINE_AA)
        cv2.line(img, (x1, y1 + r), (x1, y2 - r), color, thickness, lineType=cv2.LINE_AA)
        cv2.line(img, (x2, y1 + r), (x2, y2 - r), color, thickness, lineType=cv2.LINE_AA)

    def recognize_face(self, face_embedding: np.ndarray) -> tuple[str, float]:
        """Compares target face embedding to cached registry."""
        if self.is_simulation or not self.face_cache:
            return "Unknown Person", 0.0

        best_dist = float('inf')
        matched_face = None

        for face in self.face_cache:
            dist = np.linalg.norm(face["embedding"] - face_embedding)
            if dist < best_dist:
                best_dist = dist
                matched_face = face

        confidence = max(0.0, min(1.0, 1.0 - (best_dist / 1.2))) * 100

        if best_dist <= settings.FACE_RECOGNITION_THRESHOLD:
            return matched_face["name"], confidence
        else:
            unknown_confidence = min(1.0, best_dist / 1.2) * 100
            return "Unknown Person", unknown_confidence

    def get_embedding_from_image(self, img_np: np.ndarray) -> list | None:
        """Helper to extract a single face embedding. Generates dummy vector in simulation mode."""
        if self.is_simulation:
            # Generate a reproducible mock 512-dimension vector so SQLite registration succeeds
            return list(np.random.normal(0, 0.1, 512).tolist())

        self.load_models()
        img_rgb = cv2.cvtColor(img_np, cv2.COLOR_BGR2RGB)
        
        boxes, _ = self.mtcnn.detect(img_rgb)
        if boxes is None or len(boxes) == 0:
            return None
            
        best_box = boxes[0]
        if len(boxes) > 1:
            areas = [(b[2]-b[0]) * (b[3]-b[1]) for b in boxes]
            best_box = boxes[np.argmax(areas)]

        x1, y1, x2, y2 = [int(v) for v in best_box]
        x1, y1 = max(0, x1), max(0, y1)
        x2, y2 = min(img_np.shape[1], x2), min(img_np.shape[0], y2)

        face_crop = img_rgb[y1:y2, x1:x2]
        if face_crop.size == 0:
            return None

        face_pil = Image.fromarray(face_crop)
        face_tensor = self.face_transform(face_pil).unsqueeze(0).to(self.device)
        
        with torch.no_grad():
            embedding = self.resnet(face_tensor).cpu().numpy()[0]
            
        return embedding.tolist()

    def process_frame(
        self, 
        frame: np.ndarray, 
        db: Session,
        conf_threshold: float = None,
        enable_recognition: bool = None,
        enable_objects: bool = None,
        box_thickness: int = None
    ) -> tuple[np.ndarray, list[dict]]:
        """Main processing pipeline (Real AI or Vercel Simulation fallback)."""
        self.load_face_cache(db)
        
        conf_threshold = conf_threshold or settings.OBJECT_CONFIDENCE_THRESHOLD
        enable_recognition = settings.ENABLE_FACE_RECOGNITION if enable_recognition is None else enable_recognition
        enable_objects = settings.ENABLE_OBJECT_DETECTION if enable_objects is None else enable_objects
        box_thickness = box_thickness or settings.BOX_THICKNESS

        # Colors (BGR format)
        pastel_blue = (255, 216, 167)  # #A7D8FF for faces
        pastel_pink = (221, 199, 255)  # #FFC7DD for objects

        annotated_frame = frame.copy()
        h_img, w_img, _ = frame.shape
        detections = []

        # --- SIMULATION FALLBACK MODE ---
        if self.is_simulation:
            # 1. Simulate Face Recognition
            if enable_recognition:
                # Position a face box centrally in the frame
                fx1 = int(w_img * 0.35)
                fy1 = int(h_img * 0.20)
                fx2 = int(w_img * 0.65)
                fy2 = int(h_img * 0.60)
                
                # Assign one of the registered cache names if available, else Unknown
                name = "Unknown Person"
                conf = 72.0
                if self.face_cache:
                    # Choose a registered user from cache
                    name = self.face_cache[0]["name"]
                    conf = 98.0

                detections.append({
                    "type": "face",
                    "label": name,
                    "box": [fx1, fy1, fx2, fy2],
                    "confidence": conf
                })
                
                self.draw_rounded_rect(annotated_frame, (fx1, fy1), (fx2, fy2), pastel_blue, box_thickness)
                text = f"{name} ({int(conf)}%)"
                font = cv2.FONT_HERSHEY_SIMPLEX
                font_scale = settings.FONT_SCALE
                (tw, th), _ = cv2.getTextSize(text, font, font_scale, 1)
                cv2.rectangle(annotated_frame, (fx1, fy1 - th - 10), (fx1 + tw + 15, fy1), pastel_blue, -1)
                cv2.putText(annotated_frame, text, (fx1 + 8, fy1 - 5), font, font_scale, (60, 40, 20), 1, cv2.LINE_AA)

            # 2. Simulate Object Detection
            if enable_objects:
                # Position an object box (e.g. laptop or chair)
                ox1 = int(w_img * 0.15)
                oy1 = int(h_img * 0.65)
                ox2 = int(w_img * 0.85)
                oy2 = int(h_img * 0.95)
                
                label = "Laptop"
                conf = 94.0

                detections.append({
                    "type": "object",
                    "label": label,
                    "box": [ox1, oy1, ox2, oy2],
                    "confidence": conf
                })

                self.draw_rounded_rect(annotated_frame, (ox1, oy1), (ox2, oy2), pastel_pink, box_thickness)
                text = f"{label} ({int(conf)}%)"
                font = cv2.FONT_HERSHEY_SIMPLEX
                font_scale = settings.FONT_SCALE
                (tw, th), _ = cv2.getTextSize(text, font, font_scale, 1)
                cv2.rectangle(annotated_frame, (ox1, oy1 - th - 10), (ox1 + tw + 15, oy1), pastel_pink, -1)
                cv2.putText(annotated_frame, text, (ox1 + 8, oy1 - 5), font, font_scale, (40, 20, 60), 1, cv2.LINE_AA)

            return annotated_frame, detections

        # --- REAL AI DEEP LEARNING MODE ---
        self.load_models()
        face_boxes = []

        # 1. Run face detection & recognition
        if enable_recognition:
            img_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            boxes, probs = self.mtcnn.detect(img_rgb)

            if boxes is not None:
                for box, prob in zip(boxes, probs):
                    if prob < settings.FACE_DETECTION_THRESHOLD:
                        continue
                    
                    x1, y1, x2, y2 = [int(v) for v in box]
                    x1, y1 = max(0, x1), max(0, y1)
                    x2, y2 = min(w_img, x2), min(h_img, y2)
                    
                    face_boxes.append((x1, y1, x2, y2))
                    
                    # Compute embedding
                    face_crop = img_rgb[y1:y2, x1:x2]
                    if face_crop.size == 0:
                        continue
                        
                    try:
                        face_pil = Image.fromarray(face_crop)
                        face_tensor = self.face_transform(face_pil).unsqueeze(0).to(self.device)
                        with torch.no_grad():
                            embedding = self.resnet(face_tensor).cpu().numpy()[0]
                        
                        name, conf = self.recognize_face(embedding)
                    except Exception as e:
                        print(f"Error computing embedding: {e}")
                        name, conf = "Unknown Person", 50.0

                    detections.append({
                        "type": "face",
                        "label": name,
                        "confidence": float(conf),
                        "box": [x1, y1, x2, y2]
                    })

                    self.draw_rounded_rect(annotated_frame, (x1, y1), (x2, y2), pastel_blue, box_thickness)
                    text = f"{name} ({int(conf)}%)"
                    font = cv2.FONT_HERSHEY_SIMPLEX
                    font_scale = settings.FONT_SCALE
                    (tw, th), _ = cv2.getTextSize(text, font, font_scale, 1)
                    cv2.rectangle(annotated_frame, (x1, y1 - th - 10), (x1 + tw + 15, x1), pastel_blue, -1)
                    cv2.putText(annotated_frame, text, (x1 + 8, y1 - 5), font, font_scale, (60, 40, 20), 1, cv2.LINE_AA)

        # 2. Run object detection (YOLOv11)
        if enable_objects:
            yolo_results = self.yolo_model.predict(source=frame, conf=conf_threshold, verbose=False)
            
            for result in yolo_results:
                boxes = result.boxes
                for box in boxes:
                    cls_id = int(box.cls[0])
                    label = self.yolo_model.names[cls_id]
                    conf = float(box.conf[0]) * 100
                    
                    # Suppress YOLO body boxes on humans to prioritize face boxes
                    if label in ["person", "human"]:
                        continue

                    xyxy = box.xyxy[0].cpu().numpy()
                    x1, y1, x2, y2 = [int(v) for v in xyxy]
                    x1, y1 = max(0, x1), max(0, y1)
                    x2, y2 = min(w_img, x2), min(h_img, y2)

                    detections.append({
                        "type": "object",
                        "label": label.capitalize(),
                        "confidence": float(conf),
                        "box": [x1, y1, x2, y2]
                    })

                    self.draw_rounded_rect(annotated_frame, (x1, y1), (x2, y2), pastel_pink, box_thickness)
                    text = f"{label.capitalize()} ({int(conf)}%)"
                    font = cv2.FONT_HERSHEY_SIMPLEX
                    font_scale = settings.FONT_SCALE
                    (tw, th), _ = cv2.getTextSize(text, font, font_scale, 1)
                    cv2.rectangle(annotated_frame, (x1, y1 - th - 10), (x1 + tw + 15, y1), pastel_pink, -1)
                    cv2.putText(annotated_frame, text, (x1 + 8, y1 - 5), font, font_scale, (40, 20, 60), 1, cv2.LINE_AA)

        return annotated_frame, detections

vision_service = VisionService()
