import cv2
import numpy as np
import torch
from PIL import Image
import os
import json
import time
from pathlib import Path
from sqlalchemy.orm import Session
from facenet_pytorch import MTCNN, InceptionResnetV1
from ultralytics import YOLO
import torchvision.transforms as transforms

from app.backend.config.settings import settings
from app.backend.database.models import RegisteredFace

class VisionService:
    def __init__(self):
        self.device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
        
        # Lazy load models to keep startup fast
        self.yolo_model = None
        self.mtcnn = None
        self.resnet = None
        
        # Face registry cache
        # List of dicts: {"id": int, "name": str, "embedding": np.ndarray}
        self.face_cache = []
        self.cache_loaded = False
        
        # Image transformation for face embeddings
        self.face_transform = transforms.Compose([
            transforms.Resize((160, 160)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
        ])

    def load_models(self):
        """Pre-loads all models. Runs on first inference or initialization."""
        # Load YOLOv11
        if self.yolo_model is None:
            print(f"Loading YOLOv11 Model on {self.device}...")
            # If the weights file does not exist, YOLO will download it automatically
            self.yolo_model = YOLO(settings.YOLO_MODEL_PATH)
            print("YOLOv11 loaded successfully.")

        # Load MTCNN face detector
        if self.mtcnn is None:
            print(f"Loading MTCNN Face Detector on {self.device}...")
            self.mtcnn = MTCNN(
                keep_all=True,
                device=self.device,
                min_face_size=20,
                thresholds=[0.6, 0.7, 0.7] # MTCNN default stage thresholds
            )
            print("MTCNN loaded successfully.")

        # Load InceptionResnetV1 face embedding extractor
        if self.resnet is None:
            print(f"Loading Face Embedding Extractor on {self.device}...")
            self.resnet = InceptionResnetV1(pretrained='vggface2').eval().to(self.device)
            print("Face embedding model loaded successfully.")

    def load_face_cache(self, db: Session, force_reload: bool = False):
        """Loads registered faces from DB into local memory cache for O(1) matching."""
        if self.cache_loaded and not force_reload:
            return
            
        faces = db.query(RegisteredFace).all()
        self.face_cache = []
        for f in faces:
            try:
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
        print(f"Loaded {len(self.face_cache)} faces into embedding cache.")

    def add_to_face_cache(self, face_id: int, name: str, employee_id: str, embedding: list):
        """Adds or updates a face embedding in the cache."""
        self.face_cache.append({
            "id": face_id,
            "name": name,
            "employee_id": employee_id,
            "embedding": np.array(embedding, dtype=np.float32)
        })

    def draw_rounded_rect(self, img, pt1, pt2, color, thickness, r=12):
        """Draws a beautiful rounded rectangle border for the bounding box."""
        x1, y1 = pt1
        x2, y2 = pt2
        
        # Ensure radius isn't too large for the bounding box size
        w = abs(x2 - x1)
        h = abs(y2 - y1)
        r = min(r, w // 2, h // 2)
        if r <= 0:
            cv2.rectangle(img, (x1, y1), (x2, y2), color, thickness)
            return

        # Top-left corner arc
        cv2.ellipse(img, (x1 + r, y1 + r), (r, r), 180, 0, 90, color, thickness, lineType=cv2.LINE_AA)
        # Top-right corner arc
        cv2.ellipse(img, (x2 - r, y1 + r), (r, r), 270, 0, 90, color, thickness, lineType=cv2.LINE_AA)
        # Bottom-left corner arc
        cv2.ellipse(img, (x1 + r, y2 - r), (r, r), 90, 0, 90, color, thickness, lineType=cv2.LINE_AA)
        # Bottom-right corner arc
        cv2.ellipse(img, (x2 - r, y2 - r), (r, r), 0, 0, 90, color, thickness, lineType=cv2.LINE_AA)

        # Connecting straight lines
        cv2.line(img, (x1 + r, y1), (x2 - r, y1), color, thickness, lineType=cv2.LINE_AA) # top
        cv2.line(img, (x1 + r, y2), (x2 - r, y2), color, thickness, lineType=cv2.LINE_AA) # bottom
        cv2.line(img, (x1, y1 + r), (x1, y2 - r), color, thickness, lineType=cv2.LINE_AA) # left
        cv2.line(img, (x2, y1 + r), (x2, y2 - r), color, thickness, lineType=cv2.LINE_AA) # right

    def recognize_face(self, face_embedding: np.ndarray) -> tuple[str, float]:
        """Compares target face embedding to cached registry. Returns name and similarity/confidence."""
        if not self.face_cache:
            return "Unknown Person", 0.0

        best_dist = float('inf')
        matched_face = None

        # Compare using Euclidean distance
        for face in self.face_cache:
            dist = np.linalg.norm(face["embedding"] - face_embedding)
            if dist < best_dist:
                best_dist = dist
                matched_face = face

        # Convert distance to confidence percentage
        # Dynamic calculation based on typical face thresholds
        # Standard threshold is 0.60. Anything below 0.60 is a match.
        # Mapping dist [0.0, 1.2] to confidence [100%, 0%]
        confidence = max(0.0, min(1.0, 1.0 - (best_dist / 1.2))) * 100

        if best_dist <= settings.FACE_RECOGNITION_THRESHOLD:
            return matched_face["name"], confidence
        else:
            # For unknown faces, still report a mock or actual distance-based confidence
            # representing how sure we are that it is an unknown/new face
            unknown_confidence = min(1.0, best_dist / 1.2) * 100
            return "Unknown Person", unknown_confidence

    def get_embedding_from_image(self, img_np: np.ndarray) -> list | None:
        """Helper to extract a single face embedding from a registry photo."""
        self.load_models()
        img_rgb = cv2.cvtColor(img_np, cv2.COLOR_BGR2RGB)
        
        # Detect boxes
        boxes, _ = self.mtcnn.detect(img_rgb)
        if boxes is None or len(boxes) == 0:
            return None
            
        # Take the largest face if multiple detected
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

        # Convert crop to PIL and calculate embedding
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
        """
        Main vision pipeline. Processes a single frame and returns the annotated frame
        and a list of detections metadata.
        """
        # Load models dynamically if not pre-loaded
        self.load_models()
        self.load_face_cache(db)

        # Set variables from parameters or global settings
        conf_threshold = conf_threshold or settings.OBJECT_CONFIDENCE_THRESHOLD
        enable_recognition = settings.ENABLE_FACE_RECOGNITION if enable_recognition is None else enable_recognition
        enable_objects = settings.ENABLE_OBJECT_DETECTION if enable_objects is None else enable_objects
        box_thickness = box_thickness or settings.BOX_THICKNESS

        # Detections log
        detections = []
        annotated_frame = frame.copy()
        h_img, w_img, _ = frame.shape

        # Colors (BGR format)
        pastel_blue = (255, 216, 167)  # #A7D8FF for faces
        pastel_pink = (221, 199, 255)  # #FFC7DD for objects
        white = (255, 255, 255)

        # Flag to indicate face detection ran
        has_face_run = False
        face_boxes = []

        # 1. RUN FACE DETECTION FIRST (if enabled)
        if enable_recognition:
            has_face_run = True
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
                    
                    # Generate embedding
                    face_crop = img_rgb[y1:y2, x1:x2]
                    if face_crop.size == 0:
                        continue
                        
                    try:
                        face_pil = Image.fromarray(face_crop)
                        face_tensor = self.face_transform(face_pil).unsqueeze(0).to(self.device)
                        with torch.no_grad():
                            embedding = self.resnet(face_tensor).cpu().numpy()[0]
                        
                        # Match name
                        name, conf = self.recognize_face(embedding)
                    except Exception as e:
                        print(f"Error computing face embedding: {e}")
                        name, conf = "Unknown Person", 50.0

                    detections.append({
                        "type": "face",
                        "label": name,
                        "confidence": float(conf),
                        "box": [x1, y1, x2, y2]
                    })

                    # Draw Custom Rounded Pastel Blue Bounding Box
                    self.draw_rounded_rect(annotated_frame, (x1, y1), (x2, y2), pastel_blue, box_thickness)

                    # Text label background & text
                    text = f"{name} ({int(conf)}%)"
                    font = cv2.FONT_HERSHEY_SIMPLEX
                    font_scale = settings.FONT_SCALE
                    # Calculate label size
                    (tw, th), baseline = cv2.getTextSize(text, font, font_scale, 1)
                    # Draw a neat pill background
                    cv2.rectangle(
                        annotated_frame, 
                        (x1, y1 - th - 10), 
                        (x1 + tw + 15, y1), 
                        pastel_blue, 
                        thickness=-1
                    )
                    # Draw text in contrasting dark blue/black color for readability
                    cv2.putText(
                        annotated_frame, 
                        text, 
                        (x1 + 8, y1 - 5), 
                        font, 
                        font_scale, 
                        (60, 40, 20), 
                        1, 
                        cv2.LINE_AA
                    )

        # 2. RUN OBJECT DETECTION (YOLOv11)
        if enable_objects:
            # Predict using YOLO nano
            yolo_results = self.yolo_model.predict(
                source=frame, 
                conf=conf_threshold,
                verbose=False
            )
            
            for result in yolo_results:
                boxes = result.boxes
                for box in boxes:
                    cls_id = int(box.cls[0])
                    label = self.yolo_model.names[cls_id]
                    conf = float(box.conf[0]) * 100
                    
                    # PRIORITY LOGIC:
                    # If object is a 'person' or 'human', do NOT use the object detector label.
                    # Instead, suppress the YOLO person box and let the face detector display the face.
                    if label in ["person", "human"]:
                        continue

                    # Retrieve bounding box coordinates
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

                    # Draw Custom Rounded Pastel Pink Bounding Box
                    self.draw_rounded_rect(annotated_frame, (x1, y1), (x2, y2), pastel_pink, box_thickness)

                    # Text label background & text
                    text = f"{label.capitalize()} ({int(conf)}%)"
                    font = cv2.FONT_HERSHEY_SIMPLEX
                    font_scale = settings.FONT_SCALE
                    # Calculate label size
                    (tw, th), baseline = cv2.getTextSize(text, font, font_scale, 1)
                    # Draw a neat pill background
                    cv2.rectangle(
                        annotated_frame, 
                        (x1, y1 - th - 10), 
                        (x1 + tw + 15, y1), 
                        pastel_pink, 
                        thickness=-1
                    )
                    # Draw text in contrasting dark pink/red color for readability
                    cv2.putText(
                        annotated_frame, 
                        text, 
                        (x1 + 8, y1 - 5), 
                        font, 
                        font_scale, 
                        (40, 20, 60), 
                        1, 
                        cv2.LINE_AA
                    )

        return annotated_frame, detections

# Create a singleton instance of the vision service
vision_service = VisionService()
