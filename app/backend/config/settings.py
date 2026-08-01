import os
from pathlib import Path

# Base Paths
BASE_DIR = Path(__file__).resolve().parent.parent.parent.parent

# App Directories
APP_DIR = BASE_DIR / "app"
BACKEND_DIR = APP_DIR / "backend"
FRONTEND_DIR = APP_DIR / "frontend"

# static & templates paths
STATIC_DIR = FRONTEND_DIR / "static"
TEMPLATES_DIR = FRONTEND_DIR / "templates"

# Upload and Database Directories
if os.environ.get("VERCEL"):
    UPLOAD_DIR = Path("/tmp/uploads")
    DATABASE_DIR = Path("/tmp/database")
    LOGS_DIR = Path("/tmp/logs")
    MODELS_DIR = Path("/tmp/models")
else:
    UPLOAD_DIR = BASE_DIR / "uploads"
    DATABASE_DIR = BASE_DIR / "database"
    LOGS_DIR = BASE_DIR / "logs"
    MODELS_DIR = BASE_DIR / "models"

REGISTERED_FACES_DIR = UPLOAD_DIR / "registered_faces"
HISTORY_SCREENSHOTS_DIR = UPLOAD_DIR / "history_screenshots"
TEMP_DIR = UPLOAD_DIR / "temp"

# Ensure all writable directories exist
for directory in [
    UPLOAD_DIR, REGISTERED_FACES_DIR, HISTORY_SCREENSHOTS_DIR,
    TEMP_DIR, DATABASE_DIR, LOGS_DIR, MODELS_DIR
]:
    directory.mkdir(parents=True, exist_ok=True)

# Only ensure static directories exist on local machine (read-only on Vercel)
if not os.environ.get("VERCEL"):
    for directory in [
        STATIC_DIR, TEMPLATES_DIR,
        STATIC_DIR / "css", STATIC_DIR / "js", STATIC_DIR / "images"
    ]:
        directory.mkdir(parents=True, exist_ok=True)

class Settings:
    # App General Settings
    HOST: str = "0.0.0.0"
    PORT: int = 8000
    DEBUG: bool = True
    
    # DB Settings
    DATABASE_URL: str = f"sqlite:///{DATABASE_DIR}/vision_system.db"
    
    # Vision Model Settings
    YOLO_MODEL_PATH: str = str(MODELS_DIR / "yolo11n.pt")
    # Face recognition threshold (cosine or Euclidean similarity)
    # MTCNN settings
    FACE_DETECTION_THRESHOLD: float = 0.85
    # Embedding similarity threshold (Euclidean distance - lower is more similar)
    FACE_RECOGNITION_THRESHOLD: float = 0.60  # Default 0.60 is standard for InceptionResnetV1 distance
    
    # Object detection settings
    OBJECT_CONFIDENCE_THRESHOLD: float = 0.25
    
    # Render Settings
    BOX_THICKNESS: int = 3
    FONT_SCALE: float = 0.5
    
    # Toggle Options (Dynamic settings that can be updated via api)
    ENABLE_FACE_RECOGNITION: bool = True
    ENABLE_OBJECT_DETECTION: bool = True
    
    # Camera Defaults
    DEFAULT_RESOLUTION: str = "640x480"
    
settings = Settings()
