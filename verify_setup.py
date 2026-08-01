import os
import sys
import numpy as np

# Ensure path includes root
sys.path.append(os.path.abspath(os.path.dirname(__file__)))

from app.backend.config.settings import settings
from app.backend.database.models import init_db
from app.backend.database.connection import SessionLocal
from app.backend.services.vision import vision_service

def verify_system():
    print("Initiating System Integration Verification...")
    
    # 1. Initialize DB
    print("Initializing Database tables...")
    init_db()
    print("Database ready.")
    
    # 2. Warm up models
    db = SessionLocal()
    try:
        print("Preloading face registries cache...")
        vision_service.load_face_cache(db)
        
        print("Warming up neural network models (this will download weight files if missing)...")
        vision_service.load_models()
        
        # 3. Create dummy BGR image (640x480, light blue block)
        print("Generating mock frame...")
        dummy_frame = np.zeros((480, 640, 3), dtype=np.uint8)
        dummy_frame[:, :] = [255, 216, 167]  # Pastel Blue color fill
        
        print("Executing inference pipeline on mock frame...")
        annotated_frame, detections = vision_service.process_frame(
            dummy_frame, 
            db, 
            conf_threshold=0.25, 
            enable_recognition=True, 
            enable_objects=True
        )
        
        print(f"Pipeline executed successfully! Detections count: {len(detections)}")
        print(f"Annotated frame dimensions match original: {annotated_frame.shape == dummy_frame.shape}")
        
        print("\n--- SYSTEM VERIFICATION SUCCESSFUL ---")
        return True
    except Exception as e:
        print(f"\nVerification Failed! Details: {e}", file=sys.stderr)
        return False
    finally:
        db.close()

if __name__ == "__main__":
    success = verify_system()
    sys.exit(0 if success else 1)
