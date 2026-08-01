from fastapi import APIRouter, Depends, UploadFile, File, Form, WebSocket, WebSocketDisconnect, HTTPException, BackgroundTasks
from fastapi.responses import FileResponse, StreamingResponse, JSONResponse
from sqlalchemy.orm import Session
import os
import shutil
import uuid
import time
import json
import base64
import cv2
import numpy as np
import pandas as pd
import io

from app.backend.config.settings import settings
from app.backend.database.connection import get_db, SessionLocal
from app.backend.database.models import RegisteredFace
from app.backend.services.vision import vision_service
from app.backend.utils.helpers import get_system_status, base64_to_cv2, cv2_to_base64, generate_session_id
from app.backend.services import db_service

router = APIRouter()

# Global dict to track video processing tasks
video_tasks = {}

# --- VISION DETECTION API ---

@router.post("/api/detect")
async def detect_image(
    file: UploadFile = File(...),
    confidence_threshold: float = Form(None),
    enable_recognition: bool = Form(True),
    enable_objects: bool = Form(True),
    db: Session = Depends(get_db)
):
    """
    Process an uploaded image file, execute AI vision models,
    save results to history, and return annotated image with detections.
    """
    if confidence_threshold is None:
        confidence_threshold = settings.OBJECT_CONFIDENCE_THRESHOLD

    # Validate file type
    filename = file.filename.lower()
    if not (filename.endswith(('.png', '.jpg', '.jpeg', '.webp', '.bmp'))):
        raise HTTPException(status_code=400, detail="Invalid image file format. Supported: PNG, JPG, JPEG, WEBP, BMP")

    # Read image
    contents = await file.read()
    nparr = np.frombuffer(contents, np.uint8)
    frame = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    
    if frame is None:
        raise HTTPException(status_code=400, detail="Failed to decode image file.")

    # Process frame
    annotated_frame, detections = vision_service.process_frame(
        frame, db,
        conf_threshold=confidence_threshold,
        enable_recognition=enable_recognition,
        enable_objects=enable_objects
    )

    # Save screenshot to history if anything detected
    screenshot_filename = f"upload_{uuid.uuid4().hex[:12]}_{int(time.time())}.jpg"
    screenshot_path = settings.HISTORY_SCREENSHOTS_DIR / screenshot_filename
    cv2.imwrite(str(screenshot_path), annotated_frame)

    person_names = [d["label"] for d in detections if d["type"] == "face"]
    object_names = [d["label"] for d in detections if d["type"] == "object"]
    conf_values = [f"{int(d['confidence'])}%" for d in detections]

    # Save history entry in database
    db_service.create_history_entry(
        db,
        person=",".join(person_names) if person_names else None,
        objects=",".join(object_names) if object_names else None,
        screenshot=f"/uploads/history_screenshots/{screenshot_filename}",
        confidence=",".join(conf_values) if conf_values else None,
        session_id=f"upload_{int(time.time())}"
    )

    # Convert annotated image to base64
    annotated_b64 = cv2_to_base64(annotated_frame)

    return {
        "success": True,
        "image": annotated_b64,
        "detections": detections
    }


# --- FACE REGISTRATION API ---

@router.post("/api/register-face")
async def register_face(
    name: str = Form(...),
    employee_id: str = Form(None),
    image_file: UploadFile = File(None),
    image_base64: str = Form(None),
    db: Session = Depends(get_db)
):
    """
    Register a face. Accepts an uploaded photo file or a base64 string captured via camera.
    Generates a 512-d embedding using MTCNN & InceptionResnetV1 and caches it.
    """
    img_np = None

    if image_file:
        contents = await image_file.read()
        nparr = np.frombuffer(contents, np.uint8)
        img_np = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    elif image_base64:
        try:
            img_np = base64_to_cv2(image_base64)
        except Exception:
            raise HTTPException(status_code=400, detail="Failed to parse base64 image data.")
            
    if img_np is None:
        raise HTTPException(status_code=400, detail="No image provided. Upload a file or take a webcam snapshot.")

    # Extract face embedding
    embedding = vision_service.get_embedding_from_image(img_np)
    if embedding is None:
        raise HTTPException(
            status_code=400, 
            detail="No face detected in the photo. Please ensure your face is fully visible and try again."
        )

    # Save registered photo to disk
    photo_filename = f"face_{uuid.uuid4().hex[:12]}_{int(time.time())}.jpg"
    photo_path = settings.REGISTERED_FACES_DIR / photo_filename
    cv2.imwrite(str(photo_path), img_np)

    # Write to Database
    db_face = db_service.create_registered_face(
        db,
        name=name,
        employee_id=employee_id,
        embedding=embedding,
        photo_path=f"/uploads/registered_faces/{photo_filename}"
    )

    # Add to in-memory face cache
    vision_service.add_to_face_cache(db_face.id, name, employee_id, embedding)

    return {
        "success": True,
        "message": f"Successfully registered face for {name}",
        "face": {
            "id": db_face.id,
            "name": db_face.name,
            "employee_id": db_face.employee_id,
            "photo_path": db_face.photo_path
        }
    }

@router.get("/api/registered-faces")
async def list_registered_faces(db: Session = Depends(get_db)):
    """List all registered face entries in the system."""
    faces = db_service.get_registered_faces(db)
    return [
        {
            "id": f.id,
            "name": f.name,
            "employee_id": f.employee_id,
            "photo_path": f.photo_path,
            "date_added": f.date_added.strftime("%Y-%m-%d %H:%M:%S")
        } for f in faces
    ]

@router.delete("/api/registered-faces/{face_id}")
async def delete_face(face_id: int, db: Session = Depends(get_db)):
    """Deletes a registered face and reloads the memory cache."""
    face = db.query(RegisteredFace).filter(RegisteredFace.id == face_id).first()
    if not face:
        raise HTTPException(status_code=404, detail="Face entry not found")
        
    # Delete file
    if face.photo_path:
        local_path = settings.BASE_DIR / face.photo_path.lstrip("/")
        if os.path.exists(local_path):
            try:
                os.remove(local_path)
            except Exception:
                pass
                
    success = db_service.delete_registered_face(db, face_id)
    # Reload embedding cache
    vision_service.load_face_cache(db, force_reload=True)
    return {"success": success}


# --- DETECTION HISTORY API ---

@router.get("/api/history")
async def list_history(
    search: str = None,
    filter_type: str = None,
    start_date: str = None,
    end_date: str = None,
    limit: int = 100,
    offset: int = 0,
    db: Session = Depends(get_db)
):
    """Retrieve history of vision session detections."""
    items, total = db_service.get_history(db, search, filter_type, start_date, end_date, limit, offset)
    return {
        "total": total,
        "items": [
            {
                "id": i.id,
                "timestamp": i.timestamp.strftime("%Y-%m-%d %H:%M:%S"),
                "person": i.person,
                "objects": i.objects,
                "screenshot": i.screenshot,
                "confidence": i.confidence,
                "session_id": i.session_id
            } for i in items
        ]
    }

@router.delete("/api/history/{history_id}")
async def delete_history(history_id: int, db: Session = Depends(get_db)):
    """Deletes a single history log entry and its screenshot."""
    entry = db.query(db_service.DetectionHistory).filter(db_service.DetectionHistory.id == history_id).first()
    if not entry:
        raise HTTPException(status_code=404, detail="Log entry not found")

    if entry.screenshot:
        local_path = settings.BASE_DIR / entry.screenshot.lstrip("/")
        if os.path.exists(local_path):
            try:
                os.remove(local_path)
            except Exception:
                pass

    success = db_service.delete_history_entry(db, history_id)
    return {"success": success}

@router.delete("/api/history-clear")
async def clear_history(db: Session = Depends(get_db)):
    """Clear all entries in the detection history and wipe screenshot files."""
    entries = db.query(db_service.DetectionHistory).all()
    for entry in entries:
        if entry.screenshot:
            local_path = settings.BASE_DIR / entry.screenshot.lstrip("/")
            if os.path.exists(local_path):
                try:
                    os.remove(local_path)
                except Exception:
                    pass
    db_service.clear_all_history(db)
    return {"success": True}

@router.get("/api/history-export")
async def export_history(db: Session = Depends(get_db)):
    """Export the detection history logs as a CSV file stream."""
    items = db.query(db_service.DetectionHistory).order_by(db_service.DetectionHistory.timestamp.desc()).all()
    
    data = []
    for i in items:
        data.append({
            "Log ID": i.id,
            "Timestamp (UTC)": i.timestamp.strftime("%Y-%m-%d %H:%M:%S"),
            "Recognized Persons": i.person or "None",
            "Detected Objects": i.objects or "None",
            "Confidence Scores": i.confidence or "N/A",
            "Session ID": i.session_id or "N/A"
        })
        
    df = pd.DataFrame(data)
    stream = io.StringIO()
    df.to_csv(stream, index=False)
    
    response = StreamingResponse(
        iter([stream.getvalue()]),
        media_type="text/csv"
    )
    response.headers["Content-Disposition"] = "attachment; filename=detection_history.csv"
    return response


# --- SETTINGS API ---

@router.get("/api/settings")
async def get_settings():
    """Retrieve the current model configuration and detection thresholds."""
    return {
        "confidence_threshold": settings.OBJECT_CONFIDENCE_THRESHOLD,
        "enable_recognition": settings.ENABLE_FACE_RECOGNITION,
        "enable_objects": settings.ENABLE_OBJECT_DETECTION,
        "box_thickness": settings.BOX_THICKNESS,
        "face_recognition_threshold": settings.FACE_RECOGNITION_THRESHOLD
    }

@router.post("/api/settings")
async def update_settings(payload: dict):
    """Dynamically modify application confidence limits and model overlays."""
    if "confidence_threshold" in payload:
        settings.OBJECT_CONFIDENCE_THRESHOLD = float(payload["confidence_threshold"])
    if "enable_recognition" in payload:
        settings.ENABLE_FACE_RECOGNITION = bool(payload["enable_recognition"])
    if "enable_objects" in payload:
        settings.ENABLE_OBJECT_DETECTION = bool(payload["enable_objects"])
    if "box_thickness" in payload:
        settings.BOX_THICKNESS = int(payload["box_thickness"])
    if "face_recognition_threshold" in payload:
        settings.FACE_RECOGNITION_THRESHOLD = float(payload["face_recognition_threshold"])
        
    return {"success": True, "settings": await get_settings()}


# --- SYSTEM STATS & ANALYTICS ---

@router.get("/api/status")
async def server_status():
    """Fetch real-time CPU, Ram, Thread, and GPU utilization metrics."""
    return get_system_status()

@router.get("/api/analytics")
async def analytics(db: Session = Depends(get_db)):
    """Fetch compiled statistical metrics and coordinates for graphing trends."""
    return db_service.get_analytics_summary(db)


# --- WEBSOCKET REAL-TIME VIDEO FEED ROUTE ---

@router.websocket("/api/stream")
async def websocket_stream(websocket: WebSocket, db: Session = Depends(get_db)):
    """Accepts connection, reads frames, runs face/object pipeline, returns annotated image."""
    await websocket.accept()
    session_id = generate_session_id()
    
    # Track statistics
    fps_start_time = time.time()
    fps_counter = 0
    fps = 0.0
    
    try:
        while True:
            # Receive frame payload
            data = await websocket.receive_text()
            data_json = json.loads(data)
            
            frame_data = data_json.get("image")
            if not frame_data:
                continue
                
            frame = base64_to_cv2(frame_data)
            if frame is None:
                continue
                
            # Retrieve parameters from client or defaults
            conf_t = data_json.get("confidence_threshold", settings.OBJECT_CONFIDENCE_THRESHOLD)
            en_rec = data_json.get("enable_recognition", settings.ENABLE_FACE_RECOGNITION)
            en_obj = data_json.get("enable_objects", settings.ENABLE_OBJECT_DETECTION)
            box_t = data_json.get("box_thickness", settings.BOX_THICKNESS)
            
            # Process frames through vision service
            annotated_frame, detections = vision_service.process_frame(
                frame, db,
                conf_threshold=conf_t,
                enable_recognition=en_rec,
                enable_objects=en_obj,
                box_thickness=box_t
            )
            
            # Update FPS counter
            fps_counter += 1
            now = time.time()
            elapsed = now - fps_start_time
            if elapsed >= 1.0:
                fps = fps_counter / elapsed
                fps_start_time = now
                fps_counter = 0
                
            # Serialize frame to base64
            annotated_b64 = cv2_to_base64(annotated_frame)
            
            # Write to database history if client explicitly requests save AND there are detections
            if data_json.get("save_history") and len(detections) > 0:
                screenshot_filename = f"ws_{session_id}_{int(time.time() * 1000)}.jpg"
                screenshot_path = settings.HISTORY_SCREENSHOTS_DIR / screenshot_filename
                cv2.imwrite(str(screenshot_path), frame)
                
                person_names = [d["label"] for d in detections if d["type"] == "face"]
                object_names = [d["label"] for d in detections if d["type"] == "object"]
                conf_values = [f"{int(d['confidence'])}%" for d in detections]
                
                db_service.create_history_entry(
                    db,
                    person=",".join(person_names) if person_names else None,
                    objects=",".join(object_names) if object_names else None,
                    screenshot=f"/uploads/history_screenshots/{screenshot_filename}",
                    confidence=",".join(conf_values) if conf_values else None,
                    session_id=session_id
                )
                
            # Transmit back processed overlays
            await websocket.send_json({
                "image": annotated_b64,
                "detections": detections,
                "fps": round(fps, 1),
                "system_status": get_system_status()
            })
            
    except WebSocketDisconnect:
        pass
    except Exception as e:
        print(f"WS error: {e}")


# --- VIDEO FILE PROCESSING BACKEND TASK ---

def process_video_task(task_id: str, input_path: str, output_filename: str):
    """Runs frame-by-frame on uploads, draws boxes, stores history metadata."""
    db = SessionLocal()
    try:
        cap = cv2.VideoCapture(input_path)
        if not cap.isOpened():
            video_tasks[task_id] = {"progress": 100, "status": "failed", "error": "Could not open video file."}
            return
            
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = cap.get(cv2.CAP_PROP_FPS)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        
        if total_frames <= 0:
            total_frames = 1
            
        output_path = settings.TEMP_DIR / output_filename
        
        # Save output using mp4v
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(str(output_path), fourcc, fps, (width, height))
        
        frame_idx = 0
        detected_persons = set()
        detected_objects = set()
        max_conf = 0.0
        
        while True:
            ret, frame = cap.read()
            if not ret:
                break
                
            # Process frames
            annotated_frame, detections = vision_service.process_frame(
                frame, db,
                enable_recognition=settings.ENABLE_FACE_RECOGNITION,
                enable_objects=settings.ENABLE_OBJECT_DETECTION
            )
            
            out.write(annotated_frame)
            
            # Sum up detections
            for d in detections:
                if d["type"] == "face":
                    detected_persons.add(d["label"])
                elif d["type"] == "object":
                    detected_objects.add(d["label"])
                max_conf = max(max_conf, d["confidence"])
                
            frame_idx += 1
            progress_pct = int((frame_idx / total_frames) * 100)
            
            # Update progress dictionary
            video_tasks[task_id] = {"progress": progress_pct, "status": "processing"}
            
        cap.release()
        out.release()
        
        # Capture middle frame for screenshot
        cap_mid = cv2.VideoCapture(input_path)
        cap_mid.set(cv2.CAP_PROP_POS_FRAMES, total_frames // 2)
        r_mid, frame_mid = cap_mid.read()
        screenshot_filename = f"vid_snap_{task_id}.jpg"
        screenshot_path = settings.HISTORY_SCREENSHOTS_DIR / screenshot_filename
        
        if r_mid:
            annotated_mid, _ = vision_service.process_frame(frame_mid, db)
            cv2.imwrite(str(screenshot_path), annotated_mid)
        cap_mid.release()
        
        # Save historical database record
        p_str = ",".join(detected_persons) if detected_persons else None
        o_str = ",".join(detected_objects) if detected_objects else None
        c_str = f"{int(max_conf)}%" if max_conf > 0 else "85%"
        
        db_service.create_history_entry(
            db,
            person=p_str,
            objects=o_str,
            screenshot=f"/uploads/history_screenshots/{screenshot_filename}" if r_mid else None,
            confidence=c_str,
            session_id=f"video_{task_id}"
        )
        
        # Clean up input upload
        if os.path.exists(input_path):
            try:
                os.remove(input_path)
            except Exception:
                pass
                
        video_tasks[task_id] = {
            "progress": 100,
            "status": "completed",
            "output_url": f"/api/download-video/{output_filename}"
        }
        
    except Exception as e:
        print(f"Video process error: {e}")
        video_tasks[task_id] = {"progress": 100, "status": "failed", "error": str(e)}
    finally:
        db.close()

@router.post("/api/upload-video")
async def upload_video(
    background_tasks: BackgroundTasks,
    file: UploadFile = File(...)
):
    """Accepts mp4/avi/mov files, writes locally, and starts asynchronous annotation."""
    filename = file.filename.lower()
    if not (filename.endswith(('.mp4', '.avi', '.mov', '.mkv'))):
        raise HTTPException(status_code=400, detail="Invalid video format. Supported: MP4, AVI, MOV, MKV")
        
    # Save file
    task_id = str(uuid.uuid4())[:18]
    temp_input_filename = f"input_{task_id}_{file.filename}"
    temp_input_path = settings.TEMP_DIR / temp_input_filename
    
    with open(temp_input_path, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)
        
    output_filename = f"annotated_{task_id}.mp4"
    video_tasks[task_id] = {"progress": 0, "status": "processing"}
    
    # Spawn background task
    background_tasks.add_task(
        process_video_task,
        task_id,
        str(temp_input_path),
        output_filename
    )
    
    return {
        "success": True,
        "task_id": task_id,
        "message": "Video processing started successfully in the background."
    }

@router.get("/api/video-progress/{task_id}")
async def get_video_progress(task_id: str):
    """Retrieve progress metrics of a background video annotation task."""
    task = video_tasks.get(task_id)
    if not task:
        raise HTTPException(status_code=404, detail="Video task not found")
    return task

@router.get("/api/download-video/{filename}")
async def download_video(filename: str):
    """Download the completed annotated video file from disk."""
    file_path = settings.TEMP_DIR / filename
    if not os.path.exists(file_path):
        raise HTTPException(status_code=404, detail="File not found")
    return FileResponse(path=file_path, media_type="video/mp4", filename=filename)
