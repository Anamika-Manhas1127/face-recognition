import datetime
from sqlalchemy.orm import Session
from sqlalchemy import func, desc, and_, or_
import json

from app.backend.database.models import RegisteredFace, DetectionHistory

# --- FACE REGISTRATION CRUD ---

def create_registered_face(
    db: Session, 
    name: str, 
    employee_id: str | None, 
    embedding: list, 
    photo_path: str
) -> RegisteredFace:
    db_face = RegisteredFace(
        name=name,
        employee_id=employee_id,
        photo_path=photo_path
    )
    db_face.set_embedding(embedding)
    db.add(db_face)
    db.commit()
    db.refresh(db_face)
    return db_face

def get_registered_faces(db: Session):
    return db.query(RegisteredFace).order_by(desc(RegisteredFace.date_added)).all()

def delete_registered_face(db: Session, face_id: int) -> bool:
    face = db.query(RegisteredFace).filter(RegisteredFace.id == face_id).first()
    if face:
        db.delete(face)
        db.commit()
        return True
    return False


# --- DETECTION HISTORY CRUD ---

def create_history_entry(
    db: Session,
    person: str | None,
    objects: str | None,
    screenshot: str | None,
    confidence: str | None,
    session_id: str | None = None
) -> DetectionHistory:
    db_history = DetectionHistory(
        person=person,
        objects=objects,
        screenshot=screenshot,
        confidence=confidence,
        session_id=session_id,
        timestamp=datetime.datetime.utcnow()
    )
    db.add(db_history)
    db.commit()
    db.refresh(db_history)
    return db_history

def get_history(
    db: Session,
    search: str | None = None,
    filter_type: str | None = None, # "face", "object", or "all"
    start_date: str | None = None,
    end_date: str | None = None,
    limit: int = 100,
    offset: int = 0
):
    query = db.query(DetectionHistory)
    
    # Text search
    if search:
        search_filter = f"%{search}%"
        query = query.filter(
            or_(
                DetectionHistory.person.like(search_filter),
                DetectionHistory.objects.like(search_filter)
            )
        )
        
    # Filter by detection type
    if filter_type == "face":
        query = query.filter(DetectionHistory.person.isnot(None), DetectionHistory.person != "")
    elif filter_type == "object":
        query = query.filter(DetectionHistory.objects.isnot(None), DetectionHistory.objects != "")
        
    # Date filters
    if start_date:
        try:
            sd = datetime.datetime.strptime(start_date, "%Y-%m-%d")
            query = query.filter(DetectionHistory.timestamp >= sd)
        except ValueError:
            pass
    if end_date:
        try:
            # Set to end of day
            ed = datetime.datetime.strptime(end_date, "%Y-%m-%d") + datetime.timedelta(days=1)
            query = query.filter(DetectionHistory.timestamp < ed)
        except ValueError:
            pass
            
    # Ordered by newest first
    total_count = query.count()
    items = query.order_by(desc(DetectionHistory.timestamp)).limit(limit).offset(offset).all()
    return items, total_count

def delete_history_entry(db: Session, history_id: int) -> bool:
    entry = db.query(DetectionHistory).filter(DetectionHistory.id == history_id).first()
    if entry:
        db.delete(entry)
        db.commit()
        return True
    return False

def clear_all_history(db: Session):
    db.query(DetectionHistory).delete()
    db.commit()


# --- ANALYTICS ---

def get_analytics_summary(db: Session):
    now = datetime.datetime.utcnow()
    today_start = datetime.datetime(now.year, now.month, now.day)
    
    # Detections Today
    today_records = db.query(DetectionHistory).filter(DetectionHistory.timestamp >= today_start).all()
    
    faces_today = 0
    objects_today = 0
    unknown_today = 0
    
    for r in today_records:
        if r.person:
            names = [n.strip() for n in r.person.split(",") if n.strip()]
            for name in names:
                if name.lower() == "unknown person":
                    unknown_today += 1
                else:
                    faces_today += 1
        if r.objects:
            objs = [o.strip() for o in r.objects.split(",") if o.strip()]
            objects_today += len(objs)
            
    # Total historical counts
    total_detections = db.query(DetectionHistory).count()
    
    # Calculate top items (object frequency)
    all_objects = db.query(DetectionHistory.objects).filter(
        DetectionHistory.objects.isnot(None), 
        DetectionHistory.objects != ""
    ).all()
    
    object_counts = {}
    for (obj_str,) in all_objects:
        for obj in obj_str.split(","):
            obj = obj.strip().capitalize()
            if obj:
                object_counts[obj] = object_counts.get(obj, 0) + 1
                
    # Format top objects for pie chart
    top_objects = [{"name": k, "count": v} for k, v in sorted(object_counts.items(), key=lambda x: x[1], reverse=True)[:5]]
    
    # Timeline data: hourly counts for today
    hourly_data = {i: {"faces": 0, "objects": 0} for i in range(24)}
    for r in today_records:
        hour = r.timestamp.hour
        if r.person:
            # Count faces
            faces = len([n for n in r.person.split(",") if n.strip()])
            hourly_data[hour]["faces"] += faces
        if r.objects:
            # Count objects
            objs = len([o for o in r.objects.split(",") if o.strip()])
            hourly_data[hour]["objects"] += objs
            
    timeline = [{"hour": f"{h:02d}:00", "faces": hourly_data[h]["faces"], "objects": hourly_data[h]["objects"]} for h in range(24)]

    # Face matching distribution (Known vs Unknown)
    known_count = db.query(DetectionHistory).filter(
        DetectionHistory.person.isnot(None), 
        DetectionHistory.person != "",
        ~DetectionHistory.person.like("%Unknown Person%")
    ).count()
    
    unknown_count = db.query(DetectionHistory).filter(
        DetectionHistory.person.like("%Unknown Person%")
    ).count()
    
    # Average confidence calculation
    all_confidences = db.query(DetectionHistory.confidence).filter(
        DetectionHistory.confidence.isnot(None),
        DetectionHistory.confidence != ""
    ).all()
    
    conf_scores = []
    for (conf_str,) in all_confidences:
        for c in conf_str.split(","):
            c = c.replace("%", "").strip()
            try:
                conf_scores.append(float(c))
            except ValueError:
                pass
    
    avg_accuracy = round(sum(conf_scores) / len(conf_scores), 2) if conf_scores else 95.0

    return {
        "faces_today": faces_today,
        "objects_today": objects_today,
        "unknown_today": unknown_today,
        "total_detections": total_detections,
        "avg_accuracy": avg_accuracy,
        "top_objects": top_objects,
        "timeline": timeline,
        "recognition_ratio": {
            "known": known_count,
            "unknown": unknown_count
        }
    }
