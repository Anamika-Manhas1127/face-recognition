import datetime
import json
from sqlalchemy import Column, Integer, String, Float, DateTime, Text
from app.backend.database.connection import Base, engine

class RegisteredFace(Base):
    __tablename__ = "registered_faces"

    id = Column(Integer, primary_key=True, index=True)
    name = Column(String(100), nullable=False)
    employee_id = Column(String(50), nullable=True)
    embedding = Column(Text, nullable=False)  # JSON-serialized list of 512 floats
    photo_path = Column(String(255), nullable=False)
    date_added = Column(DateTime, default=datetime.datetime.utcnow)

    def get_embedding(self):
        return json.loads(self.embedding)

    def set_embedding(self, emb_list):
        self.embedding = json.dumps(emb_list)

class DetectionHistory(Base):
    __tablename__ = "detection_history"

    id = Column(Integer, primary_key=True, index=True)
    timestamp = Column(DateTime, default=datetime.datetime.utcnow)
    person = Column(String(255), nullable=True)     # Comma separated list of recognized names or "Unknown Person"
    objects = Column(String(1000), nullable=True)    # Comma separated list of detected objects (e.g. "laptop, cup")
    screenshot = Column(String(255), nullable=True)  # Path to saved screenshot
    confidence = Column(String(255), nullable=True)  # Comma separated confidence percentages (e.g. "98%, 74%")
    session_id = Column(String(100), nullable=True)  # To group frames in a single streaming session

def init_db():
    Base.metadata.create_all(bind=engine)
