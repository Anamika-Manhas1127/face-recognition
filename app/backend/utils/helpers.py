import psutil
import torch
import cv2
import numpy as np
import base64
import os
import uuid
from PIL import Image
import io

def get_system_status():
    """Retrieve current system resource utilization details."""
    cpu_percent = psutil.cpu_percent(interval=None)
    memory = psutil.virtual_memory()
    disk = psutil.disk_usage('/')
    
    # GPU detection
    gpu_available = torch.cuda.is_available()
    gpu_name = torch.cuda.get_device_name(0) if gpu_available else "None (CPU Mode)"
    
    # Model caching state
    return {
        "cpu_usage": f"{cpu_percent}%",
        "memory_usage": f"{memory.percent}%",
        "memory_details": f"{round(memory.used / (1024**3), 2)}GB / {round(memory.total / (1024**3), 2)}GB",
        "disk_free": f"{round(disk.free / (1024**3), 2)}GB / {round(disk.total / (1024**3), 2)}GB",
        "gpu_available": gpu_available,
        "gpu_device": gpu_name,
        "active_threads": psutil.Process().num_threads()
    }

def base64_to_cv2(b64_string: str) -> np.ndarray:
    """Convert a base64 encoded JPEG/PNG string to an OpenCV BGR frame."""
    if "," in b64_string:
        b64_string = b64_string.split(",")[1]
    
    img_data = base64.b64decode(b64_string)
    nparr = np.frombuffer(img_data, np.uint8)
    frame = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    return frame

def cv2_to_base64(frame: np.ndarray, quality: int = 80) -> str:
    """Convert an OpenCV frame to a JPEG base64 string."""
    _, buffer = cv2.imencode('.jpg', frame, [int(cv2.IMWRITE_JPEG_QUALITY), quality])
    b64_bytes = base64.b64encode(buffer)
    return f"data:image/jpeg;base64,{b64_bytes.decode('utf-8')}"

def cv2_to_bytes(frame: np.ndarray, quality: int = 80) -> bytes:
    """Convert an OpenCV BGR image into JPEG bytes."""
    _, buffer = cv2.imencode('.jpg', frame, [int(cv2.IMWRITE_JPEG_QUALITY), quality])
    return buffer.tobytes()

def generate_session_id() -> str:
    """Generate a random unique session string."""
    return str(uuid.uuid4())
