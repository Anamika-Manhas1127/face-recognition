import psutil
import base64
import os
import uuid
from PIL import Image
import io
import numpy as np

# Conditional imports
try:
    import torch
    HAS_TORCH = True
except ImportError:
    HAS_TORCH = False

try:
    import cv2
    HAS_CV2 = True
except ImportError:
    HAS_CV2 = False

def get_system_status():
    """Retrieve current system resource utilization details safely."""
    try:
        cpu_percent = psutil.cpu_percent(interval=None)
        memory = psutil.virtual_memory()
        disk = psutil.disk_usage('/')
        active_threads = psutil.Process().num_threads()
    except Exception:
        # Fallbacks for restricted serverless environments
        cpu_percent = 0
        memory = type('Mem', (object,), {'percent': 0, 'used': 0, 'total': 0})()
        disk = type('Disk', (object,), {'free': 0, 'total': 0})()
        active_threads = 1
        
    gpu_available = False
    gpu_name = "None (CPU Mode)"
    
    if HAS_TORCH:
        try:
            gpu_available = torch.cuda.is_available()
            gpu_name = torch.cuda.get_device_name(0) if gpu_available else "None (CPU Mode)"
        except Exception:
            pass

    return {
        "cpu_usage": f"{cpu_percent}%",
        "memory_usage": f"{memory.percent}%",
        "memory_details": f"{round(memory.used / (1024**3), 2)}GB / {round(memory.total / (1024**3), 2)}GB",
        "disk_free": f"{round(disk.free / (1024**3), 2)}GB / {round(disk.total / (1024**3), 2)}GB",
        "gpu_available": gpu_available,
        "gpu_device": gpu_name,
        "active_threads": active_threads
    }

def base64_to_cv2(b64_string: str) -> np.ndarray:
    """Convert a base64 encoded JPEG/PNG string to a BGR numpy frame (with PIL fallback)."""
    if "," in b64_string:
        b64_string = b64_string.split(",")[1]
    
    img_data = base64.b64decode(b64_string)
    
    if HAS_CV2:
        nparr = np.frombuffer(img_data, np.uint8)
        frame = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        return frame
    else:
        # Fallback to Pillow and convert to BGR numpy array
        img = Image.open(io.BytesIO(img_data))
        if img.mode != "RGB":
            img = img.convert("RGB")
        frame_rgb = np.array(img)
        # Convert RGB to BGR using numpy channel slicing
        frame_bgr = frame_rgb[:, :, ::-1]
        return frame_bgr

def cv2_to_base64(frame: np.ndarray, quality: int = 80) -> str:
    """Convert a BGR numpy frame to a JPEG base64 string (with PIL fallback)."""
    if HAS_CV2:
        _, buffer = cv2.imencode('.jpg', frame, [int(cv2.IMWRITE_JPEG_QUALITY), quality])
        b64_bytes = base64.b64encode(buffer)
        return f"data:image/jpeg;base64,{b64_bytes.decode('utf-8')}"
    else:
        # Convert BGR numpy array to PIL RGB Image
        frame_rgb = frame[:, :, ::-1]
        img = Image.fromarray(frame_rgb)
        buffered = io.BytesIO()
        img.save(buffered, format="JPEG", quality=quality)
        b64_bytes = base64.b64encode(buffered.getvalue())
        return f"data:image/jpeg;base64,{b64_bytes.decode('utf-8')}"

def cv2_to_bytes(frame: np.ndarray, quality: int = 80) -> bytes:
    """Convert a BGR numpy image into JPEG bytes (with PIL fallback)."""
    if HAS_CV2:
        _, buffer = cv2.imencode('.jpg', frame, [int(cv2.IMWRITE_JPEG_QUALITY), quality])
        return buffer.tobytes()
    else:
        frame_rgb = frame[:, :, ::-1]
        img = Image.fromarray(frame_rgb)
        buffered = io.BytesIO()
        img.save(buffered, format="JPEG", quality=quality)
        return buffered.getvalue()

def generate_session_id() -> str:
    """Generate a random unique session string."""
    return str(uuid.uuid4())
