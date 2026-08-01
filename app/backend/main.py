from fastapi import FastAPI, Request
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
import uvicorn
import time
import os

from app.backend.config.settings import settings
from app.backend.database.models import init_db
from app.backend.database.connection import SessionLocal
from app.backend.routes import views, api
from app.backend.services.vision import vision_service

# Initialize FastAPI App
app = FastAPI(
    title="AI Vision - Intelligent Face & Object Recognition System",
    description="Production-grade AI Vision web application supporting real-time face & object recognition.",
    version="1.0.0"
)

# Enable Cross-Origin Resource Sharing (CORS)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Error Handler
@app.exception_handler(Exception)
async def global_exception_handler(request: Request, exc: Exception):
    print(f"Unhandled Exception: {exc}")
    return JSONResponse(
        status_code=500,
        content={"detail": "An internal server error occurred. Please check console logs."}
    )

# Startup Event: DB creation & Model loading
@app.on_event("startup")
async def startup_event():
    print("Starting AI Vision system...")
    
    # 1. Initialize SQLite Database
    init_db()
    print("Database tables initialized.")
    
    # 2. Warm up AI models in memory
    db = SessionLocal()
    try:
        # Load registered faces into the cache
        vision_service.load_face_cache(db)
        
        # Load vision models (downloads if not present)
        vision_service.load_models()
    except Exception as e:
        print(f"Warning during model initialization: {e}")
    finally:
        db.close()
        
    print("AI Vision system is ready.")

# Mount Static Assets & Writable Upload Directory Safely
if os.path.exists(settings.STATIC_DIR):
    app.mount("/static", StaticFiles(directory=str(settings.STATIC_DIR)), name="static")
else:
    print(f"Warning: Static directory '{settings.STATIC_DIR}' not found. Serving handled by CDN/Vercel.")

if os.path.exists(settings.UPLOAD_DIR):
    app.mount("/uploads", StaticFiles(directory=str(settings.UPLOAD_DIR)), name="uploads")

# Include Routers
app.include_router(views.router)
app.include_router(api.router)

if __name__ == "__main__":
    # Start uvicorn server locally
    uvicorn.run(
        "app.backend.main:app",
        host=settings.HOST,
        port=settings.PORT,
        reload=settings.DEBUG
    )
