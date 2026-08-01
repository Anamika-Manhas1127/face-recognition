import traceback
from fastapi import FastAPI, Request
from fastapi.responses import HTMLResponse

try:
    # --- NORMAL APP STARTUP ---
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
            content={"detail": f"An internal server error occurred: {str(exc)}"}
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

except Exception as e:
    # --- VERCEL CRASH DIAGNOSTIC FALLBACK ---
    # Catch any startup/import errors and host a dummy app that prints the traceback.
    # This prevents Vercel's "FUNCTION_INVOCATION_FAILED" 500 error page,
    # and instead loads a page showing the exact traceback log.
    app = FastAPI(title="AI Vision - Startup Diagnostics")
    
    tb_str = traceback.format_exc()
    
    @app.get("/{path:path}", response_class=HTMLResponse)
    async def diagnostics_wildcard(request: Request, path: str):
        html_content = f"""
        <html>
        <head>
            <title>AI Vision - Deployment Diagnostics</title>
            <style>
                body {{ font-family: monospace; padding: 40px; background: #F5F5F7; color: #1D1D1F; line-height: 1.5; }}
                h2 {{ color: #FF3B30; }}
                pre {{ background: #FFFFFF; padding: 20px; border-radius: 12px; border: 1px solid #D1D1D6; overflow-x: auto; box-shadow: 0 4px 12px rgba(0,0,0,0.05); }}
            </style>
        </head>
        <body>
            <h2>⚠️ Deployment Startup Exception Caught</h2>
            <p>The application crashed during initialization on Vercel. Here is the traceback of the import failure:</p>
            <pre>{tb_str}</pre>
            <p>Please share this traceback, and I will fix it immediately!</p>
        </body>
        </html>
        """
        return HTMLResponse(content=html_content, status_code=200)
