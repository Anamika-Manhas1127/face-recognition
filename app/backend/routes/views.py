from fastapi import APIRouter, Request
from fastapi.responses import HTMLResponse
from fastapi.templating import Jinja2Templates
from app.backend.config.settings import settings

router = APIRouter()
templates = Jinja2Templates(directory=str(settings.TEMPLATES_DIR))

@router.get("/", response_class=HTMLResponse)
async def read_landing(request: Request):
    """Serve the premium landing page."""
    return templates.TemplateResponse(request=request, name="landing.html")

@router.get("/dashboard", response_class=HTMLResponse)
async def read_dashboard(request: Request):
    """Serve the modern glassmorphism dashboard."""
    return templates.TemplateResponse(request=request, name="dashboard.html")
