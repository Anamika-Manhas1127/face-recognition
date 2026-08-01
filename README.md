# AI Vision - Face & Object Recognition System (Python)

A production-quality real-time Face Detection, Face Recognition, and Object Detection web application powered by **FastAPI**, **Ultralytics YOLOv11**, and **facenet-pytorch (MTCNN + InceptionResnetV1)**. It features a responsive, Apple-inspired glassmorphic UI that runs on both desktop and mobile (iPhone & Android) browsers.

---

## Features

- **Real-time Face Detection**: Localizes human faces in video frames using MTCNN.
- **Dynamic Face Recognition**: Automatically maps faces to 512-dimensional vectors using a pretrained ResNet, executing database vector cross-matching to identify registered users.
- **YOLOv11 Object Detection**: Detects and logs dozens of everyday objects (chairs, laptops, bottles, etc.) in real time.
- **Priority Labeling Logic**: Prevents duplicate overlays by suppressing YOLO `person` boundaries and overlaying only specific face recognition boxes.
- **Webcam & Camera Switcher**: Seamlessly toggles between webcams, resolutions, and mobile front/rear camera lenses.
- **Upload Interfaces**: Drag and drop zones to process stationary images and videos with progress bar indicator feeds and annotated file downloads.
- **Analytics Workspace**: Generates statistics on daily face counts, top objects, recognition rates, and timeline charts using Chart.js.
- **History Viewer**: Logs sessions with automatic screenshots, search terms, date selectors, CSV exporters, and deletes.
- **Dynamic Settings**: Custom sliders for YOLO confidence, face match thresholds, box thickness, and toggle switches.
- **Docker Ready**: Pre-packaged container specifications for cross-system installations.

---

## Tech Stack

- **Backend**: Python 3.12+, FastAPI, SQLAlchemy (SQLite DB), OpenCV (Headless), PyTorch, Torchvision, Ultralytics YOLOv11, Facenet-PyTorch, Pandas, Psutil.
- **Frontend**: HTML5, Vanilla CSS3 (Glassmorphism), JavaScript (ES6+), Bootstrap 5, Bootstrap Icons, Chart.js.

---

## Project Structure

```
├── app/
│   ├── backend/
│   │   ├── config/
│   │   │   └── settings.py
│   │   ├── database/
│   │   │   ├── connection.py
│   │   │   └── models.py
│   │   ├── routes/
│   │   │   ├── api.py
│   │   │   └── views.py
│   │   ├── services/
│   │   │   ├── db_service.py
│   │   │   └── vision.py
│   │   ├── utils/
│   │   │   └── helpers.py
│   │   └── main.py
│   └── frontend/
│       ├── static/
│       │   ├── css/
│       │   │   └── style.css
│       │   └── js/
│       │       └── dashboard.js
│       └── templates/
│           ├── base.html
│           ├── landing.html
│           └── dashboard.html
├── database/
├── models/
├── uploads/
│   ├── history_screenshots/
│   ├── registered_faces/
│   └── temp/
├── Dockerfile
├── requirements.txt
└── README.md
```

---

## Local Installation Quick Start

### 1. Prerequisite Checklist
Make sure you have **Python 3.12+** installed on your system. 

### 2. Clone and Setup Environment
Navigate into the directory and create a virtual environment:
```bash
python -m venv venv
```
Activate the environment:
- **Windows**: `venv\Scripts\activate`
- **macOS / Linux**: `source venv/bin/activate`

### 3. Install Dependencies
Run:
```bash
pip install -r requirements.txt
```
*(Note: OpenCV and PyTorch are large libraries. Installation may take a few minutes depending on your network speed).*

### 4. Run the Application
Start the FastAPI server:
```bash
python -m app.backend.main
```
The application will launch. Open your browser and navigate to **`http://localhost:8000`** to access the landing page, or **`http://localhost:8000/dashboard`** for the dashboard.

---

## Docker Deployment

To build and run the system inside a Docker container:

```bash
# Build the Docker image
docker build -t ai-vision-system .

# Run the container (bind port 8000)
docker run -p 8000:8000 -v vision-db:/app/database -v vision-uploads:/app/uploads ai-vision-system
```

---

## Vercel Deployment Guide

Vercel has strict limits:
1. **Serverless Function Size Limit**: Maximum 250MB uncompressed limit. Deep learning frameworks like PyTorch, torchvision, and YOLO models exceed 1GB+ when installed, leading to deployment rejection.
2. **Read-only filesystem**: Vercel functions cannot write files (such as SQLite DB and screenshot uploads) outside of the `/tmp` folder, and anything in `/tmp` is wiped between invocations.

### How to deploy to Vercel + Cloud (Best Practice)
To deploy this project to the cloud, use one of the following methods:

#### Method A: Full Docker Container Deployment (Recommended)
Deploy the single Docker container to platforms supporting Docker runtimes, such as:
- **Render.com** (Supports Docker build out of the box)
- **Railway.app**
- **Fly.io**
- **AWS App Runner** or **Google Cloud Run**

#### Method B: Split Frontend and Backend
1. **Frontend (Vercel)**:
   - Deploy the frontend assets (`landing.html`, `dashboard.html`, `style.css`, `dashboard.js`) as a static site on Vercel.
   - Adjust `dashboard.js` to point WebSocket (`wsUrl`) and Fetch URLs to your hosted backend API.
2. **Backend (Render/Fly/AWS)**:
   - Run the FastAPI backend in a Docker container on Render, Railway, or AWS.
   - Attach a persistent storage volume to save the SQLite database and upload directories.
