// Global States
let ws = null;
let localStream = null;
let isStreaming = false;
let isRecording = false;
let currentFacingMode = "user"; // "user" or "environment"
let currentSection = "live-detection";
let recordingInterval = null;
let saveHistoryFlag = false;

// Global settings cached locally
let localSettings = {
    confidence_threshold: 0.25,
    enable_recognition: true,
    enable_objects: true,
    box_thickness: 3,
    face_recognition_threshold: 0.60
};

// Analytics Chart Instances
let chartObjects = null;
let chartFaceProfile = null;
let chartTimeline = null;

// Initialize Dashboard when loaded
document.addEventListener("DOMContentLoaded", () => {
    // Start Clock
    setInterval(updateClock, 1000);
    
    // Parse URL params for default startup actions
    const params = new URLSearchParams(window.location.search);
    const action = params.get("action");
    if (action) {
        switchSection(action);
    } else {
        switchSection("live-detection");
    }
    
    // Load config settings
    loadSettings();
    
    // Scan for available webcams
    detectWebcams();
    
    // Poll system statistics details every 3 seconds
    setInterval(pollSystemStats, 3000);
});

// Clock Updater
function updateClock() {
    const clockEl = document.getElementById("top-clock");
    if (clockEl) {
        const now = new Date();
        clockEl.innerText = now.toLocaleTimeString();
    }
}

// Switch Sections (SPA navigation)
function switchSection(sectionId) {
    currentSection = sectionId;
    
    // Update Title
    const titleEl = document.getElementById("current-section-title");
    titleEl.innerText = sectionId.split("-").map(w => w.charAt(0).toUpperCase() + w.slice(1)).join(" ");

    // Toggle active link highlights
    document.querySelectorAll(".sidebar-nav a").forEach(link => {
        link.classList.remove("active");
    });
    const activeLink = document.getElementById(`nav-${sectionId}`);
    if (activeLink) activeLink.classList.add("active");

    // Hide all sections, display active one
    const sections = ["live-detection", "upload-image", "upload-video", "registered-faces", "history", "analytics", "settings"];
    sections.forEach(s => {
        const el = document.getElementById(`section-${s}`);
        if (el) {
            if (s === sectionId) {
                el.classList.remove("d-none");
            } else {
                el.classList.add("d-none");
            }
        }
    });

    // If leaving live camera, stop webcam to free device
    if (sectionId !== "live-detection" && isStreaming) {
        stopCamera();
    }

    // Trigger actions based on section
    if (sectionId === "registered-faces") {
        fetchRegisteredFaces();
    } else if (sectionId === "history") {
        fetchHistory();
    } else if (sectionId === "analytics") {
        renderAnalytics();
    }
}

// --- SYSTEM DIAGNOSTICS & SETTINGS ---

async function loadSettings() {
    try {
        const res = await fetch("/api/settings");
        const data = await res.json();
        localSettings = data;
        
        // Update Form Inputs
        document.getElementById("slider-conf-threshold").value = data.confidence_threshold;
        document.getElementById("label-conf-slider").innerText = data.confidence_threshold;
        
        document.getElementById("slider-face-threshold").value = data.face_recognition_threshold;
        document.getElementById("label-face-slider").innerText = data.face_recognition_threshold.toFixed(2);
        
        document.getElementById("slider-box-thickness").value = data.box_thickness;
        document.getElementById("label-thickness-slider").innerText = data.box_thickness + "px";
        
        document.getElementById("toggle-enable-recognition").checked = data.enable_recognition;
        document.getElementById("toggle-enable-objects").checked = data.enable_objects;
    } catch (e) {
        console.error("Error loading settings:", e);
    }
}

async function saveSettings() {
    const payload = {
        confidence_threshold: parseFloat(document.getElementById("slider-conf-threshold").value),
        face_recognition_threshold: parseFloat(document.getElementById("slider-face-threshold").value),
        box_thickness: parseInt(document.getElementById("slider-box-thickness").value),
        enable_recognition: document.getElementById("toggle-enable-recognition").checked,
        enable_objects: document.getElementById("toggle-enable-objects").checked
    };
    
    try {
        const res = await fetch("/api/settings", {
            method: "POST",
            headers: { "Content-Type": "application/json" },
            body: JSON.stringify(payload)
        });
        const data = await res.json();
        if (data.success) {
            localSettings = data.settings;
            showToast("Settings updated successfully!");
        }
    } catch (e) {
        showToast("Failed to save settings.", true);
    }
}

async function pollSystemStats() {
    try {
        const res = await fetch("/api/status");
        if (!res.ok) return;
        const data = await res.json();
        
        // Update Right panel metrics
        document.getElementById("sys-cpu").innerText = data.cpu_usage;
        document.getElementById("sys-ram").innerText = data.memory_details.split(" / ")[0];
        document.getElementById("sys-disk").innerText = data.disk_free.split(" / ")[0];
        
        const gpuEl = document.getElementById("sys-gpu");
        if (data.gpu_available) {
            gpuEl.innerText = "CUDA (GPU)";
            gpuEl.className = "badge bg-success text-white small";
        } else {
            gpuEl.innerText = "CPU MODE";
            gpuEl.className = "badge bg-secondary text-white small";
        }
    } catch (e) {
        // Silently catch
    }
}

// --- WEBCAM & REAL-TIME WS INTERACTION ---

async function detectWebcams() {
    const select = document.getElementById("camera-select");
    try {
        const devices = await navigator.mediaDevices.enumerateDevices();
        const videoDevices = devices.filter(d => d.kind === "videoinput");
        
        select.innerHTML = "";
        if (videoDevices.length === 0) {
            select.innerHTML = '<option value="">No Webcams Found</option>';
            return;
        }
        
        videoDevices.forEach((device, index) => {
            const option = document.createElement("option");
            option.value = device.deviceId;
            option.text = device.label || `Camera ${index + 1}`;
            select.appendChild(option);
        });
    } catch (e) {
        console.error("Camera scan error:", e);
        select.innerHTML = '<option value="">Permission denied</option>';
    }
}

async function startCamera() {
    if (isStreaming) stopCamera();

    const video = document.getElementById("webcam-video");
    const deviceId = document.getElementById("camera-select").value;
    const resString = document.getElementById("resolution-select").value;
    const [w, h] = resString.split("x").map(Number);

    const constraints = {
        video: {
            deviceId: deviceId ? { exact: deviceId } : undefined,
            width: { ideal: w },
            height: { ideal: h },
            facingMode: currentFacingMode
        },
        audio: false
    };

    try {
        document.getElementById("viewport-loader").classList.remove("d-none");
        localStream = await navigator.mediaDevices.getUserMedia(constraints);
        video.srcObject = localStream;
        await video.play();
        isStreaming = true;
        
        // Open WebSocket link
        initWebSocket();
        
        // Hide Loader
        document.getElementById("viewport-loader").classList.add("d-none");
        
        // Log event
        addLogEntry("System - Camera connection active.");
    } catch (e) {
        console.error("Camera access failed:", e);
        showToast("Webcam initiation failed. Check browser permissions.", true);
        document.getElementById("viewport-loader").classList.add("d-none");
    }
}

function stopCamera() {
    if (localStream) {
        localStream.getTracks().forEach(track => track.stop());
    }
    if (ws) {
        ws.close();
    }
    
    // Clear canvas
    const canvas = document.getElementById("canvas-stream");
    const ctx = canvas.getContext("2d");
    ctx.clearRect(0, 0, canvas.width, canvas.height);
    
    isStreaming = false;
    document.getElementById("top-fps").innerText = "FPS: --";
    updateConnectionStatus("Offline", "offline");
    addLogEntry("System - Camera stopped.");
}

function changeCamera() {
    if (isStreaming) startCamera();
}

function changeResolution() {
    if (isStreaming) startCamera();
}

function switchCameraFacing() {
    currentFacingMode = currentFacingMode === "user" ? "environment" : "user";
    if (isStreaming) startCamera();
}

// WebSocket connection
function initWebSocket() {
    const loc = window.location;
    const proto = loc.protocol === "https:" ? "wss:" : "ws:";
    const wsUrl = `${proto}//${loc.host}/api/stream`;
    
    ws = new WebSocket(wsUrl);
    
    ws.onopen = () => {
        updateConnectionStatus("Connected", "online");
        addLogEntry("System - Pipeline Socket active.");
        
        // Trigger capture frames loop
        setTimeout(captureFrameAndSend, 100);
    };
    
    ws.onmessage = (event) => {
        const data = JSON.parse(event.data);
        
        // Render Frame
        renderAnnotatedFrame(data.image);
        
        // Update Detections sidebar logs
        renderFrameDetections(data.detections);
        
        // Update FPS & Counters
        document.getElementById("top-fps").innerText = `FPS: ${data.fps}`;
        document.getElementById("top-detections").innerText = `Detections: ${data.detections.length}`;
        
        // Trigger next frame
        if (isStreaming && ws.readyState === WebSocket.OPEN) {
            setTimeout(captureFrameAndSend, 30); // limit to roughly 30 FPS transmission
        }
    };
    
    ws.onclose = () => {
        updateConnectionStatus("Disconnected", "offline");
    };
    
    ws.onerror = (e) => {
        console.error("WS error:", e);
        updateConnectionStatus("Socket Error", "warning");
    };
}

// Grabs frame, writes to canvas, gets base64, pushes to server
function captureFrameAndSend() {
    if (!isStreaming || !ws || ws.readyState !== WebSocket.OPEN) return;
    
    const video = document.getElementById("webcam-video");
    const canvas = document.getElementById("canvas-stream");
    
    // Match dimensions
    if (canvas.width !== video.videoWidth) {
        canvas.width = video.videoWidth;
        canvas.height = video.videoHeight;
    }
    
    const ctx = canvas.getContext("2d");
    // Draw raw image in-memory first to extract base64
    ctx.drawImage(video, 0, 0, canvas.width, canvas.height);
    const b64Image = canvas.toDataURL("image/jpeg", 0.6); // compress slightly to speed up pipeline
    
    const payload = {
        image: b64Image,
        confidence_threshold: localSettings.confidence_threshold,
        enable_recognition: localSettings.enable_recognition,
        enable_objects: localSettings.enable_objects,
        box_thickness: localSettings.box_thickness,
        save_history: saveHistoryFlag
    };
    
    ws.send(JSON.stringify(payload));
}

function renderAnnotatedFrame(b64Data) {
    const canvas = document.getElementById("canvas-stream");
    const ctx = canvas.getContext("2d");
    const img = new Image();
    img.onload = () => {
        ctx.drawImage(img, 0, 0, canvas.width, canvas.height);
    };
    img.src = b64Data;
}

function updateConnectionStatus(text, statusClass) {
    const dot = document.getElementById("connection-status-dot");
    const txt = document.getElementById("connection-status-text");
    txt.innerText = text;
    dot.className = `status-dot ${statusClass}`;
}

// Side detections logs updates
function renderFrameDetections(detections) {
    const listEl = document.getElementById("frame-detections-list");
    listEl.innerHTML = "";
    
    if (detections.length === 0) {
        listEl.innerHTML = '<div class="text-secondary small">No objects detected.</div>';
        return;
    }
    
    detections.forEach(d => {
        const pill = document.createElement("div");
        pill.className = `detection-pill ${d.type}`;
        
        const typeIcon = d.type === "face" ? '<i class="bi bi-person-fill"></i>' : '<i class="bi bi-box"></i>';
        
        pill.innerHTML = `
            <span class="small fw-bold">${typeIcon} ${d.label}</span>
            <span class="badge bg-white text-dark small font-monospace">${parseInt(d.confidence)}%</span>
        `;
        listEl.appendChild(pill);
        
        // Real-time Event Logger trigger
        if (d.type === "face" && d.label !== "Unknown Person") {
            addLogEntry(`Detected registered user: ${d.label}`);
        }
    });
}

function addLogEntry(text) {
    const container = document.getElementById("live-logs-container");
    const entry = document.createElement("div");
    entry.className = "log-entry-row";
    
    const timeStr = new Date().toLocaleTimeString();
    entry.innerHTML = `
        <span class="text-secondary">[${timeStr}]</span>
        <span class="text-dark font-monospace">${text}</span>
    `;
    container.insertBefore(entry, container.firstChild);
    
    // Prune logs to 30 items
    if (container.children.length > 30) {
        container.removeChild(container.lastChild);
    }
}

// Control Actions
function captureSnapshot() {
    const canvas = document.getElementById("canvas-stream");
    if (!isStreaming) return;
    
    const link = document.createElement('a');
    link.download = `snapshot_${Date.now()}.jpg`;
    link.href = canvas.toDataURL("image/jpeg");
    link.click();
    addLogEntry("Action - Captured frame snapshot.");
}

function toggleRecording() {
    const btn = document.getElementById("record-btn");
    if (!isRecording) {
        isRecording = true;
        saveHistoryFlag = true;
        btn.classList.add("btn-danger");
        btn.innerHTML = '<i class="bi bi-stop-circle text-white"></i>';
        showToast("Continuous logging enabled. Snapshots are registering in Database.");
        addLogEntry("Action - Started database tracking logs session.");
        
        // Flash indicator logic
        recordingInterval = setInterval(() => {
            btn.style.opacity = btn.style.opacity === "0.5" ? "1" : "0.5";
        }, 500);
    } else {
        isRecording = false;
        saveHistoryFlag = false;
        clearInterval(recordingInterval);
        btn.style.opacity = "1";
        btn.classList.remove("btn-danger");
        btn.innerHTML = '<i class="bi bi-record-circle text-danger"></i>';
        showToast("Database session tracking disabled.");
        addLogEntry("Action - Ended tracking log session.");
    }
}

// --- FILE UPLOADS: IMAGE & VIDEO ---

async function handleImageUpload(e) {
    const file = e.target.files[0];
    if (!file) return;

    // Show progress bar animation
    const progressContainer = document.getElementById("image-progress-container");
    const progressBar = document.getElementById("image-progress-bar");
    progressContainer.classList.remove("d-none");
    progressBar.style.width = "40%";

    const fd = new FormData();
    fd.append("file", file);
    fd.append("confidence_threshold", localSettings.confidence_threshold);
    fd.append("enable_recognition", localSettings.enable_recognition);
    fd.append("enable_objects", localSettings.enable_objects);

    try {
        const res = await fetch("/api/detect", {
            method: "POST",
            body: fd
        });
        progressBar.style.width = "80%";
        
        const data = await res.json();
        progressBar.style.width = "100%";
        
        if (data.success) {
            document.getElementById("image-result-container").classList.remove("d-none");
            const resultImg = document.getElementById("image-result-img");
            resultImg.src = data.image;
            
            // Setup download anchor
            const dlBtn = document.getElementById("btn-download-image");
            dlBtn.href = data.image;
            dlBtn.download = `annotated_${file.name}`;
            
            showToast("Image processed successfully!");
        } else {
            showToast("Failed to process image.", true);
        }
    } catch (e) {
        showToast("Upload error occurred.", true);
    } finally {
        setTimeout(() => {
            progressContainer.classList.add("d-none");
            progressBar.style.width = "0%";
        }, 1500);
    }
}

async function handleVideoUpload(e) {
    const file = e.target.files[0];
    if (!file) return;

    const progressContainer = document.getElementById("video-progress-container");
    const progressBar = document.getElementById("video-progress-bar");
    const statusText = document.getElementById("video-status-text");

    progressContainer.classList.remove("d-none");
    statusText.classList.remove("d-none");
    progressBar.style.width = "10%";
    progressBar.innerText = "Uploading...";

    const fd = new FormData();
    fd.append("file", file);

    try {
        const res = await fetch("/api/upload-video", {
            method: "POST",
            body: fd
        });
        const data = await res.json();
        
        if (data.success) {
            const taskId = data.task_id;
            progressBar.style.width = "25%";
            progressBar.innerText = "25% (Processing)";
            
            // Poll for progress updates
            const pollInterval = setInterval(async () => {
                const progRes = await fetch(`/api/video-progress/${taskId}`);
                const progData = await progRes.json();
                
                progressBar.style.width = `${progData.progress}%`;
                progressBar.innerText = `${progData.progress}% (${progData.status})`;
                
                if (progData.status === "completed") {
                    clearInterval(pollInterval);
                    statusText.innerText = "Processing completed!";
                    progressBar.classList.remove("progress-bar-animated");
                    
                    // Display player
                    document.getElementById("video-result-container").classList.remove("d-none");
                    const player = document.getElementById("video-result-player");
                    player.src = progData.output_url;
                    
                    const dlBtn = document.getElementById("btn-download-video");
                    dlBtn.href = progData.output_url;
                    
                    showToast("Video rendered successfully!");
                } else if (progData.status === "failed") {
                    clearInterval(pollInterval);
                    statusText.innerText = `Error: ${progData.error}`;
                    progressBar.className = "progress-bar bg-danger";
                    showToast("Video processing failed.", true);
                }
            }, 2000);
        } else {
            showToast("Failed to upload video file.", true);
            progressContainer.classList.add("d-none");
        }
    } catch (e) {
        showToast("Video request error.", true);
        progressContainer.classList.add("d-none");
    }
}

// --- FACE REGISTRATION LOGIC ---

let regModal = null;
let regMethod = "camera"; // camera or upload
let regStream = null;

function openRegistrationModal() {
    regModal = new bootstrap.Modal(document.getElementById('modal-register-face'));
    regModal.show();
    setRegMethod('camera'); // default to camera
}

async function setRegMethod(method) {
    regMethod = method;
    const camArea = document.getElementById("reg-method-camera");
    const uploadArea = document.getElementById("reg-method-upload");
    const errBox = document.getElementById("reg-error-box");
    errBox.classList.add("d-none");
    
    if (method === "camera") {
        camArea.classList.remove("d-none");
        uploadArea.classList.add("d-none");
        
        // Start registration webcam
        try {
            regStream = await navigator.mediaDevices.getUserMedia({ video: { width: 320, height: 240 }, audio: false });
            document.getElementById("reg-video").srcObject = regStream;
        } catch (e) {
            console.error("Reg camera error:", e);
            showToast("Webcam initiation failed for face registration.", true);
        }
    } else {
        camArea.classList.add("d-none");
        uploadArea.classList.remove("d-none");
        
        // Stop registration webcam
        if (regStream) {
            regStream.getTracks().forEach(t => t.stop());
            regStream = null;
        }
    }
}

// Cleanup webcam when registration modal closes
document.getElementById('modal-register-face').addEventListener('hidden.bs.modal', function () {
    if (regStream) {
        regStream.getTracks().forEach(t => t.stop());
        regStream = null;
    }
    document.getElementById("reg-snap-indicator").classList.add("d-none");
});

// Snaps in-memory frames for registration
let registeredBase64Frame = null;
function captureRegistrationFrame() {
    const video = document.getElementById("reg-video");
    const canvas = document.getElementById("reg-canvas");
    canvas.width = video.videoWidth;
    canvas.height = video.videoHeight;
    
    const ctx = canvas.getContext("2d");
    ctx.drawImage(video, 0, 0, canvas.width, canvas.height);
    
    registeredBase64Frame = canvas.toDataURL("image/jpeg");
    document.getElementById("reg-snap-indicator").classList.remove("d-none");
    addLogEntry("System - Registration photo snap captured.");
}

// Submit registration payload
document.getElementById("face-registration-form").addEventListener("submit", async (e) => {
    e.preventDefault();
    const name = document.getElementById("reg-name").value;
    const empId = document.getElementById("reg-emp-id").value;
    const errBox = document.getElementById("reg-error-box");
    
    errBox.classList.add("d-none");
    
    const fd = new FormData();
    fd.append("name", name);
    if (empId) fd.append("employee_id", empId);
    
    if (regMethod === "camera") {
        if (!registeredBase64Frame) {
            errBox.classList.remove("d-none");
            errBox.innerText = "Please capture a webcam snapshot first.";
            return;
        }
        fd.append("image_base64", registeredBase64Frame);
    } else {
        const fileInput = document.getElementById("reg-file-input");
        if (fileInput.files.length === 0) {
            errBox.classList.remove("d-none");
            errBox.innerText = "Please choose a photo file to upload.";
            return;
        }
        fd.append("image_file", fileInput.files[0]);
    }

    // Disable register button
    const submitBtn = document.getElementById("btn-submit-registration");
    submitBtn.disabled = true;
    submitBtn.innerText = "Registering...";

    try {
        const res = await fetch("/api/register-face", {
            method: "POST",
            body: fd
        });
        const data = await res.json();
        
        if (res.ok && data.success) {
            showToast(`Registered successfully: ${name}`);
            regModal.hide();
            fetchRegisteredFaces();
        } else {
            errBox.classList.remove("d-none");
            errBox.innerText = data.detail || "Registration failed. Verify face is clearly visible.";
        }
    } catch (err) {
        errBox.classList.remove("d-none");
        errBox.innerText = "Internal connection error. Please try again.";
    } finally {
        submitBtn.disabled = false;
        submitBtn.innerText = "Register Person";
    }
});

async function fetchRegisteredFaces() {
    const grid = document.getElementById("registered-faces-grid");
    grid.innerHTML = '<div class="text-center text-secondary py-5"><div class="spinner-border text-primary" role="status"></div></div>';
    
    try {
        const res = await fetch("/api/registered-faces");
        const faces = await res.json();
        
        grid.innerHTML = "";
        if (faces.length === 0) {
            grid.innerHTML = `
                <div class="col-12 text-center text-secondary py-5">
                    <i class="bi bi-person-slash display-3 mb-3 d-block"></i>
                    <h5>No Registered Faces</h5>
                    <p class="small">Add users to enable customized real-time face tagging.</p>
                </div>
            `;
            return;
        }
        
        faces.forEach(f => {
            const card = document.createElement("div");
            card.className = "col-sm-6 col-md-4 col-lg-3";
            card.innerHTML = `
                <div class="glass history-card p-3 d-flex flex-column align-items-center position-relative">
                    <button class="btn btn-sm btn-light position-absolute top-0 end-0 m-2 rounded-circle shadow-sm" onclick="deleteFace(${f.id})" style="width:28px; height:28px; padding:0; display:flex; align-items:center; justify-content:center;"><i class="bi bi-trash text-danger"></i></button>
                    <img src="${f.photo_path}" class="rounded-circle mb-3 border object-fit-cover shadow-sm" style="width: 100px; height: 100px;">
                    <h6 class="fw-bold mb-1 text-dark text-center text-truncate w-100">${f.name}</h6>
                    <span class="text-secondary small font-monospace mb-1">${f.employee_id || "No ID"}</span>
                    <span class="text-secondary font-monospace" style="font-size:0.75rem;">Registered: ${f.date_added.split(" ")[0]}</span>
                </div>
            `;
            grid.appendChild(card);
        });
    } catch (e) {
        grid.innerHTML = '<div class="col-12 text-center text-danger py-5">Failed to fetch face registrations.</div>';
    }
}

async function deleteFace(id) {
    if (!confirm("Are you sure you want to delete this face registration embedding?")) return;
    try {
        const res = await fetch(`/api/registered-faces/${id}`, { method: "DELETE" });
        const data = await res.json();
        if (data.success) {
            showToast("Face registration deleted successfully.");
            fetchRegisteredFaces();
        }
    } catch (e) {
        showToast("Error deleting face registration.", true);
    }
}

// --- LOGS HISTORY VIEW LOGIC ---

async function fetchHistory() {
    const grid = document.getElementById("history-logs-grid");
    grid.innerHTML = '<div class="text-center text-secondary py-5"><div class="spinner-border text-primary" role="status"></div></div>';

    const search = document.getElementById("history-search").value;
    const filter = document.getElementById("history-filter").value;
    const sDate = document.getElementById("history-start-date").value;
    const eDate = document.getElementById("history-end-date").value;

    let url = `/api/history?limit=30`;
    if (search) url += `&search=${encodeURIComponent(search)}`;
    if (filter && filter !== "all") url += `&filter_type=${filter}`;
    if (sDate) url += `&start_date=${sDate}`;
    if (eDate) url += `&end_date=${eDate}`;

    try {
        const res = await fetch(url);
        const data = await res.json();
        
        document.getElementById("history-count").innerText = `Showing ${data.items.length} records (Total: ${data.total})`;
        
        grid.innerHTML = "";
        if (data.items.length === 0) {
            grid.innerHTML = `
                <div class="col-12 text-center text-secondary py-5">
                    <i class="bi bi-folder-x display-3 mb-3 d-block"></i>
                    <h5>No History Log Entries</h5>
                </div>
            `;
            return;
        }

        data.items.forEach(log => {
            const card = document.createElement("div");
            card.className = "col-md-6 col-lg-4";
            
            const personsMarkup = log.person ? 
                `<div><i class="bi bi-person-fill text-primary"></i> <span class="fw-bold small">${log.person}</span></div>` : '';
            const objectsMarkup = log.objects ? 
                `<div><i class="bi bi-box-fill text-danger"></i> <span class="text-secondary small text-truncate d-inline-block w-75 align-middle">${log.objects}</span></div>` : '';
            
            card.innerHTML = `
                <div class="glass history-card">
                    <div style="height: 180px; overflow:hidden; background: #000; position:relative;">
                        <img src="${log.screenshot || '/static/images/placeholder.svg'}" class="w-100 h-100 object-fit-cover">
                        <span class="badge bg-dark text-white position-absolute bottom-0 end-0 m-2 small font-monospace">${log.confidence || '95%'}</span>
                    </div>
                    <div class="p-3">
                        <div class="d-flex justify-content-between align-items-center mb-2">
                            <span class="text-secondary small font-monospace">${log.timestamp}</span>
                            <button class="btn btn-sm btn-light rounded-circle shadow-sm" onclick="deleteHistoryItem(${log.id})"><i class="bi bi-trash text-danger"></i></button>
                        </div>
                        <div class="d-flex flex-column gap-1">
                            ${personsMarkup}
                            ${objectsMarkup}
                        </div>
                    </div>
                </div>
            `;
            grid.appendChild(card);
        });
    } catch (e) {
        grid.innerHTML = '<div class="col-12 text-center text-danger py-5">Failed to fetch history logs.</div>';
    }
}

async function deleteHistoryItem(id) {
    if (!confirm("Are you sure you want to delete this log?")) return;
    try {
        const res = await fetch(`/api/history/${id}`, { method: "DELETE" });
        const data = await res.json();
        if (data.success) {
            showToast("Log entry deleted successfully.");
            fetchHistory();
        }
    } catch (e) {
        showToast("Error deleting history entry.", true);
    }
}

async function clearAllHistory() {
    if (!confirm("CRITICAL WARNING: This will permanently wipe all logs and screenshots from the system database. Proceed?")) return;
    try {
        const res = await fetch("/api/history-clear", { method: "DELETE" });
        const data = await res.json();
        if (data.success) {
            showToast("System history database cleared.");
            fetchHistory();
        }
    } catch (e) {
        showToast("Error clearing history database.", true);
    }
}

function clearHistoryFilters() {
    document.getElementById("history-search").value = "";
    document.getElementById("history-filter").value = "all";
    document.getElementById("history-start-date").value = "";
    document.getElementById("history-end-date").value = "";
    fetchHistory();
}

// --- ANALYTICS GRAPHICS CHARTS ---

async function renderAnalytics() {
    try {
        const res = await fetch("/api/analytics");
        const data = await res.json();
        
        // Update Stats Counters
        document.getElementById("an-faces-today").innerText = data.faces_today;
        document.getElementById("an-objects-today").innerText = data.objects_today;
        document.getElementById("an-unknown-today").innerText = data.unknown_today;
        document.getElementById("an-avg-accuracy").innerText = `${data.avg_accuracy}%`;
        
        // Render 1: Top Objects Pie Chart
        const ctxObjects = document.getElementById("chart-top-objects").getContext("2d");
        if (chartObjects) chartObjects.destroy();
        chartObjects = new Chart(ctxObjects, {
            type: "pie",
            data: {
                labels: data.top_objects.map(o => o.name),
                datasets: [{
                    data: data.top_objects.map(o => o.count),
                    backgroundColor: ["#A7D8FF", "#FFC7DD", "#DCCEFF", "#FFEAA7", "#E0F2F1"]
                }]
            },
            options: {
                responsive: true,
                maintainAspectRatio: false,
                plugins: { legend: { position: 'bottom' } }
            }
        });

        // Render 2: Face Profile Known vs Unknown Bar Chart
        const ctxFace = document.getElementById("chart-face-profile").getContext("2d");
        if (chartFaceProfile) chartFaceProfile.destroy();
        chartFaceProfile = new Chart(ctxFace, {
            type: "bar",
            data: {
                labels: ["Known Tags", "Unknown Faces"],
                datasets: [{
                    label: "Recognition Distribution",
                    data: [data.recognition_ratio.known, data.recognition_ratio.unknown],
                    backgroundColor: ["#A7D8FF", "#FFC7DD"]
                }]
            },
            options: {
                responsive: true,
                maintainAspectRatio: false,
                plugins: { legend: { display: false } },
                scales: { y: { beginAtZero: true } }
            }
        });

        // Render 3: Timeline Line Chart
        const ctxTimeline = document.getElementById("chart-timeline").getContext("2d");
        if (chartTimeline) chartTimeline.destroy();
        chartTimeline = new Chart(ctxTimeline, {
            type: "line",
            data: {
                labels: data.timeline.map(t => t.hour),
                datasets: [
                    {
                        label: "Faces Detected",
                        data: data.timeline.map(t => t.faces),
                        borderColor: "#A7D8FF",
                        backgroundColor: "rgba(167,216,255,0.2)",
                        tension: 0.3,
                        fill: true
                    },
                    {
                        label: "Objects Detected",
                        data: data.timeline.map(t => t.objects),
                        borderColor: "#FFC7DD",
                        backgroundColor: "rgba(255,199,221,0.2)",
                        tension: 0.3,
                        fill: true
                    }
                ]
            },
            options: {
                responsive: true,
                maintainAspectRatio: false,
                scales: { y: { beginAtZero: true } }
            }
        });
    } catch (e) {
        console.error("Error drawing analytics:", e);
    }
}

// --- THEME & VIEWPORT EFFECTS ---

function toggleTheme() {
    const html = document.documentElement;
    const icon = document.getElementById("theme-toggle-icon");
    const currentTheme = html.getAttribute("data-theme");
    
    if (currentTheme === "dark") {
        html.setAttribute("data-theme", "light");
        icon.className = "bi bi-moon";
    } else {
        html.setAttribute("data-theme", "dark");
        icon.className = "bi bi-sun";
    }
}

function toggleFullscreen() {
    if (!document.fullscreenElement) {
        document.documentElement.requestFullscreen().catch(err => {
            showToast(`Fullscreen request failed: ${err.message}`, true);
        });
    } else {
        document.exitFullscreen();
    }
}

// Toast Notification helper
function showToast(message, isError = false) {
    const toastEl = document.getElementById("app-toast");
    const bodyEl = document.getElementById("toast-message");
    bodyEl.innerText = message;
    
    if (isError) {
        toastEl.className = "toast align-items-center text-white bg-danger border-0";
    } else {
        toastEl.className = "toast align-items-center text-white bg-dark border-0";
    }
    
    const toast = new bootstrap.Toast(toastEl, { delay: 4000 });
    toast.show();
}
