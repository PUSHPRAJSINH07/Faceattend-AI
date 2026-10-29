/**
 * Live AI Webcam Face Recognition & Automatic Attendance Controller
 * -----------------------------------------------------------------
 * Handles camera streaming, frame capture, backend AI API calls,
 * bounding box rendering, audio feedback, and real-time DOM updates.
 */

let videoElement = null;
let overlayCanvas = null;
let overlayCtx = null;
let mediaStream = null;
let recognitionLoopTimer = null;
let isProcessingFrame = false;
let currentFacingMode = 'user'; // 'user' (selfie/front) or 'environment' (rear camera on mobile)

// Audio context for non-intrusive synthesized chime
let audioCtx = null;

// Throttling and cooldown to prevent rapid multi-recording
let lastRecognizedStudentId = null;
let lastRecognizedTimestamp = 0;
const RECOGNITION_COOLDOWN_MS = 6000; // 6 seconds cooldown per student before re-notifying
const FRAME_INTERVAL_MS = 750; // Capture frame every 750ms

document.addEventListener('DOMContentLoaded', () => {
    videoElement = document.getElementById('webcamVideo');
    overlayCanvas = document.getElementById('overlayCanvas');
    if (overlayCanvas) {
        overlayCtx = overlayCanvas.getContext('2d');
    }
});

/**
 * Initialize Web Audio API synth beep
 */
function playRecognitionChime() {
    const chk = document.getElementById('chkAudioFeedback');
    if (chk && !chk.checked) return;

    try {
        if (!audioCtx) {
            audioCtx = new (window.AudioContext || window.webkitAudioContext)();
        }
        if (audioCtx.state === 'suspended') {
            audioCtx.resume();
        }

        const osc = audioCtx.createOscillator();
        const gain = audioCtx.createGain();

        // Pleasant two-tone chime
        osc.type = 'sine';
        osc.frequency.setValueAtTime(587.33, audioCtx.currentTime); // D5
        osc.frequency.exponentialRampToValueAtTime(880.00, audioCtx.currentTime + 0.15); // A5

        gain.gain.setValueAtTime(0.2, audioCtx.currentTime);
        gain.gain.exponentialRampToValueAtTime(0.001, audioCtx.currentTime + 0.35);

        osc.connect(gain);
        gain.connect(audioCtx.destination);

        osc.start();
        osc.stop(audioCtx.currentTime + 0.36);
    } catch (e) {
        console.warn('Audio chime error:', e);
    }
}

/**
 * Synthesizes an alert buzzer for unrecognized / unregistered persons
 */
let lastBuzzerTimestamp = 0;
function playUnregisteredBuzzer() {
    const chk = document.getElementById('chkAudioFeedback');
    if (chk && !chk.checked) return;

    const now = Date.now();
    if (now - lastBuzzerTimestamp < 2000) return; // limit buzzer frequency
    lastBuzzerTimestamp = now;

    try {
        if (!audioCtx) {
            audioCtx = new (window.AudioContext || window.webkitAudioContext)();
        }
        if (audioCtx.state === 'suspended') {
            audioCtx.resume();
        }

        const osc = audioCtx.createOscillator();
        const gain = audioCtx.createGain();

        // Low urgent warning buzzer
        osc.type = 'sawtooth';
        osc.frequency.setValueAtTime(220, audioCtx.currentTime);
        osc.frequency.linearRampToValueAtTime(130, audioCtx.currentTime + 0.28);

        gain.gain.setValueAtTime(0.25, audioCtx.currentTime);
        gain.gain.exponentialRampToValueAtTime(0.001, audioCtx.currentTime + 0.3);

        osc.connect(gain);
        gain.connect(audioCtx.destination);

        osc.start();
        osc.stop(audioCtx.currentTime + 0.31);
    } catch (e) {
        console.warn('Audio buzzer error:', e);
    }
}

async function flipCamera() {
    currentFacingMode = (currentFacingMode === 'user') ? 'environment' : 'user';
    const isRunning = (mediaStream !== null);
    if (isRunning) {
        stopWebcam();
        setTimeout(startWebcam, 350);
    }
}

/**
 * Start webcam stream using HTML5 getUserMedia
 */
async function startWebcam() {
    const streamDot = document.getElementById('streamDot');
    const streamStatusText = document.getElementById('streamStatusText');
    const btnStart = document.getElementById('btnStartCam');
    const btnStop = document.getElementById('btnStopCam');
    const idleOverlay = document.getElementById('idleOverlay');
    const targetReticle = document.getElementById('targetReticle');

    try {
        updateBanner('scanning', 'Connecting to camera...', 'Please allow webcam permissions if prompted.');

        mediaStream = await navigator.mediaDevices.getUserMedia({
            video: {
                width: { ideal: 640 },
                height: { ideal: 480 },
                facingMode: { ideal: currentFacingMode }
            },
            audio: false
        });

        videoElement.srcObject = mediaStream;

        // On mobile, if rear camera is active, don't mirror image; mirror only for front selfie camera
        if (currentFacingMode === 'user') {
            videoElement.style.transform = 'scaleX(-1)';
            overlayCanvas.style.transform = 'scaleX(-1)';
        } else {
            videoElement.style.transform = 'none';
            overlayCanvas.style.transform = 'none';
        }

        videoElement.onloadedmetadata = () => {
            videoElement.play();
            // Sync canvas dimensions with video stream
            overlayCanvas.width = videoElement.videoWidth || 640;
            overlayCanvas.height = videoElement.videoHeight || 480;

            if (idleOverlay) idleOverlay.classList.add('hidden');
            if (targetReticle) targetReticle.classList.remove('reticle-hidden');

            streamDot.classList.add('active');
            streamStatusText.textContent = (currentFacingMode === 'user') ? 'Front Camera Active' : 'Rear Camera Active';
            btnStart.disabled = true;
            btnStop.disabled = false;

            document.getElementById('fpsDisplay').textContent = '~1.3 FPS Active';

            // Start recognition capture loop
            startRecognitionLoop();
        };

    } catch (err) {
        console.error('Camera access error:', err);
        let errorMsg = 'Could not access camera.';
        if (err.name === 'NotAllowedError' || err.name === 'PermissionDeniedError') {
            errorMsg = 'Camera permission was denied. Please grant camera permission in your browser URL bar.';
        } else if (err.name === 'NotFoundError' || err.name === 'DevicesNotFoundError') {
            errorMsg = 'No camera device found on this system.';
        } else if (err.name === 'NotReadableError' || err.name === 'TrackStartError') {
            errorMsg = 'Camera is in use by another application. Please close other apps using the webcam.';
        }
        alert(errorMsg);
        updateBanner('idle', 'Camera Unavailable', errorMsg);
    }
}

/**
 * Stop webcam stream and clear canvas
 */
function stopWebcam() {
    if (recognitionLoopTimer) {
        clearInterval(recognitionLoopTimer);
        recognitionLoopTimer = null;
    }

    if (mediaStream) {
        mediaStream.getTracks().forEach(track => track.stop());
        mediaStream = null;
    }

    if (videoElement) {
        videoElement.srcObject = null;
    }

    if (overlayCtx && overlayCanvas) {
        overlayCtx.clearRect(0, 0, overlayCanvas.width, overlayCanvas.height);
    }

    const idleOverlay = document.getElementById('idleOverlay');
    const targetReticle = document.getElementById('targetReticle');
    const streamDot = document.getElementById('streamDot');
    const streamStatusText = document.getElementById('streamStatusText');
    const btnStart = document.getElementById('btnStartCam');
    const btnStop = document.getElementById('btnStopCam');

    if (idleOverlay) idleOverlay.classList.remove('hidden');
    if (targetReticle) targetReticle.classList.add('reticle-hidden');

    streamDot.classList.remove('active');
    streamStatusText.textContent = 'Camera Inactive';
    btnStart.disabled = false;
    btnStop.disabled = true;

    document.getElementById('fpsDisplay').textContent = 'Idle';
    updateBanner('idle', 'Camera Turned Off', 'Click Start Camera to resume attendance.');
}

/**
 * Interval loop to grab frames and send to Flask backend
 */
function startRecognitionLoop() {
    if (recognitionLoopTimer) clearInterval(recognitionLoopTimer);
    recognitionLoopTimer = setInterval(captureAndAnalyzeFrame, FRAME_INTERVAL_MS);
}

/**
 * Capture current frame from video element and send to /api/recognize
 */
async function captureAndAnalyzeFrame() {
    if (!mediaStream || isProcessingFrame || !videoElement || videoElement.paused || videoElement.ended) {
        return;
    }

    isProcessingFrame = true;

    try {
        // Draw video frame to an offscreen canvas
        const tempCanvas = document.createElement('canvas');
        tempCanvas.width = videoElement.videoWidth || 640;
        tempCanvas.height = videoElement.videoHeight || 480;
        const ctx = tempCanvas.getContext('2d');
        ctx.drawImage(videoElement, 0, 0, tempCanvas.width, tempCanvas.height);

        // Convert to base64 JPEG
        const base64Data = tempCanvas.toDataURL('image/jpeg', 0.85);

        // Send to backend API
        const response = await fetch('/api/recognize', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({
                image: base64Data
            })
        });

        if (!response.ok) {
            throw new Error(`HTTP error ${response.status}`);
        }

        const data = await response.json();
        handleRecognitionResponse(data, tempCanvas.width, tempCanvas.height);

    } catch (err) {
        console.error('Frame recognition error:', err);
    } finally {
        isProcessingFrame = false;
    }
}

/**
 * Process AI backend response, render boxes and update UI
 */
function handleRecognitionResponse(res, frameWidth, frameHeight) {
    if (!overlayCtx || !overlayCanvas) return;

    // Clear previous bounding box
    overlayCtx.clearRect(0, 0, overlayCanvas.width, overlayCanvas.height);

    const now = Date.now();
    const hud = document.getElementById('unregisteredAlertHUD');
    const viewport = document.querySelector('.viewport-container');

    // 1. RECOGNIZED REGISTERED STUDENT
    if (res.status === 'recognized' && res.student) {
        if (hud) hud.classList.add('hud-hidden');
        if (viewport) viewport.classList.remove('viewport-hazard-warning');

        const student = res.student;
        const confidence = res.confidence;
        const box = res.box; // [x, y, w, h]

        // Draw bounding box in bright verified green
        if (box) {
            drawBoundingBox(box, `${student.name} (${confidence}%)`, '#10b981');
        }

        const newlyMarked = res.newly_marked;
        const isCooldownActive = (student.id === lastRecognizedStudentId && (now - lastRecognizedTimestamp) < RECOGNITION_COOLDOWN_MS);

        const daysAttended = (res.attendance_record && res.attendance_record.total_days_attended) 
            ? ` &bull; Attended ${res.attendance_record.total_days_attended} Days Total` 
            : '';

        if (newlyMarked) {
            playRecognitionChime();
            updateBanner('recognized', `Attendance Marked: ${student.name}`, `Enrollment: ${student.enrollment_no} | Class: ${student.student_class}${daysAttended}`, `${confidence}% Match`);
            appendAttendanceToFeed(student, res.attendance_record || { time: 'Just now', confidence: confidence });
            lastRecognizedStudentId = student.id;
            lastRecognizedTimestamp = now;
        } else if (!isCooldownActive) {
            updateBanner('recognized', `Already Present: ${student.name}`, `${res.message}${daysAttended}`, `${confidence}% Match`);
            lastRecognizedStudentId = student.id;
            lastRecognizedTimestamp = now;
        }
    }
    // 2. UNKNOWN / UNREGISTERED PERSON (SHOW DRAMATIC WARNING ANIMATION & BUZZER)
    else if (res.status === 'unknown') {
        if (hud) hud.classList.remove('hud-hidden');
        if (viewport) viewport.classList.add('viewport-hazard-warning');

        playUnregisteredBuzzer();

        const box = res.box;
        if (box) {
            drawBoundingBox(box, '⚠️ NOT REGISTERED', '#ef4444');
        }
        updateBanner('unknown', '⚠️ ACCESS DENIED — UNREGISTERED PERSON', 'Face does not match any student profile in the database. Attendance not marked.', `${res.confidence || 0}% Sim`);
    }
    // 3. MULTIPLE FACES
    else if (res.status === 'multiple_faces') {
        if (hud) hud.classList.add('hud-hidden');
        if (viewport) viewport.classList.remove('viewport-hazard-warning');

        if (res.boxes && Array.isArray(res.boxes)) {
            res.boxes.forEach((b, idx) => {
                drawBoundingBox(b, `Face ${idx+1}`, '#f59e0b');
            });
        }
        updateBanner('multiple', 'Multiple Faces Detected', res.message, 'Ambiguous');
    }
    // 4. NO FACE DETECTED
    else if (res.status === 'no_face') {
        if (hud) hud.classList.add('hud-hidden');
        if (viewport) viewport.classList.remove('viewport-hazard-warning');
        updateBanner('scanning', 'Scanning Active...', 'Position your face inside the camera guide box.', '--');
    }
}

/**
 * Draw bounding box and label on overlay canvas
 */
function drawBoundingBox(box, label, color) {
    if (!overlayCtx || !box) return;

    const [x, y, w, h] = box;

    overlayCtx.save();
    overlayCtx.lineWidth = 3;
    overlayCtx.strokeStyle = color;
    overlayCtx.fillStyle = color;

    // Rounded rectangle
    overlayCtx.beginPath();
    overlayCtx.roundRect(x, y, w, h, 8);
    overlayCtx.stroke();

    // Top label badge background
    overlayCtx.font = 'bold 15px Inter, sans-serif';
    const textWidth = overlayCtx.measureText(label).width;
    const badgeHeight = 26;
    const badgeY = Math.max(0, y - badgeHeight);

    overlayCtx.fillRect(x, badgeY, textWidth + 16, badgeHeight);

    // Label text
    overlayCtx.fillStyle = '#ffffff';
    overlayCtx.fillText(label, x + 8, badgeY + 18);

    overlayCtx.restore();
}

/**
 * Update recognition status banner below the video feed
 */
function updateBanner(type, title, subtitle, metric = null) {
    const banner = document.getElementById('recognitionStatusBanner');
    const bannerTitle = document.getElementById('bannerTitle');
    const bannerSubtitle = document.getElementById('bannerSubtitle');
    const metricElem = document.getElementById('metricConfidence');
    const iconElem = document.getElementById('bannerIcon');

    if (!banner) return;

    banner.className = `recognition-banner banner-${type}`;
    if (bannerTitle) bannerTitle.textContent = title;
    if (bannerSubtitle) bannerSubtitle.textContent = subtitle;

    if (metricElem) {
        metricElem.textContent = metric ? `Similarity: ${metric}` : '';
    }

    // Dynamic icon SVGs
    if (iconElem) {
        if (type === 'recognized') {
            iconElem.innerHTML = '<svg width="24" height="24" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2.5"><polyline points="20 6 9 17 4 12"></polyline></svg>';
        } else if (type === 'unknown') {
            iconElem.innerHTML = '<svg width="24" height="24" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><circle cx="12" cy="12" r="10"></circle><line x1="12" y1="8" x2="12" y2="12"></line><line x1="12" y1="16" x2="12.01" y2="16"></line></svg>';
        } else if (type === 'multiple') {
            iconElem.innerHTML = '<svg width="24" height="24" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><path d="M17 21v-2a4 4 0 0 0-4-4H5a4 4 0 0 0-4 4v2"></path><circle cx="9" cy="7" r="4"></circle><path d="M23 21v-2a4 4 0 0 0-3-3.87"></path><path d="M16 3.13a4 4 0 0 1 0 7.75"></path></svg>';
        } else {
            iconElem.innerHTML = '<svg width="24" height="24" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><circle cx="12" cy="12" r="10"></circle><line x1="12" y1="16" x2="12" y2="12"></line><line x1="12" y1="8" x2="12.01" y2="8"></line></svg>';
        }
    }
}

/**
 * Prepend a newly marked attendance record to the live feed in real-time
 */
function appendAttendanceToFeed(student, record) {
    const feedList = document.getElementById('liveFeedList');
    const emptyState = document.getElementById('emptyFeedState');
    const countBadge = document.getElementById('todayCountBadge');

    if (!feedList) return;

    if (emptyState) {
        emptyState.remove();
    }

    // Check if element for this student already exists in feed
    const existingElem = document.getElementById(`feed-student-${student.id}`);
    if (existingElem) {
        existingElem.remove();
    }

    const item = document.createElement('div');
    item.className = 'feed-item';
    item.id = `feed-student-${student.id}`;

    const timeStr = record.time || new Date().toLocaleTimeString();
    const confStr = record.confidence ? `${record.confidence}% Match` : 'Verified';

    item.innerHTML = `
        <img src="/uploads/students/${student.photo_filename}" alt="${student.name}" class="feed-avatar">
        <div class="feed-info">
            <div class="feed-name">${student.name}</div>
            <div class="feed-roll">${student.enrollment_no} &bull; ${student.student_class}</div>
            <div class="feed-meta">
                <span class="feed-time">${timeStr}</span>
                <span class="badge-conf-sm">${confStr}</span>
            </div>
        </div>
        <div class="feed-status">
            <span class="badge-status status-present">&check; Present</span>
        </div>
    `;

    feedList.prepend(item);

    // Update count badge
    if (countBadge) {
        const currentCount = feedList.querySelectorAll('.feed-item').length;
        countBadge.textContent = `${currentCount} Present`;
    }
}
