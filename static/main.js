const video = document.getElementById('webcam');
const canvas = document.getElementById('canvas');
const statusDiv = document.getElementById('status');
const ctx = canvas.getContext('2d');

// Inputs
const durationInput = document.getElementById('duration');
const minFreqInput = document.getElementById('min_freq');
const maxFreqInput = document.getElementById('max_freq');
const contrastInput = document.getElementById('contrast');
const waveformInput = document.getElementById('waveform');
const quantizeInput = document.getElementById('quantize');
const stereoInput = document.getElementById('stereo_envelope');
const continuousModeInput = document.getElementById('continuous_mode');

let isContinuousMode = false;
let currentAudio = null;

continuousModeInput.addEventListener('change', (e) => {
    isContinuousMode = e.target.checked;
    if (isContinuousMode && statusDiv.textContent === "Ready") {
        captureAndProcess();
    } else if (!isContinuousMode && currentAudio) {
        // Optional: Stop audio immediately when disabling continuous mode?
        // For now, let's just let it finish but not loop.
        currentAudio.onended = null;
    }
});

// Value displays
durationInput.addEventListener('input', (e) => document.getElementById('dur-val').textContent = e.target.value);
contrastInput.addEventListener('input', (e) => document.getElementById('con-val').textContent = e.target.value);

// Initialize Webcam
async function initCamera() {
    try {
        const stream = await navigator.mediaDevices.getUserMedia({ video: true });
        video.srcObject = stream;
    } catch (err) {
        statusDiv.textContent = "Error accessing webcam: " + err.message;
    }
}

initCamera();

// Capture and Process
async function captureAndProcess() {
    // If audio is playing, stop it to restart the cycle
    if (currentAudio) {
        currentAudio.pause();
        currentAudio.onended = null; // Prevent previous loop callback
        currentAudio = null;
    }

    if (statusDiv.textContent === "Processing...") return;

    statusDiv.textContent = "Processing...";

    // Draw video to canvas
    canvas.width = video.videoWidth;
    canvas.height = video.videoHeight;
    ctx.drawImage(video, 0, 0, canvas.width, canvas.height);

    // Show canvas, hide placeholder
    canvas.style.display = 'block';
    document.getElementById('placeholder').style.display = 'none';

    // Get Base64 image
    const imageData = canvas.toDataURL('image/jpeg');

    // Prepare payload
    const payload = {
        image: imageData,
        duration: durationInput.value,
        min_freq: minFreqInput.value,
        max_freq: maxFreqInput.value,
        contrast: contrastInput.value,
        waveform: waveformInput.value,
        quantize: quantizeInput.checked,
        stereo_envelope: stereoInput.checked
    };

    try {
        const response = await fetch('/process', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify(payload)
        });

        if (!response.ok) throw new Error("Server error");

        const data = await response.json();

        // Display Spectrogram
        const spectrogramImage = new Image();
        spectrogramImage.src = "data:image/png;base64," + data.spectrogram;
        spectrogramImage.onload = () => {
            ctx.drawImage(spectrogramImage, 0, 0, canvas.width, canvas.height);
        };

        // Play Audio
        const audioUrl = "data:audio/wav;base64," + data.audio;
        currentAudio = new Audio(audioUrl);

        statusDiv.textContent = "Playing...";

        const playhead = document.getElementById('playhead');
        playhead.style.display = 'block';

        currentAudio.play();

        // Animation Loop
        function updatePlayhead() {
            if (!currentAudio || currentAudio.paused || currentAudio.ended) return;
            const progress = currentAudio.currentTime / currentAudio.duration;
            playhead.style.left = (progress * 100) + '%';
            requestAnimationFrame(updatePlayhead);
        }
        requestAnimationFrame(updatePlayhead);

        currentAudio.onended = () => {
            statusDiv.textContent = "Ready";
            playhead.style.display = 'none';
            playhead.style.left = '0%';
            currentAudio = null;

            if (isContinuousMode) {
                captureAndProcess();
            }
        };

    } catch (err) {
        statusDiv.textContent = "Error: " + err.message;
        console.error(err);
        // If error occurs in continuous mode, should we stop or retry?
        // For now, let's stop to avoid infinite error loops.
        if (isContinuousMode) {
            continuousModeInput.checked = false;
            isContinuousMode = false;
        }
    }
}

// Capture and Process
document.addEventListener('keydown', async (e) => {
    if (e.code === 'Space') {
        e.preventDefault(); // Prevent scrolling
        captureAndProcess();
    }
});

// Add capture button click handler
const captureBtn = document.getElementById('capture-btn');
if (captureBtn) {
    captureBtn.addEventListener('click', captureAndProcess);
}

// Modal Logic
const modal = document.getElementById("info-modal");
const btn = document.getElementById("info-btn");
const span = document.getElementsByClassName("close")[0];

// Show modal on load
if (modal) {
    modal.style.display = "block";
}

if (btn) {
    btn.onclick = function () {
        modal.style.display = "block";
    }
}

if (span) {
    span.onclick = function () {
        modal.style.display = "none";
    }
}

window.onclick = function (event) {
    if (event.target == modal) {
        modal.style.display = "none";
    }
}
