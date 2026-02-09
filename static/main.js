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

// Web Audio API setup
let audioContext = null;
let currentSource = null;
let currentGain = null;
let continuationOscillator = null;
let continuationGain = null;
let lastAudioSegment = null; // Store last 1 second of audio for looping
let isContinuousMode = false;
let isProcessing = false;
let animationFrameId = null;

// Initialize Audio Context on user interaction
function initAudioContext() {
    if (!audioContext) {
        audioContext = new (window.AudioContext || window.webkitAudioContext)();
    }
    return audioContext;
}

continuousModeInput.addEventListener('change', (e) => {
    isContinuousMode = e.target.checked;
    if (isContinuousMode && statusDiv.textContent === "Ready") {
        initAudioContext();
        captureAndProcess();
    } else if (!isContinuousMode) {
        stopContinuationTone();
        if (currentSource) {
            currentSource.stop();
            currentSource = null;
        }
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

// Decode base64 WAV to AudioBuffer
async function decodeWavData(base64Data) {
    const binaryString = atob(base64Data);
    const bytes = new Uint8Array(binaryString.length);
    for (let i = 0; i < binaryString.length; i++) {
        bytes[i] = binaryString.charCodeAt(i);
    }
    const audioBuffer = await audioContext.decodeAudioData(bytes.buffer);
    return audioBuffer;
}

// Extract last 1 second of audio for looping
function extractLastAudioSegment(audioBuffer) {
    try {
        // Get the last 1 second of audio
        const sampleRate = audioBuffer.sampleRate;
        const segmentDuration = 1.0; // 1 second
        const segmentLength = Math.floor(sampleRate * segmentDuration);
        const startIndex = Math.max(0, audioBuffer.length - segmentLength);

        // Create a new buffer for the segment
        const segmentBuffer = audioContext.createBuffer(
            audioBuffer.numberOfChannels,
            segmentLength,
            sampleRate
        );

        // Copy the last segment
        for (let channel = 0; channel < audioBuffer.numberOfChannels; channel++) {
            const sourceData = audioBuffer.getChannelData(channel);
            const destData = segmentBuffer.getChannelData(channel);
            for (let i = 0; i < segmentLength; i++) {
                destData[i] = sourceData[startIndex + i];
            }
        }

        return segmentBuffer;
    } catch (err) {
        console.error('Error extracting audio segment:', err);
        return null;
    }
}

// Start continuation by looping the last audio segment
function startContinuationTone() {
    if (!audioContext || !lastAudioSegment) {
        console.log('Cannot start continuation:', { audioContext: !!audioContext, lastAudioSegment: !!lastAudioSegment });
        return;
    }

    // Stop any existing continuation tone first
    if (continuationOscillator) {
        try {
            continuationOscillator.stop();
            continuationOscillator.disconnect();
        } catch (e) {
            // Ignore errors from already stopped sources
        }
        continuationOscillator = null;
    }
    if (continuationGain) {
        try {
            continuationGain.disconnect();
        } catch (e) { }
        continuationGain = null;
    }

    const now = audioContext.currentTime;

    // Create a looping buffer source with the last 1 second of audio
    continuationOscillator = audioContext.createBufferSource();
    continuationGain = audioContext.createGain();

    continuationOscillator.buffer = lastAudioSegment;
    continuationOscillator.loop = true;
    continuationOscillator.loopStart = 0;
    continuationOscillator.loopEnd = lastAudioSegment.duration;

    // Start at zero volume and fade in smoothly to avoid clicks
    continuationGain.gain.setValueAtTime(0, now);
    continuationGain.gain.linearRampToValueAtTime(0.5, now + 0.2); // 200ms linear fade

    continuationOscillator.connect(continuationGain);
    continuationGain.connect(audioContext.destination);

    continuationOscillator.start(now);

    console.log('Continuation tone started - looping last 1 second of audio');
}

// Stop continuation tone with fade out
function stopContinuationTone(fadeTime = 0.5) {
    if (!continuationOscillator || !continuationGain) return;

    const now = audioContext.currentTime;

    // Fade out
    continuationGain.gain.cancelScheduledValues(now);
    continuationGain.gain.setValueAtTime(continuationGain.gain.value, now);
    continuationGain.gain.exponentialRampToValueAtTime(0.01, now + fadeTime);

    // Stop and clean up
    continuationOscillator.stop(now + fadeTime);

    setTimeout(() => {
        if (continuationOscillator) {
            continuationOscillator.disconnect();
            continuationOscillator = null;
        }
        if (continuationGain) {
            continuationGain.disconnect();
            continuationGain = null;
        }
    }, fadeTime * 1000 + 100);
}

// Play audio with crossfade
async function playAudioBuffer(audioBuffer) {
    if (!audioContext) return;

    // Extract and store the last 1 second for continuation
    lastAudioSegment = extractLastAudioSegment(audioBuffer);
    console.log('Extracted last 1 second of audio for continuation');

    const now = audioContext.currentTime;
    const fadeTime = 0.5; // 500ms crossfade

    // Create source and gain for new audio
    const source = audioContext.createBufferSource();
    const gain = audioContext.createGain();

    source.buffer = audioBuffer;
    source.connect(gain);
    gain.connect(audioContext.destination);

    // Start with low volume for fade in
    gain.gain.setValueAtTime(0.01, now);
    gain.gain.exponentialRampToValueAtTime(1.0, now + fadeTime);

    // If continuation tone is playing, fade it out
    if (continuationOscillator) {
        stopContinuationTone(fadeTime);
    }

    // Start playback
    source.start(now);

    // Update global references
    currentSource = source;
    currentGain = gain;

    // Update playhead
    const playhead = document.getElementById('playhead');
    playhead.style.display = 'block';
    const startTime = now;
    const duration = audioBuffer.duration;

    function updatePlayhead() {
        if (!currentSource) {
            playhead.style.display = 'none';
            playhead.style.left = '0%';
            return;
        }

        const elapsed = audioContext.currentTime - startTime;
        const progress = Math.min(elapsed / duration, 1.0);
        playhead.style.left = (progress * 100) + '%';

        if (progress < 1.0) {
            animationFrameId = requestAnimationFrame(updatePlayhead);
        } else {
            playhead.style.display = 'none';
            playhead.style.left = '0%';
            onAudioEnded();
        }
    }

    animationFrameId = requestAnimationFrame(updatePlayhead);
}

// Handle audio end
function onAudioEnded() {
    console.log('Audio ended, isContinuousMode:', isContinuousMode);
    currentSource = null;
    currentGain = null;

    if (isContinuousMode) {
        console.log('Starting continuation tone and processing next frame');
        // Start continuation tone immediately
        startContinuationTone();
        statusDiv.textContent = "Processing...";

        // Start processing next frame
        captureAndProcess();
    } else {
        statusDiv.textContent = "Ready";
    }
}

// Capture and Process
async function captureAndProcess() {
    // In continuous mode, if already processing, just return (continuation tone will play)
    if (isProcessing) {
        console.log('Already processing, continuation tone should be playing');
        return;
    }

    isProcessing = true;

    // Initialize audio context if needed
    if (!audioContext) {
        initAudioContext();
    }

    // Only update status if not in continuous mode (continuation tone handles it)
    if (!isContinuousMode || !continuationOscillator) {
        statusDiv.textContent = "Processing...";
    }

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

        // Decode and play audio
        const audioBuffer = await decodeWavData(data.audio);

        console.log('Audio decoded, playing now');
        statusDiv.textContent = "Playing...";
        await playAudioBuffer(audioBuffer);

    } catch (err) {
        console.error('Error in captureAndProcess:', err);
        statusDiv.textContent = "Error: " + err.message;
        console.error(err);

        stopContinuationTone();

        if (isContinuousMode) {
            continuousModeInput.checked = false;
            isContinuousMode = false;
        }
    } finally {
        console.log('Processing complete, resetting isProcessing flag');
        isProcessing = false;
    }
}

// Capture on spacebar
document.addEventListener('keydown', async (e) => {
    if (e.code === 'Space') {
        e.preventDefault();
        if (!isContinuousMode) {
            captureAndProcess();
        }
    }
});

// Add capture button click handler
const captureBtn = document.getElementById('capture-btn');
if (captureBtn) {
    captureBtn.addEventListener('click', () => {
        if (!isContinuousMode) {
            captureAndProcess();
        }
    });
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
