import io
import wave
import numpy as np
from PIL import Image, ImageFilter

class SpectroGraphic:
    def __init__(
        self,
        image_source,
        height: int = 100,
        duration: int = 10,
        min_freq: int = 500,
        max_freq: int = 5000,
        sample_rate: int = 44100,
        contrast: float = 5,
        waveform: str = "sine",
        quantize: bool = False,
        stereo_envelope: bool = False,
    ):
        if isinstance(image_source, (str, bytes)):
             self.image = Image.open(image_source)
        else:
            self.image = image_source

        self.HEIGHT = height
        self.DURATION = duration
        self.SAMPLE_RATE = sample_rate
        self.MIN_FREQ = min_freq
        self.MAX_FREQ = max_freq
        self.CONTRAST = contrast
        self.WAVEFORM = waveform
        self.QUANTIZE = quantize
        self.STEREO_ENVELOPE = stereo_envelope

        # Calculate width to maintain aspect ratio
        self.WIDTH = int(self.image.width * (self.HEIGHT / self.image.height))
        self.SCALE = np.array([
            261.63, 293.66, 329.63, 349.23, 392.00, 440.00, 493.88,  # C4-B4
            523.25, 587.33, 659.25, 698.46, 783.99, 880.00, 987.77,  # C5-B5
            1046.50, 1174.66, 1318.51, 1396.91, 1567.98, 1760.00, 1975.53, # C6-B6
            2093.00, 2349.32, 2637.02, 2793.83, 3135.96, 3520.00, 3951.07, # C7-B7
            4186.01 # C8
        ])

        self._sound_array = None

    def _preprocess_image(self):
        """
        Resize, blur, and convert the input image to a normalized grayscale numpy array.
        """
        # Resize and convert to grayscale
        img = self.image.resize((self.WIDTH, self.HEIGHT), Image.Resampling.LANCZOS)
        img = img.filter(ImageFilter.GaussianBlur(radius=1))
        img = img.convert('L')
        return np.array(img) / 255.0

    def _generate_frequencies(self):
        """
        Generate an array of frequencies mapped to image rows
        quantized to a musical scale.
        """
        # Map Y-axis rows to frequencies
        freqs = np.geomspace(self.MAX_FREQ, self.MIN_FREQ, self.HEIGHT)
        if self.QUANTIZE:
            idx = np.abs(freqs[:, None] - self.SCALE[None, :]).argmin(axis=1)
            freqs = self.SCALE[idx]
        return freqs

    def _generate_waveform(self, phases):
        """Generate a waveform (sine, square, sawtooth, or triangle) for the given phase values."""
        if self.WAVEFORM == "sine":
            return np.sin(phases)
        elif self.WAVEFORM == "square":
            return np.sign(np.sin(phases))
        elif self.WAVEFORM == "sawtooth":
            return 2 * (phases / (2 * np.pi) - np.floor(phases / (2 * np.pi) + 0.5))
        elif self.WAVEFORM == "triangle":
            return 2 * np.abs(2 * (phases / (2 * np.pi) - np.floor(phases / (2 * np.pi) + 0.5))) - 1
        else:
            return np.sin(phases)

    def _process(self):
        """
        Convert the processed image into an audio waveform array.
        """
        pixels = self._preprocess_image()
        pixels = pixels ** self.CONTRAST
        total_samples = int(self.DURATION * self.SAMPLE_RATE)
        samples_per_column = total_samples // self.WIDTH
        total_samples = samples_per_column * self.WIDTH
        freqs = self._generate_frequencies()[:, np.newaxis]
        audio_chunks = []
        t_col = np.linspace(0, samples_per_column / self.SAMPLE_RATE, samples_per_column, endpoint=False)
        current_time = 0
        for col_idx in range(self.WIDTH):
            col_intensities = pixels[:, col_idx:col_idx+1]
            if np.max(col_intensities) < 0.01:
                audio_chunks.append(np.zeros(samples_per_column))
                current_time += samples_per_column / self.SAMPLE_RATE
                continue
            t_global = t_col + current_time
            phases = 2 * np.pi * freqs * t_global
            waves = self._generate_waveform(phases)
            weighted_waves = waves * col_intensities
            col_audio = np.sum(weighted_waves, axis=0)
            audio_chunks.append(col_audio)
            current_time += samples_per_column / self.SAMPLE_RATE
        audio = np.concatenate(audio_chunks)
        if self.STEREO_ENVELOPE:
            audio_left = []
            audio_right = []
            current_time = 0
            pan = (freqs - self.MIN_FREQ) / (self.MAX_FREQ - self.MIN_FREQ)
            left_gain = np.cos(pan * np.pi / 2)
            right_gain = np.sin(pan * np.pi / 2)
            for col_idx in range(self.WIDTH):
                col_intensities = pixels[:, col_idx:col_idx+1]
                t_global = t_col + current_time
                phases = 2 * np.pi * freqs * t_global
                waves = self._generate_waveform(phases)
                weighted = waves * col_intensities
                chunk_l = np.sum(weighted * left_gain, axis=0)
                chunk_r = np.sum(weighted * right_gain, axis=0)
                audio_left.append(chunk_l)
                audio_right.append(chunk_r)
                current_time += samples_per_column / self.SAMPLE_RATE
            audio_l = np.concatenate(audio_left)
            audio_r = np.concatenate(audio_right)
            audio = np.column_stack((audio_l, audio_r))
        else:
            pass
        max_val = np.max(np.abs(audio))
        if max_val > 0:
            audio = audio / max_val
        audio = (audio * 32767).astype(np.int16)
        return audio

    @property
    def sound_array(self):
        """Return the generated audio array, computing it if not already cached."""
        if self._sound_array is None:
            self._sound_array = self._process()
        return self._sound_array

    def get_wav_bytes(self):
        """Return the audio as a WAV file in a BytesIO buffer."""
        audio = self.sound_array
        
        # Check if stereo
        channels = 1
        if len(audio.shape) > 1:
            channels = audio.shape[1]
        
        byte_io = io.BytesIO()
        
        with wave.open(byte_io, 'wb') as wav_file:
            wav_file.setnchannels(channels)
            wav_file.setsampwidth(2) # 16-bit
            wav_file.setframerate(self.SAMPLE_RATE)
            wav_file.writeframes(audio.tobytes())
            
        byte_io.seek(0)
        return byte_io

    def get_spectrogram_plot(self):
        """Return a PNG image buffer of the processed image with a colormap applied as a spectrogram plot."""
        pixels = self._preprocess_image()
        pixels = np.clip(pixels, 0, 1)
        h, w = pixels.shape
        rgb_img = np.zeros((h, w, 3), dtype=np.uint8)
        rgb_img[..., 0] = (pixels * 255).astype(np.uint8)
        rgb_img[..., 1] = (np.sin(pixels * np.pi) * 200).astype(np.uint8)
        rgb_img[..., 2] = ((1 - pixels) * 100 + 50).astype(np.uint8)
        mask = pixels < 0.05
        rgb_img[mask] = 0
        img = Image.fromarray(rgb_img)
        buf = io.BytesIO()
        img.save(buf, format='PNG')
        buf.seek(0)
        return buf
