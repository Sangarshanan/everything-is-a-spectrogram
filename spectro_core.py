import numpy as np
from PIL import Image, ImageFilter
import io
import wave
import math

class SpectroGraphic:
    def __init__(
        self,
        image_source,
        height: int = 100,
        duration: int = 10,
        min_freq: int = 500,
        max_freq: int = 5000,
        sample_rate: int = 44100,
        num_tones: int = 3, # Unused in vectorized version but kept for API compatibility
        contrast: float = 5,
        waveform: str = "sine",
        quantize: bool = False,
        stereo_envelope: bool = False,
    ):
        if isinstance(image_source, (str, bytes)): # Handle path or bytes if needed, though main.py passes PIL Image
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
        
        # C Major Scale Frequencies for quantization
        self.SCALE = np.array([
            261.63, 293.66, 329.63, 349.23, 392.00, 440.00, 493.88,  # C4-B4
            523.25, 587.33, 659.25, 698.46, 783.99, 880.00, 987.77,  # C5-B5
            1046.50, 1174.66, 1318.51, 1396.91, 1567.98, 1760.00, 1975.53, # C6-B6
            2093.00, 2349.32, 2637.02, 2793.83, 3135.96, 3520.00, 3951.07, # C7-B7
            4186.01 # C8
        ])

        self._sound_array = None

    def _preprocess_image(self):
        # Resize and convert to grayscale
        img = self.image.resize((self.WIDTH, self.HEIGHT), Image.Resampling.LANCZOS)
        # Apply Gaussian Blur to smooth out harsh transitions
        img = img.filter(ImageFilter.GaussianBlur(radius=1))
        img = img.convert('L')
        # Flip vertically because low frequencies (bottom) correspond to higher indices in image usually
        # But in spectrograms, low freq is bottom. Let's align with standard:
        # Image (0,0) is top-left. We want top of image to be high freq?
        # Usually spectrograms have low freq at bottom.
        # So row index 0 (top) -> High Freq
        # Row index H (bottom) -> Low Freq
        # This matches standard image coords.
        return np.array(img) / 255.0

    def _generate_frequencies(self):
        # Map Y-axis rows to frequencies
        # Row 0 = Max Freq, Row H = Min Freq
        # Use geometric progression (logarithmic scale) for musicality
        freqs = np.geomspace(self.MAX_FREQ, self.MIN_FREQ, self.HEIGHT)
        
        if self.QUANTIZE:
            # Vectorized quantization: find nearest scale freq for each freq
            # abs(freqs[:, None] - scale[None, :]).argmin(axis=1)
            idx = np.abs(freqs[:, None] - self.SCALE[None, :]).argmin(axis=1)
            freqs = self.SCALE[idx]
            
        return freqs

    def _generate_waveform(self, phases):
        if self.WAVEFORM == "sine":
            return np.sin(phases)
        elif self.WAVEFORM == "square":
            return np.sign(np.sin(phases))
        elif self.WAVEFORM == "sawtooth":
            # (x + 1) % 2 - 1 maps to -1..1
            return 2 * (phases / (2 * np.pi) - np.floor(phases / (2 * np.pi) + 0.5))
        elif self.WAVEFORM == "triangle":
            return 2 * np.abs(2 * (phases / (2 * np.pi) - np.floor(phases / (2 * np.pi) + 0.5))) - 1
        else:
            return np.sin(phases)

    def _process(self):
        # 1. Prepare Image Data
        # Shape: (HEIGHT, WIDTH)
        pixels = self._preprocess_image()
        
        # Apply contrast
        pixels = pixels ** self.CONTRAST

        # 2. Prepare Time and Frequencies
        total_samples = int(self.DURATION * self.SAMPLE_RATE)
        samples_per_column = total_samples // self.WIDTH
        
        # Adjust total samples to match exact column division
        total_samples = samples_per_column * self.WIDTH
        
        # Frequencies for each row. Shape: (HEIGHT, 1)
        freqs = self._generate_frequencies()[:, np.newaxis]
        
        # 3. Generate Audio
        # We process column by column to save memory, but vectorize within the column
        
        audio_chunks = []
        
        # Time array for one column
        t_col = np.linspace(0, samples_per_column / self.SAMPLE_RATE, samples_per_column, endpoint=False)
        
        # Global time offset tracker
        current_time = 0
        
        for col_idx in range(self.WIDTH):
            # Get column intensities. Shape: (HEIGHT, 1)
            col_intensities = pixels[:, col_idx:col_idx+1]
            
            # Skip silent columns
            if np.max(col_intensities) < 0.01:
                audio_chunks.append(np.zeros(samples_per_column))
                current_time += samples_per_column / self.SAMPLE_RATE
                continue

            # Calculate phases: 2 * pi * freq * t
            # We need continuous phase to avoid clicking between columns?
            # For simplicity in this "pixel-synth" approach, we often reset phase or accept clicks.
            # To do it properly, we'd need global time `t`.
            
            # Global time for this column
            t_global = t_col + current_time
            
            # Phases: (HEIGHT, samples_per_column)
            phases = 2 * np.pi * freqs * t_global
            
            # Generate waves
            waves = self._generate_waveform(phases)
            
            # Apply intensities
            weighted_waves = waves * col_intensities
            
            # Sum all frequencies (rows) to get the sound for this column
            # Shape: (samples_per_column,)
            col_audio = np.sum(weighted_waves, axis=0)
            
            audio_chunks.append(col_audio)
            current_time += samples_per_column / self.SAMPLE_RATE

        # Concatenate all chunks
        audio = np.concatenate(audio_chunks)
        
        # 4. Apply Envelope (Stereo or Mono)
        if self.STEREO_ENVELOPE:
            # Simple panning based on frequency (already summed, so we can't do freq-based panning easily post-sum)
            # Wait, the original code did panning PER PIXEL.
            # To support that, we need to apply panning BEFORE summing.
            
            # Let's redo the loop slightly for stereo
            # Re-initialize
            audio_left = []
            audio_right = []
            current_time = 0
            
            # Pan map: 0 (Left) to 1 (Right) based on freq
            # High freq = Right, Low freq = Left? Or vice versa.
            # Original: (freq - min) / (max - min)
            pan = (freqs - self.MIN_FREQ) / (self.MAX_FREQ - self.MIN_FREQ)
            left_gain = np.cos(pan * np.pi / 2)
            right_gain = np.sin(pan * np.pi / 2)
            
            for col_idx in range(self.WIDTH):
                col_intensities = pixels[:, col_idx:col_idx+1]
                t_global = t_col + current_time
                phases = 2 * np.pi * freqs * t_global
                waves = self._generate_waveform(phases)
                
                # Apply intensity
                weighted = waves * col_intensities
                
                # Apply Panning
                chunk_l = np.sum(weighted * left_gain, axis=0)
                chunk_r = np.sum(weighted * right_gain, axis=0)
                
                audio_left.append(chunk_l)
                audio_right.append(chunk_r)
                current_time += samples_per_column / self.SAMPLE_RATE
            
            audio_l = np.concatenate(audio_left)
            audio_r = np.concatenate(audio_right)
            audio = np.column_stack((audio_l, audio_r))
            
        else:
            # Already computed mono above? No, I put it in a block.
            # Let's just use the stereo logic but with equal gain if mono
            # Actually, let's just stick to the first loop for mono and second for stereo logic
            # to keep it clean.
            # (The code above "audio = np.concatenate(audio_chunks)" was for mono)
            pass # audio is already set

        # Normalize
        max_val = np.max(np.abs(audio))
        if max_val > 0:
            audio = audio / max_val
            
        # Convert to 16-bit PCM
        audio = (audio * 32767).astype(np.int16)
        
        return audio

    @property
    def sound_array(self):
        if self._sound_array is None:
            self._sound_array = self._process()
        return self._sound_array

    def get_wav_bytes(self):
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
        # Generate a simple spectrogram image using PIL
        # Since we already have the source image which IS the spectrogram (essentially),
        # we can just return the processed source image with a colormap!
        # This is much faster and "truer" to the generation process.
        
        # 1. Get processed grayscale image
        pixels = self._preprocess_image() # (HEIGHT, WIDTH)
        
        # 2. Apply colormap (Viridis-like)
        # Viridis approximation: Dark Purple -> Blue -> Green -> Yellow
        # We can use a simple lookup table or just map RGB channels
        
        # Normalize pixels 0..1
        pixels = np.clip(pixels, 0, 1)
        
        # Simple heatmap: 
        # Low: (0, 0, 0)
        # Mid: (0, 0, 255)
        # High: (255, 255, 0)
        
        # Let's use a custom palette for "Neon" look (matches app theme)
        # Background: Black
        # Low intensity: Dark Blue
        # Mid: Cyan
        # High: Magenta/White
        
        h, w = pixels.shape
        rgb_img = np.zeros((h, w, 3), dtype=np.uint8)
        
        # Vectorized color mapping
        # R: Increases with intensity (Purple -> Pink)
        # G: Peaked at mid-high (Cyan -> White)
        # B: High at low-mid (Blue -> Cyan)
        
        # Simple "Inferno" style
        rgb_img[..., 0] = (pixels * 255).astype(np.uint8) # R
        rgb_img[..., 1] = (np.sin(pixels * np.pi) * 200).astype(np.uint8) # G
        rgb_img[..., 2] = ((1 - pixels) * 100 + 50).astype(np.uint8) # B
        
        # Apply mask for zero intensity (Black)
        mask = pixels < 0.05
        rgb_img[mask] = 0
        
        img = Image.fromarray(rgb_img)
        
        # Do not flip. The array is already Top=HighFreq, Bottom=LowFreq (Row 0 = Max, Row H = Min)
        # And the input image was Top=Top.
        # So we want the spectrogram to match the input image orientation.
        
        buf = io.BytesIO()
        img.save(buf, format='PNG')
        buf.seek(0)
        return buf
