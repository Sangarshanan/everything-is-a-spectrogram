from pathlib import Path
from scipy import signal
import numpy as np
from PIL import Image
import io
import scipy.io.wavfile
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

class ColumnToSound(object):
    def __init__(
        self,
        duration: int,
        sample_rate: int = 44100,
        min_freq: int = 10000,
        max_freq: int = 17000,
        y_resolution: int = 1000,
        num_tones: int = 3,
        contrast: float = 5,
        waveform: str = "sine",
        quantize: bool = False,
        stereo_envelope: bool = False,
    ):
        super(ColumnToSound, self).__init__()

        # saving imporant parameters
        self.Y_RESOLUTION = y_resolution
        self.CONTRAST = contrast

        # sample rate; 44100 is a good default
        self.SAMPLE_RATE = sample_rate

        # region in which to draw the pixel sound
        self.MIN_FREQ = min_freq
        self.MAX_FREQ = max_freq

        # Number of tones used to fill the pixel sound
        self.NUM_TONES = num_tones

        # frequency window for each pixel sound
        self.HEIGHT = (max_freq - min_freq) / (y_resolution)

        # frequency delta between each tone that fills the pixel sound
        self.tone_delta = self.HEIGHT / num_tones

        # duration in seconds
        self.DURATION = duration

        self.WAVEFORM = waveform
        self.QUANTIZE = quantize
        self.STEREO_ENVELOPE = stereo_envelope

        # C Major Scale Frequencies (approximate)
        self.SCALE = [
            261.63, 293.66, 329.63, 349.23, 392.00, 440.00, 493.88,  # C4-B4
            523.25, 587.33, 659.25, 698.46, 783.99, 880.00, 987.77,  # C5-B5
            1046.50, 1174.66, 1318.51, 1396.91, 1567.98, 1760.00, 1975.53, # C6-B6
            2093.00, 2349.32, 2637.02, 2793.83, 3135.96, 3520.00, 3951.07, # C7-B7
            4186.01 # C8
        ]

    def _quantize_freq(self, freq):
        if not self.QUANTIZE:
            return freq
        return min(self.SCALE, key=lambda x: abs(x - freq))

    def _apply_envelope(self, wave, duration):
        if not self.STEREO_ENVELOPE:
            return wave
        
        # Simple ADSR-like envelope
        num_samples = len(wave)
        attack_len = int(num_samples * 0.1)
        decay_len = int(num_samples * 0.1)
        sustain_len = int(num_samples * 0.6)
        release_len = num_samples - attack_len - decay_len - sustain_len
        
        envelope = np.concatenate([
            np.linspace(0, 1, attack_len),
            np.linspace(1, 0.8, decay_len),
            np.full(sustain_len, 0.8),
            np.linspace(0.8, 0, release_len)
        ])
        
        # Ensure envelope matches wave length exactly
        if len(envelope) != len(wave):
             envelope = np.resize(envelope, len(wave))

        return wave * envelope

    def _get_wave(self, freq: int, intensity: float = 1, duration: float = 1, pan: float = 0.5):
        # get timesteps
        t = np.linspace(
            start=0, stop=duration, num=int(duration * self.SAMPLE_RATE), endpoint=False
        )

        # generate corresponding sine wave.
        # this is the only place that CONTRAST acts in
        freq = self._quantize_freq(freq)

        if self.WAVEFORM == "sine":
            wave = np.cos(freq * t * 2 * np.pi)
        elif self.WAVEFORM == "square":
            wave = signal.square(freq * t * 2 * np.pi)
        elif self.WAVEFORM == "sawtooth":
            wave = signal.sawtooth(freq * t * 2 * np.pi)
        elif self.WAVEFORM == "triangle":
            wave = signal.sawtooth(freq * t * 2 * np.pi, width=0.5)
        else:
            raise ValueError(f"Unknown waveform: {self.WAVEFORM}")

        sound_wave = (intensity ** self.CONTRAST) * wave
        
        sound_wave = self._apply_envelope(sound_wave, duration)

        if self.STEREO_ENVELOPE:
            # Stereo panning
            left_gain = np.cos(pan * np.pi / 2)
            right_gain = np.sin(pan * np.pi / 2)
            return np.column_stack((sound_wave * left_gain, sound_wave * right_gain))

        return sound_wave

    def pixel_to_sound(self, y: int, intensity: float = 1):

        # pixel position (count) from the top must be
        # positive and not larger than the number of pixels
        # in the column.
        if y < 0 or y > self.Y_RESOLUTION:
            raise ValueError("y must be between 0 and 1.")

        # Loudness should be between 0 and 1
        if not (0 <= intensity <= 1):
            raise ValueError("Intensity must be between 0 and 1.")

        # Duration should be positive
        if self.DURATION < 0:
            raise ValueError("Duration must be positive.")

        # calculating base frequency for pixel sound
        base_freq = (self.MAX_FREQ - self.MIN_FREQ) / (self.Y_RESOLUTION) * (
            self.Y_RESOLUTION - y
        ) + self.MIN_FREQ
        # Pan based on frequency (Low=Left, High=Right)
        pan = (base_freq - self.MIN_FREQ) / (self.MAX_FREQ - self.MIN_FREQ)

        # get base wave
        wave = self._get_wave(base_freq, intensity, self.DURATION, pan)

        # add tones to fill up pixel sound
        # first tone:
        tone_freq = base_freq

        # iterating over tones, adding up the sounds.
        for _ in range(self.NUM_TONES):

            tone_freq += self.tone_delta
            wave += self._get_wave(tone_freq, intensity, self.DURATION, pan)

        return wave

    def gen_soundwall(self, column: np.ndarray):
        # empty wave that we will add individual
        # pixel sounds onto
        if self.STEREO_ENVELOPE:
             wave = np.zeros((int(self.DURATION * self.SAMPLE_RATE), 2))
        else:
             wave = np.zeros(int(self.DURATION * self.SAMPLE_RATE))

        # iterating over column, adding the pixel sounds
        # of all pixels together to get the final wave.
        for idx, val in enumerate(column):
            # val is intensity
            wave += self.pixel_to_sound(idx, intensity=val)

        return wave
    
# https://github.com/LeviBorodenko/spectrographic
class SpectroGraphic(object):
    def __init__(
        self,
        image_source, # Can be Path or PIL Image
        height: int = 100,
        duration: int = 20,
        min_freq: int = 1000,
        max_freq: int = 8000,
        sample_rate: int = 44100,
        num_tones: int = 3,
        contrast: float = 5,
        waveform: str = "sine",
        quantize: bool = False,
        stereo_envelope: bool = False,
    ):

        super(SpectroGraphic, self).__init__()
        
        if isinstance(image_source, (str, Path)):
            self.image = Image.open(image_source)
        else:
            self.image = image_source
            
        self.HEIGHT = height
        self.DURATION = duration
        self.SAMPLE_RATE = sample_rate
        self.STEREO_ENVELOPE = stereo_envelope

        # Width after setting height to self.HEIGHT
        self.WIDTH = int(self.image.width * (self.HEIGHT / self.image.height))

        # Duration per column
        self.DURATION_COL = self.DURATION / self.WIDTH

        # Sounds for each column of the image
        self.col_to_sound = ColumnToSound(
            duration=self.DURATION_COL,
            sample_rate=sample_rate,
            min_freq=min_freq,
            max_freq=max_freq,
            y_resolution=height,
            num_tones=num_tones,
            contrast=contrast,
            waveform=waveform,
            quantize=quantize,
            stereo_envelope=stereo_envelope,
        )

        # Flag whether we have processed the image yet
        self.is_processed = False

    def _resize(self):
        # resizing image
        self.image = self.image.resize(
            size=(self.WIDTH, self.HEIGHT), resample=Image.LANCZOS
        )

    def _preprocess(self):
        # resize image
        self._resize()

        # convert to gray scale
        self.image = self.image.convert(mode="L")

        # get pixels as array and normalise them to be
        # between 0 and 1
        self.image_array = np.array(self.image) / 255

        # transpose image to get list of columns
        self.columns = np.transpose(self.image_array)

    def _process(self):
        self._preprocess()

        if self.STEREO_ENVELOPE:
             audio_array = np.vstack(
                [self.col_to_sound.gen_soundwall(col) for col in self.columns]
            )
        else:
            audio_array = np.hstack(
                [self.col_to_sound.gen_soundwall(col) for col in self.columns]
            )

        # convert to 16-bit data
        audio_array *= 32767 / np.max(np.abs(audio_array))
        audio_array = audio_array.astype(np.int16)

        return audio_array

    @property
    def sound_array(self):
        if self.is_processed:
            return self._sound_array
        else:
            self.is_processed = True
            self._sound_array = self._process()
            return self._sound_array

    def get_wav_bytes(self):
        audio = self.sound_array
        byte_io = io.BytesIO()
        scipy.io.wavfile.write(byte_io, self.SAMPLE_RATE, audio)
        byte_io.seek(0)
        return byte_io

    def get_spectrogram_plot(self):
        audio = self.sound_array
        
        # If stereo, take mean for visualization
        if len(audio.shape) > 1:
            audio = np.mean(audio, axis=1)
            
        # Normalize audio
        audio_normal = audio / (np.max(np.abs(audio)) + 1e-10)

        frequencies, times, Sxx = signal.spectrogram(
            audio_normal,
            self.SAMPLE_RATE,
            nperseg=1024
        )

        # Create the plot
        plt.figure(figsize=(10, 4))
        ax = plt.gca()
        ax.set_facecolor('black')
        
        plt.pcolormesh(times, frequencies, 10 * np.log10(Sxx + 1e-10), 
                    shading='gouraud', cmap='viridis')
        
        plt.axis('off') # Hide axes for cleaner look
        plt.ylim([self.col_to_sound.MIN_FREQ, self.col_to_sound.MAX_FREQ])
        plt.tight_layout(pad=0)
        
        buf = io.BytesIO()
        plt.ylim([500, 5000])
        plt.savefig(buf, format='png')
        plt.close()
        buf.seek(0)
        return buf
