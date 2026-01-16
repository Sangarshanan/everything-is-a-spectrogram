from fastapi import FastAPI, Request
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from pydantic import BaseModel
from spectro_core import SpectroGraphic
import base64
import io
from PIL import Image

app = FastAPI()

app.mount("/static", StaticFiles(directory="static"), name="static")
templates = Jinja2Templates(directory="templates")

class ProcessRequest(BaseModel):
    image: str
    duration: int = 10
    min_freq: int = 500
    max_freq: int = 5000
    contrast: float = 5.0
    waveform: str = "sine"
    quantize: bool = False
    stereo_envelope: bool = False

@app.get("/")
async def index(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})

@app.post("/process")
async def process(data: ProcessRequest):
    image_data = data.image
    
    # Decode base64 image
    # Format: "data:image/png;base64,iVBORw0KGgo..."
    try:
        header, encoded = image_data.split(",", 1)
        image_bytes = base64.b64decode(encoded)
        image = Image.open(io.BytesIO(image_bytes))
    except Exception as e:
        return {"error": f"Invalid image data: {str(e)}"}
    
    # Generate Sound
    sg = SpectroGraphic(
        image_source=image,
        duration=data.duration,
        min_freq=data.min_freq,
        max_freq=data.max_freq,
        contrast=data.contrast,
        waveform=data.waveform,
        quantize=data.quantize,
        stereo_envelope=data.stereo_envelope
    )
    
    wav_io = sg.get_wav_bytes()
    wav_b64 = base64.b64encode(wav_io.read()).decode('utf-8')
    
    # Generate Spectrogram Plot
    plot_io = sg.get_spectrogram_plot()
    plot_b64 = base64.b64encode(plot_io.read()).decode('utf-8')
    
    return {
        "audio": wav_b64,
        "spectrogram": plot_b64
    }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
