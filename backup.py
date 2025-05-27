from fastapi import FastAPI, Request, Form, UploadFile, File
from fastapi.templating import Jinja2Templates
from fastapi.staticfiles import StaticFiles
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List
from pathlib import Path
import uuid
import subprocess
import os
import base64
import time
import logging
import shutil
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm
from ai.validate import test
from ai.data.datasets import AVLip
from ai import preprocess
from ai.trainer.trainer import Trainer
from ai.yolov5face.detect_face import detect, load_model
from preprocess import process_video
from crop import load_crop_and_save_image
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Configuration
BASE_DIR = Path(os.getenv("BASE_DIR", "/media/hoangtv/0f9d3910-0ff9-406c-92e1-c2c8170ca6f4/Dat/WEBDF/WebFAPI"))
STATIC_DIR = BASE_DIR / "static"
TEMP_DIR = STATIC_DIR / "temp"
VIDEO_DIR = TEMP_DIR / "video"
AUDIO_DIR = TEMP_DIR / "wav"
OUTPUT_DIR = STATIC_DIR / "preprocessing"
CUT_DIR = BASE_DIR / "android" / "cut"
CROP_DIR = BASE_DIR / "android" / "crop"
MODEL_WEIGHTS = os.getenv("MODEL_WEIGHTS", str(BASE_DIR / "ai/yolov5face/weights/yolov5n-0.5.pt"))
ALLOWED_ORIGINS = os.getenv("ALLOWED_ORIGINS", "").split(",")

# Ensure directories exist
for directory in [TEMP_DIR, VIDEO_DIR, AUDIO_DIR, OUTPUT_DIR, CUT_DIR, CROP_DIR]:
    directory.mkdir(parents=True, exist_ok=True)

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# FastAPI setup
app = FastAPI(docs_url=None, redoc_url=None)
app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")
app.mount("/android", StaticFiles(directory=BASE_DIR / "android"), name="android")
templates = Jinja2Templates(directory=BASE_DIR / "templates")

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=ALLOWED_ORIGINS or ["http://localhost:36000"],  # Restrict in production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Mock user database (replace with proper auth in production)
FAKE_USERS = {
    "admin": {"password": "1", "position": "admin"},
    "guest": {"password": "1", "position": "lecture"},
}

# Response model
class ResultResponse(BaseModel):
    result_text: str
    ai_processing_time: float
    video_url: str | None
    audio_url: str | None
    face_video_url: str | None

def clear_directory(directory: Path) -> None:
    """Remove all files and subdirectories in the specified directory, keeping the directory itself."""
    try:
        for item in directory.iterdir():
            try:
                if item.is_file():
                    item.unlink()
                elif item.is_dir():
                    shutil.rmtree(item)
            except Exception as e:
                logger.error(f"Error deleting {item}: {e}")
    except Exception as e:
        logger.error(f"Error clearing directory {directory}: {e}")

def convert_webm_to_mp4(input_path: Path, output_path: Path) -> bool:
    """Convert WebM video to MP4 using ffmpeg."""
    cmd = [
        "ffmpeg", "-y", "-i", str(input_path),
        "-c:v", "libx264", "-preset", "ultrafast", "-crf", "23",
        str(output_path)
    ]
    try:
        result = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
        if result.returncode != 0:
            logger.error(f"FFmpeg error: {result.stderr}")
            return False
        return True
    except subprocess.CalledProcessError as e:
        logger.error(f"Error converting video: {e.stderr}")
        return False

def get_video_duration(video_path: Path) -> float:
    """Get the duration of the video in seconds using ffprobe."""
    command = [
        "ffprobe",
        "-v", "error",
        "-show_entries", "format=duration",
        "-of", "default=noprint_wrappers=1:nokey=1",
        str(video_path)
    ]
    try:
        result = subprocess.run(command, capture_output=True, text=True, check=True)
        return float(result.stdout.strip())
    except subprocess.CalledProcessError as e:
        logger.error(f"Error getting video duration: {e.stderr}")
        return 0.0

def clip_video(video_path: Path, output_path: Path, duration_seconds: int = 10) -> Path | None:
    """Clip the middle `duration_seconds` of the video using FFmpeg."""
    duration = get_video_duration(video_path)
    if duration < duration_seconds:
        logger.warning(f"Video too short ({duration}s) to clip {duration_seconds}s.")

    start = max(0, (duration - duration_seconds) / 2)
    command = [
        "ffmpeg",
        "-y",
        "-ss", str(start),
        "-i", str(video_path),
        "-t", str(duration_seconds),
        "-c:v", "libx264",
        "-preset", "ultrafast",
        "-c:a", "aac",
        str(output_path)
    ]

    try:
        result = subprocess.run(command, capture_output=True, text=True, check=True)
        logger.info(f"FFmpeg clipped video saved to {output_path}")
        return output_path
    except subprocess.CalledProcessError as e:
        logger.error(f"FFmpeg error during clipping: {e.stderr}")
        return None

def extract_audio(input_path: Path, output_audio_path: Path) -> bool:
    """Extract audio using FFmpeg."""
    command = [
        "ffmpeg",
        "-y",
        "-i", str(input_path),
        "-q:a", "0",
        "-map", "a",
        str(output_audio_path)
    ]
    try:
        subprocess.run(command, check=True, capture_output=True, text=True)
        logger.info(f"Audio extracted to {output_audio_path}")
        return True
    except subprocess.CalledProcessError as e:
        logger.error(f"Error extracting audio: {e.stderr}")
        return False

def process_video_common(video_content: bytes, video_dir: Path, audio_dir: Path, output_dir: Path, cut_dir: Path) -> tuple[Path, str | None, Path]:
    """Common video processing logic for both endpoints."""
    # Save uploaded MP4 file
    mp4_filename = f"{uuid.uuid4().hex}.mp4"
    mp4_path = video_dir / mp4_filename
    with open(mp4_path, "wb") as f:
        f.write(video_content)

    # Clip video
    cut_path = cut_dir / mp4_filename
    clip_path = clip_video(mp4_path, cut_path)
    if not clip_path:
        raise ValueError("Failed to clip video")

    # Extract audio
    audio_filename = mp4_filename.replace(".mp4", ".wav")
    audio_path = audio_dir / audio_filename
    audio_url = f"/android/wav/{audio_filename}" if extract_audio(clip_path, audio_path) else None

    return mp4_path, audio_url, clip_path

@app.get("/")
async def root(request: Request):
    """Render the homepage."""
    return templates.TemplateResponse("index.html", {"request": request})

@app.get("/deepfake")
async def deepfake(request: Request):
    """Render the deepfake page."""
    return templates.TemplateResponse("deepfake.html", {"request": request})

@app.post("/login")
async def login(username: str = Form(...), password: str = Form(...)):
    """Handle user login."""
    user = FAKE_USERS.get(username)
    if user and user["password"] == password:
        return {"success": True, "position": user["position"]}
    return JSONResponse(
        {"success": False, "message": "Invalid username or password"},
        status_code=401
    )

@app.post("/process_ai", response_model=ResultResponse)
async def process_ai(video: UploadFile = File(...)):
    """Process uploaded WebM video for deepfake detection."""
    try:
        # Clear temporary directories
        for directory in [VIDEO_DIR, AUDIO_DIR, OUTPUT_DIR]:
            clear_directory(directory)

        # Save uploaded WebM file
        raw_filename = f"{uuid.uuid4().hex}.webm"
        raw_path = TEMP_DIR / raw_filename
        with open(raw_path, "wb") as f:
            f.write(await video.read())

        # Convert to MP4
        mp4_filename = raw_filename.replace(".webm", ".mp4")
        mp4_path = VIDEO_DIR / mp4_filename
        if not convert_webm_to_mp4(raw_path, mp4_path):
            return JSONResponse(
                {"result_text": "Error converting video format"},
                status_code=500
            )

        # Clip video
        cut_path = VIDEO_DIR / f"{Path(mp4_filename).stem}_cut.mp4"
        clip_path = clip_video(mp4_path, cut_path)
        if not clip_path:
            return JSONResponse(
                {"result_text": "Failed to clip video"},
                status_code=500
            )

        # Extract audio
        audio_filename = mp4_filename.replace(".mp4", ".wav")
        audio_path = AUDIO_DIR / audio_filename
        audio_url = f"/static/temp/wav/{audio_filename}" if extract_audio(clip_path, audio_path) else None

        # Clean up raw file
        raw_path.unlink(missing_ok=True)

        # AI Processing
        ai_start = time.perf_counter()

        # Run face detection
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        project = OUTPUT_DIR
        name = "faces"
        face_video_path = OUTPUT_DIR / name / f"{Path(mp4_filename).stem}_faces.mp4"

        clear_directory(OUTPUT_DIR / name if (OUTPUT_DIR / name).exists() else OUTPUT_DIR)
        model = load_model(MODEL_WEIGHTS, device)
        detect(
            model=model,
            source=str(clip_path),
            device=device,
            project=str(project),
            name=name,
            exist_ok=True,
            save_img=True,
            view_img=False
        )

        # if not face_video_path.exists():
        #     return JSONResponse(
        #         {"result_text": "No faces detected in the video"},
        #         status_code=400
        #     )

        # Run preprocessing
        preprocess.run(video_root=str(OUTPUT_DIR / name))

        # Load dataset and run model
        dataset = AVLip(OUTPUT_DIR)
        data_loader = DataLoader(dataset, batch_size=10, shuffle=False)
        model = Trainer()
        model.to("cuda:1")
        result = test(model.model, data_loader)
        result_text = "Video Real" if result else "Video Fake"
        ai_processing_time = round(time.perf_counter() - ai_start, 2)

        return ResultResponse(
            result_text=result_text,
            ai_processing_time=ai_processing_time,
            video_url=f"/static/temp/video/{mp4_filename}",
            audio_url=audio_url,
            face_video_url=f"/static/preprocessing/{name}/{Path(mp4_filename).stem}_faces.mp4"
        )

    except Exception as e:
        logger.error(f"Error processing video: {str(e)}", exc_info=True)
        return JSONResponse(
            {"result_text": f"Error processing video: {str(e)}"},
            status_code=500
        )

@app.post("/upload_video")
async def upload_video(video: UploadFile = File(...)):
    """Process uploaded MP4 video for deepfake detection."""
    try:
        video_content = await video.read()
        file_size = len(video_content)
        print(f"Received file with size: {file_size} bytes")

        # Define directory paths
        BASE_DIR = Path("/media/hoangtv/0f9d3910-0ff9-406c-92e1-c2c8170ca6f4/Dat/WEBDF/WebFAPI/android")
        VIDEO_DIR = BASE_DIR / "vid"
        AUDIO_DIR = BASE_DIR / "wav"
        OUTPUT_DIR = BASE_DIR / "preprocessing"
        CUT_DIR = BASE_DIR / "cut"
        CROP_DIR = BASE_DIR / "crop"
        # Ensure directories exist
        VIDEO_DIR.mkdir(parents=True, exist_ok=True)
        AUDIO_DIR.mkdir(parents=True, exist_ok=True)
        OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
        CUT_DIR.mkdir(parents=True, exist_ok=True)
        CROP_DIR.mkdir(parents=True, exist_ok=True)

        # Clear existing videos in VIDEO_DIR
        for file in VIDEO_DIR.glob("*.mp4"):
            file.unlink()
        for file in CUT_DIR.glob("*.mp4"):
            file.unlink()
        for file in AUDIO_DIR.glob("*.wav"):
            file.unlink()
        for file in OUTPUT_DIR.glob("*.png"):
            file.unlink()
        for file in CROP_DIR.glob("*.jpg"):
            file.unlink()
        # Save uploaded MP4 file
        mp4_filename = f"{uuid.uuid4().hex}.mp4"
        mp4_path = VIDEO_DIR / mp4_filename
        with open(mp4_path, "wb") as f:
            f.write(video_content)
        print(f"Saved uploaded MP4 to {mp4_path}")

        # Clip video
        # cut_path = VIDEO_DIR / mp4_filename  # same name, can overwrite
        # cut_clip = clip_video(mp4_path, cut_path)
        cut_path = CUT_DIR / mp4_filename  # same name, can overwrite
        clip_path = clip_video(mp4_path, cut_path)

        # Extract audio from the clip
        audio_filename = mp4_filename.replace(".mp4", ".wav")
        audio_path = AUDIO_DIR / audio_filename
        # audio_url = f"/android/wav/{audio_filename}" if extract_audio(cut_clip, audio_path) else None
        audio_url = f"/media/hoangtv/0f9d3910-0ff9-406c-92e1-c2c8170ca6f4/Dat/WEBDF/WebFAPI/android/wav/{audio_filename}" if extract_audio(clip_path, audio_path) else None


        # Process the video
        process_video(clip_path, audio_url, OUTPUT_DIR)

    except Exception as e:
        logger.error(f"Error processing video: {str(e)}", exc_info=True)
        return JSONResponse(
            {"result_text": f"Error processing video: {str(e)}"},
            status_code=500
        )

def get_frames_as_base64(folder_path: Path) -> List[str]:
    """Convert PNG frames in folder to base64 strings."""
    frames = []
    files = sorted(folder_path.glob("*.png"))
    for file_path in files:
        with open(file_path, "rb") as image_file:
            encoded_string = base64.b64encode(image_file.read()).decode("utf-8")
            frames.append(encoded_string)
    return frames

def get_crops_as_base64(folder_path: Path) -> List[str]:
    """Convert JPG crops in folder to base64 strings."""
    frames = []
    files = sorted(folder_path.glob("*.jpg"))
    for file_path in files:
        with open(file_path, "rb") as image_file:
            encoded_string = base64.b64encode(image_file.read()).decode("utf-8")
            frames.append(encoded_string)
    return frames

@app.get("/get_frames")
async def get_frames_endpoint():
    """Return base64-encoded PNG frames."""
    folder_path = BASE_DIR / "android" / "preprocessing"
    if not folder_path.exists():
        return JSONResponse(status_code=404, content={"message": "No output found"})
    frame_list = get_frames_as_base64(folder_path)
    return {"frames": frame_list}

@app.get("/get_crops")
async def get_crops_endpoint():
    """Return base64-encoded JPG crops."""
    folder_path = BASE_DIR / "android" / "crop"
    if not folder_path.exists():
        return JSONResponse(status_code=404, content={"message": "No output found"})
    frame_list = get_crops_as_base64(folder_path)
    return {"frames": frame_list}

def crop_all():
    """Crop all images in preprocessing directory and save to crop directory."""
    folder_path = BASE_DIR / "android" / "preprocessing"
    output_dir = BASE_DIR / "android" / "crop"
    for file in folder_path.glob("*.png"):
        load_crop_and_save_image(str(file), str(output_dir))

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("app:app", host="0.0.0.0", port=int(os.getenv("PORT", 36000)), reload=True)