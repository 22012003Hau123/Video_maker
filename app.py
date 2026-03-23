"""
Video Subtitles Automation - FastAPI Application
"""
import os
import logging
import shutil
import uuid
import hashlib
import subprocess
import json
from pathlib import Path
from typing import Optional, List, Dict, Any
from datetime import datetime

from fastapi import FastAPI, UploadFile, File, Form, HTTPException
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from fastapi.responses import FileResponse, HTMLResponse, StreamingResponse, Response
from fastapi.requests import Request
from pydantic import BaseModel, Field
from dotenv import load_dotenv
from src.queue.tasks import enqueue_task
from src.queue.job_store import (
    create_job as create_job_record,
    get_job as get_job_record,
    update_job as update_job_record,
    set_job_meta as set_job_meta_record,
    get_job_meta as get_job_meta_record,
)

# Load environment variables
load_dotenv()

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Create directories
UPLOAD_DIR = Path(os.getenv("UPLOAD_DIR", "./uploads"))
OUTPUT_DIR = Path(os.getenv("OUTPUT_DIR", "./outputs"))
PREVIEW_CACHE_DIR = OUTPUT_DIR / ".preview_cache"
try:
    CLEANUP_MAX_AGE_HOURS = int(os.getenv("CLEANUP_MAX_AGE_HOURS", "24"))
except ValueError:
    CLEANUP_MAX_AGE_HOURS = 24
try:
    CLEANUP_INTERVAL_HOURS = int(os.getenv("CLEANUP_INTERVAL_HOURS", "24"))
except ValueError:
    CLEANUP_INTERVAL_HOURS = 24
UPLOAD_DIR.mkdir(parents=True, exist_ok=True)
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
PREVIEW_CACHE_DIR.mkdir(parents=True, exist_ok=True)

# Default Whisper Prompt for better brand recognition
# (Helps avoid mis-transcriptions like "Remember to attend" for "Belvedere 10")
WHISPER_PROMPT = "Belvedere 10, Belvedere Vodka, Moët & Chandon, Hennessy, luxury spirit brands."

# Initialize FastAPI
app = FastAPI(
    title="Video Subtitles Automation",
    description="Tự động hoá sản xuất video: phụ đề, nội dung pháp lý, master hoá",
    version="1.0.0"
)

# Static files and templates
static_dir = Path(__file__).parent / "static"
templates_dir = Path(__file__).parent / "templates"

if static_dir.exists():
    app.mount("/static", StaticFiles(directory=str(static_dir)), name="static")

# Mount outputs for direct serving (enables Range requests/Seeking)
app.mount("/api/outputs", StaticFiles(directory=str(OUTPUT_DIR)), name="outputs")

# Cleanup Task
@app.on_event("startup")
async def schedule_periodic_cleanup():
    """Schedule daily cleanup task"""
    import asyncio
    from src.utils.cleanup import cleanup_old_files
    
    async def cleanup_loop():
        while True:
            try:
                logger.info("Running scheduled cleanup...")
                cleanup_old_files(UPLOAD_DIR, max_age_hours=CLEANUP_MAX_AGE_HOURS)
                cleanup_old_files(OUTPUT_DIR, max_age_hours=CLEANUP_MAX_AGE_HOURS)
            except Exception as e:
                logger.error(f"Cleanup failed: {e}")
            
            interval_seconds = max(CLEANUP_INTERVAL_HOURS, 1) * 3600
            await asyncio.sleep(interval_seconds)
    
    asyncio.create_task(cleanup_loop())

templates = Jinja2Templates(directory=str(templates_dir)) if templates_dir.exists() else None


# ============ Models ============

class ResultFile(BaseModel):
    path: str
    language: str
    type: str  # "video" or "srt"
    label: str  # e.g. "English - Video", "Vietnamese - SRT"


class JobStatus(BaseModel):
    job_id: str
    status: str  # pending, processing, completed, failed
    progress: int = 0
    result_path: Optional[str] = None
    result_files: List[ResultFile] = Field(default_factory=list)
    error: Optional[str] = None
    message: Optional[str] = None
    created_at: str
    completed_at: Optional[str] = None


class SubtitleRequest(BaseModel):
    language: str = "vi"
    video_format: str = "16x9"
    use_ai_timing: bool = True


class LegalRequest(BaseModel):
    country_code: str
    media_type: str = "social"
    usage_type: str = "shareable"
    auto_position: bool = True
    sub_type: Optional[str] = "reels"  # stories or reels


# ============ Job Storage (Redis) ============

def _normalize_job_updates(kwargs: Dict[str, Any]) -> Dict[str, Any]:
    normalized = {}
    for key, value in kwargs.items():
        if key == "result_files" and isinstance(value, list):
            normalized[key] = [
                item.model_dump() if isinstance(item, ResultFile) else item
                for item in value
            ]
        else:
            normalized[key] = value
    return normalized


def create_job(job_id: str, status: str = "pending", **kwargs) -> None:
    payload = JobStatus(
        job_id=job_id,
        status=status,
        created_at=datetime.now().isoformat(),
        **kwargs,
    ).model_dump()
    try:
        create_job_record(job_id, payload)
    except Exception as e:
        logger.error(f"Redis unavailable while creating job {job_id}: {e}")
        raise HTTPException(
            status_code=503,
            detail="Queue/Redis is unavailable. Please start Redis or set REDIS_URL correctly.",
        )


def get_job(job_id: str) -> Optional[JobStatus]:
    try:
        raw = get_job_record(job_id)
    except Exception as e:
        logger.error(f"Redis unavailable while reading job {job_id}: {e}")
        return None
    if not raw:
        return None
    try:
        return JobStatus(**raw)
    except Exception:
        logger.warning(f"Malformed job payload in Redis for job_id={job_id}")
        return None


def update_job(job_id: str, **kwargs):
    try:
        update_job_record(job_id, _normalize_job_updates(kwargs))
    except Exception as e:
        logger.error(f"Redis unavailable while updating job {job_id}: {e}")


def set_job_meta(job_id: str, key: str, value: Any) -> None:
    try:
        set_job_meta_record(job_id, key, value)
    except Exception as e:
        logger.error(f"Redis unavailable while setting job meta ({job_id}/{key}): {e}")


def get_job_meta(job_id: str, key: str, default: Any = None) -> Any:
    try:
        return get_job_meta_record(job_id, key, default)
    except Exception as e:
        logger.error(f"Redis unavailable while getting job meta ({job_id}/{key}): {e}")
        return default


# ============ Helper Functions ============

def save_upload(file: UploadFile, prefix: str = "") -> Path:
    """Save uploaded file and return path"""
    ext = Path(file.filename).suffix if file.filename else ""
    
    # Fallback: detect extension from content_type if filename has no extension
    if not ext and file.content_type:
        content_type_map = {
            "video/mp4": ".mp4",
            "video/quicktime": ".mov",
            "video/x-msvideo": ".avi",
            "video/webm": ".webm",
            "video/x-matroska": ".mkv",
            "image/jpeg": ".jpg",
            "image/png": ".png",
            "image/webp": ".webp",
            "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet": ".xlsx",
            "application/vnd.ms-excel": ".xls",
        }
        ext = content_type_map.get(file.content_type, "")
    
    filename = f"{prefix}{uuid.uuid4().hex}{ext}"
    filepath = UPLOAD_DIR / filename
    
    with open(filepath, "wb") as f:
        shutil.copyfileobj(file.file, f)
        
    file_size = os.path.getsize(filepath)
    logger.info(f"Saved upload: {filepath} (Size: {file_size} bytes, Content-Type: {file.content_type}, Original: {file.filename})")
    
    if file_size == 0:
        if filepath.exists():
            filepath.unlink()  # Delete empty file
        raise HTTPException(status_code=400, detail="Uploaded file is empty. Please check your video file and try again.")
        
    return filepath


# ============ Background Tasks ============

LANG_NAMES = {
    "en": "US English",
    "uk": "UK English",
    "de": "German",
    "it": "Italian",
    "fr": "French",
    "es": "Spanish",
    "nl": "Dutch",
    "ja": "Japanese"
}


def translate_subtitles_gpt(aligned, source_lang, target_lang):
    """Dịch phụ đề sang ngôn ngữ khác bằng GPT"""
    from openai import OpenAI
    client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
    
    texts = [text for _, _, text in aligned]
    joined = "\n---\n".join(texts)
    
    src_name = LANG_NAMES.get(source_lang, source_lang)
    tgt_name = LANG_NAMES.get(target_lang, target_lang)
    
    import re
    
    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {"role": "system", "content": f"Translate these subtitles from {src_name} to {tgt_name}. Maintain the exact number of lines. Use '---' as a separator between each line. Do not add numbering or extra text."},
            {"role": "user", "content": joined}
        ],
        temperature=0.3
    )
    
    content = response.choices[0].message.content.strip()
    
    # Robust splitting using regex to handle variations like " \n---\n ", " --- ", etc.
    translated_texts = re.split(r'\s*---\s*', content)
    
    # Remove empty strings from split result (e.g. at start/end) if needed, but keep empty translations
    # However, re.split might produce empty if '---' is at start/end.
    # We trust GPT generally but let's be safe.
    if translated_texts and not translated_texts[0]:
        translated_texts.pop(0)
    if translated_texts and not translated_texts[-1]:
        translated_texts.pop(-1)
    
    # Pad or trim to match original count
    if len(translated_texts) < len(aligned):
        logger.warning(f"Translation count mismatch: got {len(translated_texts)}, expected {len(aligned)}")
        while len(translated_texts) < len(aligned):
            translated_texts.append("")
    translated_texts = translated_texts[:len(aligned)]
    
    return [
        (start, end, text.strip())
        for (start, end, _), text in zip(aligned, translated_texts)
    ]


def generate_srt(aligned, output_path):
    """Tạo file SRT từ aligned subtitles"""
    def fmt(seconds):
        h = int(seconds // 3600)
        m = int((seconds % 3600) // 60)
        s = int(seconds % 60)
        ms = int((seconds % 1) * 1000)
        return f"{h:02d}:{m:02d}:{s:02d},{ms:03d}"
    
    with open(output_path, "w", encoding="utf-8") as f:
        for i, (start, end, text) in enumerate(aligned, 1):
            f.write(f"{i}\n{fmt(start)} --> {fmt(end)}\n{text}\n\n")
    return output_path


def enqueue_app_job(task_name: str, job_id: str, *args, **kwargs) -> None:
    try:
        enqueue_task(task_name, *args, job_id=job_id, **kwargs)
    except Exception as e:
        logger.error(f"Failed to enqueue task '{task_name}' for job {job_id}: {e}")
        update_job(job_id, status="failed", error=f"Queue unavailable: {e}")
        raise HTTPException(status_code=503, detail="Queue service unavailable")


def serialize_scene(scene: Any) -> Dict[str, Any]:
    return {
        "start_time": float(getattr(scene, "start_time", 0.0)),
        "end_time": float(getattr(scene, "end_time", 0.0)),
        "start_frame": int(getattr(scene, "start_frame", 0)),
        "end_frame": int(getattr(scene, "end_frame", 0)),
        "scene_type": str(getattr(scene, "scene_type", "unknown")),
        "keyframe_path": str(getattr(scene, "keyframe_path", "")),
    }


def deserialize_scenes(scenes_data: List[Dict[str, Any]]):
    from src.mastering.scene_detection import Scene

    scenes = []
    for item in scenes_data:
        scenes.append(
            Scene(
                start_time=float(item.get("start_time", 0.0)),
                end_time=float(item.get("end_time", 0.0)),
                start_frame=int(item.get("start_frame", 0)),
                end_frame=int(item.get("end_frame", 0)),
                scene_type=str(item.get("scene_type", "unknown")),
                keyframe_path=str(item.get("keyframe_path", "")),
            )
        )
    return scenes


def _create_ae_package_dirs(job_id: str) -> Dict[str, Path]:
    root = OUTPUT_DIR / "ae_packages" / f"ae_package_{job_id}"
    paths = {
        "root": root,
        "video": root / "video",
        "audio": root / "audio",
        "subtitles": root / "subtitles",
        "timeline": root / "timeline",
    }
    for path in paths.values():
        path.mkdir(parents=True, exist_ok=True)
    return paths


def _probe_video_metadata(video_path: str) -> Dict[str, Any]:
    cmd = [
        "ffprobe",
        "-v", "quiet",
        "-print_format", "json",
        "-show_streams",
        "-show_format",
        video_path,
    ]
    try:
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            check=True,
            stdin=subprocess.DEVNULL,
        )
        info = json.loads(result.stdout or "{}")
    except Exception:
        return {}

    video_stream = None
    for stream in info.get("streams", []):
        if stream.get("codec_type") == "video":
            video_stream = stream
            break

    fps = None
    if video_stream:
        frame_rate = video_stream.get("r_frame_rate") or video_stream.get("avg_frame_rate")
        if isinstance(frame_rate, str) and "/" in frame_rate:
            n, d = frame_rate.split("/", 1)
            try:
                n_val = float(n)
                d_val = float(d)
                if d_val != 0:
                    fps = round(n_val / d_val, 3)
            except Exception:
                fps = None

    duration = None
    try:
        duration = float((info.get("format") or {}).get("duration"))
    except Exception:
        duration = None

    return {
        "fps": fps,
        "duration": duration,
        "width": (video_stream or {}).get("width"),
        "height": (video_stream or {}).get("height"),
    }


def _export_wav_for_ae(video_path: str, output_path: Path) -> str:
    cmd = [
        "ffmpeg", "-y", "-nostdin",
        "-i", video_path,
        "-vn",
        "-acodec", "pcm_s16le",
        "-ar", "48000",
        "-ac", "2",
        str(output_path),
    ]
    result = subprocess.run(
        cmd,
        capture_output=True,
        text=True,
        stdin=subprocess.DEVNULL,
    )
    if result.returncode != 0:
        raise RuntimeError(f"WAV export failed: {result.stderr}")
    return str(output_path)


def _write_timeline_json_for_ae(
    output_path: Path,
    aligned: List[tuple],
    language: str,
    source_language: str,
    video_path: str,
    metadata: Dict[str, Any],
) -> str:
    segments = []
    for idx, (start, end, text) in enumerate(aligned, start=1):
        start_val = float(start)
        end_val = float(end)
        segments.append({
            "index": idx,
            "start": round(start_val, 3),
            "end": round(end_val, 3),
            "duration": round(max(0.0, end_val - start_val), 3),
            "text": text,
        })

    payload = {
        "schema": "ae_timeline_v1",
        "language": language,
        "source_language": source_language,
        "video": {
            "path": str(video_path),
            "fps": metadata.get("fps"),
            "duration": metadata.get("duration"),
            "width": metadata.get("width"),
            "height": metadata.get("height"),
        },
        "segments": segments,
    }
    output_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    return str(output_path)


def _zip_ae_package(root_dir: Path) -> str:
    zip_path = shutil.make_archive(str(root_dir), "zip", root_dir=str(root_dir.parent), base_dir=root_dir.name)
    return str(zip_path)


async def process_subtitles(
    job_id: str,
    video_path: str,
    source_lang: str,
    target_langs: List[str],
    video_format: str,
    export_video: bool = True,
    export_srt: bool = True,
    export_format: str = "mp4_standard"
):
    """Background task để xử lý phụ đề đa ngôn ngữ"""
    try:
        update_job(job_id, status="processing", progress=5, message="Initializing...")
        
        from src.subtitle.renderer import SubtitleRenderer
        from src.subtitle.timing_sync import TimingSync, extract_audio
        
        sync = TimingSync()
        result_files = []
        ae_package_mode = export_format == "ae_package"
        render_export_format = "prores" if ae_package_mode else export_format
        if ae_package_mode:
            export_video = True
            export_srt = True
        ae_paths = _create_ae_package_dirs(job_id) if ae_package_mode else {}
        video_meta = _probe_video_metadata(video_path) if ae_package_mode else {}
        
        # 1. Trích xuất audio
        update_job(job_id, progress=10, message="Extracting audio from video...")
        audio_path = extract_audio(video_path)
        
        # 2. Whisper transcribe
        update_job(job_id, progress=20, message="AI is transcribing speech (Whisper)...")
        transcript, source_lang = sync.transcribe(audio_path, source_lang, prompt=WHISPER_PROMPT)
        
        if not transcript:
            raise ValueError("Không nhận diện được lời thoại trong video")
        
        # 3. Tạo aligned subtitles từ transcript
        aligned_source = [
            (seg.start, seg.end, sync.smart_line_break(seg.text))
            for seg in transcript
        ]

        if ae_package_mode:
            update_job(job_id, progress=25, message="Preparing AE package audio (WAV)...")
            wav_output = ae_paths["audio"] / f"{Path(video_path).stem}_master.wav"
            wav_path = _export_wav_for_ae(video_path, wav_output)
            result_files.append(ResultFile(
                path=wav_path,
                language="multi",
                type="audio",
                label="AE Package - Master WAV"
            ))
        
        update_job(job_id, progress=30)
        
        # 4. Xử lý từng ngôn ngữ
        total_langs = len(target_langs)
        renderer = SubtitleRenderer(str(OUTPUT_DIR))
        
        for i, lang in enumerate(target_langs):
            lang_name = LANG_NAMES.get(lang, lang)
            progress = 30 + int((i / total_langs) * 60)
            update_job(job_id, progress=progress, message=f"Processing language {lang_name} ({i+1}/{total_langs})...")
            
            # Dịch nếu khác ngôn ngữ gốc
            if lang == source_lang:
                aligned = aligned_source
            else:
                try:
                    aligned = translate_subtitles_gpt(aligned_source, source_lang, lang)
                except Exception as e:
                    logger.warning(f"Translation to {lang} failed: {e}")
                    continue
            
            # Export video với phụ đề
            if export_video:
                try:
                    # Chọn đuôi file phù hợp (ProRes cần .mov)
                    ext = ".mov" if render_export_format == "prores" else ".mp4"
                    output_filename = f"{Path(video_path).stem}_{lang}{ext}"
                    output_path = (
                        str(ae_paths["video"] / output_filename)
                        if ae_package_mode else
                        str(OUTPUT_DIR / output_filename)
                    )
                    
                    output_path = renderer.render(
                        video_path,
                        aligned,
                        output_path,
                        language=lang,
                        video_format=video_format,
                        export_format=render_export_format
                    )
                    result_files.append(ResultFile(
                        path=output_path,
                        language=lang,
                        type="video",
                        label=f"{lang_name} - {'ProRes Video' if ae_package_mode else 'Video'}"
                    ))
                except Exception as e:
                    logger.warning(f"Render {lang} video failed: {e}")
            
            # Export SRT
            if export_srt:
                srt_path = (
                    str(ae_paths["subtitles"] / f"{Path(video_path).stem}_{lang}.srt")
                    if ae_package_mode else
                    str(OUTPUT_DIR / f"subtitle_{job_id}_{lang}.srt")
                )
                generate_srt(aligned, srt_path)
                result_files.append(ResultFile(
                    path=srt_path,
                    language=lang,
                    type="srt",
                    label=f"{lang_name} - SRT"
                ))

            if ae_package_mode:
                timeline_path = _write_timeline_json_for_ae(
                    ae_paths["timeline"] / f"{Path(video_path).stem}_{lang}.json",
                    aligned,
                    language=lang,
                    source_language=source_lang,
                    video_path=video_path,
                    metadata=video_meta,
                )
                result_files.append(ResultFile(
                    path=timeline_path,
                    language=lang,
                    type="timeline",
                    label=f"{lang_name} - Timeline JSON"
                ))
        
        if not result_files:
            raise ValueError("Không tạo được output nào")

        primary_result_path = result_files[0].path
        if ae_package_mode:
            package_zip = _zip_ae_package(ae_paths["root"])
            result_files.insert(0, ResultFile(
                path=package_zip,
                language="multi",
                type="zip",
                label="AE Package - ZIP"
            ))
            primary_result_path = package_zip
        
        update_job(
            job_id,
            status="completed",
            progress=100,
            result_path=primary_result_path,
            result_files=result_files,
            completed_at=datetime.now().isoformat()
        )
        
    except Exception as e:
        logger.error(f"Job {job_id} failed: {e}")
        update_job(job_id, status="failed", error=str(e))


async def process_legal(
    job_id: str,
    video_path: str,
    country_code: str,
    media_type: str,
    usage_type: str,
    auto_position: bool,
    position: Optional[str] = None,
    auto_detect_product: bool = True,
    sub_type: Optional[str] = "reels"
):
    """Background task để thêm nội dung pháp lý"""
    try:
        update_job(job_id, status="processing", progress=5, message="Initializing legal processing...")
        
        from src.legal.overlay import LegalOverlay
        from src.legal.database import MediaType, UsageType
        from src.legal.product_detector import ProductDetector
        
        product_type = None
        
        # 1. Detect product type from video (if enabled)
        if auto_detect_product:
            try:
                detector = ProductDetector()
                product_type, confidence, label = detector.detect_from_video(video_path)
                logger.info(f"Video detected as: {product_type} ({label}) - {confidence:.2%}")
            except Exception as e:
                logger.error(f"Product detection failed: {e}")
            
        update_job(job_id, progress=40, message=f"Applying legal rules: {country_code} ({media_type})...")
        
        overlay = LegalOverlay(str(OUTPUT_DIR))
        
        update_job(job_id, progress=60, message="Running FFmpeg to overlay legal text and logos...")
        
        output_path = overlay.add_legal_content(
            video_path,
            country_code,
            MediaType(media_type),
            UsageType(usage_type),
            auto_detect_position=auto_position,
            product_type=product_type,
            manual_position=position,
            sub_type=sub_type
        )
        
        update_job(
            job_id,
            status="completed",
            progress=100,
            result_path=output_path,
            completed_at=datetime.now().isoformat()
        )
        
    except Exception as e:
        logger.error(f"Job {job_id} failed: {e}")
        update_job(job_id, status="failed", error=str(e))


def process_replace_logo_job(
    job_id: str,
    video_path: str,
    logo_path: str,
    x: Optional[int] = None,
    y: Optional[int] = None,
):
    try:
        update_job(job_id, status="processing", progress=30)

        from src.mastering.element_replacer import ElementReplacer

        replacer = ElementReplacer(str(OUTPUT_DIR))
        position = (x, y) if x is not None and y is not None else None
        output_path = replacer.replace_logo(
            str(video_path),
            str(logo_path),
            position=position,
        )
        update_job(
            job_id,
            status="completed",
            progress=100,
            result_path=output_path,
            completed_at=datetime.now().isoformat(),
        )
    except Exception as e:
        update_job(job_id, status="failed", error=str(e))


def process_add_packshot_job(
    job_id: str,
    video_path: str,
    packshot_path: str,
    position: str = "center",
    duration: float = 2.0,
):
    try:
        update_job(job_id, status="processing", progress=30)

        from src.mastering.element_replacer import ElementReplacer

        replacer = ElementReplacer(str(OUTPUT_DIR))
        output_path = replacer.add_packshot(
            str(video_path),
            str(packshot_path),
            position=position,
            duration=duration,
        )
        update_job(
            job_id,
            status="completed",
            progress=100,
            result_path=output_path,
            completed_at=datetime.now().isoformat(),
        )
    except Exception as e:
        update_job(job_id, status="failed", error=str(e))


def process_analyze_scenes_job(job_id: str, video_path: str):
    try:
        update_job(job_id, status="processing", progress=10, message="Starting scene analysis...")

        from src.mastering.scene_detection import SceneDetector

        detector = SceneDetector(output_dir=str(OUTPUT_DIR / "scenes" / job_id))
        compatible_path = detector.ensure_compatible_video(str(video_path))
        update_job(job_id, progress=20)

        scenes = detector.detect_scenes(compatible_path)
        update_job(job_id, progress=50)

        detector.classify_scenes(scenes, video_path=compatible_path)
        update_job(job_id, progress=90)

        scene_data = []
        serialized_scenes = []
        for i, s in enumerate(scenes):
            serialized = serialize_scene(s)
            serialized_scenes.append(serialized)

            kf_url = ""
            if serialized["keyframe_path"] and os.path.exists(serialized["keyframe_path"]):
                kf_url = f"/api/master/keyframe/{job_id}/{i}"
            scene_data.append(
                {
                    "index": i,
                    "start_time": round(serialized["start_time"], 2),
                    "end_time": round(serialized["end_time"], 2),
                    "duration": round(serialized["end_time"] - serialized["start_time"], 2),
                    "scene_type": serialized["scene_type"],
                    "keyframe_url": kf_url,
                }
            )

        set_job_meta(job_id, "scenes", serialized_scenes)
        set_job_meta(job_id, "video_path", compatible_path)
        set_job_meta(job_id, "scene_data", scene_data)

        update_job(
            job_id,
            status="completed",
            progress=100,
            result_path=str(OUTPUT_DIR / "scenes" / job_id),
            completed_at=datetime.now().isoformat(),
        )
    except Exception as e:
        logger.error(f"Scene analysis failed: {e}")
        import traceback
        traceback.print_exc()
        update_job(job_id, status="failed", error=str(e))


def process_smart_cut_job(
    job_id: str,
    scenes_data: List[Dict[str, Any]],
    video_path: str,
    remove_intro: bool = True,
    remove_outro: bool = True,
    remove_logo: bool = False,
    remove_product: bool = False,
):
    try:
        update_job(job_id, status="processing", progress=20, message="Analyzing cut segments...")

        from src.mastering.smart_cut import SmartCutter

        scenes = deserialize_scenes(scenes_data)
        cutter = SmartCutter(output_dir=str(OUTPUT_DIR / "cuts"))
        keep_segments = cutter.generate_cut_list(
            scenes,
            remove_intro=remove_intro,
            remove_outro=remove_outro,
            remove_product=remove_product,
            remove_logo=remove_logo,
        )
        update_job(job_id, progress=40, message="Rendering trimmed video...")

        output_filename = f"smartcut_{job_id}.mp4"
        output_path = cutter.render_video(video_path, keep_segments, output_filename)

        update_job(
            job_id,
            status="completed",
            progress=100,
            result_path=output_path,
            completed_at=datetime.now().isoformat(),
        )
    except Exception as e:
        logger.error(f"Smart cut failed: {e}")
        import traceback
        traceback.print_exc()
        update_job(job_id, status="failed", error=str(e))


def process_logo_auto_remove_job(
    job_id: str,
    scenes_data: List[Dict[str, Any]],
    video_path: str,
    mask_path: Optional[str] = None,
):
    try:
        update_job(job_id, status="processing", progress=10, message="Initializing logo removal...")

        from src.mastering.logo_removal import LogoRemovalService
        from src.mastering.inpainting import InpaintingService

        logo_service = LogoRemovalService(output_dir=str(OUTPUT_DIR / "logo_removal"))
        inpaint_service = InpaintingService(output_dir=str(OUTPUT_DIR / "inpaint"))
        scenes = deserialize_scenes(scenes_data)

        update_job(job_id, progress=20, message="Detecting logo with AI...")
        result = logo_service.auto_remove_logo(
            video_path=video_path,
            scenes=scenes,
            output_name=f"logo_removed_{job_id}.mp4",
            mask_path=mask_path,
            inpainting_service=inpaint_service,
        )

        if result.get("status") == "success":
            update_job(
                job_id,
                status="completed",
                progress=100,
                message="Completed!",
                result_path=result["output_path"],
                completed_at=datetime.now().isoformat(),
            )
        else:
            update_job(
                job_id,
                status="completed",
                progress=100,
                result_path=video_path,
                completed_at=datetime.now().isoformat(),
            )
    except Exception as e:
        logger.error(f"Logo removal failed: {e}")
        import traceback
        traceback.print_exc()
        update_job(job_id, status="failed", error=str(e))


def process_inpaint_video_job(job_id: str, video_path: str, mask_path: str):
    try:
        from src.mastering.inpainting import InpaintingService
        from src.mastering.scene_detection import SceneDetector

        detector = SceneDetector()
        compatible_path = detector.ensure_compatible_video(video_path)
        service = InpaintingService(output_dir=str(OUTPUT_DIR / "inpaint"))
        output_name = f"inpainted_{job_id}.mp4"

        def update_progress(p):
            update_job(job_id, progress=int(p), message=f"Processing: {int(p)}%")

        service.inpaint_video(
            str(compatible_path),
            str(mask_path),
            output_name=output_name,
            progress_callback=update_progress,
        )
        update_job(
            job_id,
            status="completed",
            progress=100,
            message="Completed!",
            result_path=str(OUTPUT_DIR / "inpaint" / output_name),
            completed_at=datetime.now().isoformat(),
        )
    except Exception as e:
        logger.error(f"Video inpainting failed: {e}")
        import traceback
        traceback.print_exc()
        update_job(job_id, status="failed", error=str(e))


def process_subtitle_batch_job(
    job_id: str,
    video_path: str,
    lines: List[str],
    source_lang: str,
    targets: List[str],
    video_format: str,
):
    try:
        update_job(job_id, status="processing", progress=5)

        from src.subtitle.translator import SubtitleTranslator
        from src.subtitle.timing_sync import TimingSync, extract_audio
        from src.subtitle.renderer import SubtitleRenderer
        import zipfile

        sync = TimingSync()
        translator = SubtitleTranslator()
        renderer = SubtitleRenderer(str(OUTPUT_DIR))

        if len(lines) == 1 and len(lines[0]) > 100:
            update_job(job_id, progress=10)
            subtitle_lines = sync.segment_long_text(lines[0], source_lang, 80, True)
        else:
            subtitle_lines = lines

        update_job(job_id, progress=20)
        try:
            audio_path = extract_audio(str(video_path))
            update_job(job_id, progress=30)
            transcript, source_lang = sync.transcribe(audio_path, source_lang)
            aligned = sync.align_subtitles(subtitle_lines, transcript)
        except Exception as e:
            logger.warning(f"Timing failed: {e}, using fallback")
            aligned = [(i * 3, (i + 1) * 3, t) for i, t in enumerate(subtitle_lines)]

        update_job(job_id, progress=40)
        output_files = []
        total_langs = len(targets)

        for idx, target_lang in enumerate(targets):
            lang_progress = 40 + int(50 * (idx / total_langs))
            update_job(job_id, progress=lang_progress)

            if target_lang != source_lang:
                translated_aligned = translator.translate_with_timing(
                    aligned, source_lang, target_lang
                )
            else:
                translated_aligned = aligned

            translated_aligned = [
                (start, end, sync.smart_line_break(text))
                for start, end, text in translated_aligned
            ]

            output_path = renderer.render(
                str(video_path),
                translated_aligned,
                language=target_lang,
                video_format=video_format,
                export_format="mp4_standard",
            )

            new_path = output_path.replace("_subtitled.mp4", f"_{target_lang}.mp4")
            if output_path != new_path:
                shutil.move(output_path, new_path)
                output_path = new_path

            output_files.append((target_lang, output_path))
            logger.info(f"Generated {target_lang} version: {output_path}")

        update_job(job_id, progress=90, message="Finalizing and packaging output...")
        if len(output_files) > 1:
            zip_path = str(OUTPUT_DIR / f"batch_{job_id}.zip")
            with zipfile.ZipFile(zip_path, "w") as zf:
                for _, path in output_files:
                    zf.write(path, Path(path).name)
            result_path = zip_path
        else:
            result_path = output_files[0][1]

        update_job(
            job_id,
            status="completed",
            progress=100,
            result_path=result_path,
            completed_at=datetime.now().isoformat(),
        )
    except Exception as e:
        logger.error(f"Batch job {job_id} failed: {e}")
        update_job(job_id, status="failed", error=str(e))


def process_subtitle_export_job(
    job_id: str,
    video_path: str,
    lines: List[str],
    language: str,
    export_format: str,
    translate_langs: List[str],
):
    try:
        update_job(job_id, status="processing", progress=5)

        from src.subtitle.timing_sync import TimingSync, extract_audio
        from src.subtitle.exporter import SubtitleExporter
        from src.subtitle.translator import SubtitleTranslator
        import zipfile

        sync = TimingSync()
        exporter = SubtitleExporter(str(OUTPUT_DIR))

        if len(lines) == 1 and len(lines[0]) > 100:
            subtitle_lines = sync.segment_long_text(lines[0], language, 80, True)
        else:
            subtitle_lines = lines

        update_job(job_id, progress=20)
        try:
            audio_path = extract_audio(str(video_path))
            update_job(job_id, progress=40, message="Aligning subtitle timing...")
            transcript, language = sync.transcribe(audio_path, language)
            aligned = sync.align_subtitles(subtitle_lines, transcript)
        except Exception as e:
            logger.warning(f"Timing failed: {e}, using fallback")
            aligned = [(i * 3, (i + 1) * 3, t) for i, t in enumerate(subtitle_lines)]

        aligned = [(start, end, sync.smart_line_break(text)) for start, end, text in aligned]
        update_job(job_id, progress=60)

        output_files = []
        base_name = Path(video_path).stem
        if export_format in ["srt", "both"]:
            output_files.append(exporter.export_srt(aligned, f"{base_name}_{language}"))
        if export_format in ["ass", "both"]:
            output_files.append(exporter.export_ass(aligned, f"{base_name}_{language}"))

        if translate_langs:
            translator = SubtitleTranslator()
            for target_lang in translate_langs:
                if target_lang != language:
                    translated_aligned = translator.translate_with_timing(
                        aligned, language, target_lang
                    )
                    if export_format in ["srt", "both"]:
                        output_files.append(
                            exporter.export_srt(translated_aligned, f"{base_name}_{target_lang}")
                        )
                    if export_format in ["ass", "both"]:
                        output_files.append(
                            exporter.export_ass(translated_aligned, f"{base_name}_{target_lang}")
                        )

        update_job(job_id, progress=90)
        if len(output_files) > 1:
            zip_path = str(OUTPUT_DIR / f"subtitles_{job_id}.zip")
            with zipfile.ZipFile(zip_path, "w") as zf:
                for path in output_files:
                    zf.write(path, Path(path).name)
            result_path = zip_path
        else:
            result_path = output_files[0]

        update_job(
            job_id,
            status="completed",
            progress=100,
            result_path=result_path,
            completed_at=datetime.now().isoformat(),
        )
    except Exception as e:
        logger.error(f"Export job {job_id} failed: {e}")
        update_job(job_id, status="failed", error=str(e))


# ============ Routes ============

@app.get("/", response_class=HTMLResponse)
async def home(request: Request):
    """Trang chủ"""
    if templates:
        return templates.TemplateResponse("index.html", {"request": request})
    return HTMLResponse("""
    <html>
        <head><title>Video Subtitles Automation</title></head>
        <body>
            <h1>Video Subtitles Automation</h1>
            <p>API documentation: <a href="/docs">/docs</a></p>
        </body>
    </html>
    """)


@app.get("/favicon.ico", include_in_schema=False)
async def favicon():
    """Suppress favicon 404 when no icon is configured."""
    return Response(status_code=204)


@app.get("/health")
async def health_check():
    """Health check endpoint"""
    return {"status": "healthy", "timestamp": datetime.now().isoformat()}


@app.post("/api/subtitle")
async def add_subtitle(
    video: UploadFile = File(...),
    source_lang: str = Form("en"),
    target_languages: str = Form("en"),
    video_format: str = Form("16x9"),
    export_video: bool = Form(True),
    export_srt: bool = Form(False),
    export_format: str = Form("mp4_standard")
):
    """
    Thêm phụ đề đa ngôn ngữ vào video bằng AI
    """
    logger.info(f"Received add_subtitle request: source={source_lang}, targets={target_languages}, export_video={export_video}, export_srt={export_srt}")
    
    video_path = save_upload(video, "video_")
    
    # Parse target languages
    langs = [l.strip() for l in target_languages.split(",") if l.strip()]
    if not langs:
        langs = [source_lang]
    
    logger.info(f"Parsed languages: {langs}")
    
    job_id = uuid.uuid4().hex
    create_job(job_id, status="pending")
    enqueue_app_job(
        "process_subtitles",
        job_id,
        job_id,
        str(video_path),
        source_lang,
        langs,
        video_format,
        export_video,
        export_srt,
        export_format,
    )
    
    lang_names = [LANG_NAMES.get(l, l) for l in langs]
    return {"job_id": job_id, "message": f"AI đang tạo phụ đề: {', '.join(lang_names)}"}


async def process_subtitles_from_text(
    job_id: str,
    video_path: str,
    subtitle_lines: Optional[List[str]],
    source_lang: str,
    target_langs: List[str],
    video_format: str,
    export_video: bool = True,
    export_srt: bool = True,
    export_format: str = "mp4_standard",
    raw_text: Optional[str] = None,
    auto_segment_rhythm: bool = True,
    manual_translations: Optional[Dict[str, List[str]]] = None
):
    """Background task để xử lý phụ đề từ text trực tiếp (đa ngôn ngữ)"""
    try:
        logger.info(f"Starting process_subtitles_from_text for job {job_id}")
        logger.info(f"Source: {source_lang}, Targets: {target_langs}")
        
        update_job(job_id, status="processing", progress=5)
        
        from src.subtitle.renderer import SubtitleRenderer
        from src.subtitle.timing_sync import TimingSync, extract_audio
        
        sync = TimingSync()
        result_files = []
        ae_package_mode = export_format == "ae_package"
        render_export_format = "prores" if ae_package_mode else export_format
        if ae_package_mode:
            export_video = True
            export_srt = True
        ae_paths = _create_ae_package_dirs(job_id) if ae_package_mode else {}
        video_meta = _probe_video_metadata(video_path) if ae_package_mode else {}
        
        # Nếu có raw_text (1 đoạn dài) và không yêu cầu ngắt theo nhịp điệu (hoặc fallback)
        if raw_text and not subtitle_lines and not auto_segment_rhythm:
            update_job(job_id, progress=10, message="AI is segmenting static text...")
            logger.info(f"Segmenting long text ({len(raw_text)} chars) with AI (static)...")
            
            subtitle_lines = sync.segment_long_text(
                raw_text,
                language=source_lang,
                max_chars=80,
                use_ai=True
            )
            logger.info(f"AI created {len(subtitle_lines)} subtitle segments (static)")
        
        if not subtitle_lines:
            raise ValueError("Không có phụ đề nào được tạo")
        
        update_job(job_id, progress=20)
        
        try:
            # Trích xuất audio từ video
            audio_path = extract_audio(video_path)
            update_job(job_id, progress=35, message="AI is analyzing audio...")
            
            # Dùng Whisper transcribe để lấy timing
            transcript, source_lang = sync.transcribe(audio_path, source_lang, prompt=WHISPER_PROMPT)
            update_job(job_id, progress=55, message="AI is aligning text with speech rhythm...")
            
            if auto_segment_rhythm and raw_text:
                # Rhythm-aware alignment: chia nhỏ raw_text theo transcript
                logger.info("Using rhythm-aware alignment for long text")
                aligned_source = sync.align_long_text_to_transcript(raw_text, transcript, source_lang)
            else:
                # Căn timing phụ đề (đã được chia sẵn) theo transcript
                if not subtitle_lines:
                    subtitle_lines = [raw_text] if raw_text else [""]
                aligned_source = sync.align_subtitles(subtitle_lines, transcript)
            
            # Áp dụng ngắt dòng thông minh cho mỗi segment, giữ lại original_index
            aligned_source = [
                (start, end, sync.smart_line_break(text), original_idx)
                for start, end, text, original_idx in aligned_source
            ]
            
        except Exception as e:
            logger.warning(f"AI timing failed: {e}, using fallback")
            # Fallback: chia đều 3 giây mỗi phụ đề, cũng cần original_index
            aligned_source = [
                (i * 3, (i + 1) * 3, text, i)
                for i, text in enumerate(subtitle_lines)
            ]

        if ae_package_mode:
            update_job(job_id, progress=62, message="Preparing AE package audio (WAV)...")
            wav_output = ae_paths["audio"] / f"{Path(video_path).stem}_master.wav"
            wav_path = _export_wav_for_ae(video_path, wav_output)
            result_files.append(ResultFile(
                path=wav_path,
                language="multi",
                type="audio",
                label="AE Package - Master WAV"
            ))
        
        update_job(job_id, progress=70, message="Starting output export...")
        
        # 4. Xử lý từng ngôn ngữ (tương tự như process_subtitles)
        total_langs = len(target_langs)
        renderer = SubtitleRenderer(str(OUTPUT_DIR))
        
        for i, lang in enumerate(target_langs):
            lang_name = LANG_NAMES.get(lang, lang)
            progress = 70 + int((i / total_langs) * 20)
            update_job(job_id, progress=progress, message=f"Rendering output for {lang_name}...")
            
            # Dịch hoặc dùng bản dịch thủ công
            if lang == source_lang:
                # Strip index for renderer compatibility
                aligned = [(s, e, t) for s, e, t, idx in aligned_source]
            elif manual_translations and lang in manual_translations:
                logger.info(f"Using manual translation for {lang} with index mapping...")
                m_texts = manual_translations[lang]
                
                # Use the original_idx to pick the correct translation from m_texts
                aligned = []
                for start, end, _, original_idx in aligned_source:
                    trans_text = ""
                    if 0 <= original_idx < len(m_texts):
                        trans_text = m_texts[original_idx].strip()
                    aligned.append((start, end, trans_text))
            else:
                try:
                    logger.info(f"Translating subtitles from {source_lang} to {lang}...")
                    # translate_subtitles_gpt expects (start, end, text) so strip index
                    source_for_trans = [(s, e, t) for s, e, t, idx in aligned_source]
                    aligned = translate_subtitles_gpt(source_for_trans, source_lang, lang)
                    logger.info(f"Translation to {lang} completed.")
                except Exception as e:
                    logger.warning(f"Translation to {lang} failed: {e}")
                    continue
            
            # Export video với phụ đề
            if export_video:
                try:
                    logger.info(f"Rendering video for {lang}...")
                    
                    # Chọn đuôi file phù hợp (ProRes cần .mov)
                    ext = ".mov" if render_export_format == "prores" else ".mp4"
                    output_filename = f"{Path(video_path).stem}_{lang}{ext}"
                    output_path = (
                        str(ae_paths["video"] / output_filename)
                        if ae_package_mode else
                        str(OUTPUT_DIR / output_filename)
                    )
                    
                    output_path = renderer.render(
                        video_path,
                        aligned,
                        output_path,
                        language=lang,
                        video_format=video_format,
                        export_format=render_export_format
                    )
                    result_files.append(ResultFile(
                        path=output_path,
                        language=lang,
                        type="video",
                        label=f"{lang_name} - {'ProRes Video' if ae_package_mode else 'Video'}"
                    ))
                    logger.info(f"Rendered video for {lang} saved to {output_path}")
                except Exception as e:
                    logger.warning(f"Render {lang} video failed: {e}")
            
            # Export SRT
            if export_srt:
                try:
                    srt_path = (
                        str(ae_paths["subtitles"] / f"{Path(video_path).stem}_{lang}.srt")
                        if ae_package_mode else
                        str(OUTPUT_DIR / f"subtitle_{job_id}_{lang}.srt")
                    )
                    generate_srt(aligned, srt_path)
                    result_files.append(ResultFile(
                        path=srt_path,
                        language=lang,
                        type="srt",
                        label=f"{lang_name} - SRT"
                    ))
                    logger.info(f"Generated SRT for {lang} saved to {srt_path}")
                except Exception as e:
                    logger.warning(f"Generate {lang} SRT failed: {e}")

            if ae_package_mode:
                timeline_path = _write_timeline_json_for_ae(
                    ae_paths["timeline"] / f"{Path(video_path).stem}_{lang}.json",
                    aligned,
                    language=lang,
                    source_language=source_lang,
                    video_path=video_path,
                    metadata=video_meta,
                )
                result_files.append(ResultFile(
                    path=timeline_path,
                    language=lang,
                    type="timeline",
                    label=f"{lang_name} - Timeline JSON"
                ))
        
        if not result_files:
            raise ValueError("Không tạo được output nào")

        primary_result_path = result_files[0].path
        if ae_package_mode:
            package_zip = _zip_ae_package(ae_paths["root"])
            result_files.insert(0, ResultFile(
                path=package_zip,
                language="multi",
                type="zip",
                label="AE Package - ZIP"
            ))
            primary_result_path = package_zip
        
        update_job(
            job_id,
            status="completed",
            progress=100,
            result_path=primary_result_path,
            result_files=result_files,
            completed_at=datetime.now().isoformat()
        )
        
    except Exception as e:
        logger.error(f"Job {job_id} failed: {e}")
        update_job(job_id, status="failed", error=str(e))


@app.post("/api/parse-translation-excel")
async def parse_translation_excel(file: UploadFile = File(...)):
    """
    Parse an Excel file containing horizontal translations (EN, FR, SP, etc.)
    Returns JSON rows for the frontend table.
    """
    try:
        from src.subtitle.excel_reader import ExcelReader
        
        # Save temp file
        ext = Path(file.filename).suffix
        temp_path = UPLOAD_DIR / f"temp_trans_{uuid.uuid4().hex}{ext}"
        with open(temp_path, "wb") as f:
            shutil.copyfileobj(file.file, f)
            
        reader = ExcelReader(str(temp_path))
        rows = reader.parse_horizontal_translations()
        
        # Cleanup temp file
        if temp_path.exists():
            temp_path.unlink()
            
        return {"rows": rows}
    except Exception as e:
        logger.error(f"Failed to parse translation excel: {e}")
        raise HTTPException(status_code=400, detail=f"Failed to parse Excel: {str(e)}")


@app.post("/api/subtitle-text")
async def add_subtitle_from_text(
    video: UploadFile = File(...),
    subtitle_text: str = Form(""), # Make optional if JSON is provided
    source_lang: str = Form("en"),
    video_format: str = Form("16x9"),
    target_languages: str = Form("en"),
    export_video: bool = Form(True),
    export_srt: bool = Form(True),
    export_format: str = Form("mp4_standard"),
    auto_segment: bool = Form(True),
    auto_segment_rhythm: bool = Form(True),
    manual_translations_json: Optional[str] = Form(None)
):
    """
    Thêm phụ đề vào video từ text nhập trực tiếp (đa ngôn ngữ)
    """
    logger.info(f"Received add_subtitle_from_text request: source={source_lang}, targets={target_languages}")

    # Force rhythm-aware timing alignment for manual text flow.
    auto_segment_rhythm = True

    manual_trans = None
    if manual_translations_json:
        try:
            import json
            manual_trans = json.loads(manual_translations_json)
            # Nếu có manual_trans, source_text thường là cột đầu tiên (ví dụ 'en')
            if not subtitle_text and 'en' in manual_trans:
                subtitle_text = "\n".join(manual_trans['en'])
        except Exception as e:
            logger.error(f"Failed to parse manual_translations_json: {e}")

    text = subtitle_text.strip()
    
    if not text and not manual_trans:
        raise HTTPException(status_code=400, detail="Không có phụ đề nào được nhập hoặc upload")
    
    # Parse target languages
    langs = [l.strip() for l in target_languages.split(",") if l.strip()]
    if not langs:
        langs = [source_lang]
    
    # Kiểm tra nếu là nhiều dòng hay 1 đoạn dài
    lines = [l.strip() for l in text.split('\n') if l.strip()]
    
    # Nếu chỉ có 1 dòng và dài hơn 100 ký tự -> dùng AI ngắt câu
    if auto_segment and len(lines) == 1 and len(lines[0]) > 100:
        # Sẽ được xử lý trong background task
        subtitle_lines = None
        raw_text = lines[0]
    else:
        subtitle_lines = lines
        raw_text = None
    
    # Save video
    video_path = save_upload(video, "video_")
    
    # Create job
    job_id = uuid.uuid4().hex
    create_job(job_id, status="pending")
    enqueue_app_job(
        "process_subtitles_from_text",
        job_id,
        job_id,
        str(video_path),
        subtitle_lines,
        source_lang,
        langs,
        video_format,
        export_video,
        export_srt,
        export_format,
        raw_text,
        auto_segment_rhythm,
        manual_trans,
    )
    
    lang_names = [LANG_NAMES.get(l, l) for l in langs]
    if raw_text:
        return {"job_id": job_id, "message": f"AI đang ngắt câu, canh timing và dịch sang: {', '.join(lang_names)}"}
    else:
        return {"job_id": job_id, "message": f"Processing {len(subtitle_lines)} dòng phụ đề và dịch sang: {', '.join(lang_names)}"}


@app.post("/api/legal")
async def add_legal_content(
    video: UploadFile = File(...),
    country_code: str = Form(...),
    media_type: str = Form("social"),
    usage_type: str = Form("shareable"),
    auto_position: bool = Form(True),
    position: Optional[str] = Form(None),
    auto_detect_product: bool = Form(False),
    sub_type: Optional[str] = Form("reels")
):
    """
    Thêm nội dung pháp lý vào video
    
    - **video**: File video
    - **country_code**: Mã quốc gia (VN, FR, US, HK)
    - **media_type**: Loại media (social, tv, ooh, digital, print)
    - **usage_type**: Hình thức sử dụng (shareable, non_shareable, paid, organic)
    - **auto_position**: Tự động phát hiện vị trí tối ưu (deprecated in logic if position is set, but kept for API compat)
    - **position**: Vị trí thủ công (bottom_left, bottom_right, etc.) overrides auto_position
    - **auto_detect_product**: Tự động phát hiện loại sản phẩm bằng CLIP
    """
    video_path = save_upload(video, "video_")
    
    job_id = uuid.uuid4().hex
    create_job(job_id, status="pending")
    enqueue_app_job(
        "process_legal",
        job_id,
        job_id,
        str(video_path),
        country_code.upper(),
        media_type,
        usage_type,
        auto_position,
        position,
        auto_detect_product,
        sub_type,
    )
    
    return {"job_id": job_id, "message": "Processing started"}


@app.post("/api/detect-product")
async def detect_product(
    file: UploadFile = File(...)
):
    """
    Tự động nhận diện loại sản phẩm trong ảnh/video bằng CLIP
    
    - **file**: File ảnh (jpg, png) hoặc video (mp4, mov)
    
    Returns:
        - product_type: alcohol, tobacco, pharmaceutical, food, unknown
        - confidence: độ tin cậy (0-1)
        - detected_label: nhãn cụ thể được phát hiện
    """
    from src.legal.product_detector import get_product_detector
    
    # Save uploaded file
    file_path = save_upload(file, "detect_")
    
    try:
        detector = get_product_detector()
        
        # Check if video or image
        if file.content_type.startswith('video/'):
            category, confidence, label = detector.detect_from_video(str(file_path))
        else:
            category, confidence, label = detector.detect_product(str(file_path))
        
        return {
            "product_type": category,
            "confidence": round(confidence, 3),
            "detected_label": label,
            "legal_rules_available": category != "unknown"
        }
        
    except Exception as e:
        logger.error(f"Product detection failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/api/master/replace-logo")
async def replace_logo(
    video: UploadFile = File(...),
    logo: UploadFile = File(...),
    x: Optional[int] = Form(None),
    y: Optional[int] = Form(None)
):
    """Thay thế logo trong video"""
    video_path = save_upload(video, "video_")
    logo_path = save_upload(logo, "logo_")
    
    job_id = uuid.uuid4().hex
    create_job(job_id, status="pending")
    enqueue_app_job(
        "process_replace_logo_job",
        job_id,
        job_id,
        str(video_path),
        str(logo_path),
        x,
        y,
    )
    return {"job_id": job_id, "message": "Processing started"}


@app.post("/api/master/add-packshot")
async def add_packshot(
    video: UploadFile = File(...),
    packshot: UploadFile = File(...),
    position: str = Form("center"),
    duration: float = Form(2.0)
):
    """Thêm packshot vào cuối video"""
    video_path = save_upload(video, "video_")
    packshot_path = save_upload(packshot, "packshot_")
    
    job_id = uuid.uuid4().hex
    create_job(job_id, status="pending")
    enqueue_app_job(
        "process_add_packshot_job",
        job_id,
        job_id,
        str(video_path),
        str(packshot_path),
        position,
        duration,
    )
    return {"job_id": job_id, "message": "Processing started"}


# ============ Mastering AI (Scene Analysis, Smart Cut, Logo Removal, Inpainting) ============

@app.post("/api/master/analyze-scenes")
async def analyze_scenes(
    video: UploadFile = File(...)
):
    """Phân tích scenes trong video bằng CLIP AI"""
    video_path = save_upload(video, "video_")
    
    job_id = uuid.uuid4().hex
    create_job(job_id, status="pending")
    enqueue_app_job("process_analyze_scenes_job", job_id, job_id, str(video_path))
    return {"job_id": job_id, "message": "Scene analysis started"}


@app.get("/api/master/scene-results/{job_id}")
async def get_scene_results(job_id: str):
    """Lấy kết quả phân tích scene"""
    job = get_job(job_id)
    if not job:
        raise HTTPException(status_code=404, detail="Job not found")
    
    if job.status != "completed":
        return {"status": job.status, "progress": job.progress, "error": job.error}
    
    scene_data = get_job_meta(job_id, "scene_data", [])
    return {
        "status": "completed",
        "scenes": scene_data,
        "total_scenes": len(scene_data),
    }


@app.get("/api/master/keyframe/{job_id}/{scene_index}")
async def get_keyframe(job_id: str, scene_index: int):
    """Serve keyframe image cho scene cụ thể"""
    job = get_job(job_id)
    if not job:
        raise HTTPException(status_code=404, detail="Job not found")
    
    scenes = get_job_meta(job_id, "scenes", [])
    if scene_index < 0 or scene_index >= len(scenes):
        raise HTTPException(status_code=404, detail="Scene not found")

    keyframe_path = scenes[scene_index].get("keyframe_path", "")
    if not keyframe_path or not os.path.exists(keyframe_path):
        raise HTTPException(status_code=404, detail="Keyframe not found")
    
    return FileResponse(keyframe_path, media_type="image/jpeg")


@app.post("/api/master/smart-cut")
async def smart_cut(
    scene_job_id: str = Form(...),
    remove_intro: bool = Form(True),
    remove_outro: bool = Form(True),
    remove_logo: bool = Form(False),
    remove_product: bool = Form(False),
):
    """Smart cut dựa trên kết quả scene analysis"""
    # Get scene analysis results
    scene_job = get_job(scene_job_id)
    if not scene_job or scene_job.status != "completed":
        raise HTTPException(status_code=400, detail="Scene analysis job not completed")
    
    scenes = get_job_meta(scene_job_id, "scenes", None)
    video_path = get_job_meta(scene_job_id, "video_path", None)
    if not scenes or not video_path:
        raise HTTPException(status_code=400, detail="Scene data not available")
    
    job_id = uuid.uuid4().hex
    create_job(job_id, status="pending")
    enqueue_app_job(
        "process_smart_cut_job",
        job_id,
        job_id,
        scenes,
        video_path,
        remove_intro,
        remove_outro,
        remove_logo,
        remove_product,
    )
    return {"job_id": job_id, "message": "Smart cut started"}


@app.post("/api/master/logo-detect")
async def logo_detect(
    video: UploadFile = File(...)
):
    """Phát hiện logo trong video bằng Grounding DINO AI"""
    video_path = save_upload(video, "video_")
    
    try:
        from src.mastering.logo_removal import LogoRemovalService
        service = LogoRemovalService(output_dir=str(OUTPUT_DIR / "logo_removal"))
        
        result = service.detect_logo_region(str(video_path))
        
        return result
    except Exception as e:
        logger.error(f"Logo detection failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/api/master/logo-remove")
async def logo_auto_remove(
    scene_job_id: str = Form(...),
    mask: Optional[UploadFile] = File(None),
):
    """Tự động xóa logo bằng AI (Grounding DINO + LaMa inpainting)"""
    scene_job = get_job(scene_job_id)
    if not scene_job or scene_job.status != "completed":
        raise HTTPException(status_code=400, detail="Scene analysis job not completed")
    
    scenes = get_job_meta(scene_job_id, "scenes", None)
    video_path = get_job_meta(scene_job_id, "video_path", None)
    if not scenes or not video_path:
        raise HTTPException(status_code=400, detail="Scene data not available")
    
    mask_path = None
    if mask:
        mask_path = str(save_upload(mask, "mask_"))
    
    job_id = uuid.uuid4().hex
    create_job(job_id, status="pending")
    enqueue_app_job(
        "process_logo_auto_remove_job",
        job_id,
        job_id,
        scenes,
        video_path,
        mask_path,
    )
    return {"job_id": job_id, "message": "Logo removal started"}


@app.post("/api/master/inpaint-image")
async def inpaint_image(
    image: UploadFile = File(...),
    mask: UploadFile = File(...),
):
    """Xóa đối tượng khỏi ảnh bằng LaMa AI inpainting"""
    image_path = save_upload(image, "img_")
    mask_path = save_upload(mask, "mask_")
    
    try:
        from src.mastering.inpainting import InpaintingService
        service = InpaintingService(output_dir=str(OUTPUT_DIR / "inpaint"))
        
        output_name = f"inpainted_{uuid.uuid4().hex}.png"
        result_path = service.inpaint(
            str(image_path),
            str(mask_path),
            output_name=output_name,
        )
        
        return {
            "result_url": f"/api/master/inpaint-result/{os.path.basename(result_path)}",
            "message": "Inpainting completed"
        }
    except Exception as e:
        logger.error(f"Image inpainting failed: {e}")
        import traceback
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/api/master/inpaint-video-preview")
async def inpaint_video_preview(video: UploadFile = File(...)):
    """Upload video và trích xuất frame đầu tiên để vẽ mask"""
    video_path = save_upload(video, "vid_")
    
    try:
        from src.mastering.scene_detection import SceneDetector
        detector = SceneDetector()
        # Đảm bảo video tương thích (chuyển đổi AV1 nếu cần)
        compatible_path = detector.ensure_compatible_video(str(video_path))
        
        import cv2
        cap = cv2.VideoCapture(str(compatible_path))
        ret, frame = cap.read()
        fps = cap.get(cv2.CAP_PROP_FPS)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        duration = total_frames / fps if fps > 0 else 0
        cap.release()

        if not ret:
            raise ValueError("Could not read video frame")

        # Save preview frame
        file_id = uuid.uuid4().hex
        preview_name = f"preview_{file_id}.png"
        preview_path = OUTPUT_DIR / "inpaint" / preview_name
        os.makedirs(preview_path.parent, exist_ok=True)
        cv2.imwrite(str(preview_path), frame)

        return {
            "video_path": str(compatible_path), # Trả về path đã convert
            "preview_url": f"/api/master/inpaint-result/{preview_name}",
            "info": {
                "fps": round(fps, 1),
                "duration": round(duration, 1)
            }
        }
    except Exception as e:
        logger.error(f"Video preview extraction failed: {e}")
        import traceback
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/api/master/inpaint-video")
async def inpaint_video(
    video_path: str = Form(...),
    mask: UploadFile = File(...),
):
    """Xóa đối tượng khỏi toàn bộ video"""
    mask_path = save_upload(mask, "mask_")
    job_id = uuid.uuid4().hex
    
    # Save job info
    create_job(
        job_id,
        status="processing",
        progress=0,
        message="Starting object removal from video...",
    )
    enqueue_app_job("process_inpaint_video_job", job_id, job_id, video_path, str(mask_path))
    return {"job_id": job_id}


@app.post("/api/video-thumbnail")
async def get_video_thumbnail(video: UploadFile = File(...)):
    """Trích xuất thumbnail từ video (dùng cho preview MOV/không hỗ trợ)"""
    file_path = save_upload(video, "thumb_")
    try:
        from src.mastering.scene_detection import SceneDetector
        detector = SceneDetector()
        # Đảm bảo video tương thích (ví dụ: chuyển MOV sang định dạng opencv đọc được tốt hơn nếu cần)
        compatible_path = detector.ensure_compatible_video(str(file_path))
        
        import cv2
        cap = cv2.VideoCapture(str(compatible_path))
        ret, frame = cap.read()
        cap.release()

        if not ret:
            raise ValueError("Could not read video frame")

        # Encode frame to memory
        _, buffer = cv2.imencode(".png", frame)
        from fastapi.responses import Response
        
        # Get dimensions
        height, width = frame.shape[:2]
        
        return Response(
            content=buffer.tobytes(), 
            media_type="image/png",
            headers={
                "X-Video-Width": str(width),
                "X-Video-Height": str(height)
            }
        )
    except Exception as e:
        logger.error(f"Thumbnail extraction failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))
    finally:
        # Cleanup temporary upload
        if file_path.exists():
            os.remove(file_path)


@app.get("/api/master/inpaint-result/{filename}")
async def get_inpaint_result(filename: str):
    """Serve kết quả inpainting"""
    result_path = OUTPUT_DIR / "inpaint" / filename
    if not result_path.exists():
        raise HTTPException(status_code=404, detail="Result not found")
    
    ext = result_path.suffix.lower()
    media_type = {"png": "image/png", ".jpg": "image/jpeg", ".jpeg": "image/jpeg"}.get(ext, "image/png")
    return FileResponse(str(result_path), media_type=media_type)


@app.post("/api/subtitle-batch")
async def add_subtitle_batch(
    video: UploadFile = File(...),
    subtitle_text: str = Form(...),
    source_lang: str = Form("en"),
    target_languages: str = Form("vi"),  # Comma-separated: "vi,ja,ko"
    video_format: str = Form("16x9")
):
    """
    Tạo nhiều video với phụ đề dịch sang nhiều ngôn ngữ
    
    - **video**: File video gốc
    - **subtitle_text**: Phụ đề gốc (text hoặc từng dòng)
    - **source_lang**: Ngôn ngữ gốc của phụ đề (en, vi...)
    - **target_languages**: Các ngôn ngữ cần dịch, phân cách bằng dấu phẩy (vi,ja,ko)
    - **video_format**: Định dạng video (16x9, 9x16, 1x1, 4x5)
    
    Returns:
        Job ID để theo dõi tiến trình. Kết quả là ZIP chứa nhiều video.
    """
    # Parse target languages
    targets = [lang.strip() for lang in target_languages.split(",") if lang.strip()]
    if not targets:
        targets = ["vi"]
    
    # Parse subtitle text
    text = subtitle_text.strip()
    if not text:
        raise HTTPException(status_code=400, detail="Không có phụ đề nào được nhập")
    
    lines = [l.strip() for l in text.split('\n') if l.strip()]
    
    # Save video
    video_path = save_upload(video, "video_")
    
    # Create job
    job_id = uuid.uuid4().hex
    create_job(job_id, status="pending")
    enqueue_app_job(
        "process_subtitle_batch_job",
        job_id,
        job_id,
        str(video_path),
        lines,
        source_lang,
        targets,
        video_format,
    )
    
    return {
        "job_id": job_id, 
        "message": f"Creating {len(targets)} language versions: {', '.join(targets)}"
    }


@app.post("/api/subtitle-export")
async def export_subtitle_files(
    video: UploadFile = File(...),
    subtitle_text: str = Form(...),
    language: str = Form("vi"),
    export_format: str = Form("both"),  # srt, ass, both
    translate_to: str = Form("")  # Optional: comma-separated languages to translate
):
    """
    Xuất file phụ đề SRT/ASS để chỉnh sửa (không render video)
    
    - **video**: File video để lấy timing
    - **subtitle_text**: Text phụ đề (đoạn dài hoặc từng dòng)
    - **language**: Ngôn ngữ của phụ đề
    - **export_format**: Định dạng xuất (srt, ass, both)
    - **translate_to**: Ngôn ngữ cần dịch, phân cách bằng dấu phẩy (optional)
    
    Returns:
        File SRT/ASS hoặc ZIP chứa nhiều file nếu dịch nhiều ngôn ngữ
    """
    text = subtitle_text.strip()
    if not text:
        raise HTTPException(status_code=400, detail="Không có phụ đề nào được nhập")
    
    lines = [l.strip() for l in text.split('\n') if l.strip()]
    
    # Save video
    video_path = save_upload(video, "video_")
    
    # Create job
    job_id = uuid.uuid4().hex
    create_job(job_id, status="pending")
    translate_langs = [l.strip() for l in translate_to.split(",") if l.strip()]
    enqueue_app_job(
        "process_subtitle_export_job",
        job_id,
        job_id,
        str(video_path),
        lines,
        language,
        export_format,
        translate_langs,
    )
    
    return {
        "job_id": job_id,
        "message": f"Exporting {export_format.upper()} files" + (f" + translations: {translate_to}" if translate_langs else "")
    }


@app.get("/api/jobs/{job_id}")
async def get_job_status(job_id: str):
    """Lấy trạng thái của job"""
    job = get_job(job_id)
    if not job:
        raise HTTPException(status_code=404, detail="Job not found")
    return job


def _resolve_output_video_path(video_path: str) -> Path:
    """Resolve output video path and prevent path traversal."""
    output_root = OUTPUT_DIR.resolve()
    full_path = (OUTPUT_DIR / video_path).resolve()
    if output_root not in full_path.parents and full_path != output_root:
        raise HTTPException(status_code=403, detail="Invalid video path")
    return full_path


def _stream_video_file(full_path: Path, request: Request) -> StreamingResponse:
    if not full_path.exists() or not full_path.is_file():
        raise HTTPException(status_code=404, detail="Video not found")

    file_size = full_path.stat().st_size
    range_header = request.headers.get("range")

    ext = full_path.suffix.lower()
    media_types = {
        ".mp4": "video/mp4",
        ".mov": "video/quicktime",
        ".qt": "video/quicktime",
        ".webm": "video/webm",
        ".mkv": "video/x-matroska",
    }
    content_type = media_types.get(ext, "video/mp4")

    headers = {
        "Accept-Ranges": "bytes",
        "Content-Type": content_type,
        "Cache-Control": "no-cache",
    }

    if range_header:
        # Range format: "bytes=start-end"
        try:
            start, end = range_header.replace("bytes=", "").split("-")
            start = int(start)
            end = int(end) if end else file_size - 1
        except ValueError:
            raise HTTPException(status_code=400, detail="Invalid range format")

        if start >= file_size:
            raise HTTPException(status_code=416, detail="Requested range not satisfiable")

        chunk_size = (end - start) + 1

        def iter_file():
            with open(full_path, "rb") as f:
                f.seek(start)
                remaining = chunk_size
                while remaining > 0:
                    chunk = f.read(min(1024 * 1024, remaining))  # 1MB chunks
                    if not chunk:
                        break
                    yield chunk
                    remaining -= len(chunk)

        headers.update({
            "Content-Range": f"bytes {start}-{end}/{file_size}",
            "Content-Length": str(chunk_size),
        })

        return StreamingResponse(iter_file(), status_code=206, headers=headers)

    # Normal full-file response
    headers["Content-Length"] = str(file_size)

    def iter_full_file():
        with open(full_path, "rb") as f:
            while chunk := f.read(1024 * 1024):
                yield chunk

    return StreamingResponse(iter_full_file(), headers=headers)


def _get_preview_cache_path(source_path: Path) -> Path:
    """Create deterministic cache filename from source file path + metadata."""
    stat = source_path.stat()
    fingerprint = f"{source_path}:{int(stat.st_mtime)}:{stat.st_size}"
    cache_name = hashlib.sha1(fingerprint.encode("utf-8")).hexdigest() + ".mp4"
    return PREVIEW_CACHE_DIR / cache_name


def _ensure_web_preview(source_path: Path) -> Path:
    """Transcode source video to web-compatible MP4 if cache does not exist."""
    preview_path = _get_preview_cache_path(source_path)
    if preview_path.exists() and preview_path.stat().st_size > 0:
        return preview_path

    PREVIEW_CACHE_DIR.mkdir(parents=True, exist_ok=True)
    cmd = [
        "ffmpeg", "-y", "-nostdin",
        "-i", str(source_path),
        "-map", "0:v:0",
        "-map", "0:a:0?",
        "-c:v", "libx264",
        "-preset", "veryfast",
        "-crf", "23",
        "-pix_fmt", "yuv420p",
        "-movflags", "+faststart",
        "-c:a", "aac",
        "-b:a", "128k",
        str(preview_path),
    ]
    try:
        subprocess.run(cmd, check=True, capture_output=True, text=True)
    except Exception as exc:
        logger.error(f"Preview transcode failed for {source_path}: {exc}")
        if preview_path.exists():
            preview_path.unlink(missing_ok=True)
        raise

    return preview_path


@app.api_route("/api/video/{video_path:path}", methods=["GET", "HEAD"])
async def stream_video(video_path: str, request: Request):
    """
    Custom video streaming route that supports Range requests (Seeking/Tua).
    This is the most robust way to ensure seeking works in all browsers.
    """
    full_path = _resolve_output_video_path(video_path)
    return _stream_video_file(full_path, request)


@app.api_route("/api/video-preview/{video_path:path}", methods=["GET", "HEAD"])
async def stream_video_preview(video_path: str, request: Request):
    """
    Stream a browser-compatible preview (H.264/AAC) generated on demand and cached.
    Falls back to original file if transcoding fails.
    """
    source_path = _resolve_output_video_path(video_path)
    try:
        preview_path = _ensure_web_preview(source_path)
        return _stream_video_file(preview_path, request)
    except Exception:
        logger.warning(f"Using original stream as fallback for {source_path}")
        return _stream_video_file(source_path, request)


@app.get("/api/download/{job_id}")
async def download_result(job_id: str, request: Request):
    """Download kết quả xử lý (file đầu tiên)"""
    job = get_job(job_id)
    if not job:
        raise HTTPException(status_code=404, detail="Job not found")
    
    if job.status != "completed":
        raise HTTPException(status_code=400, detail=f"Job status: {job.status}")
    
    if not job.result_path or not Path(job.result_path).exists():
        raise HTTPException(status_code=404, detail="Result file not found")
    
    result_path = Path(job.result_path)
    
    # Nếu result_path là thư mục (vd: scene analysis), zip lại rồi trả về
    if result_path.is_dir():
        import zipfile
        zip_path = result_path.parent / f"{result_path.name}.zip"
        with zipfile.ZipFile(str(zip_path), 'w', zipfile.ZIP_DEFLATED) as zf:
            for file in sorted(result_path.iterdir()):
                if file.is_file():
                    zf.write(str(file), file.name)
        return FileResponse(
            str(zip_path),
            filename=f"{result_path.name}.zip",
            media_type="application/zip"
        )
    
    ext = result_path.suffix.lower()
    media_types = {
        ".mp4": "video/mp4", 
        ".mov": "video/quicktime", 
        ".qt": "video/quicktime", 
        ".srt": "text/plain", 
        ".zip": "application/zip"
    }

    # If a browser tries to preview via the download endpoint, support Range requests
    # so HTML5 <video> can seek and load reliably.
    if ext in {".mp4", ".mov", ".qt", ".webm", ".mkv"}:
        range_header = request.headers.get("range")
        if range_header:
            return _stream_video_file(result_path, request)
    
    return FileResponse(
        str(result_path),
        filename=result_path.name,
        media_type=media_types.get(ext, "application/octet-stream")
    )


@app.get("/api/download/{job_id}/{file_index}")
async def download_result_file(job_id: str, file_index: int):
    """Download file cụ thể từ job (theo index)"""
    job = get_job(job_id)
    if not job:
        raise HTTPException(status_code=404, detail="Job not found")
    
    if job.status != "completed":
        raise HTTPException(status_code=400, detail=f"Job status: {job.status}")
    
    if file_index < 0 or file_index >= len(job.result_files):
        raise HTTPException(status_code=404, detail="File index out of range")
    
    rf = job.result_files[file_index]
    if not Path(rf.path).exists():
        raise HTTPException(status_code=404, detail="File not found")
    
    ext = Path(rf.path).suffix.lower()
    media_types = {".mp4": "video/mp4", ".srt": "text/plain"}
    
    return FileResponse(
        rf.path,
        filename=Path(rf.path).name,
        media_type=media_types.get(ext, "application/octet-stream")
    )


@app.get("/api/countries")
async def get_supported_countries():
    """Lấy danh sách quốc gia được hỗ trợ cho legal content"""
    from src.legal.database import get_legal_database
    db = get_legal_database()
    return {"countries": db.get_countries()}


@app.get("/api/languages")
async def get_supported_languages():
    """Lấy danh sách ngôn ngữ được hỗ trợ cho font"""
    from src.subtitle.font_manager import get_font_manager
    fm = get_font_manager()
    return {"languages": fm.get_available_languages()}


# ============ Run ============

if __name__ == "__main__":
    import uvicorn
    
    host = os.getenv("HOST", "0.0.0.0")
    port = int(os.getenv("PORT", 8000))
    debug = os.getenv("DEBUG", "True").lower() == "true"
    
    uvicorn.run(
        "app:app",
        host=host,
        port=port,
        reload=debug
    )
