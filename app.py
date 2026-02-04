"""
Video Subtitles Automation - FastAPI Application
"""
import os
import logging
import shutil
import uuid
from pathlib import Path
from typing import Optional, List
from datetime import datetime

from fastapi import FastAPI, UploadFile, File, Form, HTTPException, BackgroundTasks
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from fastapi.responses import FileResponse, HTMLResponse
from fastapi.requests import Request
from pydantic import BaseModel
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Create directories
UPLOAD_DIR = Path(os.getenv("UPLOAD_DIR", "./uploads"))
OUTPUT_DIR = Path(os.getenv("OUTPUT_DIR", "./outputs"))
UPLOAD_DIR.mkdir(parents=True, exist_ok=True)
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

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

templates = Jinja2Templates(directory=str(templates_dir)) if templates_dir.exists() else None


# ============ Models ============

class JobStatus(BaseModel):
    job_id: str
    status: str  # pending, processing, completed, failed
    progress: int = 0
    result_path: Optional[str] = None
    error: Optional[str] = None
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


# ============ Job Storage (in-memory) ============

jobs = {}


def get_job(job_id: str) -> Optional[JobStatus]:
    return jobs.get(job_id)


def update_job(job_id: str, **kwargs):
    if job_id in jobs:
        for key, value in kwargs.items():
            setattr(jobs[job_id], key, value)


# ============ Helper Functions ============

def save_upload(file: UploadFile, prefix: str = "") -> Path:
    """Save uploaded file and return path"""
    ext = Path(file.filename).suffix
    filename = f"{prefix}{uuid.uuid4().hex}{ext}"
    filepath = UPLOAD_DIR / filename
    
    with open(filepath, "wb") as f:
        shutil.copyfileobj(file.file, f)
    
    return filepath


# ============ Background Tasks ============

async def process_subtitles(
    job_id: str,
    video_path: str,
    excel_path: str,
    language: str,
    video_format: str
):
    """Background task để xử lý phụ đề - luôn dùng AI canh timing"""
    try:
        update_job(job_id, status="processing", progress=10)
        
        from src.subtitle.excel_reader import ExcelReader
        from src.subtitle.renderer import SubtitleRenderer
        from src.subtitle.timing_sync import TimingSync, extract_audio
        
        # Read Excel - chỉ cần cột text
        reader = ExcelReader(excel_path)
        subtitles_data = reader.parse()
        update_job(job_id, progress=20)
        
        if not subtitles_data:
            raise ValueError("Không tìm thấy phụ đề trong file Excel")
        
        # Luôn dùng AI để canh timing
        update_job(job_id, progress=30)
        sync = TimingSync()
        
        try:
            # Trích xuất audio từ video
            audio_path = extract_audio(video_path)
            update_job(job_id, progress=40)
            
            # Dùng Whisper transcribe để lấy timing
            transcript = sync.transcribe(audio_path, language)
            update_job(job_id, progress=60)
            
            # Căn timing phụ đề theo transcript
            subtitle_texts = [s.text for s in subtitles_data]
            aligned = sync.align_subtitles(subtitle_texts, transcript)
            
            # Áp dụng ngắt dòng thông minh
            aligned = [
                (start, end, sync.smart_line_break(text))
                for start, end, text in aligned
            ]
            
        except Exception as e:
            logger.warning(f"AI timing failed: {e}, using fallback")
            # Fallback: chia đều 3 giây mỗi phụ đề
            aligned = [
                (i * 3, (i + 1) * 3, s.text)
                for i, s in enumerate(subtitles_data)
            ]
        
        update_job(job_id, progress=70)
        
        # Render
        renderer = SubtitleRenderer(str(OUTPUT_DIR))
        output_path = renderer.render(
            video_path,
            aligned,
            language=language,
            video_format=video_format
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


async def process_legal(
    job_id: str,
    video_path: str,
    country_code: str,
    media_type: str,
    usage_type: str,
    auto_position: bool
):
    """Background task để thêm nội dung pháp lý"""
    try:
        update_job(job_id, status="processing", progress=10)
        
        from src.legal.overlay import LegalOverlay
        from src.legal.database import MediaType, UsageType
        
        overlay = LegalOverlay(str(OUTPUT_DIR))
        
        update_job(job_id, progress=50)
        
        output_path = overlay.add_legal_content(
            video_path,
            country_code,
            MediaType(media_type),
            UsageType(usage_type),
            auto_detect_position=auto_position
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


@app.get("/health")
async def health_check():
    """Health check endpoint"""
    return {"status": "healthy", "timestamp": datetime.now().isoformat()}


@app.post("/api/subtitle")
async def add_subtitle(
    background_tasks: BackgroundTasks,
    video: UploadFile = File(...),
    excel: UploadFile = File(...),
    language: str = Form("vi"),
    video_format: str = Form("16x9")
):
    """
    Thêm phụ đề vào video từ file Excel
    
    - **video**: File video (mp4, mov, etc.)
    - **excel**: File Excel chứa phụ đề (chỉ cần cột text)
    - **language**: Mã ngôn ngữ (vi, en, ja, etc.)
    - **video_format**: Định dạng video (16x9, 9x16, 1x1, 4x5)
    
    AI tự động canh timing theo lời thoại trong video.
    """
    # Save uploads
    video_path = save_upload(video, "video_")
    excel_path = save_upload(excel, "excel_")
    
    # Create job
    job_id = uuid.uuid4().hex
    jobs[job_id] = JobStatus(
        job_id=job_id,
        status="pending",
        created_at=datetime.now().isoformat()
    )
    
    # Start background processing
    background_tasks.add_task(
        process_subtitles,
        job_id,
        str(video_path),
        str(excel_path),
        language,
        video_format
    )
    
    return {"job_id": job_id, "message": "Processing started - AI đang canh timing"}


async def process_subtitles_from_text(
    job_id: str,
    video_path: str,
    subtitle_lines: Optional[List[str]],
    language: str,
    video_format: str,
    raw_text: Optional[str] = None
):
    """Background task để xử lý phụ đề từ text trực tiếp"""
    try:
        update_job(job_id, status="processing", progress=5)
        
        from src.subtitle.renderer import SubtitleRenderer
        from src.subtitle.timing_sync import TimingSync, extract_audio
        
        sync = TimingSync()
        
        # Nếu có raw_text (1 đoạn dài), dùng AI ngắt câu trước
        if raw_text and not subtitle_lines:
            update_job(job_id, progress=10)
            logger.info(f"Segmenting long text ({len(raw_text)} chars) with AI...")
            
            subtitle_lines = sync.segment_long_text(
                raw_text,
                language=language,
                max_chars=80,
                use_ai=True
            )
            logger.info(f"AI created {len(subtitle_lines)} subtitle segments")
        
        if not subtitle_lines:
            raise ValueError("Không có phụ đề nào được tạo")
        
        update_job(job_id, progress=20)
        
        try:
            # Trích xuất audio từ video
            audio_path = extract_audio(video_path)
            update_job(job_id, progress=35)
            
            # Dùng Whisper transcribe để lấy timing
            transcript = sync.transcribe(audio_path, language)
            update_job(job_id, progress=55)
            
            # Căn timing phụ đề theo transcript
            aligned = sync.align_subtitles(subtitle_lines, transcript)
            
            # Áp dụng ngắt dòng thông minh cho mỗi segment
            aligned = [
                (start, end, sync.smart_line_break(text))
                for start, end, text in aligned
            ]
            
        except Exception as e:
            logger.warning(f"AI timing failed: {e}, using fallback")
            # Fallback: chia đều 3 giây mỗi phụ đề
            aligned = [
                (i * 3, (i + 1) * 3, text)
                for i, text in enumerate(subtitle_lines)
            ]
        
        update_job(job_id, progress=70)
        
        # Render
        renderer = SubtitleRenderer(str(OUTPUT_DIR))
        output_path = renderer.render(
            video_path,
            aligned,
            language=language,
            video_format=video_format
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


@app.post("/api/subtitle-text")
async def add_subtitle_from_text(
    background_tasks: BackgroundTasks,
    video: UploadFile = File(...),
    subtitle_text: str = Form(...),
    language: str = Form("vi"),
    video_format: str = Form("16x9"),
    auto_segment: bool = Form(True)
):
    """
    Thêm phụ đề vào video từ text nhập trực tiếp
    
    - **video**: File video (mp4, mov, etc.)
    - **subtitle_text**: Text phụ đề (có thể là 1 đoạn dài hoặc mỗi dòng 1 câu)
    - **language**: Mã ngôn ngữ (vi, en, ja, etc.)
    - **video_format**: Định dạng video (16x9, 9x16, 1x1, 4x5)
    - **auto_segment**: Tự động ngắt câu dài thành các đoạn phụ đề (mặc định: True)
    
    AI tự động:
    - Ngắt câu dài thành các đoạn phụ đề phù hợp
    - Canh timing theo lời thoại trong video
    """
    text = subtitle_text.strip()
    
    if not text:
        raise HTTPException(status_code=400, detail="Không có phụ đề nào được nhập")
    
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
    jobs[job_id] = JobStatus(
        job_id=job_id,
        status="pending",
        created_at=datetime.now().isoformat()
    )
    
    # Start background processing
    background_tasks.add_task(
        process_subtitles_from_text,
        job_id,
        str(video_path),
        subtitle_lines,
        language,
        video_format,
        raw_text  # Pass raw text for AI segmentation
    )
    
    if raw_text:
        return {"job_id": job_id, "message": "AI đang ngắt câu và canh timing..."}
    else:
        return {"job_id": job_id, "message": f"Processing {len(subtitle_lines)} dòng phụ đề - AI đang canh timing"}


async def process_auto_subtitles(
    job_id: str,
    video_path: str,
    language: str,
    video_format: str,
    font_size: Optional[int] = None
):
    """Background task để tự động trích xuất phụ đề từ audio (Auto Detect)"""
    try:
        update_job(job_id, status="processing", progress=5)
        
        from src.subtitle.renderer import SubtitleRenderer
        from src.subtitle.timing_sync import TimingSync, extract_audio
        
        sync = TimingSync()
        
        try:
            # Trích xuất audio từ video
            audio_path = extract_audio(video_path)
            update_job(job_id, progress=30)
            
            # Dùng Whisper transcribe để lấy nội dung và timing (Auto Detect Language)
            logger.info("Transcribing audio with auto-detect...")
            transcript = sync.transcribe(audio_path, language=None)
            
            if not transcript:
                raise ValueError("Không thể trích xuất lời thoại từ video")
                
            detected_lang = sync.last_detected_language
            logger.info(f"Detected audio language: {detected_lang}")
            
            update_job(job_id, progress=50)
            
            # Logic: Translate if detected language != target language
            # Normalize: Whisper usually returns 2-char code 'en', 'vi'. 
            # If detected 'en' and target 'vi', translate.
            
            # Initial aligned data (raw text)
            aligned_raw = [(seg.start, seg.end, seg.text) for seg in transcript]
            
            if detected_lang and language and detected_lang.lower() != language.lower():
                logger.info(f"Translating subtitles from {detected_lang} to {language}...")
                
                try:
                    from src.subtitle.translator import SubtitleTranslator
                    translator = SubtitleTranslator()
                    
                    aligned_raw = translator.translate_with_timing(
                        aligned_raw,
                        source_lang=detected_lang,
                        target_lang=language,
                        context="Video subtitles"
                    )
                except Exception as e:
                    logger.error(f"Translation failed: {e}")
                    # Continue with original text if translation fails
            
            update_job(job_id, progress=60)
            
            # Apply smart line break final text
            aligned = [
                (start, end, sync.smart_line_break(text))
                for start, end, text in aligned_raw
            ]
            
            logger.info(f"Generated {len(aligned)} subtitle segments")
            
        except Exception as e:
            logger.error(f"Auto detect failed: {e}")
            raise e
        
        update_job(job_id, progress=70)
        
        # Render
        renderer = SubtitleRenderer(str(OUTPUT_DIR))
        output_path = renderer.render(
            video_path,
            aligned,
            language=language,
            video_format=video_format,
            font_size=font_size
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


@app.post("/api/subtitle-auto")
async def add_subtitle_auto(
    background_tasks: BackgroundTasks,
    video: UploadFile = File(...),
    language: str = Form("vi"),
    video_format: str = Form("16x9"),
    font_size: Optional[int] = Form(None)
):
    """
    Tự động trích xuất và thêm phụ đề (Auto Detect)
    
    - **video**: File video
    - **language**: Mã ngôn ngữ của video
    - **video_format**: Định dạng video
    - **font_size**: Kích thước font (optional)
    
    AI sẽ:
    - Nghe và trích xuất lời thoại (Speech-to-Text)
    - Tự động lấy timing chính xác
    - Render phụ đề lên video
    """
    # Save video
    video_path = save_upload(video, "video_")
    
    # Create job
    job_id = uuid.uuid4().hex
    jobs[job_id] = JobStatus(
        job_id=job_id,
        status="pending",
        created_at=datetime.now().isoformat()
    )
    
    # Start background processing
    background_tasks.add_task(
        process_auto_subtitles,
        job_id,
        str(video_path),
        language,
        video_format,
        font_size
    )
    
    return {"job_id": job_id, "message": "AI đang nghe và trích xuất phụ đề..."}


@app.post("/api/legal")
async def add_legal_content(
    background_tasks: BackgroundTasks,
    video: UploadFile = File(...),
    country_code: str = Form(...),
    media_type: str = Form("social"),
    usage_type: str = Form("shareable"),
    auto_position: bool = Form(True)
):
    """
    Thêm nội dung pháp lý vào video
    
    - **video**: File video
    - **country_code**: Mã quốc gia (VN, FR, US, HK)
    - **media_type**: Loại media (social, tv, ooh, digital, print)
    - **usage_type**: Hình thức sử dụng (shareable, non_shareable, paid, organic)
    - **auto_position**: Tự động phát hiện vị trí tối ưu
    """
    video_path = save_upload(video, "video_")
    
    job_id = uuid.uuid4().hex
    jobs[job_id] = JobStatus(
        job_id=job_id,
        status="pending",
        created_at=datetime.now().isoformat()
    )
    
    background_tasks.add_task(
        process_legal,
        job_id,
        str(video_path),
        country_code.upper(),
        media_type,
        usage_type,
        auto_position
    )
    
    return {"job_id": job_id, "message": "Processing started"}


@app.post("/api/master/replace-logo")
async def replace_logo(
    background_tasks: BackgroundTasks,
    video: UploadFile = File(...),
    logo: UploadFile = File(...),
    x: Optional[int] = Form(None),
    y: Optional[int] = Form(None)
):
    """Thay thế logo trong video"""
    video_path = save_upload(video, "video_")
    logo_path = save_upload(logo, "logo_")
    
    job_id = uuid.uuid4().hex
    jobs[job_id] = JobStatus(
        job_id=job_id,
        status="pending",
        created_at=datetime.now().isoformat()
    )
    
    async def process():
        try:
            update_job(job_id, status="processing", progress=30)
            
            from src.mastering.element_replacer import ElementReplacer
            replacer = ElementReplacer(str(OUTPUT_DIR))
            
            position = (x, y) if x is not None and y is not None else None
            
            output_path = replacer.replace_logo(
                str(video_path),
                str(logo_path),
                position=position
            )
            
            update_job(
                job_id,
                status="completed",
                progress=100,
                result_path=output_path,
                completed_at=datetime.now().isoformat()
            )
        except Exception as e:
            update_job(job_id, status="failed", error=str(e))
    
    background_tasks.add_task(process)
    return {"job_id": job_id, "message": "Processing started"}


@app.post("/api/master/add-packshot")
async def add_packshot(
    background_tasks: BackgroundTasks,
    video: UploadFile = File(...),
    packshot: UploadFile = File(...),
    position: str = Form("center"),
    duration: float = Form(2.0)
):
    """Thêm packshot vào cuối video"""
    video_path = save_upload(video, "video_")
    packshot_path = save_upload(packshot, "packshot_")
    
    job_id = uuid.uuid4().hex
    jobs[job_id] = JobStatus(
        job_id=job_id,
        status="pending",
        created_at=datetime.now().isoformat()
    )
    
    async def process():
        try:
            update_job(job_id, status="processing", progress=30)
            
            from src.mastering.element_replacer import ElementReplacer
            replacer = ElementReplacer(str(OUTPUT_DIR))
            
            output_path = replacer.add_packshot(
                str(video_path),
                str(packshot_path),
                position=position,
                duration=duration
            )
            
            update_job(
                job_id,
                status="completed",
                progress=100,
                result_path=output_path,
                completed_at=datetime.now().isoformat()
            )
        except Exception as e:
            update_job(job_id, status="failed", error=str(e))
    
    background_tasks.add_task(process)
    return {"job_id": job_id, "message": "Processing started"}


@app.post("/api/subtitle-batch")
async def add_subtitle_batch(
    background_tasks: BackgroundTasks,
    video: UploadFile = File(...),
    subtitle_text: str = Form(...),
    source_language: str = Form("en"),
    target_languages: str = Form("vi"),  # Comma-separated: "vi,ja,ko"
    video_format: str = Form("16x9")
):
    """
    Tạo nhiều video với phụ đề dịch sang nhiều ngôn ngữ
    
    - **video**: File video gốc
    - **subtitle_text**: Phụ đề gốc (text hoặc từng dòng)
    - **source_language**: Ngôn ngữ gốc của phụ đề (en, vi...)
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
    jobs[job_id] = JobStatus(
        job_id=job_id,
        status="pending",
        created_at=datetime.now().isoformat()
    )
    
    # Background task
    async def process_batch():
        try:
            update_job(job_id, status="processing", progress=5)
            
            from src.subtitle.translator import SubtitleTranslator
            from src.subtitle.timing_sync import TimingSync, extract_audio
            from src.subtitle.renderer import SubtitleRenderer
            import zipfile
            
            sync = TimingSync()
            translator = SubtitleTranslator()
            renderer = SubtitleRenderer(str(OUTPUT_DIR))
            
            # Step 1: Segment text if needed
            if len(lines) == 1 and len(lines[0]) > 100:
                update_job(job_id, progress=10)
                subtitle_lines = sync.segment_long_text(lines[0], source_language, 80, True)
            else:
                subtitle_lines = lines
            
            # Step 2: Get timing from video
            update_job(job_id, progress=20)
            try:
                audio_path = extract_audio(str(video_path))
                update_job(job_id, progress=30)
                transcript = sync.transcribe(audio_path, source_language)
                aligned = sync.align_subtitles(subtitle_lines, transcript)
            except Exception as e:
                logger.warning(f"Timing failed: {e}, using fallback")
                aligned = [(i * 3, (i + 1) * 3, t) for i, t in enumerate(subtitle_lines)]
            
            update_job(job_id, progress=40)
            
            # Step 3: Translate to each target language
            output_files = []
            total_langs = len(targets)
            
            for idx, target_lang in enumerate(targets):
                lang_progress = 40 + int(50 * (idx / total_langs))
                update_job(job_id, progress=lang_progress)
                
                # Translate
                if target_lang != source_language:
                    translated_aligned = translator.translate_with_timing(
                        aligned, source_language, target_lang
                    )
                else:
                    translated_aligned = aligned
                
                # Apply line breaks
                translated_aligned = [
                    (start, end, sync.smart_line_break(text))
                    for start, end, text in translated_aligned
                ]
                
                # Render video
                output_path = renderer.render(
                    str(video_path),
                    translated_aligned,
                    language=target_lang,
                    video_format=video_format
                )
                
                # Rename with language suffix
                new_path = output_path.replace("_subtitled.mp4", f"_{target_lang}.mp4")
                if output_path != new_path:
                    import shutil
                    shutil.move(output_path, new_path)
                    output_path = new_path
                
                output_files.append((target_lang, output_path))
                logger.info(f"Generated {target_lang} version: {output_path}")
            
            update_job(job_id, progress=90)
            
            # Step 4: Create ZIP if multiple files
            if len(output_files) > 1:
                zip_path = str(OUTPUT_DIR / f"batch_{job_id}.zip")
                with zipfile.ZipFile(zip_path, 'w') as zf:
                    for lang, path in output_files:
                        zf.write(path, Path(path).name)
                result_path = zip_path
            else:
                result_path = output_files[0][1]
            
            update_job(
                job_id,
                status="completed",
                progress=100,
                result_path=result_path,
                completed_at=datetime.now().isoformat()
            )
            
        except Exception as e:
            logger.error(f"Batch job {job_id} failed: {e}")
            update_job(job_id, status="failed", error=str(e))
    
    background_tasks.add_task(process_batch)
    
    return {
        "job_id": job_id, 
        "message": f"Creating {len(targets)} language versions: {', '.join(targets)}"
    }


@app.post("/api/subtitle-export")
async def export_subtitle_files(
    background_tasks: BackgroundTasks,
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
    jobs[job_id] = JobStatus(
        job_id=job_id,
        status="pending",
        created_at=datetime.now().isoformat()
    )
    
    # Parse translate languages
    translate_langs = [l.strip() for l in translate_to.split(",") if l.strip()]
    
    async def process_export():
        try:
            update_job(job_id, status="processing", progress=5)
            
            from src.subtitle.timing_sync import TimingSync, extract_audio
            from src.subtitle.exporter import SubtitleExporter
            from src.subtitle.translator import SubtitleTranslator
            import zipfile
            
            sync = TimingSync()
            exporter = SubtitleExporter(str(OUTPUT_DIR))
            
            # Step 1: Segment if needed
            if len(lines) == 1 and len(lines[0]) > 100:
                subtitle_lines = sync.segment_long_text(lines[0], language, 80, True)
            else:
                subtitle_lines = lines
            
            update_job(job_id, progress=20)
            
            # Step 2: Get timing
            try:
                audio_path = extract_audio(str(video_path))
                update_job(job_id, progress=40)
                transcript = sync.transcribe(audio_path, language)
                aligned = sync.align_subtitles(subtitle_lines, transcript)
            except Exception as e:
                logger.warning(f"Timing failed: {e}, using fallback")
                aligned = [(i * 3, (i + 1) * 3, t) for i, t in enumerate(subtitle_lines)]
            
            # Apply line breaks
            aligned = [
                (start, end, sync.smart_line_break(text))
                for start, end, text in aligned
            ]
            
            update_job(job_id, progress=60)
            
            # Step 3: Export files
            output_files = []
            base_name = Path(video_path).stem
            
            # Original language
            if export_format in ["srt", "both"]:
                srt_path = exporter.export_srt(aligned, f"{base_name}_{language}")
                output_files.append(srt_path)
            
            if export_format in ["ass", "both"]:
                ass_path = exporter.export_ass(aligned, f"{base_name}_{language}")
                output_files.append(ass_path)
            
            # Translations
            if translate_langs:
                translator = SubtitleTranslator()
                
                for target_lang in translate_langs:
                    if target_lang != language:
                        translated_aligned = translator.translate_with_timing(
                            aligned, language, target_lang
                        )
                        
                        if export_format in ["srt", "both"]:
                            srt_path = exporter.export_srt(
                                translated_aligned, f"{base_name}_{target_lang}"
                            )
                            output_files.append(srt_path)
                        
                        if export_format in ["ass", "both"]:
                            ass_path = exporter.export_ass(
                                translated_aligned, f"{base_name}_{target_lang}"
                            )
                            output_files.append(ass_path)
            
            update_job(job_id, progress=90)
            
            # Create ZIP if multiple files
            if len(output_files) > 1:
                zip_path = str(OUTPUT_DIR / f"subtitles_{job_id}.zip")
                with zipfile.ZipFile(zip_path, 'w') as zf:
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
                completed_at=datetime.now().isoformat()
            )
            
        except Exception as e:
            logger.error(f"Export job {job_id} failed: {e}")
            update_job(job_id, status="failed", error=str(e))
    
    background_tasks.add_task(process_export)
    
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


@app.get("/api/download/{job_id}")
async def download_result(job_id: str):
    """Download kết quả xử lý"""
    job = get_job(job_id)
    if not job:
        raise HTTPException(status_code=404, detail="Job not found")
    
    if job.status != "completed":
        raise HTTPException(status_code=400, detail=f"Job status: {job.status}")
    
    if not job.result_path or not Path(job.result_path).exists():
        raise HTTPException(status_code=404, detail="Result file not found")
    
    return FileResponse(
        job.result_path,
        filename=Path(job.result_path).name,
        media_type="video/mp4"
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
