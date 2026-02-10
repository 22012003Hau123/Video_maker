"""
Subtitle Renderer Module
Render phụ đề lên video bằng FFmpeg
"""
import subprocess
import os
import logging
from pathlib import Path
from typing import List, Tuple, Optional, Dict
from dataclasses import dataclass
import tempfile

from .font_manager import get_font_manager, FontConfig
from .excel_reader import SubtitleEntry

logger = logging.getLogger(__name__)


@dataclass
class VideoFormat:
    """Thông tin định dạng video"""
    name: str
    width: int
    height: int
    aspect_ratio: str
    
    # Vị trí phụ đề mặc định (% từ đáy)
    subtitle_margin_bottom: int = 50
    
    @property
    def resolution(self) -> str:
        return f"{self.width}x{self.height}"


# Các định dạng video được hỗ trợ
VIDEO_FORMATS = {
    "16x9": VideoFormat("16x9", 1920, 1080, "16:9", subtitle_margin_bottom=50),
    "9x16": VideoFormat("9x16", 1080, 1920, "9:16", subtitle_margin_bottom=100),
    "1x1": VideoFormat("1x1", 1080, 1080, "1:1", subtitle_margin_bottom=60),
    "4x5": VideoFormat("4x5", 1080, 1350, "4:5", subtitle_margin_bottom=80),
}


class SubtitleRenderer:
    """Render phụ đề lên video"""
    
    def __init__(self, output_dir: Optional[str] = None):
        self.output_dir = Path(output_dir) if output_dir else Path("./outputs")
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.font_manager = get_font_manager()
        
    def create_srt_file(
        self, 
        subtitles: List[Tuple[float, float, str]],
        output_path: str
    ) -> str:
        """
        Tạo file SRT từ danh sách phụ đề
        
        Args:
            subtitles: List of (start_time, end_time, text)
            output_path: Đường dẫn file SRT
            
        Returns:
            Đường dẫn file SRT
        """
        def format_time(seconds: float) -> str:
            hours = int(seconds // 3600)
            minutes = int((seconds % 3600) // 60)
            secs = int(seconds % 60)
            millis = int((seconds % 1) * 1000)
            return f"{hours:02d}:{minutes:02d}:{secs:02d},{millis:03d}"
        
        srt_content = []
        for i, (start, end, text) in enumerate(subtitles):
            srt_content.append(f"{i + 1}")
            srt_content.append(f"{format_time(start)} --> {format_time(end)}")
            srt_content.append(text)
            srt_content.append("")
        
        Path(output_path).write_text("\n".join(srt_content), encoding='utf-8')
        return output_path
    
    def create_ass_file(
        self,
        subtitles: List[Tuple[float, float, str]],
        output_path: str,
        font_config: FontConfig,
        video_format: VideoFormat
    ) -> str:
        """
        Tạo file ASS (Advanced SubStation Alpha) với styling
        
        Args:
            subtitles: List of (start_time, end_time, text)
            output_path: Đường dẫn file ASS
            font_config: Cấu hình font
            video_format: Định dạng video
            
        Returns:
            Đường dẫn file ASS
        """
        def format_time_ass(seconds: float) -> str:
            hours = int(seconds // 3600)
            minutes = int((seconds % 3600) // 60)
            secs = seconds % 60
            return f"{hours}:{minutes:02d}:{secs:05.2f}"
        
        # ASS header với style
        ass_content = f"""[Script Info]
Title: Auto Generated Subtitles
ScriptType: v4.00+
PlayResX: {video_format.width}
PlayResY: {video_format.height}

[V4+ Styles]
Format: Name, Fontname, Fontsize, PrimaryColour, SecondaryColour, OutlineColour, BackColour, Bold, Italic, Underline, StrikeOut, ScaleX, ScaleY, Spacing, Angle, BorderStyle, Outline, Shadow, Alignment, MarginL, MarginR, MarginV, Encoding
Style: Default,{font_config.font_family},{font_config.font_size},&H00FFFFFF,&H000000FF,&H00000000,&H80000000,{1 if font_config.bold else 0},{1 if font_config.italic else 0},0,0,100,100,0,0,1,{font_config.outline_width},1,2,10,10,{video_format.subtitle_margin_bottom},1

[Events]
Format: Layer, Start, End, Style, Name, MarginL, MarginR, MarginV, Effect, Text
"""
        
        for start, end, text in subtitles:
            # Escape các ký tự đặc biệt và xử lý xuống dòng
            text = text.replace("\\n", "\\N").replace("\n", "\\N")
            ass_content += f"Dialogue: 0,{format_time_ass(start)},{format_time_ass(end)},Default,,0,0,0,,{text}\n"
        
        Path(output_path).write_text(ass_content, encoding='utf-8')
        return output_path
    
    def get_video_info(self, video_path: str) -> Dict:
        """Lấy thông tin video bằng ffprobe"""
        cmd = [
            "ffprobe",
            "-v", "quiet",
            "-print_format", "json",
            "-show_format",
            "-show_streams",
            video_path
        ]
        
        result = subprocess.run(cmd, capture_output=True, text=True)
        if result.returncode != 0:
            raise RuntimeError(f"ffprobe failed: {result.stderr}")
        
        import json
        return json.loads(result.stdout)
    
    def detect_video_format(self, video_path: str) -> VideoFormat:
        """Tự động phát hiện định dạng video"""
        info = self.get_video_info(video_path)
        
        # Tìm video stream
        for stream in info.get("streams", []):
            if stream.get("codec_type") == "video":
                width = stream.get("width", 1920)
                height = stream.get("height", 1080)
                
                # Xác định aspect ratio
                ratio = width / height
                if abs(ratio - 16/9) < 0.1:
                    return VIDEO_FORMATS["16x9"]
                elif abs(ratio - 9/16) < 0.1:
                    return VIDEO_FORMATS["9x16"]
                elif abs(ratio - 1) < 0.1:
                    return VIDEO_FORMATS["1x1"]
                elif abs(ratio - 4/5) < 0.1:
                    return VIDEO_FORMATS["4x5"]
                else:
                    # Custom format
                    return VideoFormat(
                        name="custom",
                        width=width,
                        height=height,
                        aspect_ratio=f"{width}:{height}"
                    )
        
        return VIDEO_FORMATS["16x9"]  # Default
    
    def render(
        self,
        video_path: str,
        subtitles: List[Tuple[float, float, str]],
        output_path: Optional[str] = None,
        language: str = "vi",
        video_format: Optional[str] = None,
        use_ass: bool = True
    ) -> str:
        """
        Render phụ đề lên video
        
        Args:
            video_path: Đường dẫn video input
            subtitles: List of (start_time, end_time, text)
            output_path: Đường dẫn video output (optional)
            language: Mã ngôn ngữ để chọn font
            video_format: Định dạng video (16x9, 9x16, 1x1, 4x5)
            use_ass: Sử dụng ASS format để có styling tốt hơn
            
        Returns:
            Đường dẫn video output
        """
        video_path = Path(video_path)
        
        # Detect video format nếu không được chỉ định
        vf = VIDEO_FORMATS.get(video_format) if video_format else self.detect_video_format(str(video_path))
        
        # Lấy font config
        font_config = self.font_manager.get_font_config(language)
        
        # Output path
        if output_path is None:
            output_path = self.output_dir / f"{video_path.stem}_subtitled{video_path.suffix}"
        
        # Tạo file phụ đề tạm trong output_dir (cùng ổ đĩa với project)
        # để tránh vấn đề drive letter (C: vs D:) khi dùng relative path cho FFmpeg
        with tempfile.NamedTemporaryFile(
            mode='w', 
            suffix='.ass' if use_ass else '.srt',
            delete=False,
            encoding='utf-8',
            dir=str(self.output_dir)
        ) as tmp_file:
            if use_ass:
                self.create_ass_file(subtitles, tmp_file.name, font_config, vf)
            else:
                self.create_srt_file(subtitles, tmp_file.name)
            subtitle_file = tmp_file.name
        
        try:
            # Chuyển sang relative path + forward slash cho FFmpeg filter
            # FFmpeg filter parser coi '\' là escape char và ':' là separator
            # Dùng relative path để tránh drive letter 'C:' trên Windows
            try:
                ffmpeg_sub_path = os.path.relpath(subtitle_file).replace('\\', '/')
            except ValueError:
                # Fallback nếu khác ổ đĩa
                ffmpeg_sub_path = subtitle_file.replace('\\', '/')
            
            # Build FFmpeg command
            if use_ass:
                # ASS có styling, dùng filter ass
                subtitle_filter = f"ass={ffmpeg_sub_path}"
            else:
                # SRT dùng subtitles filter với force_style
                font_opts = self.font_manager.get_ffmpeg_font_options(language, vf.name)
                style = f"FontName={font_opts['fontname']},FontSize={font_opts['fontsize']}"
                style += f",PrimaryColour=&H00FFFFFF,OutlineColour=&H00000000"
                style += f",BorderStyle=1,Outline={font_opts['borderw']}"
                subtitle_filter = f"subtitles={ffmpeg_sub_path}:force_style='{style}'"
            
            cmd = [
                "ffmpeg", "-y",
                "-i", str(video_path),
                "-vf", subtitle_filter,
                "-c:v", "libx264",
                "-preset", "ultrafast",  # Optimize for speed
                "-crf", "23",            # Maintain good quality
                "-threads", "0",         # Use all available cores
                "-c:a", "copy",
                "-movflags", "+faststart", # Optimize for web playback
                str(output_path)
            ]
            
            logger.info(f"Running FFmpeg: {' '.join(cmd)}")
            result = subprocess.run(cmd, capture_output=True, text=True)
            
            if result.returncode != 0:
                logger.error(f"FFmpeg error: {result.stderr}")
                raise RuntimeError(f"FFmpeg failed: {result.stderr}")
            
            logger.info(f"Rendered video saved to: {output_path}")
            return str(output_path)
            
        finally:
            # Cleanup temp file
            if os.path.exists(subtitle_file):
                os.unlink(subtitle_file)
    
    def batch_render(
        self,
        video_path: str,
        subtitles_by_lang: Dict[str, List[Tuple[float, float, str]]],
        output_dir: Optional[str] = None
    ) -> Dict[str, str]:
        """
        Render nhiều phiên bản ngôn ngữ
        
        Args:
            video_path: Đường dẫn video input
            subtitles_by_lang: Dict mapping language code to subtitles
            output_dir: Thư mục output
            
        Returns:
            Dict mapping language code to output path
        """
        video_path = Path(video_path)
        out_dir = Path(output_dir) if output_dir else self.output_dir
        
        results = {}
        for lang, subs in subtitles_by_lang.items():
            output_path = out_dir / f"{video_path.stem}_{lang}{video_path.suffix}"
            results[lang] = self.render(
                str(video_path),
                subs,
                str(output_path),
                language=lang
            )
        
        return results


def render_subtitles(
    video_path: str,
    subtitles: List[Tuple[float, float, str]],
    output_path: Optional[str] = None,
    language: str = "vi"
) -> str:
    """Helper function để render phụ đề"""
    renderer = SubtitleRenderer()
    return renderer.render(video_path, subtitles, output_path, language)
