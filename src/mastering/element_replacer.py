"""
Element Replacer Module
Thay thế logo, tagline, packshot trong video
"""
import subprocess
import os
import logging
from pathlib import Path
from typing import Optional, Dict, List, Tuple
import tempfile
import json

from .scene_analysis import SceneAnalyzer, VideoElement

logger = logging.getLogger(__name__)


class ElementReplacer:
    """Thay thế elements trong video"""
    
    def __init__(self, output_dir: Optional[str] = None):
        self.output_dir = Path(output_dir) if output_dir else Path("./outputs")
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.analyzer = SceneAnalyzer()
    
    def _get_video_dimensions(self, video_path: str) -> Tuple[int, int]:
        """Lấy kích thước video"""
        cmd = [
            "ffprobe", "-v", "quiet",
            "-select_streams", "v:0",
            "-show_entries", "stream=width,height",
            "-of", "csv=p=0",
            video_path
        ]
        result = subprocess.run(cmd, capture_output=True, text=True)
        try:
            w, h = result.stdout.strip().split(',')
            return int(w), int(h)
        except:
            return 1920, 1080
    
    def replace_logo(
        self,
        video_path: str,
        new_logo_path: str,
        position: Optional[Tuple[int, int]] = None,
        size: Optional[Tuple[int, int]] = None,
        time_range: Optional[Tuple[float, float]] = None,
        output_path: Optional[str] = None
    ) -> str:
        """
        Thay thế logo trong video
        
        Args:
            video_path: Đường dẫn video input
            new_logo_path: Đường dẫn logo mới
            position: Vị trí (x, y), None = auto-detect
            size: Kích thước (width, height), None = giữ nguyên
            time_range: (start, end) thời gian hiển thị, None = toàn bộ
            output_path: Đường dẫn output
            
        Returns:
            Đường dẫn video output
        """
        video_path = Path(video_path)
        width, height = self._get_video_dimensions(str(video_path))
        
        if output_path is None:
            output_path = self.output_dir / f"{video_path.stem}_new_logo{video_path.suffix}"
        
        # Auto-detect position nếu không được chỉ định
        if position is None:
            positions = self.analyzer.find_element_positions(str(video_path), "logo")
            if positions:
                _, _, x, y = positions[0]
                position = (x, y)
            else:
                # Default: góc phải trên
                position = (width - 200, 50)
        
        # Tạo filter overlay
        x, y = position
        
        filter_parts = [f"[1:v]scale={size[0]}:{size[1]}[logo]" if size else "[1:v]copy[logo]"]
        
        enable_clause = ""
        if time_range:
            start, end = time_range
            enable_clause = f":enable='between(t,{start},{end})'"
        
        filter_parts.append(f"[0:v][logo]overlay={x}:{y}{enable_clause}")
        
        filter_complex = ";".join(filter_parts)
        
        cmd = [
            "ffmpeg", "-y",
            "-i", str(video_path),
            "-i", new_logo_path,
            "-filter_complex", filter_complex,
            "-c:a", "copy",
            str(output_path)
        ]
        
        logger.info(f"Replacing logo in video")
        result = subprocess.run(cmd, capture_output=True, text=True)
        
        if result.returncode != 0:
            logger.error(f"FFmpeg error: {result.stderr}")
            raise RuntimeError(f"FFmpeg failed: {result.stderr}")
        
        logger.info(f"Logo replaced: {output_path}")
        return str(output_path)
    
    def add_packshot(
        self,
        video_path: str,
        packshot_path: str,
        position: str = "center",
        start_time: Optional[float] = None,
        duration: float = 2.0,
        fade_in: float = 0.5,
        fade_out: float = 0.5,
        output_path: Optional[str] = None
    ) -> str:
        """
        Thêm packshot vào video
        
        Args:
            video_path: Đường dẫn video input
            packshot_path: Đường dẫn packshot image/video
            position: Vị trí (center, left, right)
            start_time: Thời điểm bắt đầu, None = cuối video
            duration: Thời lượng hiển thị
            fade_in: Thời gian fade in
            fade_out: Thời gian fade out
            output_path: Đường dẫn output
            
        Returns:
            Đường dẫn video output
        """
        video_path = Path(video_path)
        width, height = self._get_video_dimensions(str(video_path))
        
        if output_path is None:
            output_path = self.output_dir / f"{video_path.stem}_packshot{video_path.suffix}"
        
        # Lấy video duration
        cmd = [
            "ffprobe", "-v", "quiet",
            "-show_entries", "format=duration",
            "-of", "default=noprint_wrappers=1:nokey=1",
            str(video_path)
        ]
        result = subprocess.run(cmd, capture_output=True, text=True)
        video_duration = float(result.stdout.strip())
        
        if start_time is None:
            start_time = video_duration - duration
        
        end_time = start_time + duration
        
        # Xác định vị trí
        position_map = {
            "center": f"(W-w)/2:(H-h)/2",
            "left": f"50:(H-h)/2",
            "right": f"W-w-50:(H-h)/2",
        }
        overlay_pos = position_map.get(position, position_map["center"])
        
        # Build filter với fade
        filter_complex = (
            f"[1:v]format=rgba,fade=t=in:st=0:d={fade_in}:alpha=1,"
            f"fade=t=out:st={duration-fade_out}:d={fade_out}:alpha=1[pack];"
            f"[0:v][pack]overlay={overlay_pos}:enable='between(t,{start_time},{end_time})'"
        )
        
        cmd = [
            "ffmpeg", "-y",
            "-i", str(video_path),
            "-i", packshot_path,
            "-filter_complex", filter_complex,
            "-c:a", "copy",
            str(output_path)
        ]
        
        logger.info(f"Adding packshot to video")
        result = subprocess.run(cmd, capture_output=True, text=True)
        
        if result.returncode != 0:
            logger.error(f"FFmpeg error: {result.stderr}")
            raise RuntimeError(f"FFmpeg failed: {result.stderr}")
        
        logger.info(f"Packshot added: {output_path}")
        return str(output_path)
    
    def replace_text(
        self,
        video_path: str,
        old_text: str,
        new_text: str,
        font_family: str = "Arial",
        font_size: int = 48,
        position: Optional[Tuple[int, int]] = None,
        output_path: Optional[str] = None
    ) -> str:
        """
        Thay thế text/tagline trong video (cơ bản)
        Lưu ý: Việc xóa text gốc complex, cái này chỉ overlay text mới
        
        Args:
            video_path: Đường dẫn video input
            old_text: Text cần thay (để reference)
            new_text: Text mới
            font_family: Font chữ
            font_size: Kích thước font
            position: Vị trí (x, y)
            output_path: Đường dẫn output
            
        Returns:
            Đường dẫn video output
        """
        video_path = Path(video_path)
        width, height = self._get_video_dimensions(str(video_path))
        
        if output_path is None:
            output_path = self.output_dir / f"{video_path.stem}_new_text{video_path.suffix}"
        
        # Auto position if not provided
        if position is None:
            position = (width // 2, height - 100)
        
        x, y = position
        
        # Escape text
        new_text = new_text.replace("'", "'\\''").replace(":", "\\:")
        
        filter_str = (
            f"drawtext=text='{new_text}':"
            f"fontfile=/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf:"
            f"fontsize={font_size}:"
            f"fontcolor=white:"
            f"borderw=2:"
            f"bordercolor=black:"
            f"x={x}-text_w/2:"
            f"y={y}"
        )
        
        cmd = [
            "ffmpeg", "-y",
            "-i", str(video_path),
            "-vf", filter_str,
            "-c:a", "copy",
            str(output_path)
        ]
        
        result = subprocess.run(cmd, capture_output=True, text=True)
        
        if result.returncode != 0:
            logger.error(f"FFmpeg error: {result.stderr}")
            raise RuntimeError(f"FFmpeg failed: {result.stderr}")
        
        return str(output_path)
    
    def remove_intro(
        self,
        video_path: str,
        intro_duration: Optional[float] = None,
        output_path: Optional[str] = None
    ) -> str:
        """
        Xóa cảnh intro
        
        Args:
            video_path: Đường dẫn video input
            intro_duration: Thời lượng intro cần xóa, None = auto-detect
            output_path: Đường dẫn output
            
        Returns:
            Đường dẫn video output
        """
        video_path = Path(video_path)
        
        if output_path is None:
            output_path = self.output_dir / f"{video_path.stem}_no_intro{video_path.suffix}"
        
        # Auto detect intro duration
        if intro_duration is None:
            intro, _ = self.analyzer.get_intro_outro(str(video_path))
            if intro:
                intro_duration = intro.end_time
            else:
                intro_duration = 3.0  # Default
        
        cmd = [
            "ffmpeg", "-y",
            "-i", str(video_path),
            "-ss", str(intro_duration),
            "-c", "copy",
            str(output_path)
        ]
        
        result = subprocess.run(cmd, capture_output=True, text=True)
        
        if result.returncode != 0:
            raise RuntimeError(f"FFmpeg failed: {result.stderr}")
        
        return str(output_path)
    
    def remove_outro(
        self,
        video_path: str,
        outro_duration: Optional[float] = None,
        output_path: Optional[str] = None
    ) -> str:
        """
        Xóa cảnh outro
        
        Args:
            video_path: Đường dẫn video
            outro_duration: Thời lượng outro cần xóa, None = auto-detect
            output_path: Đường dẫn output
            
        Returns:
            Đường dẫn video output
        """
        video_path = Path(video_path)
        
        if output_path is None:
            output_path = self.output_dir / f"{video_path.stem}_no_outro{video_path.suffix}"
        
        # Get video duration
        cmd = [
            "ffprobe", "-v", "quiet",
            "-show_entries", "format=duration",
            "-of", "default=noprint_wrappers=1:nokey=1",
            str(video_path)
        ]
        result = subprocess.run(cmd, capture_output=True, text=True)
        video_duration = float(result.stdout.strip())
        
        # Auto detect outro duration
        if outro_duration is None:
            _, outro = self.analyzer.get_intro_outro(str(video_path))
            if outro:
                outro_duration = video_duration - outro.start_time
            else:
                outro_duration = 3.0
        
        new_duration = video_duration - outro_duration
        
        cmd = [
            "ffmpeg", "-y",
            "-i", str(video_path),
            "-t", str(new_duration),
            "-c", "copy",
            str(output_path)
        ]
        
        result = subprocess.run(cmd, capture_output=True, text=True)
        
        if result.returncode != 0:
            raise RuntimeError(f"FFmpeg failed: {result.stderr}")
        
        return str(output_path)
