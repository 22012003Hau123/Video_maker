"""
Legal Overlay Module
Thêm nội dung pháp lý lên video
"""
import subprocess
import os
import logging
from pathlib import Path
from typing import Optional, Dict, List
import tempfile

from .database import LegalContent, MediaType, UsageType, get_legal_database
from .scene_detector import SceneDetector

logger = logging.getLogger(__name__)


class LegalOverlay:
    """Thêm nội dung pháp lý lên video"""
    
    def __init__(self, output_dir: Optional[str] = None):
        self.output_dir = Path(output_dir) if output_dir else Path("./outputs")
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.database = get_legal_database()
        self.scene_detector = SceneDetector()
    
    def _get_video_duration(self, video_path: str) -> float:
        """Lấy thời lượng video"""
        cmd = [
            "ffprobe",
            "-v", "quiet",
            "-show_entries", "format=duration",
            "-of", "default=noprint_wrappers=1:nokey=1",
            video_path
        ]
        result = subprocess.run(cmd, capture_output=True, text=True)
        try:
            return float(result.stdout.strip())
        except:
            return 60.0
    
    def _get_video_dimensions(self, video_path: str) -> tuple:
        """Lấy kích thước video"""
        cmd = [
            "ffprobe",
            "-v", "quiet",
            "-select_streams", "v:0",
            "-show_entries", "stream=width,height",
            "-of", "csv=p=0",
            video_path
        ]
        result = subprocess.run(cmd, capture_output=True, text=True)
        try:
            width, height = result.stdout.strip().split(',')
            return int(width), int(height)
        except:
            return 1920, 1080
    
    def add_legal_content(
        self,
        video_path: str,
        country_code: str,
        media_type: MediaType = MediaType.SOCIAL,
        usage_type: UsageType = UsageType.SHAREABLE,
        output_path: Optional[str] = None,
        output_path: Optional[str] = None,
        auto_detect_position: bool = True,
        product_type: Optional[str] = None,
        manual_position: Optional[str] = None
    ) -> str:
        """
        Thêm nội dung pháp lý vào video
        
        Args:
            video_path: Đường dẫn video input
            country_code: Mã quốc gia (VN, FR, US, etc.)
            media_type: Loại media
            usage_type: Hình thức sử dụng
            output_path: Đường dẫn output (optional)
            auto_detect_position: Tự động phát hiện vị trí tối ưu
            
        Returns:
            Đường dẫn video output
        """
        video_path = Path(video_path)
        duration = self._get_video_duration(str(video_path))
        width, height = self._get_video_dimensions(str(video_path))
        
        # Lấy nội dung pháp lý
        legal = self.database.get_legal_content(
            country_code, media_type, usage_type, duration, product_type
        )
        
        if not legal:
            logger.warning(f"No legal content found for {country_code}")
            return str(video_path)
        
        # Output path
        if output_path is None:
            output_path = self.output_dir / f"{video_path.stem}_legal{video_path.suffix}"
        
        # Xác định vị trí
        if manual_position:
            position = manual_position
            # Nếu đã chọn vị trí thủ công thì disable AI vision positioning
            auto_detect_position = False
        elif hasattr(legal, 'position'):
            position = legal.position
        else:
            position = "bottom_left"
        
        # Safe margin ~3% của chiều cao
        safe_margin = int(height * 0.03)
        
        # Tính toán vị trí x, y theo position
        position_map = {
            "bottom_left": (f"{safe_margin}", f"{height - safe_margin}"),
            "bottom_right": (f"w-text_w-{safe_margin}", f"{height - safe_margin}"),
            "bottom_center": (f"(w-text_w)/2", f"{height - safe_margin}"),
            "top_left": (f"{safe_margin}", f"{safe_margin + 20}"),
            "top_right": (f"w-text_w-{safe_margin}", f"{safe_margin + 20}"),
            "top": (f"(w-text_w)/2", f"{safe_margin + 20}"),
        }
        
        pos_x, pos_y = position_map.get(position, position_map["bottom_left"])
        
        if auto_detect_position:
            try:
                suggestions = self.scene_detector.suggest_legal_position(str(video_path))
                if suggestions.get("position") == "top":
                    pos_y = f"{safe_margin + 20}"
            except Exception as e:
                logger.warning(f"Could not auto-detect position: {e}")
        
        # Tạo filter cho text overlay
        filters = []
        
        # Font settings - chữ nhỏ cho legal (khoảng 2-3% chiều cao video)
        fontsize = legal.font_size if hasattr(legal, 'font_size') else 14
        # Scale theo chiều cao video, chuẩn là 1080p -> fontsize 14-18
        fontsize = max(10, int(fontsize * height / 1080))
        
        # Text cuối video
        if legal.display_at in ["end", "both"]:
            end_start = duration - legal.duration_on_screen
            text_filter = (
                f"drawtext=text='{self._escape_text(legal.text)}':"
                f"fontsize={fontsize}:"
                f"fontcolor=white:"
                f"borderw=1:"
                f"bordercolor=black:"
                f"x={pos_x}:"
                f"y={pos_y}:"
                f"enable='between(t,{end_start},{duration})'"
            )
            filters.append(text_filter)
        
        # Text đầu video (nếu cần)
        if legal.display_at in ["start", "both"] and legal.min_duration <= duration:
            text_filter = (
                f"drawtext=text='{self._escape_text(legal.text)}':"
                f"fontsize={fontsize}:"
                f"fontcolor=white:"
                f"borderw=1:"
                f"bordercolor=black:"
                f"x={pos_x}:"
                f"y={pos_y}:"
                f"enable='between(t,0,{legal.duration_on_screen})'"
            )
            filters.append(text_filter)
        
        # Text secondary (nếu có) - dòng thứ 2 bên dưới
        if legal.text_secondary and legal.display_at in ["end", "both"]:
            end_start = duration - legal.duration_on_screen
            # Dòng 2 cách dòng 1 khoảng fontsize + 5px
            secondary_y_offset = fontsize + 5
            text_filter = (
                f"drawtext=text='{self._escape_text(legal.text_secondary)}':"
                f"fontsize={int(fontsize * 0.9)}:"
                f"fontcolor=white:"
                f"borderw=1:"
                f"bordercolor=black:"
                f"x={pos_x}:"
                f"y={pos_y}+{secondary_y_offset}:"
                f"enable='between(t,{end_start},{duration})'"
            )
            filters.append(text_filter)
        
        if not filters:
            logger.info("No legal overlay needed")
            return str(video_path)
        
        # Build FFmpeg command
        filter_complex = ",".join(filters)
        
        cmd = [
            "ffmpeg", "-y",
            "-i", str(video_path),
            "-vf", filter_complex,
            "-c:a", "copy",
            str(output_path)
        ]
        
        logger.info(f"Running FFmpeg for legal overlay")
        result = subprocess.run(cmd, capture_output=True, text=True)
        
        if result.returncode != 0:
            logger.error(f"FFmpeg error: {result.stderr}")
            raise RuntimeError(f"FFmpeg failed: {result.stderr}")
        
        logger.info(f"Legal overlay added: {output_path}")
        return str(output_path)
    
    def _escape_text(self, text: str) -> str:
        """Escape text cho FFmpeg drawtext filter"""
        # Escape các ký tự đặc biệt
        text = text.replace("\\", "\\\\")
        text = text.replace("'", "'\\''")
        text = text.replace(":", "\\:")
        text = text.replace("%", "\\%")
        return text
    
    def add_legal_with_image(
        self,
        video_path: str,
        legal_image_path: str,
        position: str = "bottom",
        display_at: str = "end",
        display_duration: float = 3.0,
        output_path: Optional[str] = None
    ) -> str:
        """
        Thêm nội dung pháp lý từ file hình ảnh
        
        Args:
            video_path: Đường dẫn video input
            legal_image_path: Đường dẫn hình ảnh nội dung pháp lý
            position: Vị trí (bottom, top)
            display_at: Thời điểm hiển thị (start, end, both)
            display_duration: Thời gian hiển thị (giây)
            output_path: Đường dẫn output
            
        Returns:
            Đường dẫn video output
        """
        video_path = Path(video_path)
        duration = self._get_video_duration(str(video_path))
        width, height = self._get_video_dimensions(str(video_path))
        
        if output_path is None:
            output_path = self.output_dir / f"{video_path.stem}_legal{video_path.suffix}"
        
        # Tính vị trí overlay
        overlay_y = f"H-overlay_h-20" if position == "bottom" else "20"
        
        # Build filter
        filters = []
        
        if display_at in ["end", "both"]:
            end_start = duration - display_duration
            filters.append(
                f"overlay=(W-w)/2:{overlay_y}:enable='between(t,{end_start},{duration})'"
            )
        
        if display_at in ["start", "both"]:
            filters.append(
                f"overlay=(W-w)/2:{overlay_y}:enable='between(t,0,{display_duration})'"
            )
        
        if not filters:
            return str(video_path)
        
        # Cần scale image trước
        cmd = [
            "ffmpeg", "-y",
            "-i", str(video_path),
            "-i", legal_image_path,
            "-filter_complex",
            f"[1:v]scale={width}:-1[legal];[0:v][legal]{filters[0]}",
            "-c:a", "copy",
            str(output_path)
        ]
        
        result = subprocess.run(cmd, capture_output=True, text=True)
        
        if result.returncode != 0:
            logger.error(f"FFmpeg error: {result.stderr}")
            raise RuntimeError(f"FFmpeg failed: {result.stderr}")
        
        return str(output_path)
    
    def batch_add_legal(
        self,
        video_path: str,
        countries: List[str],
        media_type: MediaType = MediaType.SOCIAL,
        usage_type: UsageType = UsageType.SHAREABLE,
        output_dir: Optional[str] = None
    ) -> Dict[str, str]:
        """
        Tạo nhiều phiên bản với legal content khác nhau
        
        Args:
            video_path: Đường dẫn video input
            countries: Danh sách mã quốc gia
            media_type: Loại media
            usage_type: Hình thức sử dụng
            output_dir: Thư mục output
            
        Returns:
            Dict mapping country code to output path
        """
        video_path = Path(video_path)
        out_dir = Path(output_dir) if output_dir else self.output_dir
        
        results = {}
        for country in countries:
            try:
                output_path = out_dir / f"{video_path.stem}_{country}{video_path.suffix}"
                results[country] = self.add_legal_content(
                    str(video_path),
                    country,
                    media_type,
                    usage_type,
                    str(output_path)
                )
            except Exception as e:
                logger.error(f"Failed to add legal for {country}: {e}")
                results[country] = None
        
        return results
