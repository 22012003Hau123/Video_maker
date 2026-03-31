"""
Legal Overlay Module
Thêm nội dung pháp lý lên video
"""
import subprocess
import os
import logging
from pathlib import Path
from typing import Optional, Dict, List, Tuple
import tempfile
from PIL import ImageFont, ImageDraw, Image

from src.utils.video import get_video_info
from src.utils.text import wrap_text_by_pixel, escape_ffmpeg_text
from .database import LegalContent, MediaType, UsageType, get_legal_database
from .scene_detector import SceneDetector

logger = logging.getLogger(__name__)
PROJECT_ROOT = Path(__file__).resolve().parents[2]
LOGO_DIR = PROJECT_ROOT / "data" / "logos"


class LegalOverlay:
    """Thêm nội dung pháp lý lên video"""
    
    def __init__(self, output_dir: Optional[str] = None):
        self.output_dir = Path(output_dir) if output_dir else Path("./outputs")
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.database = get_legal_database()
        self.scene_detector = SceneDetector()
    
    def _get_video_info_legacy(self, video_path: str) -> dict:
        """Lấy thông tin video (Dùng cho backward compatibility)"""
        return get_video_info(video_path)
    
    def add_legal_content(
        self,
        video_path: str,
        country_code: str,
        media_type: MediaType = MediaType.SOCIAL,
        usage_type: UsageType = UsageType.SHAREABLE,
        output_path: Optional[str] = None,
        auto_detect_position: bool = True,
        product_type: Optional[str] = None,
        manual_position: Optional[str] = None,
        sub_type: Optional[str] = "reels"
    ) -> str:
        """
        Thêm nội dung pháp lý vào video với styling chính xác và hỗ trợ Logo
        """
        video_path = Path(video_path)
        info = get_video_info(str(video_path))
        duration = info['duration']
        width, height = info['width'], info['height']
        ratio = width / height
        
        # Lấy danh sách nội dung pháp lý
        legal_list = self.database.get_legal_content(
            country_code, media_type, usage_type, duration, product_type, sub_type
        )
        
        if not legal_list:
            logger.warning(f"No legal content found for {country_code}")
            return str(video_path)
        
        if output_path is None:
            output_path = self.output_dir / f"{video_path.stem}_legal{video_path.suffix}"

        # Initialize FFmpeg command components
        from ..subtitle.font_manager import get_font_manager
        fm = get_font_manager()
        
        inputs = [str(video_path)]
        filters = []
        current_v = "[0:v]"
        
        for legal in legal_list:
            # 1. Detect Logo placeholder (e.g., *bedrinkaware logo*)
            logo_path = None
            clean_text = legal.text
            import re
            logo_match = re.search(r'\*([a-zA-Z0-9_\s]+)\s+logo\*', legal.text, re.IGNORECASE)
            if logo_match:
                logo_name = logo_match.group(1).strip().replace(" ", "")
                # Try common image extensions
                for ext in ['.png', '.jpg', '.jpeg']:
                    p = LOGO_DIR / f"{logo_name}{ext}"
                    if p.exists():
                        logo_path = str(p)
                        logger.info(f"Logo detected and matched: {logo_path}")
                        break
                if not logo_path:
                    logger.warning(f"Logo placeholder found but file not found under {LOGO_DIR} for base name: {logo_name}")
                # Remove placeholder from text
                clean_text = re.sub(r'\*.*logo\*', '', legal.text).strip()
                # Remove trailing newlines if any
                clean_text = clean_text.rstrip('\n')

            # 2. Get Styling from FontManager/Logic
            lang_code = country_code.lower()
            if lang_code == "uk": lang_code = "en"
            font_config = fm.get_font_config(lang_code)
            
            # Base styling logic (similar to existing but more robust)
            base_fontsize = 35
            base_y_offset = 30
            alignment = "center"
            
            if ratio < 0.6:  # 9x16
                base_fontsize = 42
                if sub_type == "stories":
                    alignment = "left"
                    base_y_offset = 250
                else: base_y_offset = 40
            elif ratio < 1.1: # 1x1
                base_fontsize = 35
                base_y_offset = 30
            else: # 16x9
                base_fontsize = 35
                base_y_offset = 30
            
            scale = height / 1080
            fontsize = int(base_fontsize * scale)
            x_margin = int(40 * scale)
            y_offset = int(base_y_offset * scale)

            # Shadow
            sx = max(1, int(2 * scale))
            sy = max(1, int(2 * scale))
            shadow_params = f"shadowcolor=black@0.5:shadowx={sx}:shadowy={sy}"
            
            logo_h = 0
            if logo_path:
                logo_h = int(30 * scale)
                # We'll use H-h-y_offset in the filter, but logo_h here is for text stacking
                logo_y = f"H-h-{y_offset}"

            # 4. Handle Text Rendering
            font_path = font_config.font_path or "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf"
            max_width = width - (x_margin * 2)
            wrapped_text = wrap_text_by_pixel(clean_text, font_path, fontsize, max_width)
            
            # Final X/Y calculations for Text
            if alignment == "left": pos_x = str(x_margin)
            else: pos_x = "(w-text_w)/2"
            
            # STACKING LOGIC: If logo exists, move text UP by logo height + padding
            if logo_h > 0:
                text_y_offset = y_offset + logo_h + int(15 * scale)
                pos_y = f"h-text_h-{text_y_offset}"
            else:
                pos_y = f"h-text_h-{y_offset}"

            # Manual override (overrides stacking logic if specific position chosen)
            if manual_position or (hasattr(legal, 'position') and legal.position != 'bottom'):
                position_map = {
                    "bottom_left": (f"{x_margin}", f"h-text_h-{x_margin}"),
                    "bottom_right": (f"w-text_w-{x_margin}", f"h-text_h-{x_margin}"),
                    "top_left": (f"{x_margin}", f"{x_margin}"),
                    "top_right": (f"w-text_w-{x_margin}", f"{x_margin}"),
                    "top": (f"(w-text_w)/2", f"{x_margin}"),
                    "center": (f"(w-text_w)/2", f"(h-text_h)/2"),
                    "bottom": (pos_x, pos_y)
                }
                pos_x, pos_y = position_map.get(manual_position or legal.position, (pos_x, pos_y))

            # Timing
            enable_str = ""
            if legal.display_at == "start": enable_str = f":enable='between(t,0,{legal.duration_on_screen})'"
            elif legal.display_at == "both": 
                enable_str = f":enable='between(t,0,{legal.duration_on_screen})+between(t,{duration-legal.duration_on_screen},{duration})'"
            elif legal.display_at == "end":
                enable_str = f":enable='between(t,{max(0, duration-legal.duration_on_screen)},{duration})'"
            
            if wrapped_text:
                out_pad = f"[v{len(filters)}]"
                filters.append(f"{current_v}drawtext=text='{escape_ffmpeg_text(wrapped_text)}':fontsize={fontsize}:fontcolor=white:fontfile='{font_path}':{shadow_params}:x={pos_x}:y={pos_y}{enable_str}{out_pad}")
                current_v = out_pad

            # 5. Handle Logo Overlay if detected
            if logo_path:
                input_idx = len(inputs)
                inputs.append(logo_path)
                out_pad = f"[v{len(filters)}]"
                # Rescale logo to logo_h before overlaying
                logo_label = f"logo{input_idx}"
                filters.append(f"[{input_idx}:v]scale=-1:{logo_h}[{logo_label}]")
                filters.append(f"{current_v}[{logo_label}]overlay=(W-w)/2:{logo_y}{enable_str}{out_pad}")
                current_v = out_pad

        if not filters:
            return str(video_path)
        
        filter_complex = ";".join(filters)
        cmd = ["ffmpeg", "-y", "-nostdin"]
        for inp in inputs: cmd.extend(["-i", inp])
        cmd.extend(["-filter_complex", filter_complex, "-map", current_v, "-map", "0:a?", "-c:v", "libx264", "-profile:v", "main", "-level", "3.1", "-pix_fmt", "yuv420p", "-c:a", "aac", "-movflags", "+faststart", str(output_path)])
        # 6. Execute FFmpeg
        # Determine display label for logs to avoid confusion
        display_label = sub_type
        if ratio > 1.7: display_label = "16x9 (Landscape)"
        elif ratio > 0.9: display_label = "1x1 (Square)"
        elif ratio < 0.6: display_label = f"9x16 ({sub_type})"
        
        logger.info(f"Running FFmpeg for legal overlay with style: {display_label}, {len(inputs)-1} logos")
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            stdin=subprocess.DEVNULL,
        )
        if result.returncode != 0:
            logger.error(f"FFmpeg failed: {result.stderr}")
            return str(video_path)
        return str(output_path)
    
    def _escape_text_legacy(self, text: str) -> str:
        """Escape text cho FFmpeg (Dùng cho backward compatibility)"""
        return escape_ffmpeg_text(text)

    def _wrap_text_legacy(self, text: str, font_path: str, fontsize: int, max_width: int) -> str:
        """Tự động ngắt dòng văn bản (Dùng cho backward compatibility)"""
        return wrap_text_by_pixel(text, font_path, fontsize, max_width)

    def _get_text_width_legacy(self, text: str, font) -> int:
        """Tính chiều rộng của văn bản (Legacy)"""
        try:
            bbox = font.getbbox(text)
            return bbox[2] - bbox[0]
        except AttributeError:
            return font.getsize(text)[0]
    
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
        info = get_video_info(str(video_path))
        duration = info['duration']
        width, height = info['width'], info['height']
        
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
            "-nostdin",
            "-i", str(video_path),
            "-i", legal_image_path,
            "-filter_complex",
            f"[1:v]scale={width}:-1[legal];[0:v][legal]{filters[0]}",
            "-c:a", "aac",
            str(output_path)
        ]
        
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            stdin=subprocess.DEVNULL,
        )
        
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
