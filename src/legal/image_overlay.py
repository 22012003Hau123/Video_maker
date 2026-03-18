"""
Image Legal Overlay Module
Thêm nội dung pháp lý lên ảnh tĩnh (print, banner, cover)
"""
import subprocess
import os
import logging
from pathlib import Path
from typing import Optional, Dict, List, Tuple
from PIL import Image, ImageDraw, ImageFont

from src.utils.text import wrap_text_by_pixel
from .database import LegalContent, MediaType, UsageType, get_legal_database

logger = logging.getLogger(__name__)


class ImageLegalOverlay:
    """Thêm nội dung pháp lý lên ảnh tĩnh"""
    
    def __init__(self, output_dir: Optional[str] = None):
        self.output_dir = Path(output_dir) if output_dir else Path("./outputs")
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.database = get_legal_database()
        
        # Default fonts - có thể tùy chỉnh
        self.default_fonts = {
            "en": "Arial",
            "vi": "Arial", 
            "fr": "Arial",
            "zh": "Noto Sans CJK SC"
        }
    
    def _get_font(self, font_name: str, size: int) -> ImageFont.FreeTypeFont:
        """Lấy font với fallback"""
        font_paths = [
            f"/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
            f"/usr/share/fonts/truetype/noto/NotoSansCJK-Regular.ttc",
            f"/usr/share/fonts/truetype/liberation/LiberationSans-Regular.ttf",
        ]
        
        for font_path in font_paths:
            if os.path.exists(font_path):
                try:
                    return ImageFont.truetype(font_path, size)
                except:
                    continue
        
        # Fallback to default
        return ImageFont.load_default()
    
    def _calculate_position(
        self, 
        img_width: int, 
        img_height: int, 
        text_width: int, 
        text_height: int,
        position: str,
        margin_percent: float = 0.03
    ) -> Tuple[int, int]:
        """Tính toán vị trí đặt text"""
        margin_x = int(img_width * margin_percent)
        margin_y = int(img_height * margin_percent)
        
        positions = {
            "bottom_left": (margin_x, img_height - margin_y - text_height),
            "bottom_right": (img_width - margin_x - text_width, img_height - margin_y - text_height),
            "bottom_center": ((img_width - text_width) // 2, img_height - margin_y - text_height),
            "top_left": (margin_x, margin_y),
            "top_right": (img_width - margin_x - text_width, margin_y),
            "top_center": ((img_width - text_width) // 2, margin_y),
        }
        
        return positions.get(position, positions["bottom_left"])
    
    def add_legal_content(
        self,
        image_path: str,
        country_code: str,
        media_type: MediaType = MediaType.PRINT,
        usage_type: UsageType = UsageType.PAID,
        output_path: Optional[str] = None,
        position: Optional[str] = None,
        font_size: Optional[int] = None,
        auto_detect_product: bool = False,
        product_type: Optional[str] = None,
        sub_type: Optional[str] = "reels"
    ) -> str:
        """
        Thêm nội dung pháp lý vào ảnh với styling chính xác
        """
        image_path = Path(image_path)
        
        # Mở ảnh
        img = Image.open(image_path)
        img_width, img_height = img.size
        ratio = img_width / img_height
        
        # Convert to RGB if needed
        if img.mode in ('RGBA', 'P'):
            background = Image.new('RGB', img.size, (255, 255, 255))
            if img.mode == 'RGBA':
                background.paste(img, mask=img.split()[3])
            else:
                background.paste(img)
            img = background
        
        # Lấy danh sách nội dung pháp lý
        legal_list = self.database.get_legal_content(
            country_code, media_type, usage_type, video_duration=0, product_type=product_type, sub_type=sub_type
        )
        
        if not legal_list:
            return str(image_path)
        
        if output_path is None:
            output_path = self.output_dir / f"{image_path.stem}_legal{image_path.suffix}"
        
        # Build list of overlays to draw
        draw = ImageDraw.Draw(img)
        
        # We need to track the Y position if multiple are at the same location, but for images it's simpler
        # Just loop through and draw
        for legal in legal_list:
            # Determine styling based on aspect ratio
            # (Reset to base for each item unless position is specified)
            cur_font_size_pt = 35
            cur_bottom_offset = 30
            cur_alignment = "center"
            
            if ratio < 0.6:  # 9x16
                cur_font_size_pt = 42
                if sub_type == "stories":
                    cur_alignment = "left"
                    cur_bottom_offset = 250
                else: # reels
                    cur_alignment = "center"
                    cur_bottom_offset = 40
            elif ratio < 1.1: # 1x1
                cur_font_size_pt = 35
                cur_bottom_offset = 30
            else: # 16x9
                cur_font_size_pt = 35
                cur_bottom_offset = 30

            # Determine font path
            font_path = "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf"
            if getattr(legal, 'font_family', '').lower() == "oswald":
                oswald_paths = ["/usr/share/fonts/truetype/oswald/Oswald-Bold.ttf", "/usr/share/fonts/truetype/google-fonts/Oswald-Bold.ttf", "data/fonts/Oswald-Bold.ttf"]
                for p in oswald_paths:
                    if Path(p).exists():
                        font_path = p
                        break
            
            # Scale
            scale = img_height / 1080
            fontsize = int(cur_font_size_pt * scale)
            y_offset = int(cur_bottom_offset * scale)
            x_margin = int(40 * scale)
            
            try:
                font = ImageFont.truetype(font_path, fontsize)
            except:
                font = ImageFont.load_default()
            
            # Wrap text
            max_width = img_width - (x_margin * 2)
            wrapped_text = wrap_text_by_pixel(legal.text, font_path, fontsize, max_width)
            
            bbox = ImageDraw.Draw(Image.new('RGB', (1,1))).textbbox((0, 0), wrapped_text, font=font)
            text_width = bbox[2] - bbox[0]
            text_height = bbox[3] - bbox[1]
            
            x = (img_width - text_width) // 2 if cur_alignment == "center" else x_margin
            y = img_height - text_height - y_offset

            # Manual override or specific rule position
            if position or (hasattr(legal, 'position') and legal.position != 'bottom'):
                position_map = {
                    "bottom_left": (x_margin, img_height - text_height - x_margin),
                    "bottom_right": (img_width - text_width - x_margin, img_height - text_height - x_margin),
                    "top_left": (x_margin, x_margin),
                    "top_right": (img_width - text_width - x_margin, x_margin),
                    "top": ((img_width - text_width) // 2, x_margin),
                    "center": ((img_width - text_width) // 2, (img_height - text_height) // 2),
                    "bottom": (x, y)
                }
                x, y = position_map.get(position or legal.position, (x, y))

            # Draw shadow
            shadow_offset = int(2 * scale)
            shadow_draw = ImageDraw.Draw(img)
            shadow_draw.text((x + shadow_offset, y + shadow_offset), wrapped_text, font=font, fill=(0, 0, 0, 200)) # 80% shadow

            # Draw main
            draw.text((x, y), wrapped_text, font=font, fill=(255, 255, 255))
            
            # Draw secondary
            if getattr(legal, 'text_secondary', None):
                wrapped_sec = wrap_text_by_pixel(legal.text_secondary, font_path, fontsize, max_width)
                try:
                    bbox_main = draw.textbbox((x, y), wrapped_text, font=font)
                    y_sec = bbox_main[1] - (fontsize * wrapped_sec.count('\n') + fontsize) - 15
                    draw.text((x, y_sec), wrapped_sec, font=font, fill=(255, 255, 255))
                except:
                    y_sec = y - fontsize - 15
                    draw.text((x, y_sec), wrapped_sec, font=font, fill=(255, 255, 255))
        
        img.save(str(output_path), quality=95)
        return str(output_path)
        
        # Lưu ảnh
            logger.info(f"Legal overlay added to image: {output_path}")
        
        return str(output_path)

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
    
    def batch_add_legal(
        self,
        image_path: str,
        countries: List[str],
        media_type: MediaType = MediaType.PRINT,
        usage_type: UsageType = UsageType.PAID,
        output_dir: Optional[str] = None
    ) -> Dict[str, str]:
        """
        Tạo nhiều phiên bản ảnh với legal content khác nhau
        
        Args:
            image_path: Đường dẫn ảnh input
            countries: Danh sách mã quốc gia
            media_type: Loại media
            usage_type: Hình thức sử dụng
            output_dir: Thư mục output
            
        Returns:
            Dict mapping country code to output path
        """
        image_path = Path(image_path)
        out_dir = Path(output_dir) if output_dir else self.output_dir
        
        results = {}
        for country in countries:
            try:
                output_path = out_dir / f"{image_path.stem}_{country}{image_path.suffix}"
                results[country] = self.add_legal_content(
                    str(image_path),
                    country,
                    media_type,
                    usage_type,
                    str(output_path)
                )
            except Exception as e:
                logger.error(f"Failed to add legal for {country}: {e}")
                results[country] = None
        
        return results
    
    def add_legal_with_ffmpeg(
        self,
        image_path: str,
        country_code: str,
        output_path: Optional[str] = None,
        position: str = "bottom_left"
    ) -> str:
        """
        Thêm legal content bằng FFmpeg (alternative method)
        Phù hợp cho batch processing lớn
        """
        image_path = Path(image_path)
        
        # Lấy nội dung pháp lý
        legal = self.database.get_legal_content(
            country_code, MediaType.PRINT, UsageType.PAID, 0
        )
        
        if not legal:
            return str(image_path)
        
        if output_path is None:
            output_path = self.output_dir / f"{image_path.stem}_legal{image_path.suffix}"
        
        # Escape text cho FFmpeg
        text = legal.text.replace("'", "'\\''").replace(":", "\\:")
        
        # Position mapping for FFmpeg
        pos_map = {
            "bottom_left": "x=20:y=h-th-20",
            "bottom_right": "x=w-tw-20:y=h-th-20",
            "bottom_center": "x=(w-tw)/2:y=h-th-20",
            "top_left": "x=20:y=20",
            "top_right": "x=w-tw-20:y=20",
        }
        
        pos_filter = pos_map.get(position, pos_map["bottom_left"])
        
        cmd = [
            "ffmpeg", "-y",
            "-i", str(image_path),
            "-vf", f"drawtext=text='{text}':fontsize=14:fontcolor=white:borderw=1:bordercolor=black:{pos_filter}",
            str(output_path)
        ]
        
        result = subprocess.run(cmd, capture_output=True, text=True)
        
        if result.returncode != 0:
            logger.error(f"FFmpeg error: {result.stderr}")
            raise RuntimeError(f"FFmpeg failed: {result.stderr}")
        
        return str(output_path)
