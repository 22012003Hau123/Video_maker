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
        product_type: Optional[str] = None
    ) -> str:
        """
        Thêm nội dung pháp lý vào ảnh
        
        Args:
            image_path: Đường dẫn ảnh input
            country_code: Mã quốc gia (VN, FR, US, HK)
            media_type: Loại media
            usage_type: Hình thức sử dụng
            output_path: Đường dẫn output (optional)
            position: Vị trí đặt text (optional, override template)
            font_size: Kích thước font (optional, override template)
            auto_detect_product: Tự động phát hiện loại sản phẩm bằng CLIP
            product_type: Loại sản phẩm (alcohol, tobacco, pharmaceutical, food)
            
        Returns:
            Đường dẫn ảnh output
        """
        image_path = Path(image_path)
        
        # Auto-detect product type using CLIP
        detected_product = None
        if auto_detect_product:
            try:
                from .product_detector import get_product_detector
                detector = get_product_detector()
                detected_product, confidence, label = detector.detect_product(str(image_path))
                logger.info(f"CLIP detected: {detected_product} ({label}) - {confidence:.2%}")
                product_type = detected_product if detected_product != "unknown" else product_type
            except Exception as e:
                logger.warning(f"CLIP detection failed: {e}")
        
        # Mở ảnh
        img = Image.open(image_path)
        img_width, img_height = img.size
        
        # Convert to RGB if needed (for PNG with transparency)
        if img.mode in ('RGBA', 'P'):
            background = Image.new('RGB', img.size, (255, 255, 255))
            if img.mode == 'RGBA':
                background.paste(img, mask=img.split()[3])
            else:
                background.paste(img)
            img = background
        
        # Lấy nội dung pháp lý (với product_type nếu có)
        legal = self.database.get_legal_content(
            country_code, media_type, usage_type, video_duration=0, product_type=product_type
        )
        
        if not legal:
            logger.warning(f"No legal content found for {country_code}")
            return str(image_path)
        
        # Output path
        if output_path is None:
            output_path = self.output_dir / f"{image_path.stem}_legal{image_path.suffix}"
        
        # Setup drawing
        draw = ImageDraw.Draw(img)
        
        # Font size - nhỏ cho ảnh (khoảng 1.5-2% chiều cao)
        if font_size is None:
            font_size = max(12, int(img_height * 0.018))
        
        font = self._get_font("Arial", font_size)
        
        # Lấy kích thước text
        text = legal.text
        bbox = draw.textbbox((0, 0), text, font=font)
        text_width = bbox[2] - bbox[0]
        text_height = bbox[3] - bbox[1]
        
        # Xác định vị trí
        pos = position if position else getattr(legal, 'position', 'bottom_left')
        x, y = self._calculate_position(
            img_width, img_height, 
            text_width, text_height, 
            pos
        )
        
        # Vẽ background semi-transparent (optional)
        padding = 5
        bg_box = [x - padding, y - padding, x + text_width + padding, y + text_height + padding]
        
        # Tạo overlay với độ trong suốt
        overlay = Image.new('RGBA', img.size, (0, 0, 0, 0))
        overlay_draw = ImageDraw.Draw(overlay)
        overlay_draw.rectangle(bg_box, fill=(0, 0, 0, 128))  # 50% opacity black
        
        # Convert img to RGBA để blend
        img_rgba = img.convert('RGBA')
        img_rgba = Image.alpha_composite(img_rgba, overlay)
        img = img_rgba.convert('RGB')
        
        # Vẽ lại text
        draw = ImageDraw.Draw(img)
        draw.text((x, y), text, font=font, fill=(255, 255, 255))
        
        # Lưu ảnh
        img.save(str(output_path), quality=95)
        logger.info(f"Legal overlay added to image: {output_path}")
        
        return str(output_path)
    
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
