"""
Font Manager Module
Quản lý font chữ theo ngôn ngữ
"""
from pathlib import Path
from typing import Dict, Optional, List
from dataclasses import dataclass
import logging
import os

logger = logging.getLogger(__name__)

# Thư mục fonts mặc định
FONTS_DIR = Path(__file__).parent.parent.parent / "data" / "fonts"


@dataclass
class FontConfig:
    """Cấu hình font cho một ngôn ngữ"""
    language: str
    font_family: str
    font_path: Optional[str] = None
    font_size: int = 48
    font_color: str = "white"
    outline_color: str = "black"
    outline_width: int = 2
    bold: bool = False
    italic: bool = False


class FontManager:
    """Quản lý font theo ngôn ngữ"""
    
    def __init__(self, fonts_dir: Optional[str] = None):
        self.fonts_dir = Path(fonts_dir) if fonts_dir else FONTS_DIR
        self.custom_fonts: Dict[str, FontConfig] = {}
        
        # Determine OS
        import platform
        self.is_windows = platform.system() == "Windows"
        
        self._init_default_fonts()
        self._scan_fonts()
    
    def _init_default_fonts(self):
        """Khởi tạo font mặc định theo OS"""
        # Base Latin fonts (Arial is safe for both if ttf-mscorefonts installed on Linux)
        base_font = 'Arial'
        
        self.default_config_map = {
            'vi': FontConfig('vi', base_font, font_size=48),
            'en': FontConfig('en', base_font, font_size=48),
            'fr': FontConfig('fr', base_font, font_size=48),
            'es': FontConfig('es', base_font, font_size=48),
            'de': FontConfig('de', base_font, font_size=48),
        }
        
        if self.is_windows:
            # Windows Specific Fonts
            self.default_config_map.update({
                'ja': FontConfig('ja', 'Meiryo', font_size=44),
                'ko': FontConfig('ko', 'Malgun Gothic', font_size=44),
                'zh': FontConfig('zh', 'Microsoft YaHei', font_size=44),
                'ar': FontConfig('ar', 'Segoe UI', font_size=44),
                'th': FontConfig('th', 'Leelawadee UI', font_size=44),
            })
        else:
            # Linux/Mac Standard (Google Noto Fonts)
            self.default_config_map.update({
                'ja': FontConfig('ja', 'Noto Sans JP', font_size=44),
                'ko': FontConfig('ko', 'Noto Sans KR', font_size=44),
                'zh': FontConfig('zh', 'Noto Sans SC', font_size=44),
                'ar': FontConfig('ar', 'Noto Sans Arabic', font_size=44),
                'th': FontConfig('th', 'Noto Sans Thai', font_size=44),
            })
    
    def _scan_fonts(self):
        """Scan thư mục fonts để tìm font files"""
        if not self.fonts_dir.exists():
            logger.warning(f"Fonts directory not found: {self.fonts_dir}")
            return
            
        for font_file in self.fonts_dir.glob("**/*.ttf"):
            logger.debug(f"Found font: {font_file}")
        for font_file in self.fonts_dir.glob("**/*.otf"):
            logger.debug(f"Found font: {font_file}")
    
    def get_font_config(self, language: str) -> FontConfig:
        """Lấy cấu hình font cho ngôn ngữ"""
        lang = language.lower()[:2]  # Normalize: vi-VN -> vi
        
        # Ưu tiên custom fonts
        if lang in self.custom_fonts:
            return self.custom_fonts[lang]
        
        # Fallback to defaults
        if lang in self.default_config_map:
            return self.default_config_map[lang]
        
        # Default fallback
        logger.warning(f"No font config for language '{language}', using English default")
        return self.default_config_map['en']
    
    def set_custom_font(self, language: str, config: FontConfig):
        """Đặt font custom cho ngôn ngữ"""
        self.custom_fonts[language.lower()[:2]] = config
        logger.info(f"Set custom font for {language}: {config.font_family}")
    
    def get_ffmpeg_font_options(self, language: str, video_format: str = "16x9") -> Dict:
        """Lấy options cho FFmpeg subtitle filter"""
        config = self.get_font_config(language)
        
        # Điều chỉnh font size theo format video
        size_multipliers = {
            "1x1": 0.8,
            "4x5": 0.85,
            "9x16": 0.9,
            "16x9": 1.0,
        }
        multiplier = size_multipliers.get(video_format, 1.0)
        adjusted_size = int(config.font_size * multiplier)
        
        return {
            "fontfile": config.font_path or "",
            "fontname": config.font_family,
            "fontsize": adjusted_size,
            "fontcolor": config.font_color,
            "bordercolor": config.outline_color,
            "borderw": config.outline_width,
            "bold": 1 if config.bold else 0,
            "italic": 1 if config.italic else 0,
        }
    
    def get_available_languages(self) -> List[str]:
        """Lấy danh sách ngôn ngữ được hỗ trợ"""
        all_langs = set(self.default_config_map.keys())
        all_langs.update(self.custom_fonts.keys())
        return sorted(list(all_langs))
    
    def get_system_fonts(self) -> List[str]:
        """Lấy danh sách font có sẵn trên hệ thống"""
        fonts = []
        
        # Font directories
        font_dirs = [
            "C:/Windows/Fonts",
            os.path.expanduser("~/.fonts"),
            os.path.expanduser("~/.local/share/fonts"),
            "/usr/share/fonts",
            "/usr/local/share/fonts",
        ]
        
        for font_dir in font_dirs:
            if os.path.exists(font_dir):
                for root, dirs, files in os.walk(font_dir):
                    for file in files:
                        if file.endswith(('.ttf', '.otf', '.TTF', '.OTF')):
                            fonts.append(os.path.join(root, file))
        
        return fonts


# Singleton instance
_font_manager: Optional[FontManager] = None


def get_font_manager() -> FontManager:
    """Lấy singleton FontManager instance"""
    global _font_manager
    if _font_manager is None:
        _font_manager = FontManager()
    return _font_manager
