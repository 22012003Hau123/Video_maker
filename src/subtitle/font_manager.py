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
    shadow_color: str = "&H80000000"  # ASS format (50% opacity black)
    shadow_width: float = 2.0
    shadow_blur: float = 6.0
    bold: bool = False
    italic: bool = False
    scale: float = 1.0


class FontManager:
    """Quản lý font theo ngôn ngữ"""
    
    # Mapping font mặc định theo ngôn ngữ 
    # Helvetica thường không có trên Linux, dùng DejaVu Sans làm fallback tương đương
    DEFAULT_FONTS: Dict[str, FontConfig] = {
        'en': FontConfig(language='en', font_family='DejaVu Sans', font_size=48),
        'uk': FontConfig(language='uk', font_family='DejaVu Sans', font_size=48),
        'de': FontConfig(language='de', font_family='DejaVu Sans', font_size=48),
        'it': FontConfig(language='it', font_family='DejaVu Sans', font_size=48),
        'fr': FontConfig(language='fr', font_family='DejaVu Sans', font_size=48),
        'es': FontConfig(language='es', font_family='DejaVu Sans', font_size=48),
        'nl': FontConfig(language='nl', font_family='DejaVu Sans', font_size=48),
        'ja': FontConfig(language='ja', font_family='Noto Sans CJK JP', font_size=48), 
        'vi': FontConfig(language='vi', font_family='DejaVu Sans', font_size=48),
    }
    
    def __init__(self, fonts_dir: Optional[str] = None):
        self.fonts_dir = Path(fonts_dir) if fonts_dir else FONTS_DIR
        self.custom_fonts: Dict[str, FontConfig] = {}
    
    def get_font_config(self, language: str) -> FontConfig:
        """Lấy cấu hình font cho ngôn ngữ"""
        lang = language.lower()
        if lang in ["en", "us", "uk"]: 
            # Handle regional variants
            if "uk" in lang: config = self.DEFAULT_FONTS['uk']
            else: config = self.DEFAULT_FONTS['en']
        else:
            lang_short = lang[:2]
            if lang_short in self.DEFAULT_FONTS:
                config = self.DEFAULT_FONTS[lang_short]
            else:
                config = self.DEFAULT_FONTS['en']
        
        # Check if custom font exists in data/fonts
        if self.fonts_dir.exists():
            # Priority 1: language-specific font (e.g. ja.ttf)
            lang_font = self.fonts_dir / f"{lang}.ttf"
            if lang_font.exists():
                config.font_path = str(lang_font)
                config.font_family = lang_font.stem
            
            # Priority 2: global custom font (e.g. custom.ttf)
            elif (self.fonts_dir / "custom.ttf").exists():
                config.font_path = str(self.fonts_dir / "custom.ttf")
                config.font_family = "custom"
                
        return config
    
    def get_ffmpeg_font_options(self, language: str, video_format: str = "16x9") -> Dict:
        """Lấy options cho FFmpeg subtitle filter"""
        config = self.get_font_config(language)
        
        # Override parameters based on user spec for each format
        # Default behavior for non-specified formats
        adjusted_size = config.font_size
        scale = 1.0
        
        if video_format == "16x9":
            adjusted_size = 48
        elif video_format == "9x16":
            adjusted_size = 55
        elif video_format == "1x1":
            adjusted_size = 48
            scale = 1.62  # Derived from user screenshot
        elif video_format == "4x5":
            adjusted_size = 48
            
        return {
            "fontname": config.font_family,
            "fontsize": adjusted_size,
            "fontcolor": config.font_color,
            "outline": 0,  # Disabled per spec
            "shadow": config.shadow_width,
            "blur": config.shadow_blur,
            "backcolor": config.shadow_color,
            "scalex": int(100 * scale),
            "scaley": int(100 * scale),
            "bold": 1 if config.bold else 0,
            "italic": 1 if config.italic else 0,
        }
    
    def get_available_languages(self) -> List[str]:
        """Lấy danh sách ngôn ngữ được hỗ trợ"""
        all_langs = set(self.DEFAULT_FONTS.keys())
        all_langs.update(self.custom_fonts.keys())
        return sorted(list(all_langs))
    
    def get_system_fonts(self) -> List[str]:
        """Lấy danh sách font có sẵn trên hệ thống"""
        fonts = []
        
        # Linux font directories
        font_dirs = [
            "/usr/share/fonts",
            "/usr/local/share/fonts",
            os.path.expanduser("~/.fonts"),
            os.path.expanduser("~/.local/share/fonts"),
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
