"""
Legal Content Database Module
Quản lý nội dung pháp lý theo quốc gia và loại media
"""
from typing import Dict, List, Optional
from dataclasses import dataclass, field
from enum import Enum
import json
from pathlib import Path
import logging

logger = logging.getLogger(__name__)


class MediaType(str, Enum):
    """Loại media"""
    SOCIAL = "social"           # Mạng xã hội
    TV = "tv"                   # Truyền hình
    OOH = "ooh"                 # Out-of-home (billboard, etc.)
    DIGITAL = "digital"         # Banner kỹ thuật số
    PRINT = "print"             # In ấn
    

class UsageType(str, Enum):
    """Hình thức sử dụng"""
    SHAREABLE = "shareable"     # Có thể chia sẻ
    NON_SHAREABLE = "non_shareable"  # Không chia sẻ
    PAID = "paid"               # Quảng cáo trả phí
    ORGANIC = "organic"         # Organic content


@dataclass
class LegalContent:
    """Nội dung pháp lý"""
    country_code: str           # VN, FR, US, etc.
    media_type: MediaType
    usage_type: UsageType
    text: str                   # Nội dung pháp lý
    text_secondary: Optional[str] = None  # Nội dung bổ sung
    min_duration: float = 0     # Thời lượng tối thiểu để áp dụng (giây)
    position: str = "bottom"    # Vị trí: top, bottom, center, bottom_left, etc.
    display_at: str = "end"     # Hiển thị: start, end, both, entire, first_5s, last_5s
    font_size: int = 24
    duration_on_screen: float = 3.0  # Thời gian hiển thị (giây)
    product_type: str = "alcohol"  # Loại sản phẩm: alcohol, tobacco, pharmaceutical, food
    font_family: str = "Arial"
    sub_type: Optional[str] = None    # reels, stories
    

@dataclass 
class CountryLegalConfig:
    """Cấu hình pháp lý theo quốc gia"""
    country_code: str
    country_name: str
    legal_contents: List[LegalContent] = field(default_factory=list)
    default_disclaimer: str = ""


class LegalDatabase:
    """Database nội dung pháp lý"""
    
    # Nội dung pháp lý mặc định cho ngành rượu
    DEFAULT_LEGAL_CONTENTS = {
        "US": CountryLegalConfig(
            country_code="US",
            country_name="United States",
            default_disclaimer="Drink Responsibly. Must be 21+ to consume.",
            legal_contents=[
                LegalContent(
                    country_code="US",
                    media_type=MediaType.SOCIAL,
                    usage_type=UsageType.SHAREABLE,
                    text="Drink Responsibly",
                    position="bottom",
                    display_at="end"
                ),
                LegalContent(
                    country_code="US",
                    media_type=MediaType.TV,
                    usage_type=UsageType.PAID,
                    text="Drink Responsibly. Must be 21+ to consume.",
                    min_duration=30,
                    position="bottom",
                    display_at="both"
                ),
            ]
        ),
    }
    
    def __init__(self, templates_dir: Optional[str] = None):
        self.templates_dir = Path(templates_dir) if templates_dir else Path(__file__).parent.parent.parent / "data" / "legal_templates"
        self.countries: Dict[str, CountryLegalConfig] = {}
        self._load_defaults()
        self._load_json_templates()
    
    def _load_defaults(self):
        """Load dữ liệu mặc định"""
        self.countries = self.DEFAULT_LEGAL_CONTENTS.copy()
    
    def _load_json_templates(self):
        """Load dữ liệu từ các file JSON template"""
        if not self.templates_dir.exists():
            logger.warning(f"Templates directory not found: {self.templates_dir}")
            return
        
        for json_file in self.templates_dir.glob("*.json"):
            try:
                with open(json_file, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                    country_code = data.get('country_code', json_file.stem.upper())
                    
                    # Parse rules into LegalContent objects
                    legal_contents = []
                    for rule in data.get('rules', []):
                        for duration_key, duration_config in rule.get('duration_rules', {}).items():
                            # Determine min_duration and sub_type
                            min_duration = 0
                            sub_type = None
                            
                            if '30s' in duration_key:
                                min_duration = 30
                            elif '60s' in duration_key:
                                min_duration = 60
                            elif duration_key == 'stories':
                                sub_type = 'stories'
                            elif duration_key == 'reels':
                                sub_type = 'reels'
                            elif duration_key == 'all':
                                min_duration = 0
                            
                            text = duration_config.get('text', '')
                            text_secondary = duration_config.get('secondary_text', None)
                            
                            # Map media types
                            media_platforms = rule.get('media_types', [])
                            if not media_platforms and rule.get('platform'):
                                media_platforms = [rule.get('platform')]
                            
                            if not media_platforms: media_platforms = ['social']

                            for platform in media_platforms:
                                media_type = MediaType.SOCIAL
                                if platform == 'tv' or platform == 'youtube':
                                    media_type = MediaType.TV
                                elif platform == 'print':
                                    media_type = MediaType.PRINT
                                elif platform == 'digital':
                                    media_type = MediaType.DIGITAL
                                elif platform == 'all':
                                    media_type = MediaType.SOCIAL # Fallback or handle specially
                                
                                # Determine display_at from timing or legacy fields
                                timing = duration_config.get('timing', 'end')
                                if timing == 'both_5s':
                                    display_at = 'both'
                                elif timing == 'first_5s':
                                    display_at = 'start'
                                elif timing == 'last_5s':
                                    display_at = 'end'
                                elif timing == 'entire_duration':
                                    display_at = 'entire'
                                else:
                                    # Legacy check
                                    if duration_config.get('show_at_start') and duration_config.get('show_at_end'):
                                        display_at = 'both'
                                    elif duration_config.get('show_at_start'):
                                        display_at = 'start'
                                    else:
                                        display_at = 'end'
                                
                                style = rule.get('style', {})
                                rule_type = rule.get('type', 'alcohol')
                                
                                usage_type_str = rule.get('usage_type', 'paid')
                                try:
                                    usage_type = UsageType(usage_type_str)
                                except:
                                    usage_type = UsageType.PAID

                                legal_contents.append(LegalContent(
                                    country_code=country_code,
                                    media_type=media_type,
                                    usage_type=usage_type,
                                    text=text,
                                    text_secondary=text_secondary,
                                    min_duration=min_duration,
                                    position=duration_config.get('position', 'bottom'),
                                    display_at=display_at,
                                    font_size=style.get('font_size', 14),
                                    duration_on_screen=duration_config.get('min_display_duration', 5.0),
                                    product_type=rule_type,
                                    font_family=style.get('font', data.get('default_font', 'Arial')),
                                    sub_type=sub_type
                                ))
                    
                    # Create or update country config
                    self.countries[country_code] = CountryLegalConfig(
                        country_code=country_code,
                        country_name=data.get('country_name', country_code),
                        default_disclaimer=legal_contents[0].text if legal_contents else '',
                        legal_contents=legal_contents
                    )
                    
                    logger.info(f"Loaded legal template: {country_code} with {len(legal_contents)} rules")
                    
            except Exception as e:
                logger.error(f"Failed to load {json_file}: {e}")
    
    def get_legal_content(
        self,
        country_code: str,
        media_type: MediaType,
        usage_type: UsageType,
        video_duration: float = 0,
        product_type: Optional[str] = None,
        sub_type: Optional[str] = None
    ) -> List[LegalContent]:
        """
        Lấy danh sách nội dung pháp lý phù hợp
        
        Args:
            country_code: Mã quốc gia (VN, FR, US, etc.)
            media_type: Loại media
            usage_type: Hình thức sử dụng
            video_duration: Thời lượng video (giây)
            product_type: Loại sản phẩm (alcohol, tobacco, pharmaceutical, food) - từ CLIP
            sub_type: stories, reels
            
        Returns:
            Danh sách LegalContent phù hợp
        """
        country_code = country_code.upper()
        if country_code not in self.countries:
            logger.warning(f"Country {country_code} not found in database")
            return None
        
        config = self.countries[country_code]
        
        # Map product_type to category names used in JSON templates
        product_map = {
            "alcohol": ["alcohol", "spirits", "wine", "beer"],
            "tobacco": ["tobacco", "smoking"],
            "pharmaceutical": ["pharmaceutical", "pharma", "medicine", "health"],
            "food": ["food", "infant_formula", "baby", "supplement"]
        }
        
        # Tìm legal content phù hợp
        matching = []
        for lc in config.legal_contents:
            # Check media type and usage type first
            if lc.media_type == media_type and lc.usage_type == usage_type:
                # Check sub_type if specified
                if sub_type and lc.sub_type and lc.sub_type != sub_type:
                    continue
                
                if video_duration >= lc.min_duration:
                    # Filter by product_type if specified
                    if product_type:
                        # Match if lc.product_type equals the detected type
                        if lc.product_type == product_type:
                            matching.append(lc)
                            logger.info(f"Matched: {product_type} -> {lc.text[:50]}...")
                    else:
                        # No product_type specified, use default (alcohol)
                        if lc.product_type == "alcohol":
                            matching.append(lc)
        
        if matching:
            # Tìm min_duration cao nhất mà video đạt được
            max_satisfied_min_dur = max(lc.min_duration for lc in matching)
            # Trả về TẤT CẢ các luật khớp với mốc thời gian đó (để hỗ trợ cả start và end)
            best_matches = [lc for lc in matching if lc.min_duration == max_satisfied_min_dur]
            return best_matches
        
        # Fallback: trả về disclaimer mặc định
        return [LegalContent(
            country_code=country_code,
            media_type=media_type,
            usage_type=usage_type,
            text=config.default_disclaimer,
            position="bottom",
            display_at="end"
        )]
    
    def get_countries(self) -> List[str]:
        """Lấy danh sách quốc gia được hỗ trợ"""
        return list(self.countries.keys())
    
    def add_country(self, config: CountryLegalConfig):
        """Thêm cấu hình quốc gia mới"""
        self.countries[config.country_code.upper()] = config
    
    def export_to_json(self, output_path: str):
        """Export database ra JSON"""
        data = {}
        for code, config in self.countries.items():
            data[code] = {
                "country_name": config.country_name,
                "default_disclaimer": config.default_disclaimer,
                "legal_contents": [
                    {
                        "media_type": lc.media_type.value,
                        "usage_type": lc.usage_type.value,
                        "text": lc.text,
                        "text_secondary": lc.text_secondary,
                        "min_duration": lc.min_duration,
                        "position": lc.position,
                        "display_at": lc.display_at,
                    }
                    for lc in config.legal_contents
                ]
            }
        
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(data, f, ensure_ascii=False, indent=2)


# Singleton instance
_database: Optional[LegalDatabase] = None


def get_legal_database() -> LegalDatabase:
    """Lấy singleton LegalDatabase instance"""
    global _database
    if _database is None:
        _database = LegalDatabase()
    return _database
