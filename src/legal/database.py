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


class MediaType(Enum):
    """Loại media"""
    SOCIAL = "social"           # Mạng xã hội
    TV = "tv"                   # Truyền hình
    OOH = "ooh"                 # Out-of-home (billboard, etc.)
    DIGITAL = "digital"         # Banner kỹ thuật số
    PRINT = "print"             # In ấn
    

class UsageType(Enum):
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
    position: str = "bottom"    # Vị trí: top, bottom, full_screen
    display_at: str = "end"     # Hiển thị: start, end, both
    font_size: int = 24
    duration_on_screen: float = 3.0  # Thời gian hiển thị (giây)
    

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
        "VN": CountryLegalConfig(
            country_code="VN",
            country_name="Việt Nam",
            default_disclaimer="Uống rượu bia có hại cho sức khỏe. Cấm bán cho người dưới 18 tuổi.",
            legal_contents=[
                LegalContent(
                    country_code="VN",
                    media_type=MediaType.SOCIAL,
                    usage_type=UsageType.SHAREABLE,
                    text="Uống rượu bia có hại cho sức khỏe",
                    min_duration=0,
                    position="bottom",
                    display_at="end"
                ),
                LegalContent(
                    country_code="VN",
                    media_type=MediaType.TV,
                    usage_type=UsageType.PAID,
                    text="Uống rượu bia có hại cho sức khỏe. Cấm bán cho người dưới 18 tuổi.",
                    text_secondary="Đã uống rượu bia, không lái xe.",
                    min_duration=30,
                    position="bottom",
                    display_at="both"
                ),
            ]
        ),
        "FR": CountryLegalConfig(
            country_code="FR",
            country_name="France",
            default_disclaimer="L'abus d'alcool est dangereux pour la santé. À consommer avec modération.",
            legal_contents=[
                LegalContent(
                    country_code="FR",
                    media_type=MediaType.SOCIAL,
                    usage_type=UsageType.SHAREABLE,
                    text="L'abus d'alcool est dangereux pour la santé",
                    position="bottom",
                    display_at="end"
                ),
                LegalContent(
                    country_code="FR",
                    media_type=MediaType.TV,
                    usage_type=UsageType.PAID,
                    text="L'abus d'alcool est dangereux pour la santé. À consommer avec modération.",
                    min_duration=30,
                    position="bottom",
                    display_at="both"
                ),
            ]
        ),
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
        "HK": CountryLegalConfig(
            country_code="HK",
            country_name="Hong Kong",
            default_disclaimer="Drinking alcohol is harmful to health.",
            legal_contents=[
                LegalContent(
                    country_code="HK",
                    media_type=MediaType.SOCIAL,
                    usage_type=UsageType.SHAREABLE,
                    text="Drinking alcohol is harmful to health",
                    position="bottom",
                    display_at="end"
                ),
            ]
        ),
    }
    
    def __init__(self, data_path: Optional[str] = None):
        self.data_path = Path(data_path) if data_path else None
        self.countries: Dict[str, CountryLegalConfig] = {}
        self._load_defaults()
        if self.data_path:
            self._load_custom_data()
    
    def _load_defaults(self):
        """Load dữ liệu mặc định"""
        self.countries = self.DEFAULT_LEGAL_CONTENTS.copy()
    
    def _load_custom_data(self):
        """Load dữ liệu custom từ file JSON"""
        if self.data_path and self.data_path.exists():
            try:
                with open(self.data_path, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                    # Parse and merge with defaults
                    for country_code, config in data.items():
                        if country_code in self.countries:
                            # Merge
                            pass
                        else:
                            # Add new
                            pass
            except Exception as e:
                logger.error(f"Failed to load custom legal data: {e}")
    
    def get_legal_content(
        self,
        country_code: str,
        media_type: MediaType,
        usage_type: UsageType,
        video_duration: float = 0
    ) -> Optional[LegalContent]:
        """
        Lấy nội dung pháp lý phù hợp
        
        Args:
            country_code: Mã quốc gia (VN, FR, US, etc.)
            media_type: Loại media
            usage_type: Hình thức sử dụng
            video_duration: Thời lượng video (giây)
            
        Returns:
            LegalContent phù hợp hoặc None
        """
        country_code = country_code.upper()
        if country_code not in self.countries:
            logger.warning(f"Country {country_code} not found in database")
            return None
        
        config = self.countries[country_code]
        
        # Tìm legal content phù hợp
        matching = []
        for lc in config.legal_contents:
            if lc.media_type == media_type and lc.usage_type == usage_type:
                if video_duration >= lc.min_duration:
                    matching.append(lc)
        
        if matching:
            # Chọn content có min_duration cao nhất phù hợp
            return max(matching, key=lambda x: x.min_duration)
        
        # Fallback: trả về disclaimer mặc định
        return LegalContent(
            country_code=country_code,
            media_type=media_type,
            usage_type=usage_type,
            text=config.default_disclaimer,
            position="bottom",
            display_at="end"
        )
    
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
