"""
Excel Reader Module
Đọc file Excel phụ đề từ khách hàng
"""
import pandas as pd
from pathlib import Path
from typing import List, Dict, Optional
from dataclasses import dataclass
import logging

logger = logging.getLogger(__name__)

@dataclass
class SubtitleEntry:
    """Định nghĩa một đoạn phụ đề"""
    index: int
    start: float
    end: float
    text: str
    metadata: Optional[Dict] = None

class ExcelReader:
    """Đọc và parse file Excel phụ đề"""
    
    def __init__(self, file_path: str):
        self.file_path = Path(file_path)
        self.df = None

    def load(self):
        """Load file Excel hoặc CSV"""
        try:
            if self.file_path.suffix.lower() == '.csv':
                # Try common encodings for CSV
                encodings = ['utf-8', 'utf-8-sig', 'shift_jis', 'cp932', 'latin1']
                for enc in encodings:
                    try:
                        self.df = pd.read_csv(self.file_path, encoding=enc)
                        logger.info(f"Successfully loaded CSV with encoding: {enc}")
                        break
                    except (UnicodeDecodeError, Exception):
                        continue
                
                if self.df is None:
                    raise ValueError(f"Could not parse CSV {self.file_path} with any common encoding.")
            else:
                # Dùng openpyxl cho .xlsx hoặc xlrd cho .xls
                self.df = pd.read_excel(self.file_path)
            
            # Fill NaN with empty string
            self.df = self.df.fillna("")
            return self.df
        except Exception as e:
            logger.error(f"Error loading file {self.file_path}: {e}")
            raise

    def parse_horizontal_translations(self) -> List[Dict[str, str]]:
        """
        Parse Excel where each column is a language and each row is a translation segment.
        Returns a list of dicts like: [{'en': 'Hello', 'fr': 'Bonjour', ...}, ...]
        """
        if self.df is None:
            self.load()

        def normalize_lang_header(raw: str) -> str:
            token = str(raw or "").strip().lower()
            if not token:
                return ""

            compact = token.replace("-", "").replace("_", "").replace(" ", "")
            aliases = {
                "en": "en",
                "eng": "en",
                "english": "en",
                "usenglish": "en",
                "ukenglish": "uk",
                "gbenglish": "uk",
                "fr": "fr",
                "fra": "fr",
                "fre": "fr",
                "french": "fr",
                "es": "es",
                "sp": "es",
                "spa": "es",
                "spanish": "es",
                "nl": "nl",
                "dut": "nl",
                "dutch": "nl",
                "ja": "ja",
                "jp": "ja",
                "jpn": "ja",
                "japanese": "ja",
                "de": "de",
                "ger": "de",
                "deu": "de",
                "german": "de",
                "it": "it",
                "ita": "it",
                "italian": "it",
                "vi": "vi",
                "vie": "vi",
                "vn": "vi",
                "vietnamese": "vi",
                "vietnam": "vi",
                "vnese": "vi",
                "vietnamesevn": "vi",
                "zh": "zh",
                "cn": "zh",
                "chi": "zh",
                "zho": "zh",
                "chinese": "zh",
                "mandarin": "zh",
                "chinesesimplified": "zh",
                "simplifiedchinese": "zh",
                "zhcn": "zh",
                "zhs": "zh",
                "chinesetraditional": "zh-hant",
                "traditionalchinese": "zh-hant",
                "zhtw": "zh-hant",
                "zht": "zh-hant",
                "ko": "ko",
                "kor": "ko",
                "korean": "ko",
                "pt": "pt",
                "por": "pt",
                "portuguese": "pt",
                "ptbr": "pt-br",
                "brazilianportuguese": "pt-br",
                "ru": "ru",
                "rus": "ru",
                "russian": "ru",
                "ar": "ar",
                "ara": "ar",
                "arabic": "ar",
                "th": "th",
                "tha": "th",
                "thai": "th",
                "id": "id",
                "ind": "id",
                "indonesian": "id",
                "ms": "ms",
                "may": "ms",
                "malay": "ms",
            }
            return aliases.get(compact, token.replace(" ", "_"))

        # Check if the first row contains language-like indicators
        lang_indicators = {
            "en", "fr", "es", "sp", "nl", "jp", "ja", "vi", "de", "it", "zh", "cn", "ko", "pt", "ru", "ar"
        }
        
        # Try to find the header row by looking for these indicators
        header_row_idx = 0
        found_header = False
        
        for i in range(min(5, len(self.df))):
            row_values = [normalize_lang_header(v) for v in self.df.iloc[i].values]
            if any(v in lang_indicators for v in row_values):
                header_row_idx = i
                found_header = True
                break
        
        data_df = self.df
        normalized_cols: List[str] = [normalize_lang_header(c) for c in data_df.columns]
        if found_header:
            # Set columns to this row and slice data
            normalized_cols = [normalize_lang_header(v) for v in self.df.iloc[header_row_idx].values]
            # Replace empty column names with temporary ones
            normalized_cols = [c if c else f"col_{idx}" for idx, c in enumerate(normalized_cols)]
            data_df = self.df.iloc[header_row_idx + 1:]
        else:
            normalized_cols = [c if c else f"col_{idx}" for idx, c in enumerate(normalized_cols)]

        # Convert to list of dicts
        results = []
        for _, row in data_df.iterrows():
            entry = {}
            has_content = False

            for idx, clean_col in enumerate(normalized_cols):
                if clean_col.startswith("col_"):
                    # Skip columns without a recognized header.
                    continue

                cell_val = row.iloc[idx] if idx < len(row) else ""
                val = str(cell_val).strip()

                # If duplicate language headers exist, keep first non-empty value.
                if clean_col in entry and entry[clean_col] and not val:
                    continue
                if clean_col in entry and not entry[clean_col] and val:
                    entry[clean_col] = val
                elif clean_col not in entry:
                    entry[clean_col] = val

                if val:
                    has_content = True
            
            if has_content:
                results.append(entry)
                
        return results

def read_excel_subtitles(file_path: str):
    """Helper function to read subtitles from Excel"""
    reader = ExcelReader(file_path)
    return reader.parse_horizontal_translations()
