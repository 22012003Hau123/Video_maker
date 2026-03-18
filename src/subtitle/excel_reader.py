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

        # Check if the first row contains common language codes
        lang_indicators = ["en", "fr", "sp", "nl", "jp", "ja", "vi", "de", "it", "es"]
        
        # Try to find the header row by looking for these indicators
        header_row_idx = 0
        found_header = False
        
        for i in range(min(5, len(self.df))):
            row_values = [str(v).lower().strip() for v in self.df.iloc[i].values]
            if any(lang in row_values for lang in lang_indicators):
                header_row_idx = i
                found_header = True
                break
        
        if found_header:
            # Set columns to this row and slice data
            new_cols = [str(v).lower().strip() for v in self.df.iloc[header_row_idx].values]
            # Replace empty column names with temporary ones
            new_cols = [c if c else f"col_{i}" for i, c in enumerate(new_cols)]
            self.df.columns = new_cols
            data_df = self.df.iloc[header_row_idx + 1:]
        else:
            data_df = self.df

        # Language mapping for frontend consistency
        LANG_MAP = {
            'sp': 'es',
            'jp': 'ja'
        }

        # Convert to list of dicts
        results = []
        for _, row in data_df.iterrows():
            entry = {}
            has_content = False
            for col in self.df.columns:
                val = str(row[col]).strip()
                # Map language codes
                clean_col = LANG_MAP.get(col, col)
                
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
