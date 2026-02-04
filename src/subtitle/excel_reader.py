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
    """Một dòng phụ đề"""
    index: int
    text: str
    start_time: Optional[float] = None  # seconds
    end_time: Optional[float] = None    # seconds
    language: str = "vi"
    speaker: Optional[str] = None
    
    def duration(self) -> Optional[float]:
        """Tính thời lượng hiển thị phụ đề"""
        if self.start_time is not None and self.end_time is not None:
            return self.end_time - self.start_time
        return None


class ExcelReader:
    """Đọc và parse file Excel phụ đề"""
    
    # Mapping các tên cột phổ biến
    COLUMN_MAPPINGS = {
        'text': ['text', 'subtitle', 'phụ đề', 'nội dung', 'content', 'dialogue'],
        'start_time': ['start', 'start_time', 'bắt đầu', 'time_start', 'in'],
        'end_time': ['end', 'end_time', 'kết thúc', 'time_end', 'out'],
        'language': ['language', 'lang', 'ngôn ngữ', 'locale'],
        'speaker': ['speaker', 'người nói', 'character', 'nhân vật']
    }
    
    def __init__(self, file_path: str):
        self.file_path = Path(file_path)
        self.df: Optional[pd.DataFrame] = None
        self.subtitles: List[SubtitleEntry] = []
        
    def load(self) -> bool:
        """Load file Excel"""
        try:
            if self.file_path.suffix.lower() in ['.xlsx', '.xls']:
                self.df = pd.read_excel(self.file_path)
            elif self.file_path.suffix.lower() == '.csv':
                self.df = pd.read_csv(self.file_path)
            else:
                raise ValueError(f"Unsupported file format: {self.file_path.suffix}")
            
            logger.info(f"Loaded {len(self.df)} rows from {self.file_path}")
            return True
        except Exception as e:
            logger.error(f"Failed to load file: {e}")
            return False
    
    def _find_column(self, target: str) -> Optional[str]:
        """Tìm tên cột thực tế dựa trên mapping"""
        if self.df is None:
            return None
            
        mappings = self.COLUMN_MAPPINGS.get(target, [])
        columns_lower = {col.lower(): col for col in self.df.columns}
        
        for mapping in mappings:
            if mapping.lower() in columns_lower:
                return columns_lower[mapping.lower()]
        return None
    
    def _parse_time(self, value) -> Optional[float]:
        """Parse thời gian từ các format khác nhau"""
        if pd.isna(value):
            return None
            
        if isinstance(value, (int, float)):
            return float(value)
            
        # Parse format HH:MM:SS hoặc MM:SS
        if isinstance(value, str):
            value = value.strip()
            parts = value.replace(',', '.').split(':')
            try:
                if len(parts) == 3:  # HH:MM:SS
                    h, m, s = parts
                    return float(h) * 3600 + float(m) * 60 + float(s)
                elif len(parts) == 2:  # MM:SS
                    m, s = parts
                    return float(m) * 60 + float(s)
                else:
                    return float(value)
            except ValueError:
                return None
        return None
    
    def parse(self) -> List[SubtitleEntry]:
        """Parse DataFrame thành danh sách SubtitleEntry"""
        if self.df is None:
            self.load()
            
        if self.df is None or self.df.empty:
            return []
        
        # Tìm các cột
        text_col = self._find_column('text')
        start_col = self._find_column('start_time')
        end_col = self._find_column('end_time')
        lang_col = self._find_column('language')
        speaker_col = self._find_column('speaker')
        
        if text_col is None:
            # Nếu không tìm thấy cột text, lấy cột đầu tiên
            text_col = self.df.columns[0]
            logger.warning(f"Text column not found, using first column: {text_col}")
        
        self.subtitles = []
        for idx, row in self.df.iterrows():
            text = str(row[text_col]) if pd.notna(row[text_col]) else ""
            if not text.strip():
                continue
                
            entry = SubtitleEntry(
                index=len(self.subtitles),
                text=text.strip(),
                start_time=self._parse_time(row.get(start_col)) if start_col else None,
                end_time=self._parse_time(row.get(end_col)) if end_col else None,
                language=str(row.get(lang_col, 'vi')) if lang_col else 'vi',
                speaker=str(row.get(speaker_col)) if speaker_col and pd.notna(row.get(speaker_col)) else None
            )
            self.subtitles.append(entry)
        
        logger.info(f"Parsed {len(self.subtitles)} subtitle entries")
        return self.subtitles
    
    def get_languages(self) -> List[str]:
        """Lấy danh sách ngôn ngữ trong file"""
        if not self.subtitles:
            self.parse()
        return list(set(s.language for s in self.subtitles))
    
    def filter_by_language(self, language: str) -> List[SubtitleEntry]:
        """Lọc phụ đề theo ngôn ngữ"""
        if not self.subtitles:
            self.parse()
        return [s for s in self.subtitles if s.language.lower() == language.lower()]
    
    def to_srt(self, output_path: Optional[str] = None) -> str:
        """Export sang định dạng SRT"""
        if not self.subtitles:
            self.parse()
            
        def format_time(seconds: float) -> str:
            """Format seconds to SRT time format"""
            hours = int(seconds // 3600)
            minutes = int((seconds % 3600) // 60)
            secs = int(seconds % 60)
            millis = int((seconds % 1) * 1000)
            return f"{hours:02d}:{minutes:02d}:{secs:02d},{millis:03d}"
        
        srt_content = []
        for i, sub in enumerate(self.subtitles):
            start = sub.start_time if sub.start_time is not None else i * 3
            end = sub.end_time if sub.end_time is not None else start + 3
            
            srt_content.append(f"{i + 1}")
            srt_content.append(f"{format_time(start)} --> {format_time(end)}")
            srt_content.append(sub.text)
            srt_content.append("")
        
        srt_text = "\n".join(srt_content)
        
        if output_path:
            Path(output_path).write_text(srt_text, encoding='utf-8')
            logger.info(f"Saved SRT to {output_path}")
            
        return srt_text


def read_excel_subtitles(file_path: str) -> List[SubtitleEntry]:
    """Helper function để đọc phụ đề từ Excel"""
    reader = ExcelReader(file_path)
    return reader.parse()
