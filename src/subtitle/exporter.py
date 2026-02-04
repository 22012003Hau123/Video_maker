"""
Subtitle Exporter Module
Xuất file phụ đề SRT/ASS để chỉnh sửa
"""
import os
import logging
from pathlib import Path
from typing import List, Tuple, Optional
from datetime import timedelta

logger = logging.getLogger(__name__)


class SubtitleExporter:
    """Xuất phụ đề sang các định dạng chuẩn"""
    
    def __init__(self, output_dir: str = "outputs"):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
    
    def export_srt(
        self,
        subtitles: List[Tuple[float, float, str]],
        output_name: str
    ) -> str:
        """
        Xuất file SRT
        
        Args:
            subtitles: List of (start_time, end_time, text)
            output_name: Tên file output (không cần extension)
            
        Returns:
            Đường dẫn file SRT
        """
        output_path = self.output_dir / f"{output_name}.srt"
        
        with open(output_path, 'w', encoding='utf-8') as f:
            for idx, (start, end, text) in enumerate(subtitles, 1):
                start_str = self._format_srt_time(start)
                end_str = self._format_srt_time(end)
                
                f.write(f"{idx}\n")
                f.write(f"{start_str} --> {end_str}\n")
                f.write(f"{text}\n")
                f.write("\n")
        
        logger.info(f"Exported SRT: {output_path}")
        return str(output_path)
    
    def export_ass(
        self,
        subtitles: List[Tuple[float, float, str]],
        output_name: str,
        video_width: int = 1920,
        video_height: int = 1080,
        font_name: str = "Be Vietnam Pro",
        font_size: int = 48,
        primary_color: str = "&H00FFFFFF",  # White
        outline_color: str = "&H00000000",  # Black
        style_name: str = "Default"
    ) -> str:
        """
        Xuất file ASS với style tùy chỉnh
        
        Args:
            subtitles: List of (start_time, end_time, text)
            output_name: Tên file output
            video_width: Chiều rộng video
            video_height: Chiều cao video
            font_name: Tên font
            font_size: Kích thước font
            primary_color: Màu chữ (ASS format)
            outline_color: Màu viền (ASS format)
            style_name: Tên style
            
        Returns:
            Đường dẫn file ASS
        """
        output_path = self.output_dir / f"{output_name}.ass"
        
        # ASS Header
        ass_content = f"""[Script Info]
Title: Subtitle Export
ScriptType: v4.00+
PlayResX: {video_width}
PlayResY: {video_height}
WrapStyle: 0
ScaledBorderAndShadow: yes

[V4+ Styles]
Format: Name, Fontname, Fontsize, PrimaryColour, SecondaryColour, OutlineColour, BackColour, Bold, Italic, Underline, StrikeOut, ScaleX, ScaleY, Spacing, Angle, BorderStyle, Outline, Shadow, Alignment, MarginL, MarginR, MarginV, Encoding
Style: {style_name},{font_name},{font_size},{primary_color},&H000000FF,{outline_color},&H80000000,0,0,0,0,100,100,0,0,1,2,1,2,10,10,50,1

[Events]
Format: Layer, Start, End, Style, Name, MarginL, MarginR, MarginV, Effect, Text
"""
        
        # Add events
        for start, end, text in subtitles:
            start_str = self._format_ass_time(start)
            end_str = self._format_ass_time(end)
            # Convert newlines to ASS format
            text_formatted = text.replace('\n', '\\N')
            
            ass_content += f"Dialogue: 0,{start_str},{end_str},{style_name},,0,0,0,,{text_formatted}\n"
        
        with open(output_path, 'w', encoding='utf-8') as f:
            f.write(ass_content)
        
        logger.info(f"Exported ASS: {output_path}")
        return str(output_path)
    
    def export_both(
        self,
        subtitles: List[Tuple[float, float, str]],
        output_name: str,
        **ass_kwargs
    ) -> dict:
        """
        Xuất cả SRT và ASS
        
        Returns:
            Dict với paths: {"srt": path, "ass": path}
        """
        srt_path = self.export_srt(subtitles, output_name)
        ass_path = self.export_ass(subtitles, output_name, **ass_kwargs)
        
        return {
            "srt": srt_path,
            "ass": ass_path
        }
    
    def _format_srt_time(self, seconds: float) -> str:
        """Format time cho SRT: 00:00:00,000"""
        td = timedelta(seconds=seconds)
        hours, remainder = divmod(td.seconds, 3600)
        minutes, secs = divmod(remainder, 60)
        ms = int((seconds % 1) * 1000)
        return f"{hours:02d}:{minutes:02d}:{secs:02d},{ms:03d}"
    
    def _format_ass_time(self, seconds: float) -> str:
        """Format time cho ASS: 0:00:00.00"""
        td = timedelta(seconds=seconds)
        hours, remainder = divmod(td.seconds, 3600)
        minutes, secs = divmod(remainder, 60)
        cs = int((seconds % 1) * 100)  # Centiseconds
        return f"{hours}:{minutes:02d}:{secs:02d}.{cs:02d}"


def parse_srt(srt_content: str) -> List[Tuple[float, float, str]]:
    """
    Parse nội dung file SRT thành list subtitles
    
    Args:
        srt_content: Nội dung file SRT
        
    Returns:
        List of (start_time, end_time, text)
    """
    import re
    
    subtitles = []
    blocks = re.split(r'\n\n+', srt_content.strip())
    
    for block in blocks:
        lines = block.strip().split('\n')
        if len(lines) < 2:
            continue
        
        # Find timestamp line
        timestamp_line = None
        text_lines = []
        
        for i, line in enumerate(lines):
            if '-->' in line:
                timestamp_line = line
                text_lines = lines[i+1:]
                break
        
        if not timestamp_line:
            continue
        
        # Parse timestamps
        match = re.match(
            r'(\d{2}):(\d{2}):(\d{2})[,.](\d{3})\s*-->\s*(\d{2}):(\d{2}):(\d{2})[,.](\d{3})',
            timestamp_line
        )
        
        if match:
            start = (
                int(match.group(1)) * 3600 +
                int(match.group(2)) * 60 +
                int(match.group(3)) +
                int(match.group(4)) / 1000
            )
            end = (
                int(match.group(5)) * 3600 +
                int(match.group(6)) * 60 +
                int(match.group(7)) +
                int(match.group(8)) / 1000
            )
            text = '\n'.join(text_lines)
            
            subtitles.append((start, end, text))
    
    return subtitles


def get_exporter(output_dir: str = "outputs") -> SubtitleExporter:
    """Factory function"""
    return SubtitleExporter(output_dir)
