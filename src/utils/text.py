"""
Text Utilities
Common functions for text wrapping, line breaking, and FFmpeg escaping
"""
from typing import List
from PIL import ImageFont

def escape_ffmpeg_text(text: str) -> str:
    """
    Escape text for FFmpeg drawtext filter
    """
    if not text:
        return ""
    # Characters that need special escaping in FFmpeg drawtext:
    # \ ' : [ ] , ; = 
    # Also handle newlines if needed, but usually FFmpeg handles \n in the string
    escaped = text.replace('\\', '\\\\')
    escaped = escaped.replace("'", "\\'")
    escaped = escaped.replace(':', '\\:')
    # Some environments need double escaping for brackets or colons depending on the shell/filter parser
    return escaped

def smart_line_break(text: str, max_chars_per_line: int = 42, max_lines: int = 2) -> str:
    """
    Tạo ngắt dòng thông minh cho phụ đề, ưu tiên ngắt tại các dấu câu
    """
    if not text or len(text) <= max_chars_per_line:
        return text

    # Keep backward compatibility for callers while making behavior explicit.
    if max_lines <= 1:
        return text

    # Danh sách các điểm ưu tiên ngắt (dấu câu)
    delimiters = [". ", "! ", "? ", ": ", "; ", ", ", " - ", " – ", " — "]
    
    # Thử tìm điểm ngắt tối ưu gần giữa văn bản
    best_break = -1
    min_dist_from_center = len(text)
    center = len(text) / 2

    for delim in delimiters:
        start_search = 0
        while True:
            idx = text.find(delim, start_search)
            if idx == -1:
                break
            
            # Tính khoảng cách từ điểm này đến trung tâm
            dist = abs((idx + len(delim)/2) - center)
            # Điểm ngắt phải nằm trong phạm vi hợp lý (không quá gần biên)
            if dist < min_dist_from_center and 10 < idx < len(text) - 10:
                min_dist_from_center = dist
                best_break = idx + len(delim)  # Ngắt sau delimiter

            start_search = idx + 1

    if best_break != -1:
        part1 = text[:best_break].strip()
        part2 = text[best_break:].strip()
        return f"{part1}\n{part2}"

    # Nếu không tìm thấy dấu câu, ngắt tại khoảng trắng gần giữa nhất
    words = text.split()
    if len(words) < 2:
        return text

    mid_word_idx = len(words) // 2
    part1 = " ".join(words[:mid_word_idx])
    part2 = " ".join(words[mid_word_idx:])
    return f"{part1}\n{part2}"

def wrap_text_by_pixel(text: str, font_path: str, fontsize: int, max_width: int) -> str:
    """
    Tự động ngắt dòng văn bản dựa trên chiều rộng pixel cho phép, 
    bảo toàn các dấu xuống dòng có sẵn.
    """
    if not text:
        return ""
        
    try:
        font = ImageFont.truetype(font_path, fontsize)
    except Exception:
        # Fallback to default if font fails
        font = ImageFont.load_default()

    # Pre-split by existing newlines to preserve manual formatting
    paragraphs = text.split('\n')
    wrapped_paragraphs = []

    for paragraph in paragraphs:
        if not paragraph.strip():
            wrapped_paragraphs.append("")
            continue
            
        words = paragraph.split(' ')
        lines = []
        current_line = []

        for word in words:
            # Check length of current line + next word
            test_line = ' '.join(current_line + [word])
            
            # Get text width (Pillow 10+ uses getlength)
            try:
                w = font.getlength(test_line)
            except AttributeError:
                # Older Pillow versions
                w, _ = font.getsize(test_line)
                
            if w <= max_width:
                current_line.append(word)
            else:
                if current_line:
                    lines.append(' '.join(current_line))
                    current_line = [word]
                else:
                    # Single word is too long, must force break or let it overflow
                    lines.append(word)
                    current_line = []
        
        if current_line:
            lines.append(' '.join(current_line))
            
        wrapped_paragraphs.append('\n'.join(lines))

    return '\n'.join(wrapped_paragraphs)

def format_srt_time(seconds: float) -> str:
    """Format time cho SRT: 00:00:00,000"""
    h = int(seconds // 3600)
    m = int((seconds % 3600) // 60)
    s = int(seconds % 60)
    ms = int((seconds % 1) * 1000)
    return f"{h:02d}:{m:02d}:{s:02d},{ms:03d}"

def generate_srt_content(subtitles: List) -> str:
    """
    Tạo nội dung file SRT từ danh sách phụ đề
    Supports (start, end, text) or (start, end, text, index)
    """
    lines = []
    for i, sub in enumerate(subtitles, 1):
        start, end, text = sub[:3]
        lines.append(f"{i}")
        lines.append(f"{format_srt_time(start)} --> {format_srt_time(end)}")
        lines.append(text)
        lines.append("")
    return "\n".join(lines)
