"""
Timing Sync Module
Sử dụng Whisper AI để canh timing phụ đề theo lời thoại
"""
import os
import logging
from pathlib import Path
from typing import List, Optional, Tuple
from dataclasses import dataclass
import json

logger = logging.getLogger(__name__)

# Import whisper (optional - use OpenAI API if not available)
try:
    import whisper
    WHISPER_LOCAL = True
except ImportError:
    WHISPER_LOCAL = False
    logger.info("Local Whisper not available, will use OpenAI API")


@dataclass
class TranscriptSegment:
    """Một đoạn transcript từ Whisper"""
    start: float
    end: float
    text: str
    confidence: float = 1.0


class TimingSync:
    """Canh timing phụ đề theo lời thoại video"""
    
    def __init__(self, use_local_whisper: bool = False, model_size: str = "base"):
        """
        Args:
            use_local_whisper: Sử dụng Whisper local thay vì API
            model_size: Kích thước model Whisper (tiny, base, small, medium, large)
        """
        self.use_local = use_local_whisper and WHISPER_LOCAL
        self.model_size = model_size
        self.model = None
        self._openai_client = None
        
    def _get_openai_client(self):
        """Lấy OpenAI client"""
        if self._openai_client is None:
            from openai import OpenAI
            self._openai_client = OpenAI()
        return self._openai_client
    
    def _load_local_model(self):
        """Load Whisper model local"""
        if self.model is None and WHISPER_LOCAL:
            logger.info(f"Loading Whisper model: {self.model_size}")
            self.model = whisper.load_model(self.model_size)
        return self.model
    
    def transcribe(self, audio_path: str, language: str = "vi") -> List[TranscriptSegment]:
        """
        Transcribe audio và lấy timing
        
        Args:
            audio_path: Đường dẫn file audio/video
            language: Mã ngôn ngữ (vi, en, ja, etc.)
            
        Returns:
            Danh sách TranscriptSegment với timing
        """
        if self.use_local:
            return self._transcribe_local(audio_path, language)
        else:
            return self._transcribe_api(audio_path, language)
    
    def _transcribe_local(self, audio_path: str, language: str) -> List[TranscriptSegment]:
        """Transcribe sử dụng Whisper local"""
        model = self._load_local_model()
        if model is None:
            raise RuntimeError("Whisper model not available")
        
        result = model.transcribe(
            audio_path,
            language=language,
            word_timestamps=True
        )
        
        segments = []
        for seg in result.get("segments", []):
            segments.append(TranscriptSegment(
                start=seg["start"],
                end=seg["end"],
                text=seg["text"].strip(),
                confidence=seg.get("no_speech_prob", 0)
            ))
        
        return segments
    
    def _transcribe_api(self, audio_path: str, language: str) -> List[TranscriptSegment]:
        """Transcribe sử dụng OpenAI Whisper API"""
        client = self._get_openai_client()
        
        with open(audio_path, "rb") as audio_file:
            response = client.audio.transcriptions.create(
                model="whisper-1",
                file=audio_file,
                language=language,
                response_format="verbose_json",
                timestamp_granularities=["segment"]
            )
        
        segments = []
        for seg in response.segments:
            # Handle both object and dict access (OpenAI API version differences)
            if isinstance(seg, dict):
                start = seg.get("start", 0.0)
                end = seg.get("end", 0.0)
                text = seg.get("text", "").strip()
            else:
                start = getattr(seg, "start", 0.0)
                end = getattr(seg, "end", 0.0)
                text = getattr(seg, "text", "").strip()
                
            segments.append(TranscriptSegment(
                start=start,
                end=end,
                text=text
            ))
        
        return segments
    
    def align_subtitles(
        self, 
        subtitles: List[str], 
        transcript: List[TranscriptSegment],
        min_duration: float = 1.5,
        max_duration: float = 7.0
    ) -> List[Tuple[float, float, str]]:
        """
        Căn timing cho danh sách phụ đề dựa trên transcript
        Sử dụng GPT để mapping chính xác từng câu phụ đề với timeline
        
        Args:
            subtitles: Danh sách phụ đề cần căn
            transcript: Transcript từ Whisper
            min_duration: Thời gian hiển thị tối thiểu (giây)
            max_duration: Thời gian hiển thị tối đa (giây)
            
        Returns:
            List of (start_time, end_time, subtitle_text)
        """
        if not transcript:
            total_duration = 60.0
            return self._distribute_evenly(subtitles, 0, total_duration, min_duration)
        
        # Tạo timeline data từ transcript
        timeline_data = []
        for seg in transcript:
            timeline_data.append({
                "start": round(seg.start, 2),
                "end": round(seg.end, 2),
                "text": seg.text.strip()
            })
        
        logger.info(f"Transcript segments: {len(timeline_data)}")
        for t in timeline_data:
            logger.info(f"  [{t['start']:.1f}s - {t['end']:.1f}s]: {t['text']}")
        
        # Thử dùng GPT để mapping thông minh
        try:
            aligned = self._align_with_gpt(subtitles, timeline_data, min_duration, max_duration)
            if aligned:
                return aligned
        except Exception as e:
            logger.warning(f"GPT alignment failed: {e}")
        
        # Fallback: phân bổ đều theo thời gian transcript
        total_duration = transcript[-1].end if transcript else 30.0
        return self._distribute_evenly(subtitles, 0, total_duration, min_duration)
    
    def _align_with_gpt(
        self,
        subtitles: List[str],
        timeline: List[dict],
        min_duration: float,
        max_duration: float
    ) -> List[Tuple[float, float, str]]:
        """Dùng GPT để mapping chính xác subtitles với timeline"""
        client = self._get_openai_client()
        
        prompt = f"""Map these subtitles to the video transcript timeline.

TRANSCRIPT TIMELINE (with timestamps):
{json.dumps(timeline, ensure_ascii=False, indent=2)}

SUBTITLES TO ALIGN:
{json.dumps(subtitles, ensure_ascii=False)}

TASK:
For each subtitle, find when it should appear based on the transcript timing.
- Match based on meaning/content, not exact words
- Each subtitle should have a unique time range
- Subtitles should appear in sequence (no overlapping)
- Duration should be {min_duration}-{max_duration} seconds

Return a JSON array with this format:
[
  {{"start": 0.0, "end": 2.5, "text": "subtitle 1"}},
  {{"start": 2.5, "end": 5.0, "text": "subtitle 2"}}
]

Return ONLY the JSON array."""

        response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {"role": "system", "content": "You are a professional subtitle timing specialist. Output only valid JSON."},
                {"role": "user", "content": prompt}
            ],
            temperature=0.2,
            max_tokens=2000
        )
        
        result = response.choices[0].message.content.strip()
        
        # Parse JSON
        if result.startswith("```"):
            result = result.split("```")[1]
            if result.startswith("json"):
                result = result[4:]
        result = result.strip()
        
        aligned_data = json.loads(result)
        
        aligned = []
        for item in aligned_data:
            start = float(item["start"])
            end = float(item["end"])
            text = item["text"]
            
            # Validate timing
            if end - start < min_duration:
                end = start + min_duration
            if end - start > max_duration:
                end = start + max_duration
            
            aligned.append((start, end, text))
            logger.info(f"GPT Aligned: '{text[:30]}' -> {start:.1f}s - {end:.1f}s")
        
        return aligned
    
    def _find_best_matching_segment(
        self,
        subtitle: str,
        transcript: List[TranscriptSegment],
        used_segments: set
    ) -> Optional[int]:
        """Tìm segment phù hợp nhất với subtitle bằng fuzzy matching"""
        subtitle_lower = subtitle.lower().strip()
        subtitle_words = set(subtitle_lower.split())
        
        best_score = 0
        best_idx = None
        
        for idx, seg in enumerate(transcript):
            if idx in used_segments:
                continue
            
            seg_text = seg.text.lower().strip()
            seg_words = set(seg_text.split())
            
            # Tính điểm dựa trên số từ chung
            common_words = subtitle_words & seg_words
            if not common_words:
                continue
            
            # Score = số từ chung / tổng số từ unique
            all_words = subtitle_words | seg_words
            score = len(common_words) / len(all_words) if all_words else 0
            
            # Bonus nếu có substring match
            if subtitle_lower in seg_text or seg_text in subtitle_lower:
                score += 0.3
            
            # Bonus cho các từ quan trọng (dài hơn 3 ký tự)
            important_common = [w for w in common_words if len(w) > 3]
            score += len(important_common) * 0.1
            
            if score > best_score:
                best_score = score
                best_idx = idx
        
        # Chỉ chấp nhận nếu score đủ cao
        if best_score >= 0.2:
            return best_idx
        return None
    
    def _fill_missing_timings(
        self,
        aligned: List[Tuple[Optional[float], Optional[float], str]],
        transcript: List[TranscriptSegment],
        min_duration: float
    ) -> List[Tuple[float, float, str]]:
        """Điền timing cho các subtitle chưa có"""
        result = []
        
        # Lấy tổng thời gian từ transcript
        if transcript:
            total_duration = transcript[-1].end
        else:
            total_duration = len(aligned) * 3
        
        # Tìm các khoảng thời gian đã được sử dụng
        used_times = [(start, end) for start, end, _ in aligned if start is not None]
        
        # Điền timing cho subtitle chưa có
        last_end = 0
        for i, (start, end, text) in enumerate(aligned):
            if start is not None:
                result.append((start, end, text))
                last_end = end
            else:
                # Tìm khoảng trống tiếp theo
                new_start = last_end + 0.1
                new_end = new_start + min_duration
                
                # Đảm bảo không vượt quá total duration
                if new_end > total_duration:
                    new_end = total_duration
                
                result.append((new_start, new_end, text))
                last_end = new_end
        
        return result
    
    def _distribute_evenly(
        self, 
        subtitles: List[str], 
        start_time: float, 
        end_time: float,
        min_duration: float
    ) -> List[Tuple[float, float, str]]:
        """Chia đều thời gian cho các phụ đề"""
        if not subtitles:
            return []
            
        total_duration = end_time - start_time
        duration_per_sub = max(total_duration / len(subtitles), min_duration)
        
        result = []
        current_time = start_time
        
        for sub in subtitles:
            sub_end = current_time + duration_per_sub
            result.append((current_time, sub_end, sub))
            current_time = sub_end
            
        return result
    
    def smart_line_break(
        self, 
        text: str, 
        max_chars_per_line: int = 42,
        max_lines: int = 2
    ) -> str:
        """
        Tạo ngắt dòng thông minh cho phụ đề
        
        Args:
            text: Văn bản cần ngắt
            max_chars_per_line: Số ký tự tối đa mỗi dòng
            max_lines: Số dòng tối đa
            
        Returns:
            Văn bản đã được ngắt dòng
        """
        # Nếu text đủ ngắn, không cần ngắt
        if len(text) <= max_chars_per_line:
            return text
        
        words = text.split()
        lines = []
        current_line = []
        current_length = 0
        
        for word in words:
            word_len = len(word)
            
            if current_length + word_len + (1 if current_line else 0) <= max_chars_per_line:
                current_line.append(word)
                current_length += word_len + (1 if len(current_line) > 1 else 0)
            else:
                if current_line:
                    lines.append(" ".join(current_line))
                current_line = [word]
                current_length = word_len
                
                if len(lines) >= max_lines - 1:
                    # Gộp tất cả từ còn lại vào dòng cuối
                    remaining_idx = words.index(word)
                    lines.append(" ".join(words[remaining_idx:]))
                    return "\n".join(lines)
        
        if current_line:
            lines.append(" ".join(current_line))
        
        return "\n".join(lines[:max_lines])
    
    def segment_long_text(
        self,
        text: str,
        language: str = "vi",
        max_chars: int = 80,
        use_ai: bool = True
    ) -> List[str]:
        """
        Tự động ngắt văn bản dài thành các đoạn phụ đề phù hợp
        
        Args:
            text: Văn bản dài cần ngắt
            language: Mã ngôn ngữ
            max_chars: Số ký tự tối đa mỗi đoạn phụ đề
            use_ai: Dùng AI để ngắt thông minh
            
        Returns:
            Danh sách các đoạn phụ đề
        """
        text = text.strip()
        
        # Nếu text ngắn, không cần ngắt
        if len(text) <= max_chars:
            return [text]
        
        if use_ai:
            try:
                return self._segment_with_ai(text, language, max_chars)
            except Exception as e:
                logger.warning(f"AI segmentation failed: {e}, using rule-based")
        
        # Fallback: ngắt theo dấu câu
        return self._segment_by_punctuation(text, max_chars, language)
    
    def _segment_with_ai(
        self,
        text: str,
        language: str,
        max_chars: int
    ) -> List[str]:
        """Ngắt câu bằng AI GPT"""
        client = self._get_openai_client()
        
        lang_names = {
            "vi": "Vietnamese",
            "en": "English",
            "ja": "Japanese",
            "ko": "Korean",
            "zh": "Chinese",
            "fr": "French",
            "es": "Spanish",
            "de": "German"
        }
        lang_name = lang_names.get(language, "Vietnamese")
        
        prompt = f"""Break this {lang_name} text into subtitle segments for video.

Rules:
- Each segment should be 1-2 short sentences
- Each segment max {max_chars} characters
- Break at natural pauses (periods, commas, "và", "nhưng", etc.)
- Keep meaning complete in each segment
- Return as JSON array of strings

Text:
{text}

Return ONLY the JSON array, no explanation."""

        response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {"role": "system", "content": "You are a subtitle editor. Output only valid JSON."},
                {"role": "user", "content": prompt}
            ],
            temperature=0.3,
            max_tokens=2000
        )
        
        result = response.choices[0].message.content.strip()
        
        # Parse JSON
        if result.startswith("```"):
            result = result.split("```")[1]
            if result.startswith("json"):
                result = result[4:]
        
        segments = json.loads(result)
        
        if isinstance(segments, list) and all(isinstance(s, str) for s in segments):
            return [s.strip() for s in segments if s.strip()]
        
        raise ValueError("Invalid AI response format")
    
    def _segment_by_punctuation(
        self,
        text: str,
        max_chars: int,
        language: str
    ) -> List[str]:
        """Ngắt theo dấu câu (fallback)"""
        import re
        
        # Các pattern ngắt câu theo ngôn ngữ
        if language in ["vi", "en", "fr", "es", "de"]:
            # Ngắt theo dấu câu và từ nối
            pattern = r'(?<=[.!?])\s+|(?<=,)\s+(?=và\s|nhưng\s|hoặc\s|nên\s|vì\s|and\s|but\s|or\s|so\s|because\s)'
        elif language in ["ja", "zh"]:
            pattern = r'(?<=[。！？])|(?<=[、，])'
        elif language == "ko":
            pattern = r'(?<=[.!?])\s+|(?<=,)\s+'
        else:
            pattern = r'(?<=[.!?])\s+'
        
        # Tách thành các câu
        sentences = re.split(pattern, text)
        sentences = [s.strip() for s in sentences if s.strip()]
        
        # Gộp các câu ngắn, tách câu dài
        segments = []
        current = ""
        
        for sentence in sentences:
            if len(sentence) > max_chars:
                # Câu quá dài, tách theo dấu phẩy hoặc khoảng cách
                if current:
                    segments.append(current.strip())
                    current = ""
                
                # Tách câu dài thành các phần nhỏ
                words = sentence.split()
                part = ""
                for word in words:
                    if len(part) + len(word) + 1 <= max_chars:
                        part = f"{part} {word}".strip()
                    else:
                        if part:
                            segments.append(part)
                        part = word
                if part:
                    segments.append(part)
                    
            elif len(current) + len(sentence) + 1 <= max_chars:
                current = f"{current} {sentence}".strip()
            else:
                if current:
                    segments.append(current.strip())
                current = sentence
        
        if current:
            segments.append(current.strip())
        
        return segments


def extract_audio(video_path: str, output_path: Optional[str] = None) -> str:
    """
    Trích xuất audio từ video để transcribe
    
    Args:
        video_path: Đường dẫn video
        output_path: Đường dẫn output (optional)
        
    Returns:
        Đường dẫn file audio
    """
    import subprocess
    
    if output_path is None:
        video_name = Path(video_path).stem
        output_path = f"/tmp/{video_name}_audio.wav"
    
    cmd = [
        "ffmpeg", "-y",
        "-i", video_path,
        "-vn",  # No video
        "-acodec", "pcm_s16le",
        "-ar", "16000",  # 16kHz for Whisper
        "-ac", "1",  # Mono
        output_path
    ]
    
    subprocess.run(cmd, capture_output=True, check=True)
    logger.info(f"Extracted audio to {output_path}")
    
    return output_path
