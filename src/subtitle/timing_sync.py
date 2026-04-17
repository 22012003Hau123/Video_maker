"""
Timing Sync Module
Sử dụng Whisper AI để canh timing phụ đề theo lời thoại
"""
import os
import logging
import tempfile
from pathlib import Path
from typing import List, Optional, Tuple
from dataclasses import dataclass
import json
from src.utils.text import smart_line_break

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
        self.initial_subtitle_delay = 0.05
        self.zero_start_epsilon = 0.02
        
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
    
    def transcribe(self, audio_path: str, language: str = "auto", prompt: Optional[str] = None) -> Tuple[List[TranscriptSegment], str, List[dict]]:
        """
        Transcribe audio và lấy timing (bao gồm word-level)
        
        Args:
            audio_path: Đường dẫn file audio/video
            language: Mã ngôn ngữ (vi, en, ja, etc.) hoặc "auto" để tự động nhận diện
            prompt: Text gợi ý cho Whisper (giúp nhận diện brand name tốt hơn)
            
        Returns:
            Tuple (Danh sách TranscriptSegment, Mã ngôn ngữ, Danh sách word-level timings)
        """
        if self.use_local:
            return self._transcribe_local(audio_path, language, prompt)
        else:
            return self._transcribe_api(audio_path, language, prompt)
    
    def _transcribe_local(self, audio_path: str, language: str, prompt: Optional[str] = None) -> Tuple[List[TranscriptSegment], str, List[dict]]:
        """Transcribe sử dụng Whisper local"""
        model = self._load_local_model()
        if model is None:
            raise RuntimeError("Whisper model not available")
        
        result = model.transcribe(
            audio_path,
            language=language if language != "auto" else None,
            word_timestamps=True,
            initial_prompt=prompt
        )
        
        detected_lang = result.get("language", language)
        segments = []
        for seg in result.get("segments", []):
            segments.append(TranscriptSegment(
                start=seg["start"],
                end=seg["end"],
                text=seg["text"].strip(),
                confidence=seg.get("no_speech_prob", 0)
            ))
        
        return segments, detected_lang, []
    
    def _transcribe_api(self, audio_path: str, language: str, prompt: Optional[str] = None) -> Tuple[List[TranscriptSegment], str, List[dict]]:
        """Transcribe sử dụng OpenAI Whisper API"""
        client = self._get_openai_client()
        
        with open(audio_path, "rb") as audio_file:
            # Add strict timeout to avoid hanging the entire pipeline if OpenAI is slow
            response = client.audio.transcriptions.create(
                model="whisper-1",
                file=audio_file,
                language=language if language != "auto" else None,
                prompt=prompt,
                response_format="verbose_json",
                timestamp_granularities=["word", "segment"],
                timeout=60.0 # 1 minute max for 20s audio
            )
        
        detected_lang = getattr(response, 'language', language)
        
        # Xây dựng word-level timing map để refine segment timing
        word_timings = []
        if hasattr(response, 'words') and response.words:
            for w in response.words:
                if isinstance(w, dict):
                    word_timings.append({
                        "word": w.get("word", ""),
                        "start": w.get("start", 0.0),
                        "end": w.get("end", 0.0)
                    })
                else:
                    word_timings.append({
                        "word": getattr(w, "word", ""),
                        "start": getattr(w, "start", 0.0),
                        "end": getattr(w, "end", 0.0)
                    })
        
        segments = []
        word_idx = 0
        
        for seg in response.segments:
            # Handle both object and dict access (OpenAI API version differences)
            if isinstance(seg, dict):
                seg_start = seg.get("start", 0.0)
                seg_end = seg.get("end", 0.0)
                text = seg.get("text", "").strip()
            else:
                seg_start = getattr(seg, "start", 0.0)
                seg_end = getattr(seg, "end", 0.0)
                text = getattr(seg, "text", "").strip()
            
            # Dùng word-level timestamps để lấy thời điểm từ đầu tiên thực sự nói
            # (segment start thường sớm hơn vì bao gồm khoảng lặng trước lời nói)
            refined_start = seg_start
            refined_end = seg_end
            
            if word_timings and word_idx < len(word_timings):
                # Tìm từ đầu tiên thuộc segment này
                while word_idx < len(word_timings):
                    wt = word_timings[word_idx]
                    if wt["start"] >= seg_start - 0.1:
                        # Từ đầu tiên của segment → dùng start của nó (chính xác hơn)
                        refined_start = wt["start"]
                        break
                    word_idx += 1
                
                # Tìm từ cuối cùng thuộc segment này
                last_word_end = refined_end
                temp_idx = word_idx
                while temp_idx < len(word_timings):
                    wt = word_timings[temp_idx]
                    if wt["start"] > seg_end + 0.1:
                        break
                    last_word_end = wt["end"]
                    temp_idx += 1
                refined_end = last_word_end
                
                # Di chuyển word_idx tới segment tiếp theo
                word_idx = temp_idx
            
            logger.info(f"Segment timing: [{seg_start:.2f}s -> {refined_start:.2f}s] ~ [{seg_end:.2f}s -> {refined_end:.2f}s]: {text[:50]}")
            
            segments.append(TranscriptSegment(
                start=refined_start,
                end=refined_end,
                text=text
            ))
        
        return segments, detected_lang, word_timings
    
    def align_subtitles_to_words(
        self,
        subtitles: List[str],
        word_timings: List[dict],
        min_duration: float = 0.5
    ) -> List[Tuple[float, float, str, int]]:
        """
        Căn timing 100% chính xác bằng cách khớp từng từ trong phụ đề với word_timings từ Whisper.
        """
        if not word_timings:
            # Fallback to segments if word timings missing
            return self._distribute_evenly(subtitles, 0, 20.0, min_duration)

        aligned = []
        word_idx = 0
        
        # Clean words from transcript for better matching
        clean_transcript_words = []
        for wt in word_timings:
            wText = wt["word"].lower().strip().strip(".,!?\"'")
            if wText:
                clean_transcript_words.append({
                    "text": wText,
                    "start": wt["start"],
                    "end": wt["end"]
                })

        for i, sub_text in enumerate(subtitles):
            sub_words = sub_text.lower().strip().split()
            if not sub_words: continue
            
            # Find the best starting word in the transcript
            line_start = None
            line_end = None
            
            # Simple matching: advance word_idx until we find the first word of the line
            while word_idx < len(clean_transcript_words):
                if clean_transcript_words[word_idx]["text"] == sub_words[0].strip(".,!?\"'"):
                    line_start = clean_transcript_words[word_idx]["start"]
                    # Found start, now advance to find the last word
                    # Advance up to len(sub_words) pieces
                    lookahead = min(len(sub_words) + 2, len(clean_transcript_words) - word_idx)
                    best_end_idx = word_idx
                    for k in range(lookahead):
                        # Match current word
                        if k < len(sub_words):
                            target_w = sub_words[k].strip(".,!?\"'")
                            actual_w = clean_transcript_words[word_idx+k]["text"]
                            if target_w == actual_w:
                                best_end_idx = word_idx+k
                    
                    line_end = clean_transcript_words[best_end_idx]["end"]
                    word_idx = best_end_idx + 1 # Next line starts after this
                    break
                word_idx += 1
            
            if line_start is not None:
                aligned.append((line_start, line_end, sub_text, i))
            else:
                # Fallback for this line: place it after last line
                prev_end = aligned[-1][1] if aligned else 0.1
                aligned.append((prev_end + 0.1, prev_end + 2.0, sub_text, i))

        return self._apply_initial_delay_with_index(aligned)

    def align_subtitles(
        self, 
        subtitles: List[str], 
        transcript: List[TranscriptSegment],
        min_duration: float = 0.5,
        max_duration: float = 8.0
    ) -> List[Tuple[float, float, str, int]]:
        """
        Căn timing cho danh sách phụ đề dựa trên transcript
        Sử dụng GPT để mapping chính xác từng câu phụ đề với timeline
        
        Args:
            subtitles: Danh sách phụ đề cần căn
            transcript: Transcript từ Whisper
            min_duration: Thời gian hiển thị tối thiểu (giây)
            max_duration: Thời gian hiển thị tối đa (giây)
            
        Returns:
            List of (start_time, end_time, subtitle_text, original_index)
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
        
        # Perform semantic alignment regardless of match counts
        # This prevents mis-alignment when counts match but content doesn't (e.g. merged segments)
        try:
            aligned = self._align_with_gpt_by_index(subtitles, timeline_data, min_duration, max_duration)
            if aligned:
                return aligned
        except Exception as e:
            logger.warning(f"GPT index mapping failed: {e}")
        
        # Fallback: distribute evenly across the found timeline slots
        if timeline_data:
            total_start = timeline_data[0]["start"]
            total_end = timeline_data[-1].get("end", total_start + 5)
            return self._distribute_evenly(subtitles, total_start, total_end, min_duration)
        
        return self._distribute_evenly(subtitles, 0, 30.0, min_duration)

    def _align_with_gpt_by_index(
        self,
        subtitles: List[str],
        timeline: List[dict],
        min_duration: float,
        max_duration: float
    ) -> List[Tuple[float, float, str, int]]:
        """Dùng GPT để khớp Index của subtitle với Index của timeline slot"""
        client = self._get_openai_client()
        
        # Đánh số các slot để AI chọn cho dễ
        indexed_timeline = []
        for i, t in enumerate(timeline):
            indexed_timeline.append({
                "slot_id": i,
                "start": t["start"],
                "end": t["end"],
                "audio_content": t["text"]
            })

        prompt = f"""Map these {len(subtitles)} SUBTITLES to the {len(indexed_timeline)} AUDIO VOICE SLOTS.

AUDIO VOICE SLOTS (Speech from video):
{json.dumps(indexed_timeline, ensure_ascii=False, indent=2)}

SUBTITLES TO MAP:
{json.dumps(subtitles, ensure_ascii=False, indent=2)}

TASK:
1. CONTENT MATCHING IS TOP PRIORITY: Compare the subtitle text with the `audio_content`. If a slot contains text for multiple consecutive subtitles, map ALL those indices to that SAME slot ID.
2. ANCHORING: Look for unique keywords (e.g. "Belvedere", names, numbers) to anchor subtitles to the correct slots. Do not map a subtitle to a slot if the text clearly doesn't match.
3. HANDLING MERGED AUDIO: If Whisper merges the first two phrases into Slot 0, you MUST map both Subtitle 0 and Subtitle 1 to Slot 0.
4. REPETITIONS: Reuse subtitle indices for every occurrence in the audio.
5. Return ONLY a JSON array: [ {{ "subtitle_index": X, "slot_indices": [Y] }}, ... ]"""

        response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {"role": "system", "content": "You are a professional subtitle technical specialist. Output ONLY valid JSON array of indices."},
                {"role": "user", "content": prompt}
            ],
            temperature=0.0
        )
        
        result_text = response.choices[0].message.content.strip()
        if result_text.startswith("```"):
            result_text = result_text.split("```")[1]
            if result_text.startswith("json"):
                result_text = result_text[4:]
        
        mapping = json.loads(result_text.strip())
        
        # 1. Gather all mapping segments
        # Split non-contiguous slot ids into separate groups to avoid
        # one subtitle being stretched across distant timeline regions.
        def split_slot_groups(slot_ids: List[int]) -> List[List[int]]:
            if not slot_ids:
                return []
            sorted_ids = sorted(set(slot_ids))
            groups = [[sorted_ids[0]]]
            max_time_gap = 1.2  # seconds; larger gap means separate spoken occurrence

            for sid in sorted_ids[1:]:
                prev = groups[-1][-1]
                prev_end = float(timeline[prev]["end"])
                cur_start = float(timeline[sid]["start"])
                is_consecutive = sid == prev + 1
                is_close_in_time = (cur_start - prev_end) <= max_time_gap

                if is_consecutive and is_close_in_time:
                    groups[-1].append(sid)
                else:
                    groups.append([sid])
            return groups

        raw_segments = []
        for item in mapping:
            sub_idx = item.get("subtitle_index")
            slot_ids = item.get("slot_indices", [])
            if sub_idx is None or sub_idx >= len(subtitles): continue
            valid_slot_ids = [s for s in slot_ids if 0 <= s < len(timeline)]
            if not valid_slot_ids: continue

            slot_groups = split_slot_groups(valid_slot_ids)
            if len(slot_groups) > 1:
                logger.info(
                    f"Split non-contiguous slot mapping for subtitle[{sub_idx}] "
                    f"from {sorted(set(valid_slot_ids))} into {slot_groups}"
                )

            for group_ids in slot_groups:
                raw_segments.append({
                    "sub_idx": sub_idx,
                    "slot_ids": group_ids,
                    "text": subtitles[sub_idx]
                })
            
        # 2. Sort by slot order (chronological) then sub_idx
        raw_segments.sort(key=lambda x: (min(x['slot_ids']), x['sub_idx']))
        
        # 3. Handle Many-to-One: Sub-divide slots if multiple segments share them
        aligned = []
        i = 0
        while i < len(raw_segments):
            # Find a group of segments that share the EXACT SAME slot range
            group = [raw_segments[i]]
            j = i + 1
            while j < len(raw_segments) and raw_segments[j]['slot_ids'] == raw_segments[i]['slot_ids']:
                group.append(raw_segments[j])
                j += 1
            
            start_t = timeline[min(raw_segments[i]['slot_ids'])]["start"]
            end_t = timeline[max(raw_segments[i]['slot_ids'])]["end"]
            duration = end_t - start_t
            
            if len(group) == 1:
                aligned.append((start_t, end_t, group[0]['text'], group[0]['sub_idx']))
            else:
                # Sub-divide duration based on text length to avoid overlapping
                total_len = sum(len(g['text']) for g in group)
                if total_len == 0: total_len = len(group)
                
                curr_t = start_t
                for group_item in group:
                    g_dur = (len(group_item['text']) / total_len) * duration
                    # Ensure it doesn't get too short
                    g_dur = max(g_dur, min(min_duration, duration / len(group)))
                    
                    final_end = curr_t + g_dur
                    if final_end > end_t and len(group) > 1: final_end = end_t # Cap to total
                    
                    aligned.append((curr_t, final_end, group_item['text'], group_item['sub_idx']))
                    curr_t += g_dur
            
                logger.info(f"Sub-divided slot {raw_segments[i]['slot_ids']} for {len(group)} subtitles.")
            
            i = j
            
        return self._apply_initial_delay_with_index(aligned)

    def _apply_initial_delay_with_index(
        self,
        aligned: List[Tuple[float, float, str, int]]
    ) -> List[Tuple[float, float, str, int]]:
        """Delay subtitles that start at ~0s to avoid instant first-frame render."""
        adjusted = []
        for start, end, text, original_idx in aligned:
            new_start = start
            new_end = end
            if start <= self.zero_start_epsilon:
                new_start = self.initial_subtitle_delay
                if new_end <= new_start:
                    new_end = new_start + 0.05
            adjusted.append((new_start, new_end, text, original_idx))
        return adjusted

    def _apply_initial_delay_no_index(
        self,
        aligned: List[Tuple[float, float, str]]
    ) -> List[Tuple[float, float, str]]:
        """Delay subtitles that start at ~0s to avoid instant first-frame render."""
        adjusted = []
        for start, end, text in aligned:
            new_start = start
            new_end = end
            if start <= self.zero_start_epsilon:
                new_start = self.initial_subtitle_delay
                if new_end <= new_start:
                    new_end = new_start + 0.05
            adjusted.append((new_start, new_end, text))
        return adjusted
    
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
        aligned: List[Tuple[Optional[float], Optional[float], str, int]],
        transcript: List[TranscriptSegment],
        min_duration: float
    ) -> List[Tuple[float, float, str, int]]:
        """Điền timing cho các subtitle chưa có"""
        result = []
        
        # Lấy tổng thời gian từ transcript
        if transcript:
            total_duration = transcript[-1].end
        else:
            total_duration = len(aligned) * 3
        
        # Tìm các khoảng thời gian đã được sử dụng
        used_times = [(start, end) for start, end, _, _ in aligned if start is not None]
        
        # Điền timing cho subtitle chưa có
        last_end = 0
        for i, (start, end, text, original_idx) in enumerate(aligned):
            if start is not None:
                result.append((start, end, text, original_idx))
                last_end = end
            else:
                # Tìm khoảng trống tiếp theo
                new_start = last_end + 0.1
                new_end = new_start + min_duration
                
                # Đảm bảo không vượt quá total duration
                if new_end > total_duration:
                    new_end = total_duration
                
                result.append((new_start, new_end, text, original_idx))
                last_end = new_end
        
        return result
    
    def _distribute_evenly(
        self, 
        subtitles: List[str], 
        start_time: float, 
        end_time: float,
        min_duration: float
    ) -> List[Tuple[float, float, str, int]]:
        """Chia đều thời gian cho các phụ đề"""
        if not subtitles:
            return []
            
        total_duration = end_time - start_time
        duration_per_sub = max(total_duration / len(subtitles), min_duration)
        
        result = []
        current_time = start_time
        
        for i, sub in enumerate(subtitles):
            sub_end = current_time + duration_per_sub
            result.append((current_time, sub_end, sub, i))
            current_time = sub_end
            
        return self._apply_initial_delay_with_index(result)
    
    def smart_line_break(
        self, 
        text: str, 
        max_chars_per_line: int = 42,
        max_lines: int = 2
    ) -> str:
        """Tạo ngắt dòng thông minh (Dùng cho backward compatibility)"""
        return smart_line_break(text, max_chars_per_line, max_lines)
    
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

    def _get_merged_timeline(self, transcript: List[TranscriptSegment]) -> List[dict]:
        """Gộp các đoạn voice gần nhau để tạo Slots"""
        if not transcript:
            return []
            
        merged = []
        curr = {
            "slot_id": 0,
            "start": round(transcript[0].start, 2),
            "end": round(transcript[0].end, 2),
            "audio_content": transcript[0].text
        }
        
        for i in range(1, len(transcript)):
            next_seg = transcript[i]
            if next_seg.start - curr["end"] < 0.4:
                curr["end"] = round(next_seg.end, 2)
                curr["audio_content"] += " " + next_seg.text
            else:
                merged.append(curr)
                curr = {
                    "slot_id": len(merged),
                    "start": round(next_seg.start, 2),
                    "end": round(next_seg.end, 2),
                    "audio_content": next_seg.text
                }
        merged.append(curr)
        return merged

    def align_long_text_to_transcript(
        self,
        raw_text: str,
        transcript: List[TranscriptSegment],
        language: str = "vi"
    ) -> List[Tuple[float, float, str]]:
        """
        Ngắt văn bản dài và khớp với nhịp điệu của transcript (audio).
        Dùng AI để chia nhỏ văn bản và gán vào các Slots có sẵn.
        """
        if not transcript:
            return self._distribute_evenly([raw_text], 0, 10.0, 2.0)

        merged_timeline = self._get_merged_timeline(transcript)
        client = self._get_openai_client()

        prompt = f"""Break this FULL TEXT into segments to fit the {len(merged_timeline)} AUDIO VOICE SLOTS.
    
FULL TEXT:
{raw_text}

AUDIO VOICE SLOTS:
{json.dumps(merged_timeline, ensure_ascii=False, indent=2)}

TASK:
1. Divide the FULL TEXT into exactly {len(merged_timeline)} pieces.
2. Each piece MUST correspond to one slot_id in the timeline.
3. Your output must be a JSON array where each object has "slot_id" and "text".
   [ {{ "slot_id": 0, "text": "Part of text for first slot..." }}, ... ]
4. Split the text naturally so the meaning flows logically across the slots.

Return ONLY the JSON array."""

        try:
            response = client.chat.completions.create(
                model="gpt-4o-mini",
                messages=[
                    {"role": "system", "content": "You are a professional video editor specializing in rhythm-aware subtitling. Output ONLY valid JSON."},
                    {"role": "user", "content": prompt}
                ],
                temperature=0.2
            )
            
            result_text = response.choices[0].message.content.strip()
            if result_text.startswith("```"):
                result_text = result_text.split("```")[1]
                if result_text.startswith("json"):
                    result_text = result_text[4:]
            
            data = json.loads(result_text.strip())
            aligned = []
            for item in data:
                try:
                    slot_id = int(item["slot_id"])
                    if 0 <= slot_id < len(merged_timeline):
                        slot = merged_timeline[slot_id]
                        aligned.append((slot["start"], slot["end"], item["text"].strip()))
                except (ValueError, KeyError):
                    continue
            
            return self._apply_initial_delay_no_index(aligned)
        except Exception as e:
            logger.error(f"Rhythm-aware alignment failed: {e}")
            # Fallback: dùng logic cũ (chia đều hoặc map cơ bản)
            segments = self.segment_long_text(raw_text, language=language)
            return self.align_subtitles(segments, transcript)


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
        output_path = os.path.join(tempfile.gettempdir(), f"{video_name}_audio.wav")
    
    cmd = [
        "ffmpeg", "-y", "-nostdin",
        "-i", video_path,
        "-vn",  # No video
        "-acodec", "pcm_s16le",
        "-ar", "16000",  # 16kHz for Whisper
        "-ac", "1",  # Mono
        "-af", "highpass=f=200,lowpass=f=3000", # Filter noise for better voice detection
        output_path
    ]
    
    subprocess.run(
        cmd,
        capture_output=True,
        check=True,
        stdin=subprocess.DEVNULL,
    )
    logger.info(f"Extracted audio to {output_path}")
    
    return output_path
