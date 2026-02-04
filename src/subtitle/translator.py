"""
Subtitle Translation Module
Dịch phụ đề sang nhiều ngôn ngữ sử dụng GPT
"""
import os
import logging
from typing import List, Dict, Optional, Tuple
import json

logger = logging.getLogger(__name__)


class SubtitleTranslator:
    """Dịch phụ đề sang nhiều ngôn ngữ"""
    
    LANGUAGE_NAMES = {
        "vi": "Vietnamese",
        "en": "English", 
        "ja": "Japanese",
        "ko": "Korean",
        "zh": "Chinese (Simplified)",
        "zh-tw": "Chinese (Traditional)",
        "fr": "French",
        "es": "Spanish",
        "de": "German",
        "th": "Thai",
        "id": "Indonesian",
        "pt": "Portuguese",
        "ar": "Arabic",
        "ru": "Russian"
    }
    
    def __init__(self):
        self._client = None
    
    def _get_client(self):
        if self._client is None:
            from openai import OpenAI
            self._client = OpenAI()
        return self._client
    
    def translate(
        self,
        subtitles: List[str],
        source_lang: str,
        target_lang: str,
        context: str = ""
    ) -> List[str]:
        """
        Dịch danh sách phụ đề
        
        Args:
            subtitles: Danh sách các câu phụ đề
            source_lang: Mã ngôn ngữ nguồn (en, vi, ja...)
            target_lang: Mã ngôn ngữ đích
            context: Ngữ cảnh bổ sung (loại video, chủ đề...)
            
        Returns:
            Danh sách phụ đề đã dịch
        """
        if source_lang == target_lang:
            return subtitles
        
        source_name = self.LANGUAGE_NAMES.get(source_lang, source_lang)
        target_name = self.LANGUAGE_NAMES.get(target_lang, target_lang)
        
        client = self._get_client()
        
        # Dịch theo batch để tối ưu API calls
        translated = []
        batch_size = 10
        
        for i in range(0, len(subtitles), batch_size):
            batch = subtitles[i:i+batch_size]
            batch_translated = self._translate_batch(
                client, batch, source_name, target_name, context
            )
            translated.extend(batch_translated)
        
        return translated
    
    def _translate_batch(
        self,
        client,
        subtitles: List[str],
        source_lang: str,
        target_lang: str,
        context: str
    ) -> List[str]:
        """Dịch một batch phụ đề"""
        
        context_prompt = f"\nContext: {context}" if context else ""
        
        prompt = f"""Translate these video subtitles from {source_lang} to {target_lang}.
{context_prompt}

SUBTITLES:
{json.dumps(subtitles, ensure_ascii=False)}

RULES:
- Keep the same number of subtitles
- Keep subtitles concise and natural for video display
- Preserve tone and style
- Each subtitle should be max 2 lines
- Return as JSON array of strings

Return ONLY the JSON array of translated subtitles."""

        response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {
                    "role": "system", 
                    "content": f"You are a professional subtitle translator. Translate naturally and concisely for video display. Output only valid JSON."
                },
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
        result = result.strip()
        
        translated = json.loads(result)
        
        if isinstance(translated, list) and len(translated) == len(subtitles):
            logger.info(f"Translated {len(subtitles)} subtitles: {source_lang} -> {target_lang}")
            return translated
        else:
            logger.warning(f"Translation mismatch: expected {len(subtitles)}, got {len(translated)}")
            return subtitles  # Return original if translation failed
    
    def translate_with_timing(
        self,
        aligned_subtitles: List[Tuple[float, float, str]],
        source_lang: str,
        target_lang: str,
        context: str = ""
    ) -> List[Tuple[float, float, str]]:
        """
        Dịch phụ đề đã có timing
        
        Args:
            aligned_subtitles: List of (start, end, text)
            source_lang: Mã ngôn ngữ nguồn
            target_lang: Mã ngôn ngữ đích
            context: Ngữ cảnh
            
        Returns:
            List of (start, end, translated_text)
        """
        # Tách text để dịch
        texts = [text for _, _, text in aligned_subtitles]
        
        # Dịch
        translated_texts = self.translate(texts, source_lang, target_lang, context)
        
        # Ghép lại với timing
        result = []
        for i, (start, end, _) in enumerate(aligned_subtitles):
            translated_text = translated_texts[i] if i < len(translated_texts) else texts[i]
            result.append((start, end, translated_text))
        
        return result
    
    def batch_translate(
        self,
        subtitles: List[str],
        source_lang: str,
        target_langs: List[str],
        context: str = ""
    ) -> Dict[str, List[str]]:
        """
        Dịch phụ đề sang nhiều ngôn ngữ cùng lúc
        
        Args:
            subtitles: Danh sách phụ đề gốc
            source_lang: Mã ngôn ngữ nguồn
            target_langs: Danh sách mã ngôn ngữ đích
            context: Ngữ cảnh
            
        Returns:
            Dict {language_code: [translated_subtitles]}
        """
        results = {source_lang: subtitles}  # Include original
        
        for target_lang in target_langs:
            if target_lang != source_lang:
                try:
                    translated = self.translate(subtitles, source_lang, target_lang, context)
                    results[target_lang] = translated
                    logger.info(f"Batch translated to {target_lang}")
                except Exception as e:
                    logger.error(f"Failed to translate to {target_lang}: {e}")
                    results[target_lang] = subtitles  # Fallback to original
        
        return results


def get_translator() -> SubtitleTranslator:
    """Factory function"""
    return SubtitleTranslator()
