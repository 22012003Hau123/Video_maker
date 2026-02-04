"""
Scene Analysis Module
Nhận diện và phân loại các cảnh trong video
"""
import os
import logging
from pathlib import Path
from typing import List, Optional, Dict, Tuple
from dataclasses import dataclass
import subprocess
import json

logger = logging.getLogger(__name__)


@dataclass
class VideoElement:
    """Một element trong video (logo, text, product)"""
    element_type: str  # logo, text, product, tagline
    x: int
    y: int
    width: int
    height: int
    start_time: float
    end_time: float
    content: Optional[str] = None  # Text content nếu có
    confidence: float = 1.0


@dataclass
class SceneSegment:
    """Một phân đoạn cảnh trong video"""
    start_time: float
    end_time: float
    scene_type: str  # intro, outro, product_shot, dialogue, transition
    elements: List[VideoElement]
    keyframe_path: Optional[str] = None


class SceneAnalyzer:
    """Phân tích cảnh và nhận diện elements trong video"""
    
    def __init__(self):
        self._openai_client = None
    
    def _get_openai_client(self):
        """Lấy OpenAI client"""
        if self._openai_client is None:
            from openai import OpenAI
            self._openai_client = OpenAI()
        return self._openai_client
    
    def _get_video_duration(self, video_path: str) -> float:
        """Lấy thời lượng video"""
        cmd = [
            "ffprobe", "-v", "quiet",
            "-show_entries", "format=duration",
            "-of", "default=noprint_wrappers=1:nokey=1",
            video_path
        ]
        result = subprocess.run(cmd, capture_output=True, text=True)
        try:
            return float(result.stdout.strip())
        except:
            return 60.0
    
    def _extract_keyframe(self, video_path: str, timestamp: float, output_path: str) -> str:
        """Trích xuất một keyframe tại timestamp"""
        cmd = [
            "ffmpeg", "-y",
            "-ss", str(timestamp),
            "-i", video_path,
            "-vframes", "1",
            "-q:v", "2",
            output_path
        ]
        subprocess.run(cmd, capture_output=True)
        return output_path
    
    def detect_scene_changes(self, video_path: str, threshold: float = 0.3) -> List[float]:
        """
        Phát hiện các điểm chuyển cảnh bằng FFmpeg
        
        Args:
            video_path: Đường dẫn video
            threshold: Ngưỡng phát hiện (0-1)
            
        Returns:
            Danh sách timestamps của các điểm chuyển cảnh
        """
        cmd = [
            "ffprobe", "-v", "quiet",
            "-show_entries", "frame=pts_time",
            "-of", "csv=p=0",
            "-f", "lavfi",
            f"movie={video_path},select='gt(scene,{threshold})'"
        ]
        
        result = subprocess.run(cmd, capture_output=True, text=True)
        
        timestamps = [0.0]  # Luôn có cảnh bắt đầu
        for line in result.stdout.strip().split('\n'):
            if line:
                try:
                    timestamps.append(float(line))
                except ValueError:
                    continue
        
        # Thêm điểm kết thúc
        duration = self._get_video_duration(video_path)
        timestamps.append(duration)
        
        return sorted(set(timestamps))
    
    def analyze_video(self, video_path: str, sample_interval: float = 3.0) -> List[SceneSegment]:
        """
        Phân tích toàn bộ video
        
        Args:
            video_path: Đường dẫn video
            sample_interval: Khoảng cách lấy mẫu (giây)
            
        Returns:
            Danh sách SceneSegment
        """
        import tempfile
        import base64
        
        duration = self._get_video_duration(video_path)
        scene_changes = self.detect_scene_changes(video_path)
        
        segments = []
        
        for i in range(len(scene_changes) - 1):
            start = scene_changes[i]
            end = scene_changes[i + 1]
            
            # Lấy keyframe giữa segment
            mid_time = (start + end) / 2
            
            with tempfile.NamedTemporaryFile(suffix=".jpg", delete=False) as tmp:
                keyframe_path = tmp.name
            
            self._extract_keyframe(video_path, mid_time, keyframe_path)
            
            # Phân tích keyframe bằng AI
            try:
                analysis = self._analyze_keyframe(keyframe_path)
                
                segment = SceneSegment(
                    start_time=start,
                    end_time=end,
                    scene_type=analysis.get("scene_type", "unknown"),
                    elements=self._parse_elements(analysis.get("elements", []), start, end),
                    keyframe_path=keyframe_path
                )
                segments.append(segment)
                
            except Exception as e:
                logger.error(f"Error analyzing segment {start}-{end}: {e}")
                segments.append(SceneSegment(
                    start_time=start,
                    end_time=end,
                    scene_type="unknown",
                    elements=[]
                ))
        
        return segments
    
    def _analyze_keyframe(self, keyframe_path: str) -> Dict:
        """Phân tích keyframe bằng AI Vision"""
        import base64
        
        client = self._get_openai_client()
        
        with open(keyframe_path, "rb") as f:
            base64_image = base64.b64encode(f.read()).decode('utf-8')
        
        response = client.chat.completions.create(
            model="gpt-4o",
            messages=[
                {
                    "role": "system",
                    "content": """Analyze this video frame and identify:
1. Scene type: intro, outro, product_shot, dialogue, action, transition
2. Elements present (logo, text, product/bottle, tagline, person)
3. Approximate position of each element (x, y as percentage of frame)

Respond in JSON:
{
    "scene_type": "intro|outro|product_shot|dialogue|action|transition",
    "elements": [
        {"type": "logo", "x_percent": 85, "y_percent": 10, "content": null},
        {"type": "text", "x_percent": 50, "y_percent": 90, "content": "text if readable"},
        {"type": "product", "x_percent": 50, "y_percent": 50, "content": null}
    ]
}"""
                },
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "image_url",
                            "image_url": {"url": f"data:image/jpeg;base64,{base64_image}"}
                        }
                    ]
                }
            ],
            max_tokens=500
        )
        
        content = response.choices[0].message.content
        try:
            if "```json" in content:
                content = content.split("```json")[1].split("```")[0]
            elif "```" in content:
                content = content.split("```")[1].split("```")[0]
            return json.loads(content.strip())
        except:
            return {"scene_type": "unknown", "elements": []}
    
    def _parse_elements(self, elements_data: List[Dict], start_time: float, end_time: float) -> List[VideoElement]:
        """Parse elements data thành VideoElement objects"""
        elements = []
        for elem in elements_data:
            elements.append(VideoElement(
                element_type=elem.get("type", "unknown"),
                x=int(elem.get("x_percent", 50) * 19.2),  # Convert to 1920 width
                y=int(elem.get("y_percent", 50) * 10.8),  # Convert to 1080 height
                width=100,  # Default size
                height=100,
                start_time=start_time,
                end_time=end_time,
                content=elem.get("content"),
                confidence=0.8
            ))
        return elements
    
    def find_element_positions(self, video_path: str, element_type: str) -> List[Tuple[float, float, int, int]]:
        """
        Tìm vị trí của một loại element trong video
        
        Args:
            video_path: Đường dẫn video
            element_type: Loại element (logo, product, text)
            
        Returns:
            List of (timestamp, duration, x, y)
        """
        segments = self.analyze_video(video_path)
        
        positions = []
        for seg in segments:
            for elem in seg.elements:
                if elem.element_type == element_type:
                    positions.append((
                        seg.start_time,
                        seg.end_time - seg.start_time,
                        elem.x,
                        elem.y
                    ))
        
        return positions
    
    def get_intro_outro(self, video_path: str) -> Tuple[Optional[SceneSegment], Optional[SceneSegment]]:
        """
        Lấy segment intro và outro
        
        Returns:
            Tuple (intro_segment, outro_segment)
        """
        segments = self.analyze_video(video_path)
        
        intro = None
        outro = None
        
        for seg in segments:
            if seg.scene_type == "intro":
                intro = seg
                break
        
        for seg in reversed(segments):
            if seg.scene_type == "outro":
                outro = seg
                break
        
        # Nếu không tìm thấy, dùng segment đầu/cuối
        if not intro and segments:
            intro = segments[0]
        if not outro and segments:
            outro = segments[-1]
        
        return intro, outro
