"""
Scene Detector Module
Sử dụng AI Vision để phát hiện cảnh mở đầu/kết thúc
"""
import os
import logging
from pathlib import Path
from typing import List, Optional, Tuple, Dict
from dataclasses import dataclass
import base64
import tempfile
import subprocess

logger = logging.getLogger(__name__)


@dataclass
class Scene:
    """Một cảnh trong video"""
    start_time: float
    end_time: float
    scene_type: str  # opening, closing, product_shot, dialogue, etc.
    confidence: float
    description: Optional[str] = None
    has_logo: bool = False
    has_text: bool = False
    has_product: bool = False


class SceneDetector:
    """Phát hiện và phân loại cảnh trong video"""
    
    def __init__(self, use_openai: bool = True):
        self.use_openai = use_openai
        self._openai_client = None
    
    def _get_openai_client(self):
        """Lấy OpenAI client"""
        if self._openai_client is None:
            from openai import OpenAI
            self._openai_client = OpenAI()
        return self._openai_client
    
    def extract_frames(
        self, 
        video_path: str, 
        interval: float = 1.0,
        output_dir: Optional[str] = None
    ) -> List[str]:
        """
        Trích xuất frames từ video
        
        Args:
            video_path: Đường dẫn video
            interval: Khoảng cách giữa các frame (giây)
            output_dir: Thư mục lưu frames
            
        Returns:
            Danh sách đường dẫn frame images
        """
        if output_dir is None:
            output_dir = tempfile.mkdtemp(prefix="frames_")
        else:
            Path(output_dir).mkdir(parents=True, exist_ok=True)
        
        output_pattern = os.path.join(output_dir, "frame_%04d.jpg")
        
        cmd = [
            "ffmpeg", "-y",
            "-i", video_path,
            "-vf", f"fps=1/{interval}",
            "-q:v", "2",
            output_pattern
        ]
        
        result = subprocess.run(cmd, capture_output=True, text=True)
        if result.returncode != 0:
            logger.error(f"FFmpeg error: {result.stderr}")
            raise RuntimeError(f"Failed to extract frames: {result.stderr}")
        
        # Lấy danh sách frames
        frames = sorted(Path(output_dir).glob("frame_*.jpg"))
        logger.info(f"Extracted {len(frames)} frames")
        
        return [str(f) for f in frames]
    
    def _encode_image(self, image_path: str) -> str:
        """Encode image to base64"""
        with open(image_path, "rb") as f:
            return base64.b64encode(f.read()).decode('utf-8')
    
    def analyze_frame(self, frame_path: str) -> Dict:
        """
        Phân tích một frame bằng AI Vision
        
        Args:
            frame_path: Đường dẫn frame image
            
        Returns:
            Dict với thông tin phân tích
        """
        client = self._get_openai_client()
        
        base64_image = self._encode_image(frame_path)
        
        response = client.chat.completions.create(
            model="gpt-4o",
            messages=[
                {
                    "role": "system",
                    "content": """You are a video scene analyzer. Analyze the frame and determine:
1. Scene type: opening, closing, product_shot, dialogue, action, transition
2. Whether there's a logo visible
3. Whether there's text/subtitle visible
4. Whether there's a product (bottle, packshot) visible
5. Brief description of the scene

Respond in JSON format:
{
    "scene_type": "opening|closing|product_shot|dialogue|action|transition",
    "has_logo": true/false,
    "has_text": true/false,
    "has_product": true/false,
    "description": "brief description"
}"""
                },
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "image_url",
                            "image_url": {
                                "url": f"data:image/jpeg;base64,{base64_image}"
                            }
                        }
                    ]
                }
            ],
            max_tokens=300
        )
        
        import json
        try:
            content = response.choices[0].message.content
            # Parse JSON từ response
            if "```json" in content:
                content = content.split("```json")[1].split("```")[0]
            elif "```" in content:
                content = content.split("```")[1].split("```")[0]
            return json.loads(content.strip())
        except:
            return {
                "scene_type": "unknown",
                "has_logo": False,
                "has_text": False,
                "has_product": False,
                "description": "Could not analyze frame"
            }
    
    def detect_scenes(
        self, 
        video_path: str,
        sample_interval: float = 2.0
    ) -> List[Scene]:
        """
        Phát hiện các cảnh trong video
        
        Args:
            video_path: Đường dẫn video
            sample_interval: Khoảng cách lấy mẫu (giây)
            
        Returns:
            Danh sách Scene
        """
        # Lấy thời lượng video
        duration = self._get_video_duration(video_path)
        
        # Extract frames
        frames = self.extract_frames(video_path, sample_interval)
        
        scenes = []
        current_scene_type = None
        scene_start = 0
        
        for i, frame_path in enumerate(frames):
            timestamp = i * sample_interval
            
            try:
                analysis = self.analyze_frame(frame_path)
                scene_type = analysis.get("scene_type", "unknown")
                
                # Nếu scene type thay đổi, tạo scene mới
                if scene_type != current_scene_type:
                    if current_scene_type is not None:
                        scenes.append(Scene(
                            start_time=scene_start,
                            end_time=timestamp,
                            scene_type=current_scene_type,
                            confidence=0.8,
                            description=analysis.get("description"),
                            has_logo=analysis.get("has_logo", False),
                            has_text=analysis.get("has_text", False),
                            has_product=analysis.get("has_product", False)
                        ))
                    
                    current_scene_type = scene_type
                    scene_start = timestamp
                    
            except Exception as e:
                logger.error(f"Error analyzing frame {frame_path}: {e}")
                continue
        
        # Thêm scene cuối cùng
        if current_scene_type:
            scenes.append(Scene(
                start_time=scene_start,
                end_time=duration,
                scene_type=current_scene_type,
                confidence=0.8
            ))
        
        # Cleanup frames
        for frame in frames:
            try:
                os.unlink(frame)
            except:
                pass
        
        return scenes
    
    def detect_opening_closing(self, video_path: str) -> Tuple[Optional[Scene], Optional[Scene]]:
        """
        Chỉ phát hiện cảnh mở đầu và kết thúc
        
        Args:
            video_path: Đường dẫn video
            
        Returns:
            Tuple (opening_scene, closing_scene)
        """
        duration = self._get_video_duration(video_path)
        
        # Chỉ phân tích 10 giây đầu và cuối
        opening = None
        closing = None
        
        # Analyze opening (first 5 seconds)
        frames = self.extract_frames(video_path, 1.0)
        if frames:
            first_frames = frames[:5]
            for i, frame in enumerate(first_frames):
                analysis = self.analyze_frame(frame)
                if analysis.get("scene_type") in ["opening", "product_shot"]:
                    opening = Scene(
                        start_time=0,
                        end_time=min(5, duration),
                        scene_type="opening",
                        confidence=0.9,
                        description=analysis.get("description"),
                        has_logo=analysis.get("has_logo", False),
                        has_product=analysis.get("has_product", False)
                    )
                    break
            
            # Analyze closing (last 5 seconds)
            last_frames = frames[-5:] if len(frames) > 5 else []
            for i, frame in enumerate(last_frames):
                analysis = self.analyze_frame(frame)
                if analysis.get("scene_type") in ["closing", "product_shot"]:
                    closing = Scene(
                        start_time=max(0, duration - 5),
                        end_time=duration,
                        scene_type="closing",
                        confidence=0.9,
                        description=analysis.get("description"),
                        has_logo=analysis.get("has_logo", False),
                        has_product=analysis.get("has_product", False)
                    )
                    break
            
            # Cleanup
            for frame in frames:
                try:
                    os.unlink(frame)
                except:
                    pass
        
        return opening, closing
    
    def _get_video_duration(self, video_path: str) -> float:
        """Lấy thời lượng video"""
        cmd = [
            "ffprobe",
            "-v", "quiet",
            "-show_entries", "format=duration",
            "-of", "default=noprint_wrappers=1:nokey=1",
            video_path
        ]
        
        result = subprocess.run(cmd, capture_output=True, text=True)
        try:
            return float(result.stdout.strip())
        except:
            return 60.0  # Default
    
    def suggest_legal_position(
        self, 
        video_path: str,
        legal_duration: float = 3.0
    ) -> Dict:
        """
        Đề xuất vị trí đặt nội dung pháp lý
        
        Args:
            video_path: Đường dẫn video
            legal_duration: Thời gian hiển thị nội dung pháp lý
            
        Returns:
            Dict với thông tin vị trí đề xuất
        """
        opening, closing = self.detect_opening_closing(video_path)
        duration = self._get_video_duration(video_path)
        
        suggestions = {
            "start_legal": None,
            "end_legal": None,
            "position": "bottom",  # hoặc "top" nếu bottom bị che
            "recommendations": []
        }
        
        # Đề xuất vị trí đầu video
        if opening and opening.has_product:
            suggestions["start_legal"] = {
                "time": opening.start_time,
                "duration": legal_duration,
                "reason": "Đặt cùng với packshot mở đầu"
            }
        
        # Đề xuất vị trí cuối video
        if closing:
            suggestions["end_legal"] = {
                "time": max(0, duration - legal_duration),
                "duration": legal_duration,
                "reason": "Đặt ở cảnh kết thúc"
            }
            
            # Nếu closing có logo ở bottom, đề xuất đặt legal ở top
            if closing.has_logo:
                suggestions["position"] = "top"
                suggestions["recommendations"].append(
                    "Logo xuất hiện ở cuối video, đề xuất đặt nội dung pháp lý ở phía trên"
                )
        else:
            suggestions["end_legal"] = {
                "time": max(0, duration - legal_duration),
                "duration": legal_duration,
                "reason": "Vị trí mặc định cuối video"
            }
        
        return suggestions
