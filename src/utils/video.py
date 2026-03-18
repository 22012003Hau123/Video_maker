"""
Video Utilities
Common functions for FFmpeg-based video inspection
"""
import subprocess
import json
import logging
from pathlib import Path
from typing import Dict, Optional, Tuple
from dataclasses import dataclass

logger = logging.getLogger(__name__)

@dataclass
class VideoInfo:
    """Video metadata container"""
    path: str
    width: int
    height: int
    duration: float
    aspect_ratio: str
    fps: float = 25.0
    v_codec: str = "h264"
    a_codec: Optional[str] = None

    def __getitem__(self, key: str):
        """
        Backward compatibility for legacy dict-style access.
        Example: info["duration"] in older modules.
        """
        if hasattr(self, key):
            return getattr(self, key)
        raise KeyError(key)

    def get(self, key: str, default=None):
        """Dict-like .get() helper for legacy callers."""
        return getattr(self, key, default)

def get_video_info(video_path: str) -> VideoInfo:
    """
    Get detailed video information using ffprobe
    """
    try:
        cmd = [
            "ffprobe", "-v", "quiet", 
            "-print_format", "json", 
            "-show_format", "-show_streams", 
            str(video_path)
        ]
        result = subprocess.run(cmd, capture_output=True, text=True, check=True)
        data = json.loads(result.stdout)
        
        # Format info
        fmt = data.get("format", {})
        duration = float(fmt.get("duration", 0))
        
        # Stream info
        v_stream = next((s for s in data.get("streams", []) if s.get("codec_type") == "video"), {})
        a_stream = next((s for s in data.get("streams", []) if s.get("codec_type") == "audio"), None)
        
        width = int(v_stream.get("width", 0))
        height = int(v_stream.get("height", 0))
        v_codec = v_stream.get("codec_name", "unknown")
        a_codec = a_stream.get("codec_name") if a_stream else None
        
        # Calculate aspect ratio
        if width > 0 and height > 0:
            ratio = width / height
            if abs(ratio - (16/9)) < 0.1: ar = "16:9"
            elif abs(ratio - (9/16)) < 0.1: ar = "9:16"
            elif abs(ratio - 1.0) < 0.1: ar = "1:1"
            elif abs(ratio - (4/5)) < 0.1: ar = "4:5"
            else: ar = f"{width}:{height}"
        else:
            ar = "unknown"
            
        # FPS
        fps_item = v_stream.get("avg_frame_rate", "25/1")
        if "/" in fps_item:
            num, den = map(int, fps_item.split("/"))
            fps = num / den if den > 0 else 25.0
        else:
            fps = float(fps_item) if fps_item else 25.0
            
        return VideoInfo(
            path=str(video_path),
            width=width,
            height=height,
            duration=duration,
            aspect_ratio=ar,
            fps=fps,
            v_codec=v_codec,
            a_codec=a_codec
        )
    except Exception as e:
        logger.error(f"Failed to get video info for {video_path}: {e}")
        # Return a default/empty info to prevent crash, but log error
        return VideoInfo(str(video_path), 1920, 1080, 0.0, "16:9")

def detect_video_format(video_path: str) -> str:
    """
    Detect video format shorthand (16x9, 9x16, etc.)
    """
    info = get_video_info(video_path)
    ratio_map = {
        "16:9": "16x9",
        "9:16": "9x16",
        "1:1": "1x1",
        "4:5": "4x5"
    }
    return ratio_map.get(info.aspect_ratio, "16x9")
