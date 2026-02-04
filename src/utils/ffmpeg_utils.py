
import os
import shutil
import subprocess
import logging
from pathlib import Path

logger = logging.getLogger(__name__)

def get_ffmpeg_path() -> str:
    """
    Get the path to the ffmpeg executable.
    Prioritizes:
    1. FFMPEG_BINARY environment variable
    2. Common Windows Winget paths
    3. System PATH
    """
    # 1. Environment variable
    env_path = os.getenv("FFMPEG_BINARY")
    if env_path and os.path.isfile(env_path):
        return env_path
        
    # 2. Common Windows Winget path
    user_profile = os.environ.get("USERPROFILE")
    if user_profile:
        winget_path = Path(user_profile) / "AppData" / "Local" / "Microsoft" / "WinGet" / "Links" / "ffmpeg.exe"
        if winget_path.exists():
            return str(winget_path)
            
    # 3. System PATH
    system_path = shutil.which("ffmpeg")
    if system_path:
        return system_path
        
    # Fallback to just "ffmpeg" and hope for the best, or return usage/error
    logger.warning("FFmpeg not found in known locations or PATH. Defaulting to 'ffmpeg'")
    return "ffmpeg"

def get_ffprobe_path() -> str:
    """
    Get the path to the ffprobe executable.
    Similar logic to get_ffmpeg_path
    """
    # 1. Environment variable
    env_path = os.getenv("FFPROBE_BINARY")
    if env_path and os.path.isfile(env_path):
        return env_path
        
    # 2. Common Windows Winget path
    user_profile = os.environ.get("USERPROFILE")
    if user_profile:
        winget_path = Path(user_profile) / "AppData" / "Local" / "Microsoft" / "WinGet" / "Links" / "ffprobe.exe"
        if winget_path.exists():
            return str(winget_path)
            
    # 3. System PATH
    system_path = shutil.which("ffprobe")
    if system_path:
        return system_path
        
    return "ffprobe"
