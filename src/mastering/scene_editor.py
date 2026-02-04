"""
Scene Editor Module
Xóa/thêm cảnh và điều chỉnh nhạc
"""
import subprocess
import os
import logging
from pathlib import Path
from typing import Optional, List, Tuple
import tempfile

logger = logging.getLogger(__name__)


class SceneEditor:
    """Chỉnh sửa cảnh video và audio"""
    
    def __init__(self, output_dir: Optional[str] = None):
        self.output_dir = Path(output_dir) if output_dir else Path("./outputs")
        self.output_dir.mkdir(parents=True, exist_ok=True)
    
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
    
    def trim_video(
        self,
        video_path: str,
        start_time: float = 0,
        end_time: Optional[float] = None,
        output_path: Optional[str] = None
    ) -> str:
        """
        Cắt video theo thời gian
        
        Args:
            video_path: Đường dẫn video input
            start_time: Thời điểm bắt đầu (giây)
            end_time: Thời điểm kết thúc, None = đến cuối
            output_path: Đường dẫn output
            
        Returns:
            Đường dẫn video output
        """
        video_path = Path(video_path)
        
        if output_path is None:
            output_path = self.output_dir / f"{video_path.stem}_trimmed{video_path.suffix}"
        
        cmd = ["ffmpeg", "-y", "-i", str(video_path), "-ss", str(start_time)]
        
        if end_time is not None:
            duration = end_time - start_time
            cmd.extend(["-t", str(duration)])
        
        cmd.extend(["-c", "copy", str(output_path)])
        
        result = subprocess.run(cmd, capture_output=True, text=True)
        
        if result.returncode != 0:
            raise RuntimeError(f"FFmpeg failed: {result.stderr}")
        
        return str(output_path)
    
    def remove_segment(
        self,
        video_path: str,
        start_time: float,
        end_time: float,
        crossfade_duration: float = 0.5,
        output_path: Optional[str] = None
    ) -> str:
        """
        Xóa một đoạn video và nối các phần còn lại
        
        Args:
            video_path: Đường dẫn video input
            start_time: Thời điểm bắt đầu đoạn cần xóa
            end_time: Thời điểm kết thúc đoạn cần xóa
            crossfade_duration: Thời gian crossfade giữa 2 phần
            output_path: Đường dẫn output
            
        Returns:
            Đường dẫn video output
        """
        video_path = Path(video_path)
        video_duration = self._get_video_duration(str(video_path))
        
        if output_path is None:
            output_path = self.output_dir / f"{video_path.stem}_edited{video_path.suffix}"
        
        # Tạo 2 phần video tạm
        with tempfile.TemporaryDirectory() as tmpdir:
            part1 = os.path.join(tmpdir, "part1.mp4")
            part2 = os.path.join(tmpdir, "part2.mp4")
            
            # Cắt phần 1 (từ đầu đến start_time)
            if start_time > 0:
                subprocess.run([
                    "ffmpeg", "-y", "-i", str(video_path),
                    "-t", str(start_time),
                    "-c", "copy", part1
                ], capture_output=True)
            
            # Cắt phần 2 (từ end_time đến cuối)
            if end_time < video_duration:
                subprocess.run([
                    "ffmpeg", "-y", "-i", str(video_path),
                    "-ss", str(end_time),
                    "-c", "copy", part2
                ], capture_output=True)
            
            # Nối các phần
            parts = []
            if start_time > 0 and os.path.exists(part1):
                parts.append(part1)
            if end_time < video_duration and os.path.exists(part2):
                parts.append(part2)
            
            if not parts:
                raise ValueError("Cannot remove entire video")
            
            if len(parts) == 1:
                # Chỉ có 1 phần, copy trực tiếp
                subprocess.run([
                    "ffmpeg", "-y", "-i", parts[0],
                    "-c", "copy", str(output_path)
                ], capture_output=True)
            else:
                # Nối 2 phần với crossfade
                self._concat_with_crossfade(parts[0], parts[1], str(output_path), crossfade_duration)
        
        return str(output_path)
    
    def _concat_with_crossfade(
        self, 
        video1: str, 
        video2: str, 
        output_path: str, 
        crossfade_duration: float
    ):
        """Nối 2 video với crossfade"""
        duration1 = self._get_video_duration(video1)
        
        filter_complex = (
            f"[0:v][1:v]xfade=transition=fade:duration={crossfade_duration}:"
            f"offset={duration1 - crossfade_duration}[v];"
            f"[0:a][1:a]acrossfade=d={crossfade_duration}[a]"
        )
        
        cmd = [
            "ffmpeg", "-y",
            "-i", video1,
            "-i", video2,
            "-filter_complex", filter_complex,
            "-map", "[v]",
            "-map", "[a]",
            output_path
        ]
        
        result = subprocess.run(cmd, capture_output=True, text=True)
        
        if result.returncode != 0:
            # Fallback: simple concat without crossfade
            self._simple_concat([video1, video2], output_path)
    
    def _simple_concat(self, videos: List[str], output_path: str):
        """Nối các video đơn giản"""
        with tempfile.NamedTemporaryFile(mode='w', suffix='.txt', delete=False) as f:
            for video in videos:
                f.write(f"file '{video}'\n")
            listfile = f.name
        
        try:
            subprocess.run([
                "ffmpeg", "-y",
                "-f", "concat",
                "-safe", "0",
                "-i", listfile,
                "-c", "copy",
                output_path
            ], capture_output=True)
        finally:
            os.unlink(listfile)
    
    def concat_videos(
        self,
        video_paths: List[str],
        crossfade_duration: float = 0.5,
        output_path: Optional[str] = None
    ) -> str:
        """
        Nối nhiều video lại với nhau
        
        Args:
            video_paths: Danh sách đường dẫn video
            crossfade_duration: Thời gian crossfade
            output_path: Đường dẫn output
            
        Returns:
            Đường dẫn video output
        """
        if not video_paths:
            raise ValueError("No videos to concat")
        
        if len(video_paths) == 1:
            return video_paths[0]
        
        if output_path is None:
            output_path = self.output_dir / "concatenated.mp4"
        
        # Nối tuần tự
        current = video_paths[0]
        
        with tempfile.TemporaryDirectory() as tmpdir:
            for i, next_video in enumerate(video_paths[1:]):
                temp_output = os.path.join(tmpdir, f"concat_{i}.mp4")
                self._concat_with_crossfade(current, next_video, temp_output, crossfade_duration)
                current = temp_output
            
            # Copy kết quả cuối cùng
            subprocess.run([
                "ffmpeg", "-y",
                "-i", current,
                "-c", "copy",
                str(output_path)
            ], capture_output=True)
        
        return str(output_path)
    
    def adjust_audio_duration(
        self,
        video_path: str,
        target_duration: Optional[float] = None,
        fade_out_duration: float = 1.0,
        output_path: Optional[str] = None
    ) -> str:
        """
        Điều chỉnh audio cho phù hợp với video sau khi cắt
        
        Args:
            video_path: Đường dẫn video
            target_duration: Thời lượng mong muốn, None = giữ nguyên video duration
            fade_out_duration: Thời gian fade out audio ở cuối
            output_path: Đường dẫn output
            
        Returns:
            Đường dẫn video output
        """
        video_path = Path(video_path)
        video_duration = self._get_video_duration(str(video_path))
        
        if target_duration is None:
            target_duration = video_duration
        
        if output_path is None:
            output_path = self.output_dir / f"{video_path.stem}_audio_adjusted{video_path.suffix}"
        
        fade_start = target_duration - fade_out_duration
        
        audio_filter = f"afade=t=out:st={fade_start}:d={fade_out_duration}"
        
        cmd = [
            "ffmpeg", "-y",
            "-i", str(video_path),
            "-af", audio_filter,
            "-c:v", "copy",
            str(output_path)
        ]
        
        result = subprocess.run(cmd, capture_output=True, text=True)
        
        if result.returncode != 0:
            raise RuntimeError(f"FFmpeg failed: {result.stderr}")
        
        return str(output_path)
    
    def replace_audio(
        self,
        video_path: str,
        audio_path: str,
        keep_original_audio: bool = False,
        mix_volume: float = 0.5,
        output_path: Optional[str] = None
    ) -> str:
        """
        Thay thế hoặc mix audio
        
        Args:
            video_path: Đường dẫn video
            audio_path: Đường dẫn audio mới
            keep_original_audio: Giữ lại audio gốc và mix
            mix_volume: Volume của audio mới khi mix (0-1)
            output_path: Đường dẫn output
            
        Returns:
            Đường dẫn video output
        """
        video_path = Path(video_path)
        
        if output_path is None:
            output_path = self.output_dir / f"{video_path.stem}_new_audio{video_path.suffix}"
        
        if keep_original_audio:
            # Mix 2 audio tracks
            filter_complex = (
                f"[1:a]volume={mix_volume}[a1];"
                f"[0:a][a1]amix=inputs=2:duration=first[aout]"
            )
            cmd = [
                "ffmpeg", "-y",
                "-i", str(video_path),
                "-i", audio_path,
                "-filter_complex", filter_complex,
                "-map", "0:v",
                "-map", "[aout]",
                "-c:v", "copy",
                str(output_path)
            ]
        else:
            # Replace audio hoàn toàn
            cmd = [
                "ffmpeg", "-y",
                "-i", str(video_path),
                "-i", audio_path,
                "-map", "0:v",
                "-map", "1:a",
                "-c:v", "copy",
                "-shortest",
                str(output_path)
            ]
        
        result = subprocess.run(cmd, capture_output=True, text=True)
        
        if result.returncode != 0:
            raise RuntimeError(f"FFmpeg failed: {result.stderr}")
        
        return str(output_path)
