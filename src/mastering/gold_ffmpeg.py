"""
FFmpeg helpers for Gold POS: concat segments, mux WAV, optional overlay (extend as needed).
"""
from __future__ import annotations

import glob
import os
import subprocess
import tempfile
from pathlib import Path
from typing import List, Optional, Sequence, Tuple


def _intermediate_preset() -> str:
    """libx264 preset for corps/concat/scale/overlay (not final 2-pass). veryfast = much quicker than fast."""
    p = os.getenv("GOLD_INTERMEDIATE_PRESET", "veryfast").strip()
    return p if p else "veryfast"


def _twopass_preset() -> str:
    """Final deliverable 2-pass H.264; medium = default quality/speed tradeoff. Override with GOLD_X264_2PASS_PRESET=faster."""
    p = os.getenv("GOLD_X264_2PASS_PRESET", "medium").strip()
    return p if p else "medium"


def _x264_threads() -> str:
    """0 = auto (dùng hết lõi hợp lý)."""
    return os.getenv("GOLD_X264_THREADS", "0").strip() or "0"


def run_ffmpeg(cmd: Sequence[str]) -> None:
    r = subprocess.run(cmd, capture_output=True, text=True)
    if r.returncode != 0:
        raise RuntimeError(f"ffmpeg failed: {r.stderr or r.stdout}")


def ffprobe_duration_seconds(path: str) -> float:
    r = subprocess.run(
        [
            "ffprobe",
            "-v",
            "error",
            "-show_entries",
            "format=duration",
            "-of",
            "default=noprint_wrappers=1:nokey=1",
            path,
        ],
        capture_output=True,
        text=True,
    )
    if r.returncode != 0:
        raise RuntimeError(f"ffprobe failed: {r.stderr}")
    return float(r.stdout.strip() or 0.0)


def scale_pad_filter(target_w: int, target_h: int) -> str:
    return (
        f"scale={target_w}:{target_h}:force_original_aspect_ratio=decrease,"
        f"pad={target_w}:{target_h}:(ow-iw)/2:(oh-ih)/2,setsar=1"
    )


def extract_video_segment(
    input_path: str,
    start_seconds: float,
    duration_seconds: float,
    output_path: str,
    strip_audio: bool = True,
) -> str:
    """Extract a segment; re-encode H.264 for reliable concat."""
    out = Path(output_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    
    # We use re-encoding here to ensure all segments have same params (1920x1080, H.264)
    # This prevents failures when source is ProRes .mov and target is .mp4
    cmd: List[str] = [
        "ffmpeg",
        "-nostdin",
        "-y",
        "-ss",
        str(start_seconds),
        "-t",
        str(duration_seconds),
        "-i",
        input_path,
        "-threads",
        _x264_threads(),
        "-vf",
        scale_pad_filter(1920, 1080),
        "-c:v",
        "libx264",
        "-preset",
        _intermediate_preset(),
        "-crf",
        "17",
    ]
    if strip_audio:
        cmd.extend(["-an"])
    else:
        cmd.extend(["-c:a", "aac", "-b:a", "192k"])
    cmd.append(str(out))
    run_ffmpeg(cmd)
    return str(out)


def concat_three_videos_pad(
    path_open: str,
    path_corps: str,
    path_close: str,
    output_path: str,
    target_w: int = 1920,
    target_h: int = 1080,
) -> str:
    """Concat three video files (video-only), scale/pad each to target size, re-encode."""
    out = Path(output_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    sf = scale_pad_filter(target_w, target_h)
    fc = (
        f"[0:v]{sf}[v0];[1:v]{sf}[v1];[2:v]{sf}[v2];"
        f"[v0][v1][v2]concat=n=3:v=1:a=0[outv]"
    )
    run_ffmpeg(
        [
            "ffmpeg",
            "-nostdin",
            "-y",
            "-i",
            path_open,
            "-i",
            path_corps,
            "-i",
            path_close,
            "-filter_complex",
            fc,
            "-map",
            "[outv]",
            "-threads",
            _x264_threads(),
            "-c:v",
            "libx264",
            "-preset",
            _intermediate_preset(),
            "-crf",
            "18",
            "-an",
            str(out),
        ]
    )
    return str(out)


def scale_pad_video_file(
    input_path: str,
    output_path: str,
    target_w: int,
    target_h: int,
) -> str:
    out = Path(output_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    sf = scale_pad_filter(target_w, target_h)
    run_ffmpeg(
        [
            "ffmpeg",
            "-nostdin",
            "-y",
            "-i",
            input_path,
            "-vf",
            sf,
            "-threads",
            _x264_threads(),
            "-c:v",
            "libx264",
            "-preset",
            _intermediate_preset(),
            "-crf",
            "18",
            "-c:a",
            "copy",
            str(out),
        ]
    )
    return str(out)


def overlay_png_xy(
    video_path: str,
    png_path: str,
    output_path: str,
    x_expr: str = "(main_w-overlay_w)/2",
    y_expr: str = "main_h-overlay_h-48",
    enable: Optional[str] = None,
) -> str:
    out = Path(output_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    
    ov_params = f"{x_expr}:{y_expr}"
    if enable:
        ov_params += f":enable='{enable}'"
        
    fc = f"[0:v][1:v]overlay={ov_params}[v]"
    run_ffmpeg(
        [
            "ffmpeg",
            "-nostdin",
            "-y",
            "-i",
            video_path,
            "-i",
            png_path,
            "-filter_complex",
            fc,
            "-map",
            "[v]",
            "-map",
            "0:a?",
            "-threads",
            _x264_threads(),
            "-c:v",
            "libx264",
            "-preset",
            _intermediate_preset(),
            "-crf",
            "18",
            "-c:a",
            "copy",
            str(out),
        ]
    )
    return str(out)


def encode_mp4_h264_2pass(
    input_video: str,
    output_mp4: str,
    passlogfile: str,
    video_filter: Optional[str] = None,
) -> None:
    """H.264 ~20 Mbps 2-pass; optional -vf (e.g. ass=...)."""
    out = Path(output_mp4)
    out.parent.mkdir(parents=True, exist_ok=True)
    pl = Path(passlogfile)
    pl.parent.mkdir(parents=True, exist_ok=True)
    base_cmd = [
        "ffmpeg",
        "-y",
        "-nostdin",
        "-i",
        input_video,
        "-threads",
        _x264_threads(),
        "-c:v",
        "libx264",
        "-preset",
        _twopass_preset(),
        "-b:v",
        "20M",
        "-maxrate",
        "20M",
        "-bufsize",
        "40M",
        "-passlogfile",
        str(pl),
    ]
    if video_filter:
        base_cmd.extend(["-vf", video_filter])

    cmd1 = [*base_cmd, "-pass", "1", "-an", "-f", "mp4", os.devnull]
    cmd2 = [
        *base_cmd,
        "-pass",
        "2",
        "-c:a",
        "aac",
        "-movflags",
        "+faststart",
        str(out),
    ]
    run_ffmpeg(cmd1)
    run_ffmpeg(cmd2)
    for g in glob.glob(str(pl) + "*"):
        try:
            os.unlink(g)
        except OSError:
            pass


def encode_mp4_h264_1pass_draft(
    input_video: str,
    output_mp4: str,
    video_filter: Optional[str] = None,
) -> None:
    """
    Single-pass H.264 (CRF) — nhanh hơn nhiều so với 2-pass; dùng xem thử / nháp.
    GOLD_X264_1PASS_PRESET (mặc định faster), GOLD_X264_1PASS_CRF (mặc định 21).
    """
    out = Path(output_mp4)
    out.parent.mkdir(parents=True, exist_ok=True)
    preset = os.getenv("GOLD_X264_1PASS_PRESET", "faster").strip() or "faster"
    crf = os.getenv("GOLD_X264_1PASS_CRF", "21").strip() or "21"
    cmd: List[str] = [
        "ffmpeg",
        "-y",
        "-nostdin",
        "-i",
        input_video,
        "-threads",
        _x264_threads(),
        "-c:v",
        "libx264",
        "-preset",
        preset,
        "-crf",
        crf,
    ]
    if video_filter:
        cmd.extend(["-vf", video_filter])
    cmd.extend(
        [
            "-c:a",
            "aac",
            "-movflags",
            "+faststart",
            str(out),
        ]
    )
    run_ffmpeg(cmd)


def encode_prores_mov(
    input_video: str, output_mov: str, video_filter: Optional[str] = None
) -> str:
    """ProRes 422 + AAC; optional -vf."""
    out = Path(output_mov)
    out.parent.mkdir(parents=True, exist_ok=True)
    cmd: List[str] = [
        "ffmpeg",
        "-y",
        "-nostdin",
        "-i",
        input_video,
    ]
    if video_filter:
        cmd.extend(["-vf", video_filter])
    cmd.extend(
        [
            "-c:v",
            "prores_ks",
            "-profile:v",
            "3",
            "-vendor",
            "apl0",
            "-c:a",
            "aac",
            str(out),
        ]
    )
    run_ffmpeg(cmd)
    return str(out)


def concat_videos_demuxer(paths: List[str], output_path: str) -> str:
    """Concatenate videos (same codec ideally) using concat demuxer."""
    if not paths:
        raise ValueError("No inputs")
    out = Path(output_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    if len(paths) == 1:
        # Copy container — caller may want re-encode
        run_ffmpeg(["ffmpeg", "-nostdin", "-y", "-i", paths[0], "-c", "copy", str(out)])
        return str(out)

    with tempfile.NamedTemporaryFile(mode="w", suffix=".txt", delete=False, encoding="utf-8") as f:
        for p in paths:
            ap = os.path.abspath(p).replace("'", "'\\''")
            f.write(f"file '{ap}'\n")
        list_path = f.name
    try:
        run_ffmpeg(
            [
                "ffmpeg",
                "-nostdin",
                "-y",
                "-f",
                "concat",
                "-safe",
                "0",
                "-i",
                list_path,
                "-c",
                "copy",
                str(out),
            ]
        )
    finally:
        try:
            os.unlink(list_path)
        except OSError:
            pass
    return str(out)


def mux_audio_replace(video_path: str, wav_path: str, output_path: str) -> str:
    """Replace video audio track with WAV (re-encode AAC)."""
    out = Path(output_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    run_ffmpeg(
        [
            "ffmpeg",
            "-nostdin",
            "-y",
            "-i",
            video_path,
            "-i",
            wav_path,
            "-map",
            "0:v:0",
            "-map",
            "1:a:0",
            "-c:v",
            "copy",
            "-c:a",
            "aac",
            "-shortest",
            str(out),
        ]
    )
    return str(out)


def extract_audio_for_transcription(video_path: str, output_path: str) -> str:
    """Extract mono 16khz audio (optimal for Whisper API)."""
    out = Path(output_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    run_ffmpeg(
        [
            "ffmpeg",
            "-nostdin",
            "-y",
            "-i",
            video_path,
            "-vn",
            "-acodec",
            "pcm_s16le",
            "-ar",
            "16000",
            "-ac",
            "1",
            str(out),
        ]
    )
    return str(out)


def trim_to_duration(video_path: str, duration_seconds: float, output_path: str) -> str:
    out = Path(output_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    run_ffmpeg(
        [
            "ffmpeg",
            "-nostdin",
            "-y",
            "-i",
            video_path,
            "-t",
            str(duration_seconds),
            "-c:v",
            "copy",
            "-c:a",
            "aac",
            str(out),
        ]
    )
    return str(out)


def overlay_png_bottom(
    video_path: str,
    png_path: str,
    output_path: str,
    y_expr: str = "main_h-overlay_h-40",
) -> str:
    """
    Overlay PNG on video (default: bottom; y_expr uses overlay filter names).
    """
    out = Path(output_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    vf = f"[0:v][1:v]overlay=0:{y_expr}[v]"
    run_ffmpeg(
        [
            "ffmpeg",
            "-nostdin",
            "-y",
            "-i",
            video_path,
            "-i",
            png_path,
            "-filter_complex",
            vf,
            "-map",
            "[v]",
            "-map",
            "0:a?",
            "-c:a",
            "copy",
            str(out),
        ]
    )
    return str(out)
