"""
Debug Gold Render — chạy từng bước pipeline để tìm lỗi.
Duo + tagline (THE NEW PERFUME) + subtitles, 16x9, 20s, MP4 1-pass draft.
"""
import sys, os, time, shutil
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from pathlib import Path

def step(n, msg):
    print(f"\n{'='*60}")
    print(f"  STEP {n}: {msg}")
    print(f"{'='*60}")
    return time.time()

def done(t0):
    elapsed = time.time() - t0
    print(f"  ✓ Done in {elapsed:.1f}s")
    return elapsed

# ── Config ──
PROJECT_ROOT = Path(__file__).resolve().parents[1]
WORK_DIR = PROJECT_ROOT / "outputs" / "debug_gold"
if WORK_DIR.exists():
    shutil.rmtree(WORK_DIR)
WORK_DIR.mkdir(parents=True)

print(f"Project root: {PROJECT_ROOT}")
print(f"Work dir:     {WORK_DIR}")

# ── Step 0: Load manifest & resolve assets ──
t0 = step(0, "Load manifest + resolve assets")
from src.mastering.gold_manifest import (
    load_manifest, resolve_opening_path, resolve_full_video_path,
    resolve_closing_path, resolve_sound_wav, resolve_logo_path,
    resolve_surimp_path, subtitle_lines_for,
)
data = load_manifest()
assembly = data.get("assembly", {})
corps_start = float(assembly.get("full_video_corps_start_seconds", 7.04))

open_path, open_ok = resolve_opening_path(data, PROJECT_ROOT)
full_path, full_ok = resolve_full_video_path(data, PROJECT_ROOT)
close_path, close_ok = resolve_closing_path("duo", "16x9", data, PROJECT_ROOT)
wav_path, wav_ok = resolve_sound_wav(20, data, PROJECT_ROOT)
logo_path, logo_ok = resolve_logo_path("16x9", data, PROJECT_ROOT)
sur_path, sur_ok = resolve_surimp_path("vo1", "duo", "16x9", data, PROJECT_ROOT)

print(f"  opening:  {open_ok} → {open_path}")
print(f"  full:     {full_ok} → {full_path}")
print(f"  closing:  {close_ok} → {close_path}")
print(f"  wav 20s:  {wav_ok} → {wav_path}")
print(f"  logo:     {logo_ok} → {logo_path}")
print(f"  surimp:   {sur_ok} → {sur_path}")

if not all([open_ok, full_ok, close_ok, wav_ok]):
    print("\n  ❌ MISSING CRITICAL ASSETS — cannot continue")
    sys.exit(1)
done(t0)

# ── Step 1: ffprobe durations ──
t0 = step(1, "ffprobe durations")
from src.mastering.gold_ffmpeg import ffprobe_duration_seconds
opening_dur = ffprobe_duration_seconds(open_path)
closing_dur = ffprobe_duration_seconds(close_path)
full_dur = ffprobe_duration_seconds(full_path)
wav_dur = ffprobe_duration_seconds(wav_path)
print(f"  opening: {opening_dur:.2f}s")
print(f"  closing: {closing_dur:.2f}s")
print(f"  full:    {full_dur:.2f}s")
print(f"  wav:     {wav_dur:.2f}s")

master_dur = 20.0
corps_dur = master_dur - opening_dur - closing_dur
print(f"  → corps to extract: {corps_dur:.2f}s (from {corps_start}s)")
if corps_dur < 0.1:
    print("  ❌ Corps duration too short!")
    sys.exit(1)
done(t0)

# ── Step 2: Extract corps ──
t0 = step(2, f"Extract corps ({corps_dur:.2f}s from full_video at {corps_start}s)")
from src.mastering.gold_ffmpeg import extract_video_segment
corps_mp4 = str(WORK_DIR / "corps.mp4")
extract_video_segment(full_path, corps_start, corps_dur, corps_mp4, strip_audio=True)
print(f"  Output: {corps_mp4} ({os.path.getsize(corps_mp4)} bytes)")
done(t0)

# ── Step 3: Concat opening + corps + closing ──
t0 = step(3, "Concat 3-part: opening + corps + closing")
from src.mastering.gold_ffmpeg import concat_three_videos_pad
master_concat = str(WORK_DIR / "master_20s.mp4")
concat_three_videos_pad(open_path, corps_mp4, close_path, master_concat, 1920, 1080)
print(f"  Output: {master_concat} ({os.path.getsize(master_concat)} bytes)")
concat_dur = ffprobe_duration_seconds(master_concat)
print(f"  Duration: {concat_dur:.2f}s")
done(t0)

# ── Step 4: Mux audio ──
t0 = step(4, "Mux WAV audio")
from src.mastering.gold_ffmpeg import mux_audio_replace
muxed = str(WORK_DIR / "muxed.mp4")
mux_audio_replace(master_concat, wav_path, muxed)
print(f"  Output: {muxed} ({os.path.getsize(muxed)} bytes)")
done(t0)

# ── Step 5: Overlay logo ──
current = muxed
if logo_ok:
    t0 = step(5, "Overlay logo")
    from src.mastering.gold_ffmpeg import overlay_png_xy
    ov_logo = str(WORK_DIR / "ov_logo.mp4")
    overlay_png_xy(current, logo_path, ov_logo, y_expr="48")
    print(f"  Output: {ov_logo} ({os.path.getsize(ov_logo)} bytes)")
    current = ov_logo
    done(t0)
else:
    print("\n  ⏭ Skip logo overlay (not found)")

# ── Step 6: Overlay surimp ──
if sur_ok:
    t0 = step(6, "Overlay surimp/tagline")
    ov_sur = str(WORK_DIR / "ov_sur.mp4")
    overlay_png_xy(current, sur_path, ov_sur, y_expr="main_h-overlay_h-40")
    print(f"  Output: {ov_sur} ({os.path.getsize(ov_sur)} bytes)")
    current = ov_sur
    done(t0)
else:
    print("\n  ⏭ Skip surimp overlay (not found)")

# ── Final Render using Unified Pipeline ──
t0 = step(7, "Unified Render (render_gold_job)")
from src.mastering.gold_render import render_gold_job
from src.mastering.gold_manifest import GoldJobSpec

spec = GoldJobSpec(
    vo="vo1",
    line_id="duo",
    line_label="VO1 Duo",
    layout="duo",
    branded=True,
    subtitles=True,
    video_format="16x9",
    duration_seconds=20,
    export_codec_id="h264_1pass",
    output_basename_hint="DEBUG_UNIFIED_16x9_20s"
)

def debug_status_cb(msg):
    print(f"  [UI STATUS] {msg}")

final_out, findings = render_gold_job(
    spec,
    PROJECT_ROOT,
    str(WORK_DIR),
    status_callback=debug_status_cb
)

print(f"\n{'='*60}")
print(f"  ✅ UNIFIED RENDER COMPLETE!")
print(f"  📂 Final file: {final_out}")
print(f"  🧠 AI Findings: {findings}")
print(f"{'='*60}")
