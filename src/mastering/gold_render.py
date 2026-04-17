"""
Burberry Gold POS — assemble opening + corps + closing, mux WAV, overlay, subtitles, encode.
"""
from __future__ import annotations

import logging
import os
import shutil
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple, Union

from .gold_ffmpeg import (
    concat_three_videos_pad,
    encode_mp4_h264_1pass_draft,
    encode_mp4_h264_2pass,
    encode_prores_mov,
    extract_video_segment,
    ffprobe_duration_seconds,
    mux_audio_replace,
    overlay_png_xy,
    run_ffmpeg,
    scale_pad_video_file,
    trim_to_duration,
    extract_audio_for_transcription,
)
from .gold_manifest import (
    GoldJobSpec,
    load_manifest,
    resolve_closing_path,
    resolve_full_video_path,
    resolve_logo_path,
    resolve_opening_path,
    resolve_sound_wav,
    resolve_surimp_path,
    subtitle_lines_for,
)
# Heavy AI imports moved inside render_gold_job to avoid worker hangs at init time

logger = logging.getLogger(__name__)


def _gold_step(msg: str) -> None:
    """Set GOLD_VERBOSE_STEPS=1 to print each pipeline stage for debugging."""
    if os.getenv("GOLD_VERBOSE_STEPS") == "1":
        print(f"[Gold step] {msg}", flush=True)


def distribute_subtitles_evenly(
    lines: List[str],
    total_duration: float,
    margin: float = 0.35,
) -> List[Tuple[float, float, str]]:
    if not lines or total_duration <= margin * 2:
        return []
    inner = total_duration - 2 * margin
    n = len(lines)
    slot = inner / n
    out: List[Tuple[float, float, str]] = []
    for i, text in enumerate(lines):
        start = margin + i * slot
        end = margin + (i + 1) * slot
        out.append((start, end, text.strip()))
    return out


def _export_extension(export_codec_id: str, data: Dict[str, Any]) -> str:
    for c in data.get("export_codecs", []):
        if c["id"] == export_codec_id:
            return str(c.get("extension", ".mp4"))
    return ".mp4"


def render_gold_job(
    spec: GoldJobSpec,
    project_root: Path,
    output_dir: Union[str, Path],
    shared_ai_data: Optional[Dict[str, Any]] = None,
    status_callback: Optional[Callable[[str], None]] = None,
) -> Tuple[str, Dict[str, Any]]:
    """
    Full pipeline for one GoldJobSpec. Returns (path to final file, findings metadata).
    """
    output_dir = Path(output_dir)
    project_root = Path(project_root)
    findings = {
        "ai_enabled": os.getenv("GOLD_SKIP_AI") != "1",
        "subject_detection": {"found": False},
        "logo_detection": {"found": False},
        "subtitle_sync": {"status": "skipped"}
    }
    data = load_manifest()
    assembly = data.get("assembly", {})
    corps_start = float(assembly.get("full_video_corps_start_seconds", 7.04))

    open_path, open_ok = resolve_opening_path(data, project_root)
    full_path, full_ok = resolve_full_video_path(data, project_root)
    close_path, close_ok = resolve_closing_path(spec.layout, spec.video_format, data, project_root)
    wav_path, wav_ok = resolve_sound_wav(spec.duration_seconds, data, project_root)

    if not all([open_ok, full_ok, close_ok, wav_ok]):
        raise FileNotFoundError(
            "Missing assets: "
            f"opening={open_ok} full={full_ok} closing={close_ok} wav={wav_ok}"
        )

    _gold_step(
        f"Assets OK — opening={open_path} | full={full_path} | closing={close_path} | wav={wav_path}"
    )
    if status_callback: status_callback("Step 1/7: Initializing assets")

    opening_dur = ffprobe_duration_seconds(open_path)
    closing_dur = ffprobe_duration_seconds(close_path)
    target_dur = float(spec.duration_seconds)

    # Always build 20s master (opening + corps + closing), then trim for shorter spots.
    master_dur_seconds = max(data.get("durations_seconds", [20]))
    master_dur = float(master_dur_seconds)
    needs_trim = target_dur < master_dur - 0.01

    _gold_step(
        f"Timing: opening={opening_dur:.2f}s closing={closing_dur:.2f}s target={target_dur}s "
        f"master={master_dur}s → {'trim-from-master' if needs_trim else 'full-assembly'} "
        f"(corps_start={corps_start}s)"
    )

    work: Optional[Path] = None
    work = output_dir / f"{spec.output_basename_hint}_work"
    if status_callback: status_callback("Step 2/7: Preparing work directory")
    if work.exists():
        shutil.rmtree(work)
    work.mkdir(parents=True, exist_ok=True)

    try:
        # --- NEW: Smart Duration Assembly ---
        # Priority: 
        # 1. Closing (8.24s) - Packshot/Branding
        # 2. Opening (7.04s) - Brand Hook
        # 3. Corps (Variable) - Content filler
        
        master_concat = work / f"master_{spec.video_format}.mp4"
        
        if target_dur >= opening_dur + closing_dur:
            # Case 20s or 15.28s+: Opening + Corps + Closing
            corps_dur = target_dur - opening_dur - closing_dur
            _gold_step(f"Smart Assembly: Opening({opening_dur:.2f}s) + Corps({corps_dur:.2f}s) + Closing({closing_dur:.2f}s) = {target_dur}s")
            
            corps_mp4 = work / "corps.mp4"
            extract_video_segment(full_path, corps_start, corps_dur, str(corps_mp4), strip_audio=True)
            
            assembled_closing_start = opening_dur + corps_dur
            concat_three_videos_pad(open_path, str(corps_mp4), close_path, str(master_concat), 1920, 1080)
        
        elif target_dur > closing_dur:
            # Case 15s, 10s: Partial Opening + Closing
            needed_opening = target_dur - closing_dur
            _gold_step(f"Smart Assembly: Opening_head({needed_opening:.2f}s) + Closing({closing_dur:.2f}s) = {target_dur}s")
            
            # Use head of opening for brand hook / logo focus
            trimmed_open = work / "open_trim.mp4"
            # Start at 0 to include the mandatory brand hook
            extract_video_segment(open_path, 0, needed_opening, str(trimmed_open), strip_audio=True)
            
            assembled_closing_start = needed_opening
            # Simple concat of two
            run_ffmpeg([
                "ffmpeg", "-nostdin", "-y", "-i", str(trimmed_open), "-i", close_path,
                "-filter_complex", "[0:v]scale=1920:1080:force_original_aspect_ratio=decrease,pad=1920:1080:(ow-iw)/2:(oh-ih)/2[v0];[1:v]scale=1920:1080:force_original_aspect_ratio=decrease,pad=1920:1080:(ow-iw)/2:(oh-ih)/2[v1];[v0][v1]concat=n=2:v=1:a=0[v]",
                "-map", "[v]", "-c:v", "libx264", "-preset", "veryfast", str(master_concat)
            ])
            
        else:
            # Case 6s, 5s: Hook-Packshot
            hook_dur = 1.0
            close_dur_trim = target_dur - hook_dur
            _gold_step(f"Smart Assembly: Opening_hook({hook_dur:.2f}s) + Closing_tail({close_dur_trim:.2f}s) = {target_dur}s")
            
            trimmed_open = work / "open_hook.mp4"
            # Take the very beginning of opening for definitive hook
            extract_video_segment(open_path, 0, hook_dur, str(trimmed_open), strip_audio=True)
            
            trimmed_close = work / "close_tail.mp4"
            close_start = max(0, closing_dur - close_dur_trim)
            extract_video_segment(close_path, close_start, close_dur_trim, str(trimmed_close), strip_audio=True)
            
            assembled_closing_start = hook_dur
            # Simple concat of two
            run_ffmpeg([
                "ffmpeg", "-nostdin", "-y", "-i", str(trimmed_open), "-i", str(trimmed_close),
                "-filter_complex", "[0:v]scale=1920:1080:force_original_aspect_ratio=decrease,pad=1920:1080:(ow-iw)/2:(oh-ih)/2[v0];[1:v]scale=1920:1080:force_original_aspect_ratio=decrease,pad=1920:1080:(ow-iw)/2:(oh-ih)/2[v1];[v0][v1]concat=n=2:v=1:a=0[v]",
                "-map", "[v]", "-c:v", "libx264", "-preset", "veryfast", str(master_concat)
            ])

        if status_callback: status_callback("Step 4/7: Assembling (Concat)")
        master_dur = target_dur # For downstream calculations

        # --- AI STEP 1: Smart Overlay Positioning ---
        # Share corner across formats (opening is usually same)
        detected_corner = None
        
        # Performance/Testing override: skip AI if explicitly requested
        skip_ai = os.getenv("GOLD_SKIP_AI") == "1"
        
        if shared_ai_data:
            detected_corner = shared_ai_data.get("detected_corner")

        if not skip_ai and not detected_corner:
            if status_callback: status_callback("Step 5/7: AI Processing (DINO/Whisper)")
            _gold_step("AI: Detecting logo position on opening video...")
            try:
                from .logo_removal import LogoRemovalService
                logo_svc = LogoRemovalService(output_dir=str(work))
                
                # Detect existing logo in Opening
                detect_res = logo_svc.detect_logo_region(str(open_path), num_samples=3)
                detected_corner = detect_res.get("corner", "top-center")
                if detect_res.get("found"):
                    findings["logo_detection"] = detect_res
                    _gold_step(f"AI: Found existing logo in Opening at {detected_corner}")
                    if status_callback: status_callback(f"AI: Detected existing brand logo at {detected_corner}")

                # NEW: Detect main subject (Product)
                subject_res = logo_svc.detect_subject(str(master_concat), num_samples=3)
                if subject_res.get("found"):
                    findings["subject_detection"] = subject_res
                    x_rel, y_rel = subject_res["x_rel"], subject_res["y_rel"]
                    _gold_step(f"AI: Product detected at x={x_rel:.2f}, y={y_rel:.2f}")
                    if status_callback: status_callback(f"AI: Detected product on {'Right' if x_rel > 0.6 else 'Left' if x_rel < 0.4 else 'Center'}")
                    
                    # Logic: Avoid product
                    if x_rel > 0.6: safe_side = "left"
                    elif x_rel < 0.4: safe_side = "right"
                    else: safe_side = "center"
                    
                    if y_rel < 0.4: safe_vert = "bottom"
                    elif y_rel > 0.6: safe_vert = "top"
                    else: safe_vert = "balance"
                    
                    # Adjust corner based on subject
                    if safe_side == "left":
                        detected_corner = "top-left" if safe_vert == "top" else "bottom-left"
                    elif safe_side == "right":
                        detected_corner = "top-right" if safe_vert == "top" else "bottom-right"
                    else:
                        detected_corner = "bottom-center" if safe_vert == "top" else "top-center"
                
                findings["branding_corner"] = detected_corner
                _gold_step(f"AI: Final branding corner set to {detected_corner}")
                if status_callback: status_callback(f"AI: Branding position set to {detected_corner}")

            except Exception as e:
                logger.warning(f"AI detection failed: {e}. Falling back to default center.")
                detected_corner = "top-center"
            
            # Update shared data if provided
            if shared_ai_data is not None:
                shared_ai_data["detected_corner"] = detected_corner
        
        if skip_ai or not detected_corner:
            if skip_ai: _gold_step("AI: Skipping logo detection (GOLD_SKIP_AI=1)")
            detected_corner = "top-center" # Default fallback
            
        # Default positions for tagline (Center-Left to match user sample)
        surimp_x = "40"; surimp_y = "(H-h)/2"

        # --- AI STEP 2: Subtitle Sync (Whisper + GPT) ---
        sync_subs = []
        if spec.subtitles:
            # Check cache per duration (audio is same across formats)
            # Use BASE VO for the key to share timing across languages (e.g. vo1 and vo1_fr)
            base_vo = spec.vo.split('_')[0] if '_' in spec.vo else spec.vo
            dur_key = f"subs_{base_vo}_{spec.duration_seconds}s"
            
            if shared_ai_data and dur_key in shared_ai_data:
                cached_data = shared_ai_data[dur_key]
                # If cached_data is (start, end, text, idx) list, map current lines to it by index
                current_lines = subtitle_lines_for(spec.vo, master_dur, data)
                
                for s, e, _t, original_idx in cached_data:
                    # original_idx helps if AI split/merged lines, but for direct map:
                    text = current_lines[original_idx] if 0 <= original_idx < len(current_lines) else ""
                    sync_subs.append((s, e, text))
                
                # Check for "word_timings" in cache for even better alignment
                if "word_timings" in cached_data[0] if cached_data else False:
                    # Logic to re-align words if needed
                    pass

                findings["subtitle_sync"] = {
                    "status": "cached",
                    "method": "shared_cache",
                    "base_vo": base_vo
                }
                _gold_step(f"AI: Reusing timing for {base_vo} (Excel-style matching by index)")
                if status_callback: status_callback(f"AI: Synchronized timing with {base_vo} (Matched by index)")
            else:
                # ALWAYS use Whisper for subtitle sync (regardless of GOLD_SKIP_AI)
                # GOLD_SKIP_AI only controls DINO product detection, NOT subtitle timing
                _gold_step("AI: Transcribing audio and aligning subtitles...")
                try:
                    # Use definitive WAV sound for transcription to ensure 100% sync
                    audio_sync_source = Path(wav_path) if os.path.exists(wav_path) else Path(master_concat)
                    _gold_step(f"AI: Transcribing {audio_sync_source.name}...")

                    from ..subtitle.timing_sync import TimingSync
                    syncer = TimingSync(use_local_whisper=False)
                    transcript, lang, word_timings = syncer.transcribe(str(audio_sync_source), language="en", prompt="Burberry Gold, lioness")
                    
                    sub_lines_full = subtitle_lines_for(spec.vo, master_dur, data)
                    aligned = syncer.align_subtitles_to_words(sub_lines_full, word_timings)
                    
                    # Convert to (start, end, text) and store full (start, end, text, idx) for cache
                    for s, e, t, _idx in aligned:
                        sync_subs.append((s, e, t))
                    
                    findings["subtitle_sync"] = {
                        "status": "ok",
                        "method": "whisper_gpt",
                        "line_count": len(sync_subs)
                    }
                    _gold_step(f"AI: Subtitles aligned OK ({len(sync_subs)} lines)")
                    
                    # Update shared data if provided (store with indexing info)
                    if shared_ai_data is not None:
                        shared_ai_data[dur_key] = aligned
                except Exception as e:
                    logger.error(f"Subtitle AI sync failed: {e}. Falling back to even distribution.")
                    _gold_step(f"AI: Whisper failed ({e}). Using even distribution fallback.")
                    sub_lines_fallback = subtitle_lines_for(spec.vo, master_dur, data)
                    sync_subs = distribute_subtitles_evenly(sub_lines_fallback, master_dur)
                    
                    if shared_ai_data is not None:
                         shared_ai_data[dur_key] = [(s, e, t, i) for i, (s, e, t) in enumerate(sync_subs)]

        # Step 2: Extract current duration segment
        if needs_trim:
            concat_mp4 = work / "trimmed_16x9.mp4"
            trim_to_duration(str(master_concat), target_dur, str(concat_mp4))
            _gold_step(f"Trimmed {master_dur}s → {target_dur}s: {concat_mp4}")
            
            # Trim subtitls list if needed (only keep those within target_dur)
            sync_subs = [s for s in sync_subs if s[0] < target_dur - 0.1]
        else:
            concat_mp4 = master_concat

        # Step 3: Format scaling
        if status_callback: status_callback(f"Step 5/7: Formatting ({spec.video_format})")
        fmt_w, fmt_h = 1920, 1080
        for f in data.get("formats", []):
            if f["id"] == spec.video_format:
                fmt_w = int(f["width"])
                fmt_h = int(f["height"])
                break

        scaled = work / "scaled.mp4"
        if spec.video_format == "16x9":
            shutil.copy2(str(concat_mp4), str(scaled))
        else:
            scale_pad_video_file(str(concat_mp4), str(scaled), fmt_w, fmt_h)
        _gold_step(f"Scaled {spec.video_format} → {scaled}")

        # Step 4: Add Audio
        muxed = work / "muxed.mp4"
        mux_audio_replace(str(scaled), wav_path, str(muxed))
        _gold_step(f"Mux WAV → {muxed}")

        # Step 5: Overlays (Branded mode)
        if status_callback: status_callback("Step 6/7: Applying logo & tagline overlays")
        current_video = str(muxed)
        if spec.branded:
            logo_path, logo_ok = resolve_logo_path(spec.video_format, data, project_root)
            sur_path, sur_ok = resolve_surimp_path(
                spec.vo, spec.layout, spec.video_format, data, project_root
            )
            
            # User feedback: Branding should only appear exactly when the product appears.
            # Using AI Hunter (Grounding DINO) to scan the closing segment for the first bottle shot.
            # Revised Fallback: If skip_ai=1, start at assembled_closing_start or master_dur - 4.4s
            # This ensures even without AI, the branding matches the product packshot 
            # and does NOT overlap with the Opening Hook (especially for 5s/6s).
            start_t = max(assembled_closing_start, master_dur - 4.4) 
            
            if not skip_ai:
                try:
                    # Refined prompt for Gold packshot: focus on gold bottles and golden background
                    product_prompt = "gold perfume bottles . golden bottle on yellow background . packshot bottles ."
                    _gold_step(f"AI Hunter: Scanning Closing segment for product appearance (Hybrid, Threshold=0.45)...")
                    relative_start = logo_svc.hunt_appearance_hybrid(str(close_path), prompt=product_prompt, threshold=0.45)
                    if relative_start is not None:
                        # User feedback: Delay branding a bit after detection to wait for "clear" video
                        start_t = max(assembled_closing_start + relative_start + 1.0, assembled_closing_start)
                        _gold_step(f"AI Hunter: Product detected at +{relative_start:.2f}s + 1.0s delay → Global {start_t:.2f}s")
                except Exception as e:
                    logger.warning(f"AI Hunter failed: {e}. Using fallback {start_t:.2f}s")
            
            if master_dur != 20 and start_t == 0: # Fallback for non 20s single-part
                start_t = master_dur * 0.7 
            
            _gold_step(f"Final Branding Timing: start_t={start_t:.2f}s, end_t={master_dur:.2f}s")
            end_t = master_dur

            # Skip logo overlay — surimp/tagline already contains "BURBERRY GOLD + THE NEW FRAGRANCE"
            # Only overlay the unified tagline block (matches user sample: single left-stack)
            
            if sur_ok:
                ov_sur = work / "ov_sur.mp4"
                _gold_step(f"Overlay Branding (Tagline only): {detected_corner} | Style: {spec.video_format}")
                # Massive scale (80%) per user request to "double" the size
                target_sur_w = int(fmt_w * 0.80) 
                # Margin at 5% to keep it centered-left but massive
                responsive_x = int(fmt_w * 0.05)
                sur_filter = f"[1:v]scale={target_sur_w}:-1[sur];[0:v][sur]overlay={responsive_x}:{surimp_y}:enable='between(t,{start_t},{end_t})'[v]"
                run_ffmpeg([
                    "ffmpeg", "-nostdin", "-y", "-i", current_video, "-i", sur_path,
                    "-filter_complex", sur_filter, "-map", "[v]", "-map", "0:a?", "-threads", "0", "-c:v", "libx264", "-preset", "veryfast", "-crf", "18", "-c:a", "copy", str(ov_sur)
                ])
                current_video = str(ov_sur)
                _gold_step(f"Overlay tagline ({surimp_x}, {surimp_y}, w={target_sur_w}) → {current_video}")
        
        muxed = current_video

        # Step 6: Subtitles (Burn via ASS)
        vf_burn: Optional[str] = None
        if spec.subtitles:
            subs = sync_subs
            if not subs: # Emergency fallback
                subs = distribute_subtitles_evenly(subtitle_lines_for(spec.vo, target_dur, data), target_dur)
            
            ass_path = work / "burn.ass"
            from ..subtitle.renderer import VIDEO_FORMATS, SubtitleRenderer
            renderer = SubtitleRenderer(output_dir=str(work))
            fc = renderer.font_manager.get_font_config(spec.vo.replace("vo", "en"))
            vf_obj = VIDEO_FORMATS.get(spec.video_format, VIDEO_FORMATS["16x9"])
            renderer.create_ass_file(subs, str(ass_path), fc, vf_obj)
            
            try:
                rel_ass = os.path.relpath(str(ass_path), os.getcwd()).replace("\\", "/")
            except ValueError:
                rel_ass = str(ass_path).replace("\\", "/")
            vf_burn = f"ass={rel_ass}"

        # Final Encode
        if status_callback: status_callback("Step 7/7: Final Encoding & Burning Subtitles")
        ext = _export_extension(spec.export_codec_id, data)
        final_out = output_dir / f"{spec.output_basename_hint}{ext}"

        if spec.export_codec_id == "prores_422":
            _gold_step(f"Encode ProRes → {final_out}")
            # Note: encode_prores_mov might need update if we want vf support; 
            # for now assuming no filters or direct if needed.
            # But we want to burn subs. Let's use 1pass draft logic if ProRes can't do VF (ProRes can do VF).
            encode_prores_mov(current_video, str(final_out)) # Simplified high-res export
            return str(final_out)

        if spec.export_codec_id in ["mp4_h264_1pass_draft", "h264_1pass", "mp4_standard"]:
            _gold_step(f"Encode MP4 1-pass draft → {final_out}")
            encode_mp4_h264_1pass_draft(current_video, str(final_out), vf_burn)
            return str(final_out), findings

        if spec.export_codec_id in ["mp4_20mbps_2pass", "h264_2pass", "mp4_high_quality"]:
            _gold_step(f"Encode MP4 2-pass → {final_out}")
            passlog = str(work / "x264_pass")
            encode_mp4_h264_2pass(current_video, str(final_out), passlog, vf_burn)
            return str(final_out), findings

        raise ValueError(f"Unknown export_codec_id: {spec.export_codec_id}")
    finally:
        if work is not None and work.exists():
            if os.getenv("GOLD_KEEP_WORK"):
                logger.info("Gold: GOLD_KEEP_WORK=1 — giữ thư mục tạm: %s", work)
                _gold_step(f"Giữ work dir (để xem từng file): {work}")
            else:
                try:
                    shutil.rmtree(work)
                except OSError:
                    pass
