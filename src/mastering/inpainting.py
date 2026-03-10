"""
Inpainting Service Module (from Mastering_video)
Video and image inpainting using LaMa (Large Mask Inpainting)
"""
import os
import cv2
import numpy as np
from typing import Optional
from PIL import Image


class InpaintingService:
    """
    Video and image inpainting using LaMa (Large Mask Inpainting).
    Video inpainting processes frame-by-frame with LaMa.
    """

    def __init__(self, output_dir: str = "outputs/inpaint", device: str = None):
        self.output_dir = output_dir
        self.lama_model = None
        os.makedirs(self.output_dir, exist_ok=True)

        # Detect device
        if device is None:
            import torch
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
        else:
            self.device = device

    # ---------------------------------------------------------------
    # LaMa model loading
    # ---------------------------------------------------------------
    def _load_lama(self):
        """Lazy-load LaMa model for inpainting."""
        if self.lama_model is not None:
            return
        print(f"Loading LaMa model on {self.device}...")
        from iopaint.model.lama import LaMa
        self.lama_model = LaMa(device=self.device)
        print("LaMa model loaded.")

    # ---------------------------------------------------------------
    # Single image inpainting
    # ---------------------------------------------------------------
    def inpaint(self, image_path: str, mask_path: str, output_name: Optional[str] = None) -> str:
        """Single image inpainting using LaMa."""
        self._load_lama()
        from iopaint.schema import InpaintRequest as Config

        image_bgr = cv2.imread(image_path)
        if image_bgr is None:
            raise ValueError(f"Could not read image: {image_path}")
        image = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)

        mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
        if mask is None:
            raise ValueError(f"Could not read mask: {mask_path}")
        if mask.shape[:2] != image.shape[:2]:
            mask = cv2.resize(mask, (image.shape[1], image.shape[0]))
        _, mask = cv2.threshold(mask, 127, 255, cv2.THRESH_BINARY)

        config = Config(
            ldm_steps=25, ldm_sampler="plms",
            hd_strategy="Crop", hd_strategy_crop_margin=64,
            hd_strategy_crop_trigger_size=800, hd_strategy_resize_limit=800,
        )
        result_bgr = self.lama_model(image, mask, config)

        if output_name is None:
            basename = os.path.splitext(os.path.basename(image_path))[0]
            output_name = f"{basename}_inpainted.png"
        output_path = os.path.join(self.output_dir, output_name)
        cv2.imwrite(output_path, result_bgr)
        print(f"Image inpainting complete: {output_path}")
        return output_path

    # ---------------------------------------------------------------
    # Single frame from video
    # ---------------------------------------------------------------
    def inpaint_video_frame(
        self, video_path: str, frame_number: int, mask_path: str, output_name: Optional[str] = None
    ) -> str:
        """Extract a frame from video, inpaint it with LaMa, return result."""
        cap = cv2.VideoCapture(video_path)
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_number)
        ret, frame = cap.read()
        cap.release()
        if not ret:
            raise ValueError(f"Could not read frame {frame_number} from {video_path}")

        temp_frame_path = os.path.join(self.output_dir, f"temp_frame_{frame_number}.png")
        cv2.imwrite(temp_frame_path, frame)
        if output_name is None:
            output_name = f"frame_{frame_number}_inpainted.png"
        result_path = self.inpaint(temp_frame_path, mask_path, output_name)
        if os.path.exists(temp_frame_path):
            os.remove(temp_frame_path)
        return result_path

    # ---------------------------------------------------------------
    # Full video inpainting (LaMa frame-by-frame)
    # ---------------------------------------------------------------
    def inpaint_video(
        self, video_path: str, mask_path: str, output_name: Optional[str] = None,
        start_time: float = 0.0, end_time: float = -1.0,
        resize_ratio: float = 1.0, dedup_threshold: float = 2.0,
        **kwargs,
    ) -> str:
        """
        Remove objects from video using LaMa (frame-by-frame inpainting).
        Optimized with frame dedup + resize for speed.

        Args:
            video_path: Path to the video file
            mask_path: Path to the mask (white = area to remove)
            output_name: Optional output filename
            start_time: Start processing from this time (seconds)
            end_time: End processing at this time (seconds, -1 = end)
            resize_ratio: Resize factor for processing (1.0 = full size, best quality)
            dedup_threshold: MSE threshold for frame dedup (lower = more unique frames)
        """
        import subprocess
        import shutil
        import time

        self._load_lama()
        from iopaint.schema import InpaintRequest as Config

        config = Config(
            ldm_steps=25, ldm_sampler="plms",
            hd_strategy="Resize",
            hd_strategy_crop_margin=196,
            hd_strategy_crop_trigger_size=2048,
            hd_strategy_resize_limit=1920,
        )

        # Open video
        print(f"📹 Reading video: {video_path}")
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            raise ValueError(f"Could not open video: {video_path}")

        fps = cap.get(cv2.CAP_PROP_FPS)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        vid_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        vid_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

        start_frame = int(start_time * fps) if start_time > 0 else 0
        end_frame = total_frames if end_time < 0 else min(int(end_time * fps), total_frames)
        num_frames = end_frame - start_frame

        # Calculate processing resolution
        proc_w = int(vid_width * resize_ratio)
        proc_h = int(vid_height * resize_ratio)
        proc_w = proc_w if proc_w % 2 == 0 else proc_w + 1
        proc_h = proc_h if proc_h % 2 == 0 else proc_h + 1

        print(f"   Original: {vid_width}x{vid_height}, Process at: {proc_w}x{proc_h} ({resize_ratio:.0%})")
        print(f"   {fps:.1f}fps, {num_frames} frames, dedup_threshold={dedup_threshold}")

        # Load and prepare mask
        mask_img = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
        if mask_img is None:
            raise ValueError(f"Could not read mask: {mask_path}")
        if mask_img.shape[:2] != (vid_height, vid_width):
            mask_img = cv2.resize(mask_img, (vid_width, vid_height))
        _, mask_img = cv2.threshold(mask_img, 127, 255, cv2.THRESH_BINARY)

        # Dilate mask slightly
        dilate_kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (21, 21))
        mask_dilated = cv2.dilate(mask_img, dilate_kernel, iterations=1)

        # Create feathered mask for smooth blending
        feather_radius = 21
        mask_float = mask_dilated.astype(np.float32) / 255.0
        mask_feathered = cv2.GaussianBlur(mask_float, (0, 0), feather_radius)
        mask_feathered = np.clip(mask_feathered, 0, 1)
        mask_blend = np.stack([mask_feathered] * 3, axis=-1)

        # Resize dilated mask for processing
        mask_proc = cv2.resize(mask_dilated, (proc_w, proc_h))
        _, mask_proc = cv2.threshold(mask_proc, 127, 255, cv2.THRESH_BINARY)

        # Precompute mask region for dedup comparison
        mask_coords = mask_img > 127
        has_mask = mask_coords.any()

        # Setup output paths
        if output_name is None:
            basename = os.path.splitext(os.path.basename(video_path))[0]
            output_name = f"{basename}_inpainted.mp4"

        output_path = os.path.join(self.output_dir, output_name)
        temp_raw_path = output_path + ".raw.mp4"

        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        writer = cv2.VideoWriter(temp_raw_path, fourcc, fps, (vid_width, vid_height))

        # Process frame by frame
        cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)
        print(f"🔧 Running LaMa inpainting ({num_frames} frames)...")

        prev_frame_gray = None
        prev_result_bgr = None
        inpainted_count = 0
        skipped_count = 0
        t_start = time.time()

        for i in range(num_frames):
            ret, frame = cap.read()
            if not ret:
                break

            # Frame Dedup: compare mask region only
            if has_mask and prev_frame_gray is not None:
                curr_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                diff = cv2.absdiff(curr_gray[mask_coords], prev_frame_gray[mask_coords])
                mse = float(diff.astype(float).mean())

                if mse < dedup_threshold and prev_result_bgr is not None:
                    blended = (frame.astype(np.float32) * (1 - mask_blend) +
                               prev_result_bgr.astype(np.float32) * mask_blend).astype(np.uint8)
                    writer.write(blended)
                    skipped_count += 1
                    prev_frame_gray = curr_gray
                    if (i + 1) % 50 == 0 or i == num_frames - 1:
                        pct = int((i + 1) / num_frames * 100)
                        elapsed = time.time() - t_start
                        print(f"   {pct}% ({i+1}/{num_frames}) | "
                              f"inpainted={inpainted_count} skipped={skipped_count} | "
                              f"{elapsed:.0f}s elapsed")
                    continue

            # Inpaint
            frame_small = cv2.resize(frame, (proc_w, proc_h))
            frame_small_rgb = cv2.cvtColor(frame_small, cv2.COLOR_BGR2RGB)

            result_bgr = self.lama_model(frame_small_rgb, mask_proc, config)

            # Pass 2: refine with eroded mask
            erode_kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (11, 11))
            mask_eroded = cv2.erode(mask_proc, erode_kernel, iterations=1)
            if mask_eroded.any():
                result_u8 = np.clip(result_bgr, 0, 255).astype(np.uint8)
                result_rgb_p2 = cv2.cvtColor(result_u8, cv2.COLOR_BGR2RGB)
                result_bgr = self.lama_model(result_rgb_p2, mask_eroded, config)

            # Scale result back if needed
            if resize_ratio < 1.0:
                result_bgr = cv2.resize(result_bgr, (vid_width, vid_height))

            # Blend with feathered mask
            result_final = (frame.astype(np.float32) * (1 - mask_blend) +
                            result_bgr.astype(np.float32) * mask_blend).astype(np.uint8)

            writer.write(result_final)
            inpainted_count += 1

            prev_frame_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            prev_result_bgr = result_bgr

            if (i + 1) % 30 == 0 or i == num_frames - 1:
                pct = int((i + 1) / num_frames * 100)
                elapsed = time.time() - t_start
                print(f"   {pct}% ({i+1}/{num_frames}) | "
                      f"inpainted={inpainted_count} skipped={skipped_count} | "
                      f"{elapsed:.0f}s elapsed")
                
                # Report progress
                if "progress_callback" in kwargs:
                    kwargs["progress_callback"](pct)

        cap.release()
        writer.release()

        total_time = time.time() - t_start
        skip_pct = skipped_count / max(num_frames, 1) * 100
        print(f"   ⚡ Done in {total_time:.1f}s | "
              f"Inpainted: {inpainted_count}, Skipped: {skipped_count} ({skip_pct:.0f}%)")

        # Re-encode to H.264
        print("   Re-encoding to H.264...")
        cmd = [
            "ffmpeg", "-y",
            "-i", temp_raw_path,
            "-c:v", "libx264",
            "-preset", "fast",
            "-crf", "18",
            "-pix_fmt", "yuv420p",
            output_path,
        ]
        result = subprocess.run(cmd, capture_output=True)
        if result.returncode != 0:
            print(f"   ⚠️ H.264 re-encode failed, using raw output")
            shutil.move(temp_raw_path, output_path)
        else:
            os.remove(temp_raw_path)

        # Add audio from original
        self._add_audio(output_path, video_path)

        print(f"✅ Video inpainting complete: {output_path}")
        return output_path

    # ---------------------------------------------------------------
    # Audio merge
    # ---------------------------------------------------------------
    def _add_audio(self, video_path: str, original_video: str):
        """Merge audio from original video into the processed video."""
        import subprocess
        import shutil

        temp_path = video_path + ".tmp.mp4"
        cmd = [
            "ffmpeg", "-y",
            "-i", video_path,
            "-i", original_video,
            "-c:v", "copy",
            "-c:a", "aac",
            "-map", "0:v:0",
            "-map", "1:a:0?",
            "-shortest",
            temp_path
        ]
        result = subprocess.run(cmd, capture_output=True)
        if result.returncode == 0:
            shutil.move(temp_path, video_path)
        elif os.path.exists(temp_path):
            os.remove(temp_path)
