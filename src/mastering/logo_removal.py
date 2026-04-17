"""
Logo Removal Service Module (from Mastering_video)
AI-powered logo detection (Grounding DINO) + inpainting (LaMa)
"""
import os
import cv2
import numpy as np
import subprocess
import shutil
import torch
from typing import List, Optional, Tuple
from PIL import Image
from .scene_detection import Scene


class LogoRemovalService:
    """
    Unified scene-aware logo removal:
    1. Auto-detect logo with Grounding DINO (AI zero-shot detection)
    2. Generate mask automatically
    3. Inpaint only logo/intro scenes, keep content untouched
    4. Merge everything back
    """

    def __init__(self, output_dir: str = "outputs/logo_removal"):
        self.output_dir = output_dir
        self.temp_dir = os.path.join(output_dir, "temp_segments")
        os.makedirs(self.output_dir, exist_ok=True)
        self._gdino_model = None
        self._gdino_processor = None

    def _load_grounding_dino(self):
        """Lazy-load Grounding DINO model."""
        if self._gdino_model is not None:
            return

        print("🧠 Loading Grounding DINO model...")
        from transformers import AutoProcessor, AutoModelForZeroShotObjectDetection

        model_id = "IDEA-Research/grounding-dino-tiny"
        self._gdino_processor = AutoProcessor.from_pretrained(model_id)
        self._gdino_model = AutoModelForZeroShotObjectDetection.from_pretrained(model_id)
        self._gdino_model.eval()
        print("✅ Grounding DINO loaded")

    # =========================================
    # Auto Logo Detection (Grounding DINO)
    # =========================================
    def detect_logo_region(
        self, video_path: str, num_samples: int = 10, threshold: float = 0.15,
        time_ranges: Optional[List[Tuple[float, float]]] = None,
    ) -> dict:
        """
        Detect logo/watermark using Grounding DINO (AI zero-shot detection).

        Args:
            video_path: Path to the video
            num_samples: Number of frames to try (picks best detection)
            threshold: Confidence threshold for detection (0-1)
            time_ranges: If provided, sample frames from these time ranges (logo scenes)

        Returns:
            dict with: mask_path, bbox (x, y, w, h), confidence, found
        """
        self._load_grounding_dino()

        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            raise ValueError(f"Could not open video: {video_path}")

        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        fps = cap.get(cv2.CAP_PROP_FPS)
        vid_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        vid_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

        if total_frames < 1:
            cap.release()
            return {"found": False, "message": "Video too short"}

        # Pick sample frames from logo scenes (or middle of video)
        if time_ranges and fps > 0:
            sample_indices = []
            for start_t, end_t in time_ranges:
                mid_t = (start_t + end_t) / 2
                mid_f = int(mid_t * fps)
                sample_indices.append(min(mid_f, total_frames - 1))
                t1 = start_t + (end_t - start_t) * 0.33
                t2 = start_t + (end_t - start_t) * 0.66
                sample_indices.append(min(int(t1 * fps), total_frames - 1))
                sample_indices.append(min(int(t2 * fps), total_frames - 1))
            sample_indices = sorted(set(sample_indices))
            print(f"   Sampling {len(sample_indices)} frames from {len(time_ranges)} logo scene(s)")
        else:
            sample_indices = [total_frames // 2]

        text_prompt = "logo . watermark . brand logo . text overlay ."
        best_detection = None
        best_score = 0
        best_frame = None

        for idx in sample_indices:
            cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
            ret, frame = cap.read()
            if not ret:
                continue

            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            pil_image = Image.fromarray(frame_rgb)

            inputs = self._gdino_processor(
                images=pil_image,
                text=text_prompt,
                return_tensors="pt"
            )

            with torch.no_grad():
                outputs = self._gdino_model(**inputs)

            results = self._gdino_processor.post_process_grounded_object_detection(
                outputs,
                inputs["input_ids"],
                box_threshold=threshold,
                text_threshold=threshold,
                target_sizes=[(vid_height, vid_width)]
            )[0]

            if len(results["boxes"]) > 0:
                scores = results["scores"]
                max_idx = scores.argmax().item()
                score = scores[max_idx].item()

                if score > best_score:
                    best_score = score
                    box = results["boxes"][max_idx].tolist()
                    best_detection = {
                        "x1": int(box[0]),
                        "y1": int(box[1]),
                        "x2": int(box[2]),
                        "y2": int(box[3]),
                        "label": results["labels"][max_idx],
                    }
                    best_frame = frame.copy()

        cap.release()

        if best_detection is None:
            print("⚠️ Grounding DINO found no logo, falling back to static analysis...")
            return self._detect_logo_static(video_path, time_ranges=time_ranges)

        # Convert box to x, y, w, h with padding
        pad = 20
        x = max(0, best_detection["x1"] - pad)
        y = max(0, best_detection["y1"] - pad)
        x2 = min(vid_width, best_detection["x2"] + pad)
        y2 = min(vid_height, best_detection["y2"] + pad)
        w = x2 - x
        h = y2 - y

        cx, cy = x + w // 2, y + h // 2
        pos_x = "left" if cx < vid_width / 3 else ("right" if cx > vid_width * 2 / 3 else "center")
        pos_y = "top" if cy < vid_height / 3 else ("bottom" if cy > vid_height * 2 / 3 else "middle")
        position = f"{pos_y}-{pos_x}"

        mask = np.zeros((vid_height, vid_width), dtype=np.uint8)
        mask[y:y+h, x:x+w] = 255

        import time as _time
        mask_id = int(_time.time() * 1000) % 100000
        mask_path = os.path.join(self.output_dir, f"auto_logo_mask_{mask_id}.png")
        cv2.imwrite(mask_path, mask)

        preview_path = ""
        if best_frame is not None:
            cv2.rectangle(best_frame, (x, y), (x + w, y + h), (0, 165, 255), 3)
            label_text = f"Logo ({position}) {best_score:.0%}"
            cv2.putText(best_frame, label_text, (x, y - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 165, 255), 2)
            preview_path = os.path.join(self.output_dir, "logo_preview.png")
            cv2.imwrite(preview_path, best_frame)

        print(f"🎯 Grounding DINO detected logo at {position}: ({x}, {y}, {w}x{h}), "
              f"confidence={best_score:.2%}, label='{best_detection['label']}'")

        return {
            "found": True,
            "corner": position,
            "bbox": {"x": x, "y": y, "w": w, "h": h},
            "confidence": round(best_score, 4),
            "mask_path": mask_path,
            "preview_path": preview_path,
            "method": "grounding_dino",
            "label": best_detection["label"],
        }

    def _detect_logo_static(
        self, video_path: str,
        time_ranges: Optional[List[Tuple[float, float]]] = None,
        num_samples: int = 15, threshold: int = 15,
    ) -> dict:
        """Fallback: detect logo via static pixel analysis (OpenCV only)."""
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            return {"found": False, "message": "Could not open video"}

        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        fps = cap.get(cv2.CAP_PROP_FPS)
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

        if time_ranges and fps > 0:
            indices = []
            for s, e in time_ranges:
                sf, ef = int(s * fps), min(int(e * fps), total_frames - 1)
                if ef > sf:
                    count = max(2, num_samples // len(time_ranges))
                    indices.extend(np.linspace(sf, ef, count, dtype=int).tolist())
            sample_indices = sorted(set(indices))
        else:
            sample_indices = np.linspace(
                int(total_frames * 0.1), int(total_frames * 0.9), num_samples, dtype=int
            ).tolist()

        frames = []
        for idx in sample_indices:
            cap.set(cv2.CAP_PROP_POS_FRAMES, int(idx))
            ret, frame = cap.read()
            if ret:
                frames.append(cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY).astype(np.float32))
        cap.release()

        if len(frames) < 3:
            return {"found": False, "message": "Not enough frames"}

        frame_stack = np.stack(frames, axis=0)
        pixel_std = np.std(frame_stack, axis=0)
        pixel_mean = np.mean(frame_stack, axis=0)

        static_mask = (pixel_std < threshold).astype(np.uint8) * 255
        brightness_mask = (pixel_mean > 30).astype(np.uint8) * 255
        static_mask = cv2.bitwise_and(static_mask, brightness_mask)

        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (7, 7))
        static_mask = cv2.morphologyEx(static_mask, cv2.MORPH_CLOSE, kernel)
        static_mask = cv2.morphologyEx(static_mask, cv2.MORPH_OPEN,
                                        cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3)))

        contours, _ = cv2.findContours(static_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if not contours:
            return {"found": False, "message": "No static regions"}

        min_area = width * height * 0.001
        max_area = width * height * 0.3
        valid = [(c, cv2.contourArea(c)) for c in contours
                 if min_area < cv2.contourArea(c) < max_area]

        if not valid:
            return {"found": False, "message": "No valid regions"}

        combined = np.zeros((height, width), dtype=np.uint8)
        for c, _ in valid:
            cv2.drawContours(combined, [c], -1, 255, -1)

        dilated = cv2.dilate(combined, cv2.getStructuringElement(cv2.MORPH_RECT, (20, 20)), iterations=2)
        merged, _ = cv2.findContours(dilated, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        if not merged:
            return {"found": False, "message": "No regions after merge"}

        best = max(merged, key=cv2.contourArea)
        x, y, w, h = cv2.boundingRect(best)
        pad = 15
        x, y = max(0, x - pad), max(0, y - pad)
        w = min(width - x, w + 2 * pad)
        h = min(height - y, h + 2 * pad)

        mask = np.zeros((height, width), dtype=np.uint8)
        mask[y:y+h, x:x+w] = 255

        import time as _time
        mask_id = int(_time.time() * 1000) % 100000
        mask_path = os.path.join(self.output_dir, f"auto_logo_mask_{mask_id}.png")
        cv2.imwrite(mask_path, mask)

        cx, cy = x + w // 2, y + h // 2
        pos_x = "left" if cx < width / 3 else ("right" if cx > width * 2 / 3 else "center")
        pos_y = "top" if cy < height / 3 else ("bottom" if cy > height * 2 / 3 else "middle")
        position = f"{pos_y}-{pos_x}"

        confidence = cv2.contourArea(best) / (width * height)
        print(f"🎯 Static analysis detected logo at {position}: ({x}, {y}, {w}x{h})")

        return {
            "found": True,
            "corner": position,
            "bbox": {"x": x, "y": y, "w": w, "h": h},
            "confidence": round(confidence, 4),
            "mask_path": mask_path,
            "preview_path": "",
            "method": "static_analysis",
        }

    # =========================================
    # Scene Filtering
    # =========================================
    def get_logo_scenes(self, scenes: List[Scene]) -> List[Scene]:
        """Filter scenes classified as logo or intro."""
        return [s for s in scenes if "logo" in s.scene_type or "intro" in s.scene_type]

    # =========================================
    # Unified Pipeline: detect + remove
    # =========================================
    def auto_remove_logo(
        self,
        video_path: str,
        scenes: List[Scene],
        output_name: Optional[str] = None,
        mask_path: Optional[str] = None,
        inpainting_service=None,
    ) -> dict:
        """
        Full pipeline:
        1. If no mask provided, auto-detect logo and generate mask
        2. Filter for logo/intro scenes
        3. Inpaint only those scenes
        4. Merge everything back
        """
        if inpainting_service is None:
            from .inpainting import InpaintingService
            inpainting_service = InpaintingService()

        # Filter logo scenes
        logo_scenes = self.get_logo_scenes(scenes)
        if not logo_scenes:
            return {
                "status": "no_logo_scenes",
                "output_path": video_path,
                "detection": {},
                "message": "No scenes classified as logo/intro. Nothing to process."
            }

        print(f"📋 {len(logo_scenes)}/{len(scenes)} scenes contain logo → processing only these")

        # Process segments — detect logo PER SCENE
        os.makedirs(self.temp_dir, exist_ok=True)

        try:
            segment_files = []

            for i, scene in enumerate(scenes):
                segment_path = os.path.join(self.temp_dir, f"seg_{i:03d}.mp4")
                is_logo = "logo" in scene.scene_type or "intro" in scene.scene_type

                # Extract segment
                duration = scene.end_time - scene.start_time
                cmd_extract = [
                    "ffmpeg", "-y",
                    "-ss", str(scene.start_time),
                    "-i", video_path,
                    "-t", str(duration),
                    "-c:v", "libx264",
                    "-preset", "fast",
                    "-crf", "18",
                    "-pix_fmt", "yuv420p",
                    "-an",
                    "-avoid_negative_ts", "1",
                    segment_path
                ]
                subprocess.run(cmd_extract, capture_output=True, check=True)

                if is_logo:
                    print(f"  🔧 Scene {i+1}/{len(scenes)} ({scene.scene_type}: {scene.start_time:.1f}s - {scene.end_time:.1f}s)")

                    scene_time_range = [(scene.start_time, scene.end_time)]
                    if mask_path is None:
                        print(f"    🔍 Detecting logo position for this scene...")
                        scene_detection = self.detect_logo_region(
                            video_path, num_samples=5, time_ranges=scene_time_range
                        )
                        if not scene_detection.get("found"):
                            print(f"    ⚠️ No logo found in this scene, keeping original")
                            segment_files.append(segment_path)
                            continue
                        scene_mask_path = scene_detection["mask_path"]
                        print(f"    🎯 Logo at {scene_detection.get('bbox', '?')}")
                    else:
                        scene_mask_path = mask_path

                    inpainted_name = f"seg_{i:03d}_inpainted.mp4"
                    inpainted_path = inpainting_service.inpaint_video(
                        segment_path, scene_mask_path, inpainted_name
                    )
                    segment_files.append(inpainted_path)
                else:
                    print(f"  ✓ Keeping scene {i+1}/{len(scenes)} ({scene.scene_type})")
                    segment_files.append(segment_path)

            # Merge
            if output_name is None:
                basename = os.path.splitext(os.path.basename(video_path))[0]
                output_name = f"{basename}_logo_removed.mp4"

            output_path = os.path.join(self.output_dir, output_name)
            self._merge_segments(segment_files, output_path)

            # Add audio from original
            self._add_audio(output_path, video_path)

            print(f"✅ Logo removal complete: {output_path}")
            return {
                "status": "success",
                "output_path": output_path,
                "detection": {},
                "logo_scenes_found": len(logo_scenes),
                "total_scenes": len(scenes),
                "scenes_processed": len(logo_scenes),
                "scenes_skipped": len(scenes) - len(logo_scenes),
            }

        finally:
            if os.path.exists(self.temp_dir):
                shutil.rmtree(self.temp_dir)

    def _merge_segments(self, segment_files: List[str], output_path: str):
        """Merge video segments using FFmpeg concat demuxer."""
        concat_file = os.path.join(self.output_dir, "concat_list.txt")
        with open(concat_file, "w") as f:
            for seg in segment_files:
                f.write(f"file '{os.path.abspath(seg)}'\n")

        cmd = [
            "ffmpeg", "-y",
            "-f", "concat", "-safe", "0",
            "-i", concat_file,
            "-c", "copy",
            output_path
        ]
        subprocess.run(cmd, capture_output=True, check=True)
        os.remove(concat_file)

    def _add_audio(self, video_path: str, original_video: str):
        """Merge audio from original video into the processed video."""
        temp_output = video_path + ".tmp.mp4"
        cmd = [
            "ffmpeg", "-y",
            "-i", video_path,
            "-i", original_video,
            "-c:v", "copy", "-c:a", "aac",
            "-map", "0:v:0", "-map", "1:a:0?",
            "-shortest",
            temp_output
        ]
        result = subprocess.run(cmd, capture_output=True)
        if result.returncode == 0:
            os.replace(temp_output, video_path)
        elif os.path.exists(temp_output):
            os.remove(temp_output)
    def detect_subject(
        self, video_path: str, subject_prompt: str = "perfume bottle . product .", 
        num_samples: int = 5, threshold: float = 0.2
    ) -> dict:
        """
        Detect the main subject (e.g. perfume bottle) to avoid covering it.
        Returns relative coordinates (0-1).
        """
        self._load_grounding_dino()
        cap = cv2.VideoCapture(video_path)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        v_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        v_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        
        best_score = 0
        best_box = None
        
        # Sample frames from 20%, 40%, 60%, 80% to find product
        for i in range(num_samples):
            target = int((total_frames * (i + 1)) / (num_samples + 1))
            cap.set(cv2.CAP_PROP_POS_FRAMES, target)
            ret, frame = cap.read()
            if not ret: continue
            
            img_pil = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
            inputs = self._gdino_processor(images=img_pil, text=subject_prompt, return_tensors="pt")
            with torch.no_grad():
                outputs = self._gdino_model(**inputs)
            
            target_sizes = torch.Tensor([img_pil.size[::-1]])
            results = self._gdino_processor.post_process_grounded_object_detection(
                outputs, inputs.input_ids, box_threshold=threshold, text_threshold=threshold, target_sizes=target_sizes
            )[0]
            
            for score, box in zip(results["scores"], results["boxes"]):
                if score > best_score:
                    best_score = score.item()
                    best_box = box.tolist()
        
        cap.release()
        
        if best_box:
            x1, y1, x2, y2 = best_box
            return {
                "found": True,
                "bbox": [x1, y1, x2 - x1, y2 - y1],
                "x_rel": (x1 + x2) / (2 * v_w),
                "y_rel": (y1 + y2) / (2 * v_h),
                "confidence": best_score
            }
        return {"found": False}

    def hunt_appearance_hybrid(
        self, video_path: str, prompt: str = "perfume bottle . product .", 
        threshold: float = 0.25
    ) -> Optional[float]:
        """
        Hybrid scan: 
        1. Use FFmpeg-based scene detection to find candidate frames (scene starts).
        2. Use Grounding DINO to verify which scene first contains the subject.
        Returns the timestamp in seconds.
        """
        print(f"🎬 Hybrid Hunter: Starting scene-aware scan for '{prompt}'...")
        
        try:
            from .scene_detection import SceneDetector
            sd = SceneDetector(output_dir=self.temp_dir)
            scenes = sd.detect_scenes(video_path, threshold=25.0) # More sensitive
            
            if not scenes:
                print("⚠️ Hybrid Hunter: No distinct scenes detected. Falling back to linear hunter.")
                return self.hunt_appearance(video_path, prompt, threshold)
            
            self._load_grounding_dino()
            cap = cv2.VideoCapture(video_path)
            
            for i, scene in enumerate(scenes):
                # Sample the exact start of the scene
                cap.set(cv2.CAP_PROP_POS_FRAMES, scene.start_frame)
                ret, frame = cap.read()
                if not ret: continue
                
                img_pil = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
                inputs = self._gdino_processor(images=img_pil, text=prompt, return_tensors="pt")
                with torch.no_grad():
                    outputs = self._gdino_model(**inputs)
                
                target_sizes = torch.Tensor([img_pil.size[::-1]])
                results = self._gdino_processor.post_process_grounded_object_detection(
                    outputs, inputs.input_ids, box_threshold=threshold, text_threshold=threshold, target_sizes=target_sizes
                )[0]
                
                if len(results["scores"]) > 0:
                    timestamp = scene.start_time
                    score = results["scores"][0].item()
                    print(f"🎯 Hybrid Hunter MATCH: Scene {i} starts at {timestamp:.2f}s (conf={score:.2f})")
                    cap.release()
                    return timestamp
            
            cap.release()
            print("ℹ️ Hybrid Hunter: No product found at scene starts. Trying linear scan for safety.")
            return self.hunt_appearance(video_path, prompt, threshold)
            
        except Exception as e:
            print(f"❌ Hybrid Hunter error: {e}. Falling back to linear scan.")
            return self.hunt_appearance(video_path, prompt, threshold)

    def hunt_appearance(
        self, video_path: str, prompt: str = "perfume bottle . product .", 
        threshold: float = 0.25, step_seconds: float = 0.4
    ) -> Optional[float]:
        """
        Scan video to find the first second where the subject appears.
        Returns the timestamp in seconds.
        """
        self._load_grounding_dino()
        cap = cv2.VideoCapture(video_path)
        fps = cap.get(cv2.CAP_PROP_FPS)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        if fps <= 0 or total_frames <= 0:
            cap.release()
            return None
        
        # Scan every step_seconds
        step_frames = max(1, int(step_seconds * fps))
        
        for f_idx in range(0, total_frames, step_frames):
            cap.set(cv2.CAP_PROP_POS_FRAMES, f_idx)
            ret, frame = cap.read()
            if not ret: break
            
            img_pil = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
            inputs = self._gdino_processor(images=img_pil, text=prompt, return_tensors="pt")
            with torch.no_grad():
                outputs = self._gdino_model(**inputs)
            
            target_sizes = torch.Tensor([img_pil.size[::-1]])
            results = self._gdino_processor.post_process_grounded_object_detection(
                outputs, inputs.input_ids, box_threshold=threshold, text_threshold=threshold, target_sizes=target_sizes
            )[0]
            
            if len(results["scores"]) > 0:
                timestamp = f_idx / fps
                print(f"🎯 AI Hunter: Detected '{prompt}' at {timestamp:.2f}s")
                cap.release()
                return timestamp
                
        cap.release()
        return None
