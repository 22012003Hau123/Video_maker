"""
Scene Detection Module (from Mastering_video)
Phát hiện và phân loại cảnh bằng PySceneDetect + CLIP
"""
import os
import cv2
from scenedetect import SceneManager
from scenedetect.detectors import ContentDetector
from dataclasses import dataclass
from typing import List, Optional
import time

@dataclass
class Scene:
    start_time: float
    end_time: float
    start_frame: int
    end_frame: int
    scene_type: str = "unknown"  # intro, content, outro, black_screen
    keyframe_path: str = ""

class SceneDetector:
    def __init__(self, output_dir: str = "outputs/scenes"):
        self.output_dir = output_dir
        os.makedirs(self.output_dir, exist_ok=True)

    def detect_scenes(self, video_path: str, threshold: float = 30.0) -> List[Scene]:
        """
        Detect scenes in video using ContentDetector.
        Returns a list of Scene objects.
        """
        if not os.path.exists(video_path):
            raise FileNotFoundError(f"Video not found: {video_path}")

        from scenedetect import open_video, SceneManager
        from scenedetect.detectors import AdaptiveDetector

        # Use open_video (modern API)
        video = open_video(video_path)
        
        # Calculate min_scene_len (approx 1.5 seconds)
        min_scene_len = int(video.frame_rate * 1.5)
        
        scene_manager = SceneManager()
        scene_manager.add_detector(AdaptiveDetector(adaptive_threshold=3.0, min_scene_len=min_scene_len))

        # Perform scene detection
        scene_manager.detect_scenes(video, show_progress=True)

        # Obtain list of detected scenes.
        scene_list = scene_manager.get_scene_list()
        
        results = []
        # Use cv2 for keyframe extraction (Video must be reopenable)
        cap = cv2.VideoCapture(video_path)
        
        for i, scene in enumerate(scene_list):
            start, end = scene
            start_sec = start.get_seconds()
            end_sec = end.get_seconds()
            start_frame = start.get_frames()
            end_frame = end.get_frames()
            
            # Extract keyframe (middle of scene)
            mid_frame = (start_frame + end_frame) // 2
            cap.set(cv2.CAP_PROP_POS_FRAMES, mid_frame)
            ret, frame = cap.read()
            
            keyframe_path = ""
            if ret:
                keyframe_path = os.path.join(self.output_dir, f"scene_{i}_{int(mid_frame)}.jpg")
                cv2.imwrite(keyframe_path, frame)
            
            results.append(Scene(
                start_time=start_sec,
                end_time=end_sec,
                start_frame=start_frame,
                end_frame=end_frame,
                keyframe_path=keyframe_path
            ))
            
        cap.release()
        
        return results

    def ensure_compatible_video(self, video_path: str) -> str:
        """
        Check if video is AV1 (or other problematic format) and transcode to H.264 if needed.
        Returns the path to the compatible video (original or new temp file).
        """
        try:
            import ffmpeg
            probe = ffmpeg.probe(video_path)
            video_stream = next((stream for stream in probe['streams'] if stream['codec_type'] == 'video'), None)
            
            if not video_stream:
                return video_path
                
            codec = video_stream.get('codec_name', 'unknown')
            print(f"Detected video codec: {codec}")
            
            if codec == 'av1':
                print("AV1 detected. Transcoding to H.264 for compatibility...")
                output_path = video_path.replace(".mp4", "_h264.mp4")
                if video_path == output_path:
                    output_path = video_path + "_h264.mp4"
                    
                if os.path.exists(output_path):
                    return output_path
                    
                stream = ffmpeg.input(video_path)
                stream = ffmpeg.output(stream, output_path, vcodec='libx264', acodec='aac', preset='ultrafast')
                ffmpeg.run(stream, overwrite_output=True, quiet=True)
                print(f"Transcoding complete: {output_path}")
                return output_path
                
        except Exception as e:
            print(f"Error checking/transcoding video: {e}")
            
        return video_path

    def classify_scenes(self, scenes: List['Scene'], video_path: str = ""):
        """
        Classify scenes using CLIP model with multi-frame voting + visual heuristics.
        Much more accurate than single-frame classification.
        """
        try:
            from transformers import CLIPProcessor, CLIPModel
            from PIL import Image
            import torch
            import numpy as np

            model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")
            processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")

            labels = [
                # LOGO group (0-1): ONLY full-screen logo intros/outros
                "a full-screen company logo or brand name displayed on a solid colored background",
                "an animated logo intro or outro with large text on dark background",
                # OUTRO group (2-3)
                "end credits or closing text list scrolling on dark background",
                "a completely black empty screen with no content",
                # PRODUCT group (4-6): real physical products
                "a real physical product like a bottle or box photographed on a table",
                "hands holding or demonstrating a real physical product",
                "closeup of a product label printed on an actual bottle or packaging",
                # CONTENT group (7-10): general video (including watermarked content!)
                "a person talking or presenting to camera",
                "outdoor scenery landscape or nature footage",
                "general video footage of people or everyday activities",
                "video footage with a small watermark or logo in the corner",
            ]

            LOGO_INDICES = [0, 1]
            OUTRO_INDICES = [2, 3]
            PRODUCT_INDICES = [4, 5, 6]
            CONTENT_INDICES = [7, 8, 9, 10]

            video_duration = scenes[-1].end_time if scenes else 0

            for scene in scenes:
                if not scene.keyframe_path or not os.path.exists(scene.keyframe_path):
                    scene.scene_type = "content"
                    continue

                # Multi-frame voting: sample 3 frames per scene
                scene_frames = self._extract_scene_frames(scene, video_path=video_path, num_frames=3)
                if not scene_frames:
                    scene.scene_type = "content"
                    continue

                # Classify each frame and accumulate scores
                all_logo_scores = []
                all_product_scores = []
                all_content_scores = []
                all_outro_scores = []
                all_top_labels = []

                for frame_path in scene_frames:
                    if not os.path.exists(frame_path):
                        continue

                    image = Image.open(frame_path)
                    inputs = processor(text=labels, images=image, return_tensors="pt", padding=True)

                    with torch.no_grad():
                        outputs = model(**inputs)
                    probs = outputs.logits_per_image.softmax(dim=1)[0]

                    logo_s = sum(probs[i].item() for i in LOGO_INDICES)
                    product_s = sum(probs[i].item() for i in PRODUCT_INDICES)
                    content_s = sum(probs[i].item() for i in CONTENT_INDICES)
                    outro_s = sum(probs[i].item() for i in OUTRO_INDICES)

                    all_logo_scores.append(logo_s)
                    all_product_scores.append(product_s)
                    all_content_scores.append(content_s)
                    all_outro_scores.append(outro_s)

                    top_idx = probs.argmax().item()
                    all_top_labels.append(top_idx)

                # Average scores across frames (voting)
                avg_logo = sum(all_logo_scores) / len(all_logo_scores) if all_logo_scores else 0
                avg_product = sum(all_product_scores) / len(all_product_scores) if all_product_scores else 0
                avg_content = sum(all_content_scores) / len(all_content_scores) if all_content_scores else 0
                avg_outro = sum(all_outro_scores) / len(all_outro_scores) if all_outro_scores else 0

                # Brightness/color heuristic
                main_frame = cv2.imread(scene.keyframe_path)
                brightness_boost = 1.0
                if main_frame is not None:
                    gray = cv2.cvtColor(main_frame, cv2.COLOR_BGR2GRAY)
                    mean_brightness = gray.mean()
                    color_std = main_frame.std()

                    if mean_brightness < 60 and color_std < 50:
                        brightness_boost = 1.3
                    elif mean_brightness > 120 and color_std > 40:
                        avg_product *= 1.2
                        avg_content *= 1.2

                adj_logo = avg_logo * brightness_boost

                # Positional heuristics
                is_near_start = scene.start_time < min(video_duration * 0.15, 10)
                is_near_end = (scene.start_time > video_duration * 0.75 or
                               scene.end_time >= video_duration * 0.9)
                is_edge = is_near_start or is_near_end

                # Decision logic
                detected_type = "content"  # Safe default

                logo_frame_votes = sum(1 for idx in all_top_labels if idx in LOGO_INDICES)
                total_votes = len(all_top_labels) if all_top_labels else 1
                logo_unanimity = logo_frame_votes / total_votes

                if adj_logo > 0.70 and adj_logo > avg_product * 1.5:
                    detected_type = "logo"
                elif adj_logo > 0.35 and adj_logo > avg_product and is_edge:
                    if logo_unanimity >= 0.66:
                        detected_type = "logo"
                    else:
                        detected_type = "content"
                        print(f"  ⚠️ Logo score decent but frames don't agree → content")
                elif adj_logo > 0.50 and adj_logo > avg_product * 1.5 and logo_unanimity >= 0.66:
                    detected_type = "logo"
                    print(f"  ℹ️ Mid-video logo: strong score + frame agreement")
                elif avg_outro > 0.30 and is_near_end:
                    detected_type = "outro"
                elif is_edge and avg_product > 0.25 and adj_logo > 0.15:
                    detected_type = "logo"
                    print(f"  ℹ️ Product+logo mix at edge → brand scene (logo)")
                elif avg_product > avg_content and avg_product > 0.25:
                    detected_type = "product"
                elif avg_content > 0.20:
                    detected_type = "content"

                # Black screen check
                if main_frame is not None:
                    if gray.mean() < 15:
                        detected_type = "black_screen"

                scene.scene_type = detected_type

                print(f"Scene {scene.start_time:.1f}s - {scene.end_time:.1f}s: "
                      f"Final='{detected_type}' "
                      f"[logo={adj_logo:.2f} prod={avg_product:.2f} "
                      f"content={avg_content:.2f} outro={avg_outro:.2f}] "
                      f"({len(scene_frames)} frames voted)")

                for i, idx in enumerate(all_top_labels):
                    print(f"  frame{i+1}: '{labels[idx]}' "
                          f"(logo={all_logo_scores[i]:.2f} prod={all_product_scores[i]:.2f})")

                # Cleanup extra frames
                for fp in scene_frames:
                    if fp != scene.keyframe_path and os.path.exists(fp):
                        os.remove(fp)

        except Exception as e:
            print(f"CLIP classification failed: {e}")
            import traceback
            traceback.print_exc()

    def _extract_scene_frames(self, scene: 'Scene', video_path: str = "", num_frames: int = 3) -> List[str]:
        """Extract multiple frames from a scene for multi-frame voting."""
        paths = []

        if scene.keyframe_path:
            paths.append(scene.keyframe_path)

        if not video_path or not os.path.exists(video_path):
            return paths

        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            return paths

        fps = cap.get(cv2.CAP_PROP_FPS)
        if fps <= 0:
            cap.release()
            return paths

        scene_duration = scene.end_time - scene.start_time
        sample_times = [
            scene.start_time + scene_duration * 0.25,
            scene.start_time + scene_duration * 0.75,
        ]

        for t in sample_times:
            frame_idx = int(t * fps)
            cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
            ret, frame = cap.read()
            if ret:
                extra_path = os.path.join(self.output_dir, f"_vote_{scene.start_time:.1f}_{t:.1f}.jpg")
                cv2.imwrite(extra_path, frame)
                paths.append(extra_path)

        cap.release()
        return paths
