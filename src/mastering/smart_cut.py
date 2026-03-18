"""
Smart Cut Module (from Mastering_video)
Auto-cut video theo scene classification
"""
import ffmpeg
import os
from typing import List, Tuple
from .scene_detection import Scene

class SmartCutter:
    def __init__(self, output_dir: str = "outputs/cuts"):
        self.output_dir = output_dir
        os.makedirs(self.output_dir, exist_ok=True)

    def generate_cut_list(self, scenes: List[Scene], remove_intro: bool = True, remove_outro: bool = True, remove_product: bool = False, remove_logo: bool = False) -> List[Tuple[float, float]]:
        """
        Generate a list of (start, end) timestamps to KEEP.
        """
        keep_segments = []
        
        for scene in scenes:
            should_remove = False
            
            if remove_intro and "intro" in scene.scene_type:
                should_remove = True
            if remove_outro and ("outro" in scene.scene_type or "closing" in scene.scene_type):
                should_remove = True
            if remove_product and "product" in scene.scene_type:
                should_remove = True
            if remove_logo and "logo" in scene.scene_type:
                should_remove = True
                
            if not should_remove:
                keep_segments.append((scene.start_time, scene.end_time))
        
        # Fallback: If everything was removed, keep the longest scene
        if not keep_segments and scenes:
            print("Warning: All scenes were filtered out! Falling back to keeping the longest scene.")
            longest_scene = max(scenes, key=lambda s: s.end_time - s.start_time)
            keep_segments.append((longest_scene.start_time, longest_scene.end_time))
                
        return keep_segments

    def render_video(self, video_path: str, keep_segments: List[Tuple[float, float]], output_filename: str) -> str:
        """
        Render the final video by concatenating kept segments.
        Using ffmpeg-python for constructing the filter graph.
        """
        output_path = os.path.join(self.output_dir, output_filename)
        
        if not keep_segments:
            raise ValueError("No segments to keep!")

        input_stream = ffmpeg.input(video_path)
        
        streams = []
        for start, end in keep_segments:
            duration = end - start
            if duration <= 0: continue
            
            v = input_stream.video.trim(start=start, end=end).setpts('PTS-STARTPTS')
            a = input_stream.audio.filter_('atrim', start=start, end=end).filter_('asetpts', 'PTS-STARTPTS')
            streams.append(v)
            streams.append(a)

        joined = ffmpeg.concat(*streams, v=1, a=1).node

        out = ffmpeg.output(joined[0], joined[1], output_path, preset='ultrafast', crf=23, acodec='aac')
        
        try:
            out.run(overwrite_output=True)
            return output_path
        except ffmpeg.Error as e:
            print(f"FFmpeg error: {e.stderr.decode() if e.stderr else str(e)}")
            raise e
