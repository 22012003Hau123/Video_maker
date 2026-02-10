"""
VeoEditor
---------

Wrapper don gian quanh Google Vertex AI Veo 2 de:
- Split video thanh cac doan 5-8s (bang FFmpeg)
- Upload video + mask len Google Cloud Storage
- Goi VEO 2 de xoa object khoi tung doan
- Download va concat cac doan da xu ly thanh 1 video hoan chinh

Yeu cau:
- Da cai dat google-genai, google-cloud-storage
- Da cau hinh bien moi truong:
  - GOOGLE_CLOUD_PROJECT
  - GOOGLE_CLOUD_LOCATION (vi du: "global")
  - GOOGLE_GENAI_USE_VERTEXAI=True
  - GCS_BUCKET (bucket de doc/ghi video)
"""

import json
import logging
import os
import subprocess
import tempfile
import time
from pathlib import Path
from typing import List, Optional, Sequence, Tuple

from google import genai
from google.cloud import storage
from google.genai.types import (
    GenerateVideosConfig,
    GenerateVideosSource,
    Image,
    Video,
    VideoGenerationMask,
    VideoGenerationMaskMode,
)

logger = logging.getLogger(__name__)


class VeoEditor:
    """Editor video dung Veo 2 de xoa/thay object."""

    def __init__(
        self,
        project_id: Optional[str] = None,
        gcs_bucket: Optional[str] = None,
        location: Optional[str] = None,
        model_name: str = "veo-2.0-generate-preview",
        output_dir: Optional[str] = None,
    ) -> None:
        # Thong tin GCP
        self.project_id = project_id or os.getenv("GOOGLE_CLOUD_PROJECT")
        self.location = location or os.getenv("GOOGLE_CLOUD_LOCATION", "global")
        self.gcs_bucket = gcs_bucket or os.getenv("GCS_BUCKET")
        self.model_name = model_name

        if not self.project_id:
            raise RuntimeError(
                "Chua cau hinh GOOGLE_CLOUD_PROJECT. Vui long them vao file .env."
            )
        if not self.gcs_bucket:
            raise RuntimeError("Chua cau hinh GCS_BUCKET. Vui long them vao file .env.")

        # Thu muc lam viec local
        self.output_dir = Path(output_dir) if output_dir else Path("./outputs")
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.segments_dir = self.output_dir / "veo_segments"
        self.segments_dir.mkdir(parents=True, exist_ok=True)

        # Clients
        self._genai_client = genai.Client()
        self._storage_client = storage.Client(project=self.project_id)

    # ------------------------------------------------------------------
    # FFmpeg helpers
    # ------------------------------------------------------------------
    def _get_video_duration(self, video_path: str) -> float:
        """Lay duration video (giay) bang ffprobe."""
        cmd = [
            "ffprobe",
            "-v",
            "quiet",
            "-show_entries",
            "format=duration",
            "-of",
            "default=noprint_wrappers=1:nokey=1",
            video_path,
        ]
        result = subprocess.run(cmd, capture_output=True, text=True)
        if result.returncode != 0:
            raise RuntimeError(f"ffprobe failed: {result.stderr}")
        return float(result.stdout.strip())

    def split_video(
        self,
        video_path: str,
        segment_duration: float = 7.0,
    ) -> List[str]:
        """
        Cat video thanh cac doan ~segment_duration giay.

        Luu y: Veo 2 yeu cau 5-8 giay, nen mac dinh 7 giay.

        Returns:
            Danh sach duong dan cac segment MP4.
        """
        video_path = Path(video_path)
        duration = self._get_video_duration(str(video_path))

        # Thu muc rieng cho tung video
        target_dir = self.segments_dir / video_path.stem
        target_dir.mkdir(parents=True, exist_ok=True)

        # Dung segment muxer cua FFmpeg
        pattern = target_dir / "segment_%03d.mp4"
        cmd = [
            "ffmpeg",
            "-y",
            "-i",
            str(video_path),
            "-c",
            "copy",
            "-map",
            "0",
            "-f",
            "segment",
            "-segment_time",
            str(segment_duration),
            "-reset_timestamps",
            "1",
            str(pattern),
        ]

        logger.info("Splitting video %s (duration %.2fs) into segments", video_path, duration)
        result = subprocess.run(cmd, capture_output=True, text=True)
        if result.returncode != 0:
            logger.error("FFmpeg segment error: %s", result.stderr)
            raise RuntimeError(f"FFmpeg split failed: {result.stderr}")

        segments = sorted(target_dir.glob("segment_*.mp4"))
        if not segments:
            raise RuntimeError("Khong tao duoc segment nao tu video")

        return [str(p) for p in segments]

    def _concat_segments(self, segment_paths: Sequence[str]) -> str:
        """
        Ghep cac segment MP4 thanh 1 video.

        Dung FFmpeg concat demuxer + crossfade nhe giua cac doan (0.3s).
        Crossfade phuc tap, o day tam thoi dung concat thang cho on dinh.
        """
        if not segment_paths:
            raise ValueError("segment_paths rong")

        first_segment = Path(segment_paths[0])
        output_path = self.output_dir / f"{first_segment.stem}_veo_concat.mp4"

        # Tao file list cho ffmpeg concat
        with tempfile.NamedTemporaryFile("w", suffix=".txt", delete=False) as tf:
            for p in segment_paths:
                tf.write(f"file '{Path(p).as_posix()}'\n")
            list_path = tf.name

        cmd = [
            "ffmpeg",
            "-y",
            "-f",
            "concat",
            "-safe",
            "0",
            "-i",
            list_path,
            "-c",
            "copy",
            str(output_path),
        ]

        logger.info("Concatenating %d segments", len(segment_paths))
        result = subprocess.run(cmd, capture_output=True, text=True)
        if result.returncode != 0:
            logger.error("FFmpeg concat error: %s", result.stderr)
            raise RuntimeError(f"FFmpeg concat failed: {result.stderr}")

        return str(output_path)

    # ------------------------------------------------------------------
    # GCS helpers
    # ------------------------------------------------------------------
    def upload_to_gcs(self, local_path: str, prefix: str = "veo_inputs") -> str:
        """
        Upload file len GCS.

        Returns:
            gs://bucket/path/to/object
        """
        bucket = self._storage_client.bucket(self.gcs_bucket)
        local_path_obj = Path(local_path)

        blob_name = f"{prefix}/{local_path_obj.name}"
        blob = bucket.blob(blob_name)
        logger.info("Uploading %s to gs://%s/%s", local_path_obj, self.gcs_bucket, blob_name)
        blob.upload_from_filename(str(local_path_obj))

        return f"gs://{self.gcs_bucket}/{blob_name}"

    def download_from_gcs(self, gcs_uri: str, local_path: str) -> str:
        """
        Download file tu GCS ve local_path.

        Args:
            gcs_uri: vi du 'gs://bucket/path/file.mp4'
        """
        if not gcs_uri.startswith("gs://"):
            raise ValueError(f"gcs_uri khong hop le: {gcs_uri}")

        without_scheme = gcs_uri[len("gs://") :]
        bucket_name, blob_name = without_scheme.split("/", 1)

        bucket = self._storage_client.bucket(bucket_name)
        blob = bucket.blob(blob_name)

        local_path_obj = Path(local_path)
        local_path_obj.parent.mkdir(parents=True, exist_ok=True)

        logger.info("Downloading %s to %s", gcs_uri, local_path_obj)
        blob.download_to_filename(str(local_path_obj))

        return str(local_path_obj)

    # ------------------------------------------------------------------
    # VEO 2 helpers
    # ------------------------------------------------------------------
    def _poll_operation(
        self,
        operation,
        wait_seconds: int = 15,
        timeout_seconds: int = 900,
    ):
        """Poll long-running operation cua VEO 2."""
        start = time.time()
        while not operation.done:
            if time.time() - start > timeout_seconds:
                raise TimeoutError("VEO operation timeout")
            time.sleep(wait_seconds)
            operation = self._genai_client.operations.get(operation.name)
        return operation

    def remove_object(
        self,
        video_gcs_uri: str,
        mask_gcs_uri: str,
        output_prefix: Optional[str] = None,
    ) -> str:
        """
        Goi VEO 2 de xoa object tren 1 segment video.

        Args:
            video_gcs_uri: gs://... video segment
            mask_gcs_uri: gs://... mask PNG (vung trang = object)
            output_prefix: gs://bucket/prefix/ de VEO ghi ket qua

        Returns:
            GCS URI cua video output
        """
        if output_prefix is None:
            output_prefix = f"gs://{self.gcs_bucket}/veo_outputs/"

        logger.info(
            "Calling VEO 2 remove-object: video=%s mask=%s output_prefix=%s",
            video_gcs_uri,
            mask_gcs_uri,
            output_prefix,
        )

        operation = self._genai_client.models.generate_videos(
            model=self.model_name,
            source=GenerateVideosSource(
                video=Video(uri=video_gcs_uri, mime_type="video/mp4")
            ),
            config=GenerateVideosConfig(
                mask=VideoGenerationMask(
                    image=Image(gcs_uri=mask_gcs_uri, mime_type="image/png"),
                    mask_mode=VideoGenerationMaskMode.REMOVE,
                ),
                output_gcs_uri=output_prefix,
            ),
        )

        operation = self._poll_operation(operation)

        # Lay URI video output
        try:
            generated = operation.result.generated_videos[0].video.uri
        except Exception as exc:  # pragma: no cover - phong tru thay doi API
            logger.error("Khong doc duoc URI video tu operation: %s", operation)
            raise RuntimeError("Khong doc duoc URI video output tu VEO 2") from exc

        logger.info("VEO 2 generated video: %s", generated)
        return generated

    def remove_object_full(
        self,
        video_path: str,
        mask_path: str,
        prompt: str = "",
        segment_duration: float = 7.0,
    ) -> str:
        """
        Xu ly toan bo video:
        - Cat video thanh cac segment ~7s
        - Upload tung segment + mask len GCS
        - Goi VEO 2 cho tung segment
        - Download va concat lai

        Args:
            video_path: Duong dan video goc
            mask_path: Duong dan mask PNG (static, ap dung cho toan bo video)
            prompt: Text mo ta (hien tai chi log lai, VEO 2 remove-object khong bat buoc)
            segment_duration: Do dai moi segment (giay)

        Returns:
            Duong dan video output local
        """
        logger.info(
            "Starting full remove-object pipeline for %s (prompt=%s)",
            video_path,
            prompt,
        )
        video_path = str(video_path)
        mask_path = str(mask_path)

        segments = self.split_video(video_path, segment_duration=segment_duration)
        mask_gcs_uri = self.upload_to_gcs(mask_path, prefix="veo_masks")

        output_segment_paths: List[str] = []
        for idx, seg in enumerate(segments):
            seg_gcs_uri = self.upload_to_gcs(seg, prefix="veo_segments")
            output_prefix = f"gs://{self.gcs_bucket}/veo_outputs/{Path(video_path).stem}/seg_{idx:03d}"
            out_gcs_uri = self.remove_object(
                video_gcs_uri=seg_gcs_uri,
                mask_gcs_uri=mask_gcs_uri,
                output_prefix=output_prefix,
            )

            local_out = self.segments_dir / f"{Path(video_path).stem}_out_{idx:03d}.mp4"
            self.download_from_gcs(out_gcs_uri, str(local_out))
            output_segment_paths.append(str(local_out))

        final_video = self._concat_segments(output_segment_paths)

        # Luu lai metadata de debug
        meta = {
            "video_input": video_path,
            "mask_path": mask_path,
            "prompt": prompt,
            "segments_input": segments,
            "segments_output": output_segment_paths,
            "final_output": final_video,
        }
        meta_path = Path(final_video).with_suffix(".json")
        try:
            with open(meta_path, "w", encoding="utf-8") as f:
                json.dump(meta, f, ensure_ascii=False, indent=2)
        except Exception:
            logger.exception("Khong ghi duoc metadata VEO 2")

        logger.info("VEO 2 full pipeline done, output: %s", final_video)
        return final_video

