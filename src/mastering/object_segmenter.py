"""
Object Segmenter using SAM 2
----------------------------

Chiu trach nhiem:
- Trich xuat 1 frame tu video (dung FFmpeg)
- Segmentation object tu 1 click (SAM 2 point prompt)
- (Mo rong sau) Video tracking object qua toan bo video
"""

import logging
import os
import subprocess
from pathlib import Path
from typing import List, Optional, Sequence, Tuple

import numpy as np
from PIL import Image
from ultralytics import SAM

logger = logging.getLogger(__name__)


class ObjectSegmenter:
    """
    Wrapper don gian quanh SAM 2 de:
    - Trich xuat frame
    - Tao mask PNG tu 1 click cua user

    Luu y:
    - Mac dinh dung model SAM 2 pretrain cua Ultralytics (sam2_b.pt)
      -> Ultralytics se tu dong tai ve lan dau tien (theo [docs](https://docs.ultralytics.com/vi/models/sam-2/))
    - Co the override bang bien moi truong SAM_MODEL_NAME hoac tham so model_path
    """

    def __init__(
        self,
        output_dir: Optional[str] = None,
        model_path: Optional[str] = None,
    ) -> None:
        self.output_dir = Path(output_dir) if output_dir else Path("./outputs")
        self.output_dir.mkdir(parents=True, exist_ok=True)

        # Thu muc luu frame & mask
        self.frames_dir = self.output_dir / "master_frames"
        self.masks_dir = self.output_dir / "master_masks"
        self.frames_dir.mkdir(parents=True, exist_ok=True)
        self.masks_dir.mkdir(parents=True, exist_ok=True)

        # Ten model SAM 2 (Ultralytics tu dong tai trong lan dau)
        # Vi du: sam2_b.pt, sam2_t.pt, sam2.1_b.pt ...
        self.model_name = model_path or os.getenv("SAM_MODEL_NAME", "sam2.1_b.pt")
        self._sam_model: Optional[SAM] = None

    # ------------------------------------------------------------------
    # FFmpeg helpers
    # ------------------------------------------------------------------
    def extract_frame(
        self,
        video_path: str,
        time_seconds: float,
        frame_name: Optional[str] = None,
    ) -> str:
        """
        Trich xuat 1 frame JPEG tu video tai thoi diem time_seconds.

        Args:
            video_path: Duong dan video input
            time_seconds: Thoi diem tinh bang giay
            frame_name: Ten file output (khong co duoi). Neu None -> tu sinh

        Returns:
            Duong dan toi file JPEG frame
        """
        video_path = Path(video_path)
        if frame_name is None:
            safe_time = f"{time_seconds:.2f}".replace(".", "_")
            frame_name = f"{video_path.stem}_t{safe_time}.jpg"

        frame_path = self.frames_dir / frame_name

        cmd = [
            "ffmpeg",
            "-y",
            "-ss",
            str(time_seconds),
            "-i",
            str(video_path),
            "-frames:v",
            "1",
            "-q:v",
            "2",
            str(frame_path),
        ]

        logger.info("Extracting frame at t=%s from %s", time_seconds, video_path)
        result = subprocess.run(cmd, capture_output=True, text=True)

        if result.returncode != 0:
            logger.error("FFmpeg error while extracting frame: %s", result.stderr)
            raise RuntimeError(f"FFmpeg failed to extract frame: {result.stderr}")

        return str(frame_path)

    # ------------------------------------------------------------------
    # SAM 2 helpers
    # ------------------------------------------------------------------
    def _load_sam_model(self) -> SAM:
        """Lazy load SAM 2 model."""
        if self._sam_model is None:
            logger.info("Loading SAM 2 model: %s", self.model_name)
            # Ultralytics SAM tu dong tai weights va chon device (GPU neu co)
            self._sam_model = SAM(self.model_name)
        return self._sam_model

    def segment_by_click(
        self,
        frame_path: str,
        click_x: float,
        click_y: float,
        output_name: Optional[str] = None,
    ) -> str:
        """
        Tao binary mask PNG tu 1 click cua user tren frame.

        Args:
            frame_path: Duong dan frame JPEG/PNG
            click_x, click_y: Toa do pixel (theo goc trai tren cua frame)
            output_name: Ten file mask (khong duoi). Neu None -> tu sinh

        Returns:
            Duong dan toi file mask PNG (mau trang = object)
        """
        frame_path = Path(frame_path)
        if output_name is None:
            output_name = f"{frame_path.stem}_mask.png"

        mask_path = self.masks_dir / output_name

        model = self._load_sam_model()

        # SAM 2 yeu cau diem theo dinh dang [[x, y]]
        points: Sequence[Sequence[float]] = [[float(click_x), float(click_y)]]
        labels: Sequence[int] = [1]  # 1 = foreground

        logger.info(
            "Running SAM 2 segmentation on %s at point (%s, %s)",
            frame_path,
            click_x,
            click_y,
        )

        results = model.predict(source=str(frame_path), points=points, labels=labels)
        if not results:
            raise RuntimeError("SAM 2 khong tra ve ket qua segmentation nao")

        first = results[0]
        if getattr(first, "masks", None) is None:
            raise RuntimeError("SAM 2 khong tra ve truong masks trong ket qua")

        # Lay mask dau tien (tensor HxW)
        try:
            mask_tensor = first.masks.data[0]
        except Exception as exc:  # pragma: no cover - phong tru future API
            raise RuntimeError("Khong doc duoc mask tu ket qua SAM 2") from exc

        mask = mask_tensor.cpu().numpy()

        # Chuyen thanh anh 8-bit, 0/255 va resize ve cung size voi frame goc
        mask_img = Image.fromarray((mask * 255).astype(np.uint8), mode="L")
        frame_img = Image.open(frame_path)
        fw, fh = frame_img.size
        if mask_img.size != (fw, fh):
            mask_img = mask_img.resize((fw, fh), resample=Image.NEAREST)
        mask_img.save(mask_path)

        logger.info("Saved SAM 2 mask to %s", mask_path)
        return str(mask_path)

    def segment_by_box(
        self,
        frame_path: str,
        x1: float,
        y1: float,
        x2: float,
        y2: float,
        output_name: Optional[str] = None,
    ) -> str:
        """
        Tao binary mask PNG tu 1 bounding box tren frame.

        Args:
            frame_path: Duong dan frame JPEG/PNG
            x1, y1, x2, y2: Toa do pixel bbox (goc trai tren, goc phai duoi) theo frame goc
            output_name: Ten file mask (khong duoi). Neu None -> tu sinh

        Returns:
            Duong dan toi file mask PNG (mau trang = object)
        """
        frame_path = Path(frame_path)
        if output_name is None:
            output_name = f"{frame_path.stem}_box_mask.png"

        mask_path = self.masks_dir / output_name

        model = self._load_sam_model()

        bboxes: Sequence[Sequence[float]] = [[float(x1), float(y1), float(x2), float(y2)]]

        logger.info(
            "Running SAM 2 segmentation (bbox) on %s at box (%s, %s, %s, %s)",
            frame_path,
            x1,
            y1,
            x2,
            y2,
        )

        results = model.predict(source=str(frame_path), bboxes=bboxes)
        if not results:
            raise RuntimeError("SAM 2 khong tra ve ket qua segmentation nao (bbox)")

        first = results[0]
        if getattr(first, "masks", None) is None:
            raise RuntimeError("SAM 2 khong tra ve truong masks trong ket qua (bbox)")

        try:
            mask_tensor = first.masks.data[0]
        except Exception as exc:  # pragma: no cover
            raise RuntimeError("Khong doc duoc mask tu ket qua SAM 2 (bbox)") from exc

        mask = mask_tensor.cpu().numpy()

        mask_img = Image.fromarray((mask * 255).astype(np.uint8), mode="L")
        frame_img = Image.open(frame_path)
        fw, fh = frame_img.size
        if mask_img.size != (fw, fh):
            mask_img = mask_img.resize((fw, fh), resample=Image.NEAREST)
        mask_img.save(mask_path)

        logger.info("Saved SAM 2 bbox mask to %s", mask_path)
        return str(mask_path)

    # ------------------------------------------------------------------
    # Video tracking (placeholder for future extension)
    # ------------------------------------------------------------------
    def track_object_in_video(
        self,
        video_path: str,
        click_x: float,
        click_y: float,
    ) -> List[np.ndarray]:
        """
        (Mo rong) Track object qua toan bo video.

        Hien tai de don gian va ben vung voi API VEO 2 (chi nhan 1 mask PNG),
        ta chi can mask tĩnh (2D) cho toan bo video. Do do ham nay hien
        tra ve list gom 1 mask duy nhat, de tuong thich voi design ban dau.

        Args:
            video_path: Duong dan video goc (khong dung truc tiep o day)
            click_x, click_y: Diem click tren frame tham chieu

        Returns:
            List 1 phan tu: [mask_array(HxW)]
        """
        # Thay vi chay SAM3VideoPredictor phuc tap, ta su dung lai logic
        # segment_by_click tren 1 frame dai dien. Viec chon frame nay
        # se duoc thuc hien o tang cao hon (API).
        raise NotImplementedError(
            "track_object_in_video chua duoc implement. "
            "Su dung segment_by_click + mask tinh cho VEO 2."
        )

    def export_segment_masks(
        self,
        mask_array: np.ndarray,
        segments: Sequence[Tuple[float, float]],
    ) -> List[str]:
        """
        (Mo rong) Tao mask rieng cho tung segment.

        Voi VEO 2, 1 mask tinh la du. Ham nay giu lai de tuong thich voi
        design ban dau neu sau nay muon toi uu theo segment.

        Args:
            mask_array: Binary mask 2D (0/1 hoac 0/255)
            segments: Danh sach (start, end) segment

        Returns:
            List duong dan toi cac file mask PNG (hien tai chi 1 file chung)
        """
        if mask_array.dtype != np.uint8:
            mask_img = (mask_array.astype(float) * 255).clip(0, 255).astype(np.uint8)
        else:
            mask_img = mask_array

        base_mask_path = self.masks_dir / "segment_mask_base.png"
        Image.fromarray(mask_img, mode="L").save(base_mask_path)

        # Hien tai: dung cung 1 mask cho tat ca segment
        paths: List[str] = []
        for idx, _seg in enumerate(segments):
            seg_mask_path = self.masks_dir / f"segment_{idx:03d}.png"
            if not seg_mask_path.exists():
                # Copy tu base (de don gian)
                Image.open(base_mask_path).save(seg_mask_path)
            paths.append(str(seg_mask_path))

        return paths

