"""
Product Detector Module
Sử dụng CLIP để tự động nhận diện loại sản phẩm trong ảnh/video
"""
import logging
from pathlib import Path
from typing import Optional, Tuple, List, Dict
from PIL import Image
import tempfile
import subprocess

logger = logging.getLogger(__name__)

# Product categories mapped to legal types
PRODUCT_CATEGORIES = {
    "alcohol": [
        "wine bottle",
        "beer bottle", 
        "champagne bottle",
        "whiskey bottle",
        "vodka bottle",
        "alcohol bottle",
        "liquor bottle",
        "cocktail glass",
        "beer can",
        "wine glass"
    ],
    "tobacco": [
        "cigarette pack",
        "cigarette",
        "cigar",
        "tobacco product",
        "smoking",
        "e-cigarette",
        "vape"
    ],
    "pharmaceutical": [
        "medicine bottle",
        "pill bottle",
        "medication",
        "pharmaceutical product",
        "drug packaging",
        "vitamin bottle"
    ],
    "food": [
        "baby formula",
        "infant formula",
        "milk powder",
        "food product",
        "snack package",
        "beverage can"
    ]
}


class ProductDetector:
    """Phát hiện loại sản phẩm bằng CLIP"""
    
    def __init__(self, model_name: str = "openai/clip-vit-base-patch32"):
        self.model_name = model_name
        self._model = None
        self._processor = None
        self._device = None
        
        # Build flat list of all labels
        self.labels = []
        self.label_to_category = {}
        for category, items in PRODUCT_CATEGORIES.items():
            for item in items:
                self.labels.append(item)
                self.label_to_category[item] = category
    
    def _load_model(self):
        """Lazy load CLIP model"""
        if self._model is None:
            try:
                import torch
                from transformers import CLIPProcessor, CLIPModel
                
                logger.info(f"Loading CLIP model: {self.model_name}")
                
                self._device = "cuda" if torch.cuda.is_available() else "cpu"
                self._model = CLIPModel.from_pretrained(self.model_name).to(self._device)
                self._processor = CLIPProcessor.from_pretrained(self.model_name)
                
                logger.info(f"CLIP model loaded on {self._device}")
            except ImportError as e:
                logger.error(f"Missing dependencies: {e}")
                raise RuntimeError("Please install: pip install transformers torch")
    
    def detect_product(self, image_path: str, threshold: float = 0.3) -> Tuple[str, float, str]:
        """
        Phát hiện loại sản phẩm trong ảnh
        
        Args:
            image_path: Đường dẫn ảnh
            threshold: Ngưỡng confidence tối thiểu
            
        Returns:
            Tuple (category, confidence, detected_label)
            category: alcohol, tobacco, pharmaceutical, food, unknown
        """
        self._load_model()
        
        import torch
        
        # Load image
        image = Image.open(image_path).convert("RGB")
        
        # Prepare inputs
        inputs = self._processor(
            text=self.labels,
            images=image,
            return_tensors="pt",
            padding=True
        ).to(self._device)
        
        # Get predictions
        with torch.no_grad():
            outputs = self._model(**inputs)
            logits_per_image = outputs.logits_per_image
            probs = logits_per_image.softmax(dim=1)
        
        # Get best match
        best_idx = probs.argmax().item()
        best_prob = probs[0][best_idx].item()
        best_label = self.labels[best_idx]
        
        if best_prob < threshold:
            return "unknown", best_prob, best_label
        
        category = self.label_to_category.get(best_label, "unknown")
        
        logger.info(f"Detected: {best_label} ({category}) - {best_prob:.2%}")
        
        return category, best_prob, best_label
    
    def detect_from_video(
        self, 
        video_path: str, 
        sample_frames: int = 3,
        threshold: float = 0.3
    ) -> Tuple[str, float, str]:
        """
        Phát hiện sản phẩm từ video (lấy mẫu vài frame)
        
        Args:
            video_path: Đường dẫn video
            sample_frames: Số frame để phân tích
            threshold: Ngưỡng confidence
            
        Returns:
            Tuple (category, confidence, detected_label)
        """
        # Extract sample frames
        with tempfile.TemporaryDirectory() as tmpdir:
            output_pattern = f"{tmpdir}/frame_%02d.jpg"
            
            # Get video duration
            duration = self._get_video_duration(video_path)
            
            # Extract frames at different times
            for i in range(sample_frames):
                time = (i + 1) * duration / (sample_frames + 1)
                cmd = [
                    "ffmpeg", "-y",
                    "-ss", str(time),
                    "-i", video_path,
                    "-vframes", "1",
                    "-q:v", "2",
                    f"{tmpdir}/frame_{i:02d}.jpg"
                ]
                subprocess.run(cmd, capture_output=True)
            
            # Analyze each frame
            results = []
            for frame_path in Path(tmpdir).glob("frame_*.jpg"):
                try:
                    category, prob, label = self.detect_product(str(frame_path), threshold)
                    if category != "unknown":
                        results.append((category, prob, label))
                except Exception as e:
                    logger.warning(f"Failed to analyze frame: {e}")
            
            if not results:
                return "unknown", 0.0, ""
            
            # Get most confident result
            best = max(results, key=lambda x: x[1])
            return best
    
    def _get_video_duration(self, video_path: str) -> float:
        """Get video duration using ffprobe"""
        cmd = [
            "ffprobe",
            "-v", "quiet",
            "-show_entries", "format=duration",
            "-of", "default=noprint_wrappers=1:nokey=1",
            video_path
        ]
        result = subprocess.run(cmd, capture_output=True, text=True)
        try:
            return float(result.stdout.strip())
        except:
            return 30.0
    
    def get_legal_type(self, image_or_video_path: str) -> str:
        """
        Phát hiện và trả về loại legal phù hợp
        
        Args:
            image_or_video_path: Đường dẫn file
            
        Returns:
            Legal type: alcohol, tobacco, pharmaceutical, food, unknown
        """
        path = Path(image_or_video_path)
        
        if path.suffix.lower() in ['.mp4', '.mov', '.avi', '.mkv', '.webm']:
            category, _, _ = self.detect_from_video(str(path))
        else:
            category, _, _ = self.detect_product(str(path))
        
        return category


# Singleton instance
_detector: Optional[ProductDetector] = None


def get_product_detector() -> ProductDetector:
    """Get singleton ProductDetector instance"""
    global _detector
    if _detector is None:
        _detector = ProductDetector()
    return _detector
