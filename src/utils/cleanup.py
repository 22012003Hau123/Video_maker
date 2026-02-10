import os
import time
import logging
from pathlib import Path

logger = logging.getLogger(__name__)

def cleanup_old_files(directory: Path, max_age_hours: int = 24):
    """
    Xóa các file trong thư mục cũ hơn thời gian quy định
    
    Args:
        directory: Đường dẫn thư mục cần dọn dẹp
        max_age_hours: Tuổi thọ tối đa của file (giờ)
    """
    if not directory.exists():
        return
        
    current_time = time.time()
    max_age_seconds = max_age_hours * 3600
    cleaned_count = 0
    cleaned_size = 0
    
    logger.info(f"Starting cleanup for {directory}, removing files older than {max_age_hours}h")
    
    try:
        for file_path in directory.glob("*"):
            if not file_path.is_file() or file_path.name == ".gitkeep":
                continue
                
            file_age = current_time - file_path.stat().st_mtime
            
            if file_age > max_age_seconds:
                try:
                    file_size = file_path.stat().st_size
                    file_path.unlink()
                    cleaned_count += 1
                    cleaned_size += file_size
                    logger.debug(f"Deleted old file: {file_path.name}")
                except Exception as e:
                    logger.warning(f"Could not delete {file_path.name}: {e}")
                    
        if cleaned_count > 0:
            size_mb = cleaned_size / (1024 * 1024)
            logger.info(f"Cleanup finished: Released {size_mb:.2f}MB, removed {cleaned_count} files")
            
    except Exception as e:
        logger.error(f"Error during cleanup of {directory}: {e}")
