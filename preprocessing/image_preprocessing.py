"""
Image Preprocessing Module for Tuberculosis CXR Analytics.
- Streams images one-by-one to prevent high memory consumption.
- Converts to single-channel 8-bit grayscale.
- Resizes to standard dimensions (224 x 224).
- Applies min-max intensity normalization.
- Handles corrupted or unreadable images gracefully.
"""

import sys
from pathlib import Path
from PIL import Image
import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.append(str(PROJECT_ROOT))

import config

def preprocess_single_image(image_path: Path, target_size=config.IMAGE_TARGET_SIZE) -> np.ndarray:
    """
    Loads, validates, converts to grayscale, resizes, and normalizes a single CXR image.
    Returns normalized float32 array in [0, 1] range, or None if corrupted.
    """
    try:
        with Image.open(image_path) as img:
            # Verify image integrity
            img.verify()
        
        # Re-open for actual processing (PIL verify closes the file)
        with Image.open(image_path) as img:
            # Convert to Grayscale ('L' mode: 8-bit pixels, black and white)
            gray_img = img.convert("L")
            # Resize using high-quality Lanczos resampling
            resized_img = gray_img.resize(target_size, Image.Resampling.LANCZOS)
            
            # Convert to numpy array
            arr = np.array(resized_img, dtype=np.float32)
            
            # Min-Max Normalization to [0.0, 1.0]
            arr_min = arr.min()
            arr_max = arr.max()
            if arr_max > arr_min:
                normalized = (arr - arr_min) / (arr_max - arr_min)
            else:
                normalized = np.zeros_like(arr, dtype=np.float32)
                
            return normalized
            
    except Exception as e:
        print(f"[ERROR] Corrupted or unreadable image {image_path.name}: {e}")
        return None

def preprocess_all_images(input_dir: Path = config.IMAGES_DIR, 
                          output_dir: Path = config.PROCESSED_DIR) -> pd.DataFrame:
    """
    Processes all CXR images iteratively (avoiding in-memory accumulation).
    Saves preprocessed arrays as .npy files and outputs an audit log.
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    images = sorted(list(input_dir.glob("*.png")))
    total = len(images)
    print(f"[INFO] Beginning preprocessing of {total} CXR images...")
    
    audit_records = []
    
    for idx, img_path in enumerate(images, start=1):
        filename = img_path.name
        processed_data = preprocess_single_image(img_path)
        
        if processed_data is not None:
            # Save preprocessed float32 array
            out_filename = f"{img_path.stem}_processed.npy"
            out_path = output_dir / out_filename
            np.save(out_path, processed_data)
            
            # Save visual thumbnail PNG for quick preview
            thumb_img = Image.fromarray((processed_data * 255).astype(np.uint8))
            thumb_path = output_dir / f"{img_path.stem}_thumb.png"
            thumb_img.save(thumb_path)
            
            audit_records.append({
                "original_filename": filename,
                "processed_filename": out_filename,
                "status": "Success",
                "width": processed_data.shape[1],
                "height": processed_data.shape[0],
                "mean_intensity": round(float(np.mean(processed_data)), 4),
                "std_intensity": round(float(np.std(processed_data)), 4),
                "min_val": round(float(np.min(processed_data)), 4),
                "max_val": round(float(np.max(processed_data)), 4)
            })
        else:
            audit_records.append({
                "original_filename": filename,
                "processed_filename": None,
                "status": "Corrupted_Failed",
                "width": 0,
                "height": 0,
                "mean_intensity": 0.0,
                "std_intensity": 0.0,
                "min_val": 0.0,
                "max_val": 0.0
            })
            
        if idx % 20 == 0 or idx == total:
            print(f"[PROGRESS] Processed {idx}/{total} images ({idx/total*100:.1f}%)")
            
    audit_df = pd.DataFrame(audit_records)
    audit_path = config.PROCESSED_DIR / "preprocessing_audit.csv"
    audit_df.to_csv(audit_path, index=False)
    print(f"[SUCCESS] Preprocessing completed. Audit saved to {audit_path}")
    return audit_df

if __name__ == "__main__":
    preprocess_all_images()
