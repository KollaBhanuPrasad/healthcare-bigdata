"""
Feature Extraction Module for Tuberculosis CXR Images.
Extracts:
1. Histogram of Oriented Gradients (HOG) descriptors.
2. First-order statistical metrics (mean, std, skewness, kurtosis).
3. Texture and intensity distribution measures.
Generates structured numerical feature matrices suitable for:
- Apache Mahout clustering (Module 6)
- Scikit-learn Random Forest classification (Module 7)
"""

import sys
from pathlib import Path
import numpy as np
import pandas as pd
from skimage.feature import hog
from scipy.stats import skew, kurtosis

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.append(str(PROJECT_ROOT))

import config

def extract_features_from_array(img_array: np.ndarray) -> np.ndarray:
    """
    Computes HOG descriptors and statistical texture features for an image array.
    """
    # 1. HOG Features
    hog_feats = hog(
        img_array,
        orientations=config.HOG_ORIENTATIONS,
        pixels_per_cell=config.HOG_PIXELS_PER_CELL,
        cells_per_block=config.HOG_CELLS_PER_BLOCK,
        block_norm='L2-Hys',
        visualize=False
    )
    
    # 2. First-order & Statistical Features
    flat = img_array.flatten()
    mean_val = float(np.mean(flat))
    std_val = float(np.std(flat))
    skew_val = float(skew(flat))
    kurt_val = float(kurtosis(flat))
    p25 = float(np.percentile(flat, 25))
    p50 = float(np.median(flat))
    p75 = float(np.percentile(flat, 75))
    
    # Energy and Entropy
    hist, _ = np.histogram(flat, bins=32, range=(0.0, 1.0), density=True)
    hist = hist / (np.sum(hist) + 1e-7)
    entropy = float(-np.sum([p * np.log2(p) for p in hist if p > 0]))
    energy = float(np.sum(hist ** 2))
    
    stats_feats = np.array([mean_val, std_val, skew_val, kurt_val, p25, p50, p75, entropy, energy], dtype=np.float32)
    
    # Combined feature vector
    combined = np.hstack([hog_feats, stats_feats])
    return combined

def extract_all_features() -> tuple:
    """
    Extracts features for all preprocessed CXR images, synchronizes with labels,
    and saves feature matrices and tabular CSVs.
    """
    config.FEATURES_DIR.mkdir(parents=True, exist_ok=True)
    meta_path = config.DATA_DIR / "metadata.csv"
    if not meta_path.exists():
        raise FileNotFoundError(f"Metadata file not found at {meta_path}. Run dataset_loader first.")
        
    meta_df = pd.read_csv(meta_path)
    print(f"[INFO] Extracting features for {len(meta_df)} images...")
    
    feature_vectors = []
    valid_labels = []
    patient_ids = []
    
    for idx, row in meta_df.iterrows():
        pid = row["patient_id"]
        label = row["label"]
        npy_path = config.PROCESSED_DIR / f"{pid}_processed.npy"
        
        if not npy_path.exists():
            print(f"[WARN] Processed file {npy_path.name} not found. Skipping.")
            continue
            
        img_arr = np.load(npy_path)
        feat_vec = extract_features_from_array(img_arr)
        
        feature_vectors.append(feat_vec)
        valid_labels.append(label)
        patient_ids.append(pid)
        
    X = np.array(feature_vectors, dtype=np.float32)
    y = np.array(valid_labels, dtype=np.int32)
    
    print(f"[SUCCESS] Extracted feature matrix shape: {X.shape} (Samples x Features)")
    print(f"[INFO] Class distribution: Normal (0) = {np.sum(y == 0)}, TB Positive (1) = {np.sum(y == 1)}")
    
    # Save Numpy binaries
    np.save(config.FEATURES_DIR / "X_features.npy", X)
    np.save(config.FEATURES_DIR / "y_labels.npy", y)
    
    # Save CSV representation
    feat_cols = [f"hog_feat_{i}" for i in range(X.shape[1] - 9)] + [
        "stat_mean", "stat_std", "stat_skew", "stat_kurt",
        "stat_p25", "stat_median", "stat_p75", "stat_entropy", "stat_energy"
    ]
    df_features = pd.DataFrame(X, columns=feat_cols)
    df_features.insert(0, "patient_id", patient_ids)
    df_features.insert(1, "label", y)
    df_features.insert(2, "diagnosis", ["Normal" if l == 0 else "TB_Positive" for l in y])
    
    csv_out = config.FEATURES_DIR / "features_summary.csv"
    df_features.to_csv(csv_out, index=False)
    print(f"[INFO] Features dataset saved to {csv_out}")
    
    return X, y, df_features

if __name__ == "__main__":
    extract_all_features()
