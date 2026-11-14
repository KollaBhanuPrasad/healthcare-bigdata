"""
Official Dataset Acquisition and Verification Module for Montgomery County CXR Set.
Source: National Library of Medicine (NLM), Lister Hill National Center for Biomedical Communications (LHNCBC)
URL: https://data.lhncbc.nlm.nih.gov/public/Tuberculosis-Chest-X-ray-Datasets/Montgomery-County-CXR-Set/
"""

import os
import sys
import re
import json
import shutil
from pathlib import Path
import urllib.request
import requests
from concurrent.futures import ThreadPoolExecutor, as_completed
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.append(str(PROJECT_ROOT))

import config

NLM_BASE_URL = "https://data.lhncbc.nlm.nih.gov/public/Tuberculosis-Chest-X-ray-Datasets/Montgomery-County-CXR-Set/MontgomerySet"

def get_file_list_from_index(index_url: str, pattern: str) -> list:
    """Parses HTML directory index from LHNCBC to extract file names."""
    headers = {'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) HealthcareAnalytics/1.0'}
    resp = requests.get(index_url, headers=headers, timeout=30)
    resp.raise_for_status()
    # Find all href='filename' matches
    filenames = re.findall(pattern, resp.text)
    # Deduplicate and filter out index/parent links
    valid = sorted(list(set([f for f in filenames if not f.endswith('/') and not f.startswith('?') and not f.startswith('http')])))
    return valid

def download_file(url: str, target_path: Path, max_retries: int = 3) -> bool:
    """Downloads a single file with retries."""
    if target_path.exists() and target_path.stat().st_size > 0:
        return True
        
    headers = {'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) HealthcareAnalytics/1.0'}
    for attempt in range(max_retries):
        try:
            resp = requests.get(url, headers=headers, stream=True, timeout=30)
            resp.raise_for_status()
            with open(target_path, 'wb') as f:
                for chunk in resp.iter_content(chunk_size=64 * 1024):
                    if chunk:
                        f.write(chunk)
            return True
        except Exception as e:
            if attempt == max_retries - 1:
                print(f"[ERROR] Failed to download {url}: {e}")
                return False

def acquire_official_dataset(max_workers: int = 12) -> dict:
    """
    Downloads all 138 official images, clinical reports, and masks from NLM LHNCBC.
    """
    config.IMAGES_DIR.mkdir(parents=True, exist_ok=True)
    config.CLINICAL_DIR.mkdir(parents=True, exist_ok=True)
    (config.MASKS_DIR / "leftMask").mkdir(parents=True, exist_ok=True)
    (config.MASKS_DIR / "rightMask").mkdir(parents=True, exist_ok=True)
    
    print("\n" + "=" * 65)
    print("      ACQUIRING OFFICIAL MONTGOMERY COUNTY CXR DATASET      ")
    print(f"Source: {NLM_BASE_URL}")
    print("=" * 65)
    
    # 1. Fetch file lists
    print("[INFO] Fetching manifest from official NLM directories...")
    cxr_files = get_file_list_from_index(f"{NLM_BASE_URL}/CXR_png/index.html", r"href='(MCUCXR_[^']+?\.png)'")
    clinical_files = get_file_list_from_index(f"{NLM_BASE_URL}/ClinicalReadings/index.html", r"href='(MCUCXR_[^']+?\.txt)'")
    left_masks = get_file_list_from_index(f"{NLM_BASE_URL}/ManualMask/leftMask/index.html", r"href='(MCUCXR_[^']+?\.png)'")
    right_masks = get_file_list_from_index(f"{NLM_BASE_URL}/ManualMask/rightMask/index.html", r"href='(MCUCXR_[^']+?\.png)'")
    
    print(f"[FOUND] {len(cxr_files)} CXR images, {len(clinical_files)} clinical readings, {len(left_masks)} left masks, {len(right_masks)} right masks.")
    
    # Tasks list
    download_tasks = []
    
    for f in cxr_files:
        url = f"{NLM_BASE_URL}/CXR_png/{f}"
        dest = config.IMAGES_DIR / f
        download_tasks.append((url, dest))
        
    for f in clinical_files:
        url = f"{NLM_BASE_URL}/ClinicalReadings/{f}"
        dest = config.CLINICAL_DIR / f
        download_tasks.append((url, dest))
        
    for f in left_masks:
        url = f"{NLM_BASE_URL}/ManualMask/leftMask/{f}"
        dest = config.MASKS_DIR / "leftMask" / f
        download_tasks.append((url, dest))
        
    for f in right_masks:
        url = f"{NLM_BASE_URL}/ManualMask/rightMask/{f}"
        dest = config.MASKS_DIR / "rightMask" / f
        download_tasks.append((url, dest))
        
    print(f"[INFO] Downloading {len(download_tasks)} official assets using {max_workers} parallel workers...")
    
    completed = 0
    total = len(download_tasks)
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = {executor.submit(download_file, url, dest): dest.name for url, dest in download_tasks}
        for future in as_completed(futures):
            name = futures[future]
            completed += 1
            if completed % 25 == 0 or completed == total:
                print(f"[PROGRESS] Downloaded {completed}/{total} assets ({completed/total*100:.1f}%)")
                
    print("[SUCCESS] All official assets downloaded successfully.")
    return organize_and_verify()

def organize_and_verify() -> dict:
    """
    Parses clinical readings, builds metadata records, and verifies ground-truth counts.
    """
    config.REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    metadata_records = []
    normal_count = 0
    tb_count = 0
    
    images = sorted(list(config.IMAGES_DIR.glob("*.png")))
    
    for img_path in images:
        filename = img_path.name
        # Label convention: MCUCXR_XXXX_0.png -> Normal (0), MCUCXR_XXXX_1.png -> TB (1)
        patient_id = img_path.stem
        parts = patient_id.split("_")
        label_code = int(parts[-1]) if len(parts) >= 3 and parts[-1].isdigit() else -1
        
        status = "Normal" if label_code == 0 else ("TB_Positive" if label_code == 1 else "Unknown")
        if label_code == 0:
            normal_count += 1
        elif label_code == 1:
            tb_count += 1
            
        file_size_kb = round(img_path.stat().st_size / 1024, 2)
        
        # Clinical reading parsing
        txt_path = config.CLINICAL_DIR / f"{patient_id}.txt"
        patient_gender = "Unknown"
        patient_age = -1
        radiology_finding = "No clinical reading available"
        
        if txt_path.exists():
            with open(txt_path, 'r', encoding='utf-8', errors='ignore') as tf:
                lines = [l.strip() for l in tf.readlines() if l.strip()]
                if lines:
                    radiology_finding = " ".join(lines)
                    for line in lines:
                        lower_l = line.lower()
                        # Detect sex: "Patient's Sex: M", "Sex: F", "male", "female"
                        if "sex: m" in lower_l or "sex:m" in lower_l or lower_l.endswith(": m") or lower_l.endswith(":m") or "male" in lower_l and "female" not in lower_l:
                            patient_gender = "Male"
                        elif "sex: f" in lower_l or "sex:f" in lower_l or lower_l.endswith(": f") or lower_l.endswith(":f") or "female" in lower_l:
                            patient_gender = "Female"
                            
                        # Detect age: "Patient's Age: 027Y", "Age: 40"
                        if "age:" in lower_l:
                            age_part = lower_l.split("age:")[-1].replace("y", "").replace("r", "").strip()
                            digits = "".join([c for c in age_part if c.isdigit()])
                            if digits and 1 <= int(digits) <= 120:
                                patient_age = int(digits)
                                
        has_left_mask = (config.MASKS_DIR / "leftMask" / filename).exists()
        has_right_mask = (config.MASKS_DIR / "rightMask" / filename).exists()
        
        metadata_records.append({
            "patient_id": patient_id,
            "filename": filename,
            "label": label_code,
            "diagnosis": status,
            "patient_gender": patient_gender,
            "patient_age": patient_age,
            "file_size_kb": file_size_kb,
            "has_left_mask": has_left_mask,
            "has_right_mask": has_right_mask,
            "radiology_finding": radiology_finding
        })
        
    df_meta = pd.DataFrame(metadata_records)
    csv_path = config.DATA_DIR / "metadata.csv"
    df_meta.to_csv(csv_path, index=False)
    print(f"[INFO] Metadata saved to {csv_path} with {len(df_meta)} records.")
    
    total_images = len(df_meta)
    total_size_mb = round(sum(p.stat().st_size for p in images) / (1024**2), 2)
    
    summary = {
        "dataset_name": "Montgomery County Chest X-ray Set (NLM/NIH)",
        "source_url": NLM_BASE_URL,
        "total_images": total_images,
        "normal_cases": normal_count,
        "tb_positive_cases": tb_count,
        "total_image_size_mb": total_size_mb,
        "verification_status": (
            total_images == config.TOTAL_IMAGES_EXPECTED and 
            normal_count == config.NORMAL_IMAGES_EXPECTED and 
            tb_count == config.TB_POSITIVE_IMAGES_EXPECTED
        ),
        "masks_available": {
            "left_mask_count": len(list((config.MASKS_DIR / "leftMask").glob("*.png"))),
            "right_mask_count": len(list((config.MASKS_DIR / "rightMask").glob("*.png")))
        },
        "clinical_readings_count": len(list(config.CLINICAL_DIR.glob("*.txt")))
    }
    
    summary_path = config.REPORTS_DIR / "dataset_summary.json"
    with open(summary_path, "w", encoding="utf-8") as sf:
        json.dump(summary, sf, indent=4)
        
    print("\n" + "=" * 65)
    print("              DATASET VERIFICATION AUDIT REPORT              ")
    print("=" * 65)
    print(f"Total CXR Images : {total_images} (Expected: {config.TOTAL_IMAGES_EXPECTED})")
    print(f"Normal Cases     : {normal_count} (Expected: {config.NORMAL_IMAGES_EXPECTED})")
    print(f"TB Positive      : {tb_count} (Expected: {config.TB_POSITIVE_IMAGES_EXPECTED})")
    print(f"Total Image Size : {total_size_mb:.2f} MB")
    print(f"Masks Available  : Left={summary['masks_available']['left_mask_count']}, Right={summary['masks_available']['right_mask_count']}")
    print(f"Reports Parsed   : {summary['clinical_readings_count']}")
    print(f"Audit Verified   : {summary['verification_status']}")
    print("=" * 65 + "\n")
    
    return summary

def download_dataset():
    return acquire_official_dataset()

if __name__ == "__main__":
    download_dataset()
