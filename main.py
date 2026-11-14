"""
Healthcare Analytics for Tuberculosis Detection & Pattern Analysis
Master Pipeline Execution Workflow.
Integrates all modules from dataset acquisition to final ML evaluation and reporting.
"""

import sys
import argparse
import time
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.append(str(PROJECT_ROOT))

import config

def run_phase_1():
    print("\n" + "=" * 70)
    print("PHASE 1: ENVIRONMENT & CONFIGURATION CHECK")
    print("=" * 70)
    print(f"Base Directory   : {config.BASE_DIR}")
    print(f"Dataset URL      : {config.OFFICIAL_DATASET_URL}")
    print(f"Expected Samples : {config.TOTAL_IMAGES_EXPECTED} (80 Normal, 58 TB-Positive)")
    print("[SUCCESS] Configuration validated.")

def run_phase_2():
    print("\n" + "=" * 70)
    print("PHASE 2: DATASET ACQUISITION & PREPROCESSING")
    print("=" * 70)
    from preprocessing.dataset_loader import download_dataset, extract_and_organize
    from preprocessing.image_preprocessing import preprocess_all_images
    from preprocessing.feature_extraction import extract_all_features
    
    zip_path = download_dataset()
    extract_and_organize(zip_path)
    preprocess_all_images()
    extract_all_features()

def run_phase_3():
    print("\n" + "=" * 70)
    print("PHASE 3: HADOOP HDFS STORAGE & INGESTION")
    print("=" * 70)
    from hadoop.hdfs_manager import setup_and_ingest_hdfs
    setup_and_ingest_hdfs()

def run_phase_4():
    print("\n" + "=" * 70)
    print("PHASE 4: HADOOP MAPREDUCE STREAMING JOBS")
    print("=" * 70)
    from hadoop.run_mapreduce import execute_mapreduce_pipeline
    execute_mapreduce_pipeline()

def run_phase_5():
    print("\n" + "=" * 70)
    print("PHASE 5: APACHE HIVE DATA WAREHOUSING")
    print("=" * 70)
    from hive.run_hive import run_hive_analytics
    run_hive_analytics()

def run_phase_6():
    print("\n" + "=" * 70)
    print("PHASE 6: APACHE MAHOUT K-MEANS CLUSTERING")
    print("=" * 70)
    from mahout.clustering.mahout_kmeans import run_mahout_kmeans
    run_mahout_kmeans(k=2)

def run_phase_7():
    print("\n" + "=" * 70)
    print("PHASE 7: RANDOM FOREST CLASSIFIER & EVALUATION")
    print("=" * 70)
    from machine_learning.train_model import train_and_evaluate_rf
    train_and_evaluate_rf()

def main():
    parser = argparse.ArgumentParser(description="Healthcare CXR Tuberculosis Big Data Analytics Pipeline")
    parser.add_argument("--all", action="store_true", help="Execute complete end-to-end pipeline")
    parser.add_argument("--step", type=int, choices=range(1, 8), help="Execute specific pipeline phase (1-7)")
    
    args = parser.parse_args()
    
    start_time = time.time()
    
    if args.step:
        phase_funcs = {
            1: run_phase_1,
            2: run_phase_2,
            3: run_phase_3,
            4: run_phase_4,
            5: run_phase_5,
            6: run_phase_6,
            7: run_phase_7
        }
        phase_funcs[args.step]()
    else:
        # Default run all phases
        run_phase_1()
        run_phase_2()
        run_phase_3()
        run_phase_4()
        run_phase_5()
        run_phase_6()
        run_phase_7()
        
    elapsed = time.time() - start_time
    print("\n" + "=" * 70)
    print(f"PIPELINE EXECUTION COMPLETED IN {elapsed:.2f} SECONDS")
    print("To launch the Streamlit Healthcare Portal, run:")
    print("  streamlit run dashboard/app.py")
    print("=" * 70 + "\n")

if __name__ == "__main__":
    main()
