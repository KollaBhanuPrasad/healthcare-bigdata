"""
Configuration file for Healthcare Analytics for Tuberculosis Detection
and Pattern Analysis Using Big Data Technologies.
"""

from pathlib import Path

# Base Paths
BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "data"
RAW_DATA_DIR = DATA_DIR / "raw"
IMAGES_DIR = DATA_DIR / "images"
MASKS_DIR = DATA_DIR / "masks"
CLINICAL_DIR = DATA_DIR / "clinical_readings"
PROCESSED_DIR = DATA_DIR / "processed"
FEATURES_DIR = DATA_DIR / "features"

MODELS_DIR = BASE_DIR / "models"
RESULTS_DIR = BASE_DIR / "results"
REPORTS_DIR = BASE_DIR / "reports"
HIVE_DIR = BASE_DIR / "hive"
HADOOP_DIR = BASE_DIR / "hadoop"
MAHOUT_DIR = BASE_DIR / "mahout"

# Official Dataset Parameters (Montgomery County CXR Set - NLM)
OFFICIAL_DATASET_URL = "https://openi.nlm.nih.gov/imgs/collections/NLM-MontgomeryCXRSet.zip"
TOTAL_IMAGES_EXPECTED = 138
NORMAL_IMAGES_EXPECTED = 80
TB_POSITIVE_IMAGES_EXPECTED = 58

# Image Preprocessing & Feature Extraction Parameters
IMAGE_TARGET_SIZE = (224, 224)
HOG_ORIENTATIONS = 9
HOG_PIXELS_PER_CELL = (16, 16)
HOG_CELLS_PER_BLOCK = (2, 2)
RANDOM_STATE = 42

# Random Forest Hyperparameters
RF_N_ESTIMATORS = 100
RF_MAX_DEPTH = 10
TEST_SIZE = 0.20
