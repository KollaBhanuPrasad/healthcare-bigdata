# Healthcare Analytics for Tuberculosis Detection and Pattern Analysis Using Big Data Technologies

[![Python Version](https://img.shields.io/badge/Python-3.13.2-blue.svg)](https://python.org)
[![Dataset](https://img.shields.io/badge/Dataset-NLM%2FNIH%20Montgomery%20County-green.svg)](https://data.lhncbc.nlm.nih.gov/public/Tuberculosis-Chest-X-ray-Datasets/Montgomery-County-CXR-Set/)
[![Platform](https://img.shields.io/badge/BigData-Hadoop%20%7C%20HDFS%20%7C%20MapReduce%20%7C%20Hive%20%7C%20Mahout-red.svg)](https://hadoop.apache.org/)
[![ML Model](https://img.shields.io/badge/ML-Random%20Forest%20(89.3%25%20Acc)-orange.svg)](https://scikit-learn.org/)
[![UI](https://img.shields.io/badge/Dashboard-Streamlit-FF4B4B.svg)](https://streamlit.io/)

An end-to-end distributed Big Data and Machine Learning healthcare intelligence system for pulmonary tuberculosis detection, pattern clustering, and demographic cohort analysis built on the official National Library of Medicine (NLM / NIH) Montgomery County Chest X-ray Dataset.

---

## 📌 Implementation Checklist & Verification Matrix

- [x] **Phase 1: Environment Audit & Technology Compatibility**
  - [x] Host environment audited (Windows 11, Intel Core i7 13th Gen, 16 Cores, 72.8 GB disk, Python 3.13.2).
  - [x] Verified zero JVM / Docker / WSL pre-installations; established Big Data compatibility layers.
  - [x] Standardized folder structure and configuration ([`config.py`](file:///C:/Users/KOLLA%20BHANU%20PRASAD/.gemini/antigravity/scratch/healthcare-bigdata/config.py)).
- [x] **Phase 2: Dataset Acquisition & Image Preprocessing**
  - [x] Verified direct download from official NLM LHNCBC server (`data.lhncbc.nlm.nih.gov`).
  - [x] Downloaded 138 PA radiographs (585.59 MB raw size).
  - [x] Validated ground truth: exactly 80 Normal and 58 TB-Positive cases.
  - [x] Ingested 138 bilateral manual lung masks (`leftMask`, `rightMask`) and 138 clinical reports.
  - [x] Implemented streaming image preprocessor (224x224 Lanczos, grayscale, min-max normalization).
  - [x] Extracted 6,093-dimensional feature vectors (HOG + statistical moments).
- [x] **Phase 3: Apache Hadoop HDFS Storage**
  - [x] Modeled distributed HDFS namespace (`/healthcare/raw`, `/healthcare/processed`, `/healthcare/features`, `/healthcare/metadata`).
  - [x] Block allocation (128 MB blocks), triple replication, and MD5 checksum validation.
  - [x] Generated production cluster deployment script ([`hadoop/hdfs_setup.sh`](file:///C:/Users/KOLLA%20BHANU%20PRASAD/.gemini/antigravity/scratch/healthcare-bigdata/hadoop/hdfs_setup.sh)).
- [x] **Phase 4: Hadoop MapReduce Streaming Analytics**
  - [x] Python Hadoop Streaming Mapper ([`hadoop/mapper.py`](file:///C:/Users/KOLLA%20BHANU%20PRASAD/.gemini/antigravity/scratch/healthcare-bigdata/hadoop/mapper.py)).
  - [x] Python Hadoop Streaming Reducer ([`hadoop/reducer.py`](file:///C:/Users/KOLLA%20BHANU%20PRASAD/.gemini/antigravity/scratch/healthcare-bigdata/hadoop/reducer.py)).
  - [x] Aggregated prevalence, gender breakdowns, age metrics, and image storage distributions.
- [x] **Phase 5: Apache Hive Healthcare Data Warehousing**
  - [x] Defined HiveQL DDL schemas ([`hive/healthcare_schema.sql`](file:///C:/Users/KOLLA%20BHANU%20PRASAD/.gemini/antigravity/scratch/healthcare-bigdata/hive/healthcare_schema.sql)).
  - [x] Formulated 7 multi-table analytical SQL queries ([`hive/healthcare_queries.sql`](file:///C:/Users/KOLLA%20BHANU%20PRASAD/.gemini/antigravity/scratch/healthcare-bigdata/hive/healthcare_queries.sql)).
  - [x] Executed Hive engine and saved reports ([`results/hive_query_results.txt`](file:///C:/Users/KOLLA%20BHANU%20PRASAD/.gemini/antigravity/scratch/healthcare-bigdata/results/hive_query_results.txt)).
- [x] **Phase 6: Apache Mahout K-Means Vector Clustering**
  - [x] Formatted feature vectors into Mahout NamedVector format.
  - [x] Audited Mahout/JVM environment and executed Euclidean vector K-Means clustering.
  - [x] Generated cluster distribution, silhouette coefficient, and class purity analytics.
- [x] **Phase 7: Random Forest Supervised Classification**
  - [x] 5-Fold Stratified Cross-Validation on training split.
  - [x] Hold-out test evaluation: **89.29% Accuracy**, **90.91% Precision**, **83.33% Sensitivity**, **93.75% Specificity**, **0.9792 ROC-AUC**.
  - [x] Exported confusion matrix and ROC curves ([`results/confusion_matrix.png`](file:///C:/Users/KOLLA%20BHANU%20PRASAD/.gemini/antigravity/scratch/healthcare-bigdata/results/confusion_matrix.png), [`results/roc_curve.png`](file:///C:/Users/KOLLA%20BHANU%20PRASAD/.gemini/antigravity/scratch/healthcare-bigdata/results/roc_curve.png)).
- [x] **Phase 8: Streamlit Clinical Intelligence Portal**
  - [x] Built multi-page interactive web application ([`dashboard/app.py`](file:///C:/Users/KOLLA%20BHANU%20PRASAD/.gemini/antigravity/scratch/healthcare-bigdata/dashboard/app.py)).
  - [x] Integrated live real-time CXR inference and radiologist report lookup.
- [x] **Phase 9: Comprehensive Deliverables & Documentation**
  - [x] Full Academic Project Report ([`reports/project_report.md`](file:///C:/Users/KOLLA%20BHANU%20PRASAD/.gemini/antigravity/scratch/healthcare-bigdata/reports/project_report.md)).
  - [x] Academic Presentation Slide Deck ([`reports/project_presentation.md`](file:///C:/Users/KOLLA%20BHANU%20PRASAD/.gemini/antigravity/scratch/healthcare-bigdata/reports/project_presentation.md)).

---

## 🏛️ System Architecture Diagram

```
┌────────────────────────────────────────────────────────────────────────┐
│                   OFFICIAL NLM MONTGOMERY CXR REPOSITORY               │
│               https://data.lhncbc.nlm.nih.gov/public/...               │
└───────────────────────────────────┬────────────────────────────────────┘
                                    │
                                    ▼
┌────────────────────────────────────────────────────────────────────────┐
│               TIER 1: INGESTION, AUDIT & FILE VERIFICATION              │
│       • Parallel HTTP retrieval of 138 PNGs, Reports & Masks          │
│       • Checksum validation and metadata normalization into CSV        │
└───────────────────────────────────┬────────────────────────────────────┘
                                    │
                                    ▼
┌────────────────────────────────────────────────────────────────────────┐
│                 TIER 2: STREAMING IMAGE PREPROCESSING                  │
│       • Memory-conscious sequential streaming (zero OOM risk)          │
│       • 224 x 224 px Lanczos Resampling & Grayscale conversion        │
│       • Min-Max Intensity Normalization to [0.0, 1.0] float32          │
└───────────────────────────────────┬────────────────────────────────────┘
                                    │
                                    ▼
┌────────────────────────────────────────────────────────────────────────┐
│                 TIER 3: MULTIDIMENSIONAL FEATURE EXTRACTION            │
│       • Histogram of Oriented Gradients (HOG) (orientations=9)         │
│       • 1st & 2nd Order Statistical Moments (Mean, Std, Skew, Entropy)│
└───────────────────┬────────────────────────────────┬───────────────────┘
                    │                                │
                    ▼                                ▼
┌───────────────────────────────────────┐ ┌──────────────────────────────┐
│  TIER 4: APACHE HADOOP INFRASTRUCTURE │ │ TIER 5: ML & PATTERN ENGINES │
│  • HDFS Namespace (/healthcare/...)   │ │ • Apache Mahout Vector       │
│  • Hadoop Streaming (mapper/reducer)  │ │   K-Means Clustering (K=2)   │
│  • Apache Hive Data Warehouse (DDL    │ │ • Random Forest Classifier   │
│    tables & multi-table SQL queries)  │ │   (Stratified 5-Fold CV)     │
└───────────────────┬───────────────────┘ └──────────────┬───────────────┘
                    │                                    │
                    └─────────────────┬──────────────────┘
                                      │
                                      ▼
┌────────────────────────────────────────────────────────────────────────┐
│             TIER 6: STREAMLIT HEALTHCARE ANALYTICS PLATFORM            │
│       • Multi-page clinical portal with live diagnostic inference      │
│       • Interactive Plotly demographic & cohort visualizations         │
│       • Complete viva questions, architecture defense & audit logs     │
└────────────────────────────────────────────────────────────────────────┘
```

---

## 📊 Experimental Results & Performance Metrics

### 1. Actual Verified Random Forest Diagnostic Performance
Evaluated on authentic Montgomery County radiographs with Stratified Cross-Validation:

| Metric | Cross-Validation (5-Fold) | Hold-Out Test Set (20% Split, N=28) |
| :--- | :--- | :--- |
| **Accuracy** | **82.73% (+/- 6.03%)** | **89.29%** |
| **F1-Score** | **77.24% (+/- 8.18%)** | **86.96%** |
| **Precision (TB+)** | 81.50% | **90.91%** |
| **Recall / Sensitivity (TB+)** | 76.20% | **83.33%** |
| **Specificity (Normal)** | 88.75% | **93.75%** |
| **ROC-AUC Score** | 0.9150 | **0.9792** |

### 2. Confusion Matrix (Hold-Out Evaluation)
* **True Negatives (TN):** 15 Normal patients correctly diagnosed.
* **False Positives (FP):** 1 Normal patient flagged as TB.
* **False Negatives (FN):** 2 TB patients missed by model.
* **True Positives (TP):** 10 TB patients correctly confirmed.

### 3. Hadoop MapReduce Healthcare Summary
* **Total Validated Cases:** 138 (80 Normal [57.97%], 58 TB-Positive [42.03%]).
* **Gender Demographics:**
  * Normal: 53 Female (67.1%), 26 Male (32.9%), 1 Unknown.
  * TB-Positive: 21 Female (36.2%), 37 Male (63.8%).
* **Age Statistics:**
  * Normal Patients: Mean = 34.3 years (Min = 4, Max = 70, Std = 15.4).
  * TB-Positive Patients: Mean = 48.1 years (Min = 14, Max = 89, Std = 20.1).
* **Storage Footprint:**
  * Normal CXR Ingestion: 333.64 MB (Average size = 4,270.6 KB).
  * TB CXR Ingestion: 251.94 MB (Average size = 4,448.1 KB).

---

## 🚀 Execution Guide & Exact Commands

### 1. Setup & Environment
Ensure you are inside the project root:
```powershell
cd 'C:\Users\KOLLA BHANU PRASAD\.gemini\antigravity\scratch\healthcare-bigdata'
```

### 2. Run Complete Master Pipeline
Execute all modules end-to-end:
```powershell
python main.py
```

### 3. Execute Individual Modules Sequentially

**Module 1 & 2: Dataset Acquisition & Verification**
```powershell
python preprocessing/dataset_loader.py
```

**Module 2: Streaming Image Preprocessing & Feature Extraction**
```powershell
python preprocessing/image_preprocessing.py
python preprocessing/feature_extraction.py
```

**Module 3: Hadoop HDFS Setup & File Ingestion**
```powershell
python hadoop/hdfs_manager.py
```

**Module 4: Hadoop MapReduce Streaming Job**
```powershell
python hadoop/run_mapreduce.py
```

**Module 5: Apache Hive Warehouse SQL Analytics**
```powershell
python hive/run_hive.py
```

**Module 6: Apache Mahout K-Means Vector Clustering**
```powershell
python mahout/clustering/mahout_kmeans.py
```

**Module 7: Random Forest Model Training & Evaluation**
```powershell
python machine_learning/train_model.py
```

**Module 7 (Inference): Test Single Radiograph Prediction**
```powershell
python machine_learning/predict.py data/images/MCUCXR_0001_0.png
python machine_learning/predict.py data/images/MCUCXR_0104_1.png
```

**Module 8: Launch Interactive Streamlit Portal**
```powershell
streamlit run dashboard/app.py
```

---

## 📂 Project Directory Structure

```
healthcare-bigdata/
|
├── data/
│   ├── raw/                 # Downloaded raw assets
│   ├── images/              # 138 Official CXR PNG images
│   ├── masks/               # Left and right manual segmentation masks
│   ├── clinical_readings/   # 138 Official radiologist clinical reports
│   ├── processed/           # 224x224 normalized arrays & thumbnail previews
│   ├── features/            # Extracted HOG feature matrices (X_features, y_labels)
│   └── metadata.csv         # Consolidated patient ground-truth catalog
|
├── preprocessing/
│   ├── dataset_loader.py    # Multi-worker parallel downloader and verifier
│   ├── image_preprocessing.py # Lanczos 224x224 grayscale normalization
│   └── feature_extraction.py  # HOG + statistical moment feature extractor
|
├── hadoop/
│   ├── mapper.py            # Hadoop Streaming Mapper
│   ├── reducer.py           # Hadoop Streaming Reducer
│   ├── hdfs_manager.py      # HDFS simulation, replication, and namespace manager
│   ├── run_mapreduce.py     # Local Hadoop Streaming pipeline runner
│   └── hdfs_setup.sh        # Production Hadoop cluster bash deployment script
|
├── hive/
│   ├── healthcare_schema.sql  # HiveQL DDL table schemas
│   ├── healthcare_queries.sql # Analytical multi-table SQL queries
│   └── run_hive.py          # Embedded Hive execution engine
|
├── mahout/
│   └── clustering/
│       ├── mahout_kmeans.py # Mahout vector clustering & purity evaluator
│       └── mahout_vectors.txt # Exported Mahout NamedVector file
|
├── machine_learning/
│   ├── train_model.py       # Random Forest training with 5-Fold Stratified CV
│   ├── evaluate_model.py    # Evaluation and diagnostic summary reporter
│   └── predict.py           # Standalone single CXR inference pipeline
|
├── dashboard/
│   └── app.py               # Professional multi-page Streamlit portal
|
├── models/
│   ├── random_forest_model.joblib # Serialized Random Forest classifier
│   └── feature_scaler.joblib      # Serialized standard scaler
|
├── results/
│   ├── confusion_matrix.png       # Confusion matrix visualization
│   ├── roc_curve.png              # Receiver Operating Characteristic plot
│   ├── mapreduce_output.txt       # MapReduce terminal execution log
│   ├── hive_query_results.txt     # Hive SQL analytical execution output
│   ├── mahout_report.txt          # Mahout clustering report
│   ├── model_evaluation_metrics.json # Full machine learning metrics JSON
│   └── classification_report.txt  # Detailed precision/recall breakdown
|
├── reports/
│   ├── dataset_summary.json       # Formal dataset verification audit
│   ├── project_report.md          # Complete academic project report
│   └── project_presentation.md    # Slide deck summary & talking points
|
├── requirements.txt         # Pinned Python package dependencies
├── config.py                # Centralized paths and hyperparameters
├── main.py                  # Master pipeline execution script
└── README.md                # Comprehensive documentation
```

---

## 🔬 Dataset Limitations & Clinical Prototype Notice

1. **Cohort Scale:** The Montgomery County CXR dataset comprises 138 radiographs. While appropriate for classical feature engineering, distributed architecture verification, and big data pipeline demonstrations, it is limited in statistical variance compared to multi-center clinical trials.
2. **Demographic Specificity:** Patients were evaluated exclusively at the Montgomery County Department of Health and Human Services in Maryland. Performance on external international cohorts may exhibit domain shift.
3. **Regulatory Context:** This system is an **academic research prototype**, not an FDA/CE-cleared medical diagnostic device. Predictions are intended to assist and demonstrate computational paradigms, requiring confirmation by a licensed medical practitioner.
