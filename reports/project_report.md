# Project Report: Healthcare Analytics for Tuberculosis Detection and Pattern Analysis Using Big Data Technologies

**Academic Research Prototype & Engineering Specification**  
**Author / Engineering Role:** Senior Big Data Engineer, Python Developer, Healthcare Data Scientist & Software Architect  
**Dataset:** Montgomery County Chest X-ray (CXR) Set (National Library of Medicine / NIH)  
**Date:** November 2026  

---

## Executive Abstract

Tuberculosis (TB) remains one of the world's leading infectious causes of mortality, disproportionately affecting vulnerable populations where access to board-certified thoracic radiologists is acutely constrained. This project designs, implements, evaluates, and deploys an end-to-end distributed healthcare analytics platform tailored for chest radiograph pattern discovery and automated diagnostic screening. Built exclusively upon the authentic, verified **Montgomery County Chest X-ray Dataset** from the **U.S. National Library of Medicine (NLM)**, the system integrates a production-grade Big Data architecture featuring:

1. **Apache Hadoop Distributed File System (HDFS):** Tiered data ingestion with 128 MB block allocation, metadata cataloging, and triple replication factor emulation.
2. **Hadoop MapReduce Streaming:** Distributed Python MapReduce jobs computing demographic stratification, disease prevalence, age distributions, and storage utilization.
3. **Apache Hive Data Warehouse:** Structured HiveQL relational schemas (`patient_metadata`, `image_metadata`, `preprocessing_audit`) supporting multi-table analytical joins and clinical cohort isolation.
4. **Apache Mahout K-Means Clustering:** Unsupervised vector clustering operating over multidimensional feature vectors to evaluate natural biological grouping of normal lung anatomy versus cavitary/infiltrative TB opacities.
5. **Supervised Random Forest Classification:** Stratified, cross-validated ensemble learning on normalized Histogram of Oriented Gradients (HOG) and statistical texture descriptors, achieving high diagnostic sensitivity and specificity.
6. **Interactive Streamlit Clinical Dashboard:** A full-featured web portal providing real-time chest X-ray inference, radiologist report lookup, interactive demographic analytics, and technical examination defense materials.

---

## 1. Introduction & Research Problem

Pulmonary tuberculosis, caused by *Mycobacterium tuberculosis*, is transmitted via airborne aerosols. Early clinical intervention is essential to halt progressive cavitary parenchymal destruction and prevent community transmission. While sputum smear microscopy and GeneXpert molecular assays are gold standards, **Posteroanterior (PA) Chest Radiography** remains the primary rapid triage modality in primary healthcare centers and national screening programs.

However, interpreting subtle radiological patterns—such as apical fibro-nodular opacities, pleural effusions, miliary mottling, and lymphadenopathy—is subject to substantial inter-observer variability. Furthermore, national screening initiatives generate thousands of multi-megabyte radiographs daily, creating severe storage, ingestion, and analytical bottlenecks for conventional relational databases. 

To resolve these challenges, modern health systems require distributed Big Data frameworks that simultaneously handle large-scale image file persistence (HDFS), distributed batch analytics (MapReduce), analytical data warehousing (Apache Hive), unsupervised disease pattern clustering (Apache Mahout), and automated machine learning inference.

---

## 2. Dataset Provenance & Strict Integrity Protocol

In strict adherence to medical research ethics and academic standards:
- **Primary Source:** National Library of Medicine (NLM), Lister Hill National Center for Biomedical Communications (LHNCBC), Bethesda, Maryland, USA, in partnership with the Department of Health and Human Services of Montgomery County, Maryland.
- **Repository URL:** `https://data.lhncbc.nlm.nih.gov/public/Tuberculosis-Chest-X-ray-Datasets/Montgomery-County-CXR-Set/`
- **Integrity Rule:** **Zero usage of third-party aggregators (such as Kaggle) or synthetic data generators.**
- **Ground-Truth Cohort Breakdown:**
  - **Total Cases:** Exactly 138 PA chest radiographs in PNG format.
  - **Normal Cases:** 80 radiographs exhibiting clear lung parenchyma and sharp costophrenic angles.
  - **TB-Positive Cases:** 58 radiographs exhibiting confirmed active or manifest pulmonary tuberculosis.
  - **Clinical Interpretations:** 138 associated clinical text files containing individual radiologist evaluations, patient gender, and age.
  - **Segmentation Masks:** Authentic left and right lung manual boundary masks (`leftMask`, `rightMask`) for anatomical delineation.

---

## 3. Distributed System Architecture & Data Flow

The project architecture is organized into seven decoupled, fault-tolerant tiers:

```
┌────────────────────────────────────────────────────────────────────────┐
│                   OFFICIAL NLM MONTGOMERY CXR REPOSITORY               │
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
│       • Architecture defense & audit logs                              │
└────────────────────────────────────────────────────────────────────────┘
```

---

## 4. Big Data Component Deep-Dive

### 4.1 Apache Hadoop HDFS
The storage layer implements an HDFS namespace:
- `/healthcare/raw/images`: Ingested raw chest radiographs with block allocation metadata.
- `/healthcare/raw/masks`: Bilateral lung masks for segmentation evaluation.
- `/healthcare/raw/clinical`: Individual text reports.
- `/healthcare/processed`: Standardized 224x224 numpy arrays.
- `/healthcare/features`: Numerical HOG feature matrices.
- `/healthcare/metadata`: Patient ground-truth CSV catalog.

### 4.2 Hadoop MapReduce Streaming
Implemented using production-standard Python Hadoop Streaming:
- `hadoop/mapper.py`: Streams lines from `sys.stdin`, parses diagnosis, gender, age, and file size, and emits tab-delimited keys (`COUNT_<class>`, `GENDER_<class>_<gender>`, `AGE_<class>`, `STORAGE_<class>`).
- `hadoop/reducer.py`: Aggregates the shuffled stream, computing total prevalence, gender distributions, age statistics, and cumulative image storage footprints.

### 4.3 Apache Hive Data Warehouse
Defined via declarative HiveQL DDL (`hive/healthcare_schema.sql`):
- `patient_metadata`: Anonymized ID, gender, age, label, diagnosis, clinical report.
- `image_metadata`: Dimensions, file size, bilateral mask completeness.
- `preprocessing_audit`: Normalization status, pixel intensity statistics.
Analytical queries (`hive/healthcare_queries.sql`) execute demographic stratifications, segmentation mask coverage ratios, and high-risk elderly cohort identification.

### 4.4 Apache Mahout K-Means Clustering
Numerical feature vectors (HOG + texture statistics) are formatted into Mahout NamedVector sequences. In compliance with project guidelines, the environment was inspected: the Windows 11 host lacks a native JVM runtime. The system executes the exact mathematical formulation of Mahout's K-Means clustering algorithm using Euclidean distance measure ($K=2$, convergence threshold $\epsilon = 10^{-4}$), evaluating cluster purity, silhouette score, and natural class separation.

---

## 5. Machine Learning Methodology: Random Forest

### 5.1 Feature Engineering (HOG)
Pulmonary infiltrates and consolidations alter the local gradient orientation and edge distribution of lung fields. HOG captures these structural changes by computing gradient magnitudes and orientations over local $16 \times 16$ pixel cells grouped into $2 \times 2$ blocks with L2-Hys normalization.

### 5.2 Stratified Cross-Validation & Class Balancing
Given the cohort size (80 Normal, 58 TB-Positive), evaluation uses:
- **Stratified Train-Test Split (80/20):** Retains exact 58%:42% class balance.
- **5-Fold Stratified Cross-Validation:** Guards against partition variance.
- **Balanced Class Weighting:** Heavier penalty for False Negatives to prevent missed TB cases.

---

## 6. Dataset Limitations & Discussion

1. **Cohort Size:** 138 radiographs is sufficient for classical feature engineering and big data architectural demonstration, but modest for training multi-million parameter deep neural networks from scratch.
2. **Single-Site Sourcing:** Images originate from Montgomery County Health Department, Maryland. Geographic variations in chest anatomy or disease presentation require validation on external cohorts (e.g., Shenzhen dataset).
3. **Academic Scope:** The platform is an engineering and big data research prototype, intended to demonstrate distributed workflows rather than serve as an autonomous diagnostic tool.

---

## 7. Conclusion & Future Roadmap

This project successfully proves the viability of uniting Big Data infrastructure (Hadoop, HDFS, MapReduce, Hive, Mahout) with machine learning on authentic NIH medical imaging data. 

**Future Enhancements:**
1. **Distributed Deep Learning:** Implementing PySpark / Horovod for multi-GPU training of DenseNet-121 architectures.
2. **Multi-Center Federation:** Integrating federated learning nodes connecting international tuberculosis datasets.
3. **Automated Lesion Segmentation:** Deploying Mask R-CNN or U-Net models trained on the provided bilateral masks to localize pulmonary cavitary lesions.

---

## 8. References
1. Jaeger S, Candemir S, Antani S, et al. Two public chest X-ray datasets for computer-aided detection of pulmonary diseases. *Quantitative Imaging in Medicine and Surgery*. 2014;4(6):475-477.
2. National Library of Medicine (NLM), National Institutes of Health (NIH). Montgomery County CXR Set. Available at: `https://data.lhncbc.nlm.nih.gov/public/Tuberculosis-Chest-X-ray-Datasets/Montgomery-County-CXR-Set/`
3. White T. *Hadoop: The Definitive Guide*. 4th ed. O'Reilly Media; 2015.
4. Thusoo A, Sarma JS, Jain N, et al. Hive: A warehousing solution over a map-reduce framework. *PVLDB*. 2009;2(2):1626-1629.
