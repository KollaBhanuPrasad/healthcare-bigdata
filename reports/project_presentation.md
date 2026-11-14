# Presentation: Healthcare Big Data Analytics for Tuberculosis Detection
## Slide Deck Summary & Presentation Notes

---

### Slide 1: Title & Executive Summary
* **Title:** Healthcare Analytics for Tuberculosis Detection and Pattern Analysis Using Big Data Technologies
* **Presenter:** Senior Big Data & Healthcare Data Scientist
* **Academic Domain:** Biomedical Big Data, Distributed Computing, Healthcare Machine Learning
* **Dataset:** Montgomery County Chest X-ray Set (National Library of Medicine / NIH)
* **Core Technologies:** Hadoop HDFS, MapReduce Streaming, Apache Hive, Apache Mahout, Scikit-Learn, Streamlit

---

### Slide 2: Clinical Problem Statement & Research Motivation
* **Global Health Burden:** Tuberculosis remains one of the world's deadliest infectious diseases, caused by *Mycobacterium tuberculosis*.
* **Radiological Bottleneck:** Early screening relies on posteroanterior (PA) chest radiographs. In high-burden regions, specialized radiologists are scarce.
* **Big Data Challenge:** Clinical image repositories and longitudinal electronic health records (EHR) generate massive volumes of unstructured pixel and structured clinical data requiring distributed storage, parallel processing, and automated screening models.

---

### Slide 3: Official Dataset Verification & Integrity
* **Authentic Source:** National Library of Medicine (NLM), Lister Hill Center (NIH).
* **Strict Avoidance of Third-Party Platforms:** No Kaggle datasets or synthetic patient records.
* **Ground-Truth Breakdown:**
  * **138 Total Images** (80 Normal, 58 TB-Positive).
  * 138 Clinical Radiologist Reports (`.txt`).
  * Complete Bilateral Lung Segmentation Masks (`leftMask`, `rightMask`).
* **Ethical Context:** Academic research prototype demonstrating distributed architectural patterns and feature engineering.

---

### Slide 4: End-to-End System Architecture
* **Tier 1: Ingestion & Verification:** NLM Archive Extraction, Checksums, and File Validation.
* **Tier 2: Image Processing:** 224x224 Lanczos Resizing, Grayscale, Intensity Normalization.
* **Tier 3: Distributed Storage:** Apache Hadoop HDFS namespace with block management (128 MB blocks, replication=3).
* **Tier 4: Distributed Processing:** Python Hadoop Streaming (`mapper.py` and `reducer.py`).
* **Tier 5: Data Warehousing:** Apache HiveQL multi-table schemas (`patient_metadata`, `image_metadata`, `preprocessing_audit`).
* **Tier 6: Machine Learning:**
  * **Unsupervised:** Apache Mahout K-Means Vector Clustering.
  * **Supervised:** Random Forest Classifier on HOG Feature Vectors.
* **Tier 7: Application Layer:** Multi-Page Interactive Streamlit Portal.

---

### Slide 5: Hadoop MapReduce Implementation
* **Hadoop Streaming Architecture:** Standard I/O pipelines (`sys.stdin` to `sys.stdout`).
* **Mapper Role:** Emits key-value tokens for case counts, gender stratification, and age intervals.
* **Shuffle & Sort:** Hadoop internal partitioning by diagnosis key.
* **Reducer Role:** Computes disease prevalence (57.97% Normal vs 42.03% TB), gender ratios, and average storage footprints per diagnostic cohort.

---

### Slide 6: Apache Hive Healthcare Data Warehouse
* **Hive Tables:**
  * `patient_metadata` (Demographics, diagnosis, clinical readings)
  * `image_metadata` (Storage size, mask availability)
  * `preprocessing_audit` (Image dimensions, mean/std pixel intensities)
* **Analytical Inquiries:**
  * Geriatric & high-risk cohort isolation.
  * Bilateral lung segmentation completeness rate.
  * Image file footprint comparisons.

---

### Slide 7: Unsupervised Discovery via Apache Mahout K-Means
* **Vector Transformation:** Dense and Named Vector representations of combined HOG and texture statistics.
* **Clustering Objective:** Discover intrinsic biological separation between healthy lung parenchymal patterns and pathological infiltrative opacities.
* **Evaluation:** Purity analysis, Silhouette Coefficient, and Inertia tracking across cluster assignments.

---

### Slide 8: Supervised Random Forest Classification & Clinical Metrics
* **Feature Extraction:** Histogram of Oriented Gradients (HOG) capturing directional edges and parenchymal opacities.
* **Stratified Validation:** 5-Fold Stratified Cross-Validation + 20% Hold-out test set to prevent class imbalance skew.
* **Clinical Performance:**
  * Accuracy, Sensitivity (Recall), Specificity, Precision, F1-Score, and ROC-AUC.
  * Confusion matrix analysis prioritizing minimization of False Negatives.

---

### Slide 9: Streamlit Interactive Healthcare Portal
* **Live Features:**
  * Cohort Demographic Visualizations (Plotly donuts and boxplots).
  * Side-by-side CXR viewer with manual mask overlays.
  * MapReduce and Hive Query execution consoles.
  * Live Real-Time CXR Diagnostic Inference with probability gauges.

---

### Slide 10: Conclusion & Future Roadmap
* **Conclusion:** Successfully integrated the full big data ecosystem (Hadoop, HDFS, MapReduce, Hive, Mahout) with machine learning on authentic NLM medical data.
* **Future Enhancements:**
  * Multi-GPU Distributed Deep Learning (ResNet-50 / DenseNet-121) using Apache Spark MLlib.
  * Multi-center federated dataset scaling (integrating Shenzhen CXR set).
  * Automated Lung Nodule & Lesion Localization using YOLO / Mask R-CNN.
