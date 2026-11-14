-- ====================================================================
-- APACHE HIVE HEALTHCARE BIG DATA ANALYTICS SCHEMA
-- Project: Tuberculosis Detection and Pattern Analysis
-- Dataset: Montgomery County CXR Set (NLM/NIH)
-- ====================================================================

-- 1. Create Dedicated Healthcare Database
CREATE DATABASE IF NOT EXISTS healthcare_analytics
COMMENT 'Central Database for Tuberculosis CXR Big Data Analytics'
LOCATION '/healthcare/hive/warehouse/healthcare_analytics.db';

USE healthcare_analytics;

-- 2. Patient Demographics and Diagnostic Table
CREATE TABLE IF NOT EXISTS patient_metadata (
    patient_id STRING COMMENT 'Unique anonymized patient identifier (e.g., MCUCXR_0001_0)',
    patient_gender STRING COMMENT 'Biological gender of patient (Male, Female, Unknown)',
    patient_age INT COMMENT 'Age of patient in years',
    label INT COMMENT 'Diagnostic integer flag (0: Normal, 1: TB-Positive)',
    diagnosis STRING COMMENT 'Clinical classification string (Normal, TB_Positive)',
    radiology_finding STRING COMMENT 'Full clinical reading from NLM radiologist'
)
COMMENT 'Patient-level clinical records and disease classifications'
ROW FORMAT DELIMITED
FIELDS TERMINATED BY ','
STORED AS TEXTFILE
LOCATION '/healthcare/hive/patient_metadata';

-- 3. Raw Chest X-ray Image Storage and Mask Table
CREATE TABLE IF NOT EXISTS image_metadata (
    patient_id STRING COMMENT 'Foreign key to patient_metadata.patient_id',
    filename STRING COMMENT 'CXR PNG filename',
    file_size_kb DOUBLE COMMENT 'Original image file size in kilobytes',
    has_left_mask BOOLEAN COMMENT 'Manual left lung segmentation mask availability',
    has_right_mask BOOLEAN COMMENT 'Manual right lung segmentation mask availability'
)
COMMENT 'Metadata tracking original X-ray images and segmentation ground-truth'
ROW FORMAT DELIMITED
FIELDS TERMINATED BY ','
STORED AS TEXTFILE
LOCATION '/healthcare/hive/image_metadata';

-- 4. Preprocessing and Image Statistics Table
CREATE TABLE IF NOT EXISTS preprocessing_audit (
    original_filename STRING COMMENT 'Original CXR PNG filename',
    processed_filename STRING COMMENT 'Processed NumPy serialized array file',
    status STRING COMMENT 'Preprocessing status (Success, Corrupted_Failed)',
    width INT COMMENT 'Normalized width dimension (224 px)',
    height INT COMMENT 'Normalized height dimension (224 px)',
    mean_intensity DOUBLE COMMENT 'Normalized pixel mean intensity in [0, 1]',
    std_intensity DOUBLE COMMENT 'Normalized pixel standard deviation',
    min_val DOUBLE COMMENT 'Minimum normalized pixel value',
    max_val DOUBLE COMMENT 'Maximum normalized pixel value'
)
COMMENT 'Quality control audit of image normalization and resizing pipeline'
ROW FORMAT DELIMITED
FIELDS TERMINATED BY ','
STORED AS TEXTFILE
LOCATION '/healthcare/hive/preprocessing_audit';
