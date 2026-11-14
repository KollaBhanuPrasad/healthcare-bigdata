-- ====================================================================
-- APACHE HIVE ANALYTICAL HEALTHCARE QUERIES
-- Tuberculosis CXR Patterns, Cohort Statistics and Image Metrics
-- ====================================================================

USE healthcare_analytics;

-- --------------------------------------------------------------------
-- QUERY 1: Overall Disease Distribution & Prevalence
-- --------------------------------------------------------------------
SELECT 
    diagnosis,
    COUNT(*) AS total_patients,
    ROUND(COUNT(*) * 100.0 / SUM(COUNT(*)) OVER(), 2) AS prevalence_percentage
FROM patient_metadata
GROUP BY diagnosis;

-- --------------------------------------------------------------------
-- QUERY 2: Gender Stratification Across Diagnostic Categories
-- --------------------------------------------------------------------
SELECT 
    diagnosis,
    patient_gender,
    COUNT(*) AS patient_count,
    ROUND(COUNT(*) * 100.0 / SUM(COUNT(*)) OVER(PARTITION BY diagnosis), 2) AS gender_share_pct
FROM patient_metadata
WHERE patient_gender IN ('Male', 'Female')
GROUP BY diagnosis, patient_gender
ORDER BY diagnosis, patient_count DESC;

-- --------------------------------------------------------------------
-- QUERY 3: Patient Age Demographics & Risk Analysis
-- --------------------------------------------------------------------
SELECT 
    diagnosis,
    COUNT(CASE WHEN patient_age > 0 THEN 1 END) AS recorded_age_count,
    ROUND(AVG(CASE WHEN patient_age > 0 THEN patient_age END), 2) AS mean_age,
    ROUND(STDDEV(CASE WHEN patient_age > 0 THEN patient_age END), 2) AS std_age,
    MIN(CASE WHEN patient_age > 0 THEN patient_age END) AS min_age,
    MAX(CASE WHEN patient_age > 0 THEN patient_age END) AS max_age
FROM patient_metadata
GROUP BY diagnosis;

-- --------------------------------------------------------------------
-- QUERY 4: Image Storage Footprint Analysis Across Diagnoses
-- --------------------------------------------------------------------
SELECT 
    p.diagnosis,
    COUNT(i.filename) AS image_count,
    ROUND(SUM(i.file_size_kb) / 1024.0, 2) AS total_storage_mb,
    ROUND(AVG(i.file_size_kb), 2) AS avg_image_size_kb,
    ROUND(MIN(i.file_size_kb), 2) AS min_image_size_kb,
    ROUND(MAX(i.file_size_kb), 2) AS max_image_size_kb
FROM patient_metadata p
JOIN image_metadata i ON p.patient_id = i.patient_id
GROUP BY p.diagnosis;

-- --------------------------------------------------------------------
-- QUERY 5: Segmentation Mask Coverage Rate
-- --------------------------------------------------------------------
SELECT 
    COUNT(*) AS total_images,
    SUM(CASE WHEN has_left_mask AND has_right_mask THEN 1 ELSE 0 END) AS complete_bilateral_masks,
    ROUND(SUM(CASE WHEN has_left_mask AND has_right_mask THEN 1 ELSE 0 END) * 100.0 / COUNT(*), 2) AS segmentation_readiness_pct
FROM image_metadata;

-- --------------------------------------------------------------------
-- QUERY 6: Preprocessed Image Intensity Variations (Normal vs TB)
-- --------------------------------------------------------------------
SELECT 
    p.diagnosis,
    COUNT(a.processed_filename) AS processed_image_count,
    ROUND(AVG(a.mean_intensity), 4) AS avg_pixel_mean,
    ROUND(AVG(a.std_intensity), 4) AS avg_pixel_std,
    ROUND(MIN(a.mean_intensity), 4) AS lowest_pixel_mean,
    ROUND(MAX(a.mean_intensity), 4) AS highest_pixel_mean
FROM patient_metadata p
JOIN preprocessing_audit a ON p.patient_id = REPLACE(a.original_filename, '.png', '')
WHERE a.status = 'Success'
GROUP BY p.diagnosis;

-- --------------------------------------------------------------------
-- QUERY 7: High-Risk Cohorts (Age > 45 with Tuberculosis)
-- --------------------------------------------------------------------
SELECT 
    p.patient_id,
    p.patient_gender,
    p.patient_age,
    p.radiology_finding
FROM patient_metadata p
WHERE p.diagnosis = 'TB_Positive' AND p.patient_age >= 45
ORDER BY p.patient_age DESC;
