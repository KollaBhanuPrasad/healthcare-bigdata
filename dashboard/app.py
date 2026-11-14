"""
Professional Healthcare Streamlit Analytics Dashboard
Project: Healthcare Analytics for Tuberculosis Detection and Pattern Analysis Using Big Data Technologies
Dataset: Montgomery County Chest X-ray Set (NLM/NIH)
"""

import sys
import json
from pathlib import Path
import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from PIL import Image

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.append(str(PROJECT_ROOT))

import config
from machine_learning.predict import predict_cxr

# Page Configuration
st.set_page_config(
    page_title="Tuberculosis Big Data Healthcare Analytics",
    page_icon="🫁",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom Styling
st.markdown("""
<style>
    .main-header {
        font-size: 2.2rem;
        font-weight: 700;
        color: #1E3A8A;
        margin-bottom: 0.2rem;
    }
    .sub-header {
        font-size: 1.1rem;
        color: #475569;
        margin-bottom: 1.5rem;
    }
    .metric-card {
        background-color: #F8FAFC;
        border-radius: 8px;
        padding: 15px;
        border-left: 5px solid #2563EB;
        box-shadow: 0 1px 3px rgba(0,0,0,0.1);
    }
    .clinical-alert {
        padding: 12px;
        border-radius: 6px;
        background-color: #EFF6FF;
        border: 1px solid #BFDBFE;
        color: #1E40AF;
        margin-bottom: 15px;
    }
</style>
""", unsafe_allow_html=True)

# Helper functions to load cached data
@st.cache_data
def load_metadata():
    p = config.DATA_DIR / "metadata.csv"
    if p.exists():
        return pd.read_csv(p)
    return pd.DataFrame()

@st.cache_data
def load_summary():
    p = config.REPORTS_DIR / "dataset_summary.json"
    if p.exists():
        with open(p, "r", encoding="utf-8") as f:
            return json.load(f)
    return {}

@st.cache_data
def load_evaluation_metrics():
    p = config.RESULTS_DIR / "model_evaluation_metrics.json"
    if p.exists():
        with open(p, "r", encoding="utf-8") as f:
            return json.load(f)
    return {}

@st.cache_data
def load_mahout_results():
    p = config.RESULTS_DIR / "mahout_clustering_results.json"
    if p.exists():
        with open(p, "r", encoding="utf-8") as f:
            return json.load(f)
    return {}

@st.cache_data
def load_mapreduce_report():
    p = config.RESULTS_DIR / "mapreduce_output.txt"
    if p.exists():
        with open(p, "r", encoding="utf-8") as f:
            return f.read()
    return "MapReduce report not yet generated."

@st.cache_data
def load_hive_summary():
    p = config.RESULTS_DIR / "hive_summary.json"
    if p.exists():
        with open(p, "r", encoding="utf-8") as f:
            return json.load(f)
    return {}

# Load Core Data
df_meta = load_metadata()
summary = load_summary()
metrics = load_evaluation_metrics()
mahout_data = load_mahout_results()

# Sidebar Navigation
st.sidebar.image("https://upload.wikimedia.org/wikipedia/commons/thumb/0/07/US-NLM-Logo.svg/320px-US-NLM-Logo.svg.png", width=180)
st.sidebar.title("🫁 Navigation")
section = st.sidebar.radio(
    "Select Module:",
    [
        "1. Overview & Architecture",
        "2. NLM Dataset & Demographics",
        "3. Image Preprocessing & Features",
        "4. Hadoop HDFS & MapReduce",
        "5. Apache Hive Analytics",
        "6. Apache Mahout Clustering",
        "7. Random Forest Classifier",
        "8. Clinical CXR Inference"
    ]
)

st.sidebar.markdown("---")
st.sidebar.caption("Big Data Technologies & Healthcare Analytics Prototype")
st.sidebar.caption("Data Source: U.S. National Library of Medicine (NIH)")

# ====================================================================
# SECTION 1: OVERVIEW & ARCHITECTURE
# ====================================================================
if section == "1. Overview & Architecture":
    st.markdown('<div class="main-header">Healthcare Analytics for Tuberculosis Detection</div>', unsafe_allow_html=True)
    st.markdown('<div class="sub-header">Pattern Analysis & Classification Using Apache Hadoop, Hive, Mahout and Machine Learning</div>', unsafe_allow_html=True)
    
    st.markdown("""
    <div class="clinical-alert">
        <strong>Academic Research Prototype Notice:</strong> This software is engineered for big data research, pattern discovery, and educational demonstrations. It is not approved for autonomous clinical diagnosis without board-certified radiologist verification.
    </div>
    """, unsafe_allow_html=True)
    
    # Top KPI Metrics
    c1, c2, c3, c4 = st.columns(4)
    with c1:
        st.metric("Total CXR Cohort", "138 Patients", "Montgomery County Set")
    with c2:
        st.metric("Normal X-Rays", "80 Images", "57.97%")
    with c3:
        st.metric("TB-Positive X-Rays", "58 Images", "42.03%")
    with c4:
        rf_acc = f"{metrics.get('accuracy', 0)*100:.1f}%" if metrics else "Pending"
        st.metric("Model Test Accuracy", rf_acc, "Random Forest")
        
    st.markdown("### 🏛️ System Architecture & Data Flow")
    st.code("""
    [Official NLM Dataset]
            │
            ▼
    [Image Preprocessing Pipeline] ──> (224x224 Grayscale, Min-Max Normalization)
            │
            ▼
    [Feature Engineering] ───────────> (HOG Descriptors + First/Second Order Texture Stats)
            │
            ├─────────────────────────────────────────┐
            ▼                                         ▼
    [Hadoop HDFS Storage]                     [Numerical Vectors]
            │                                         │
            ├───────────────┬───────────────┐         ├──────────────────────┐
            ▼               ▼               ▼         ▼                      ▼
    [Hadoop MapReduce] [Apache Hive] [Data Warehouse] [Apache Mahout K-Means] [Random Forest ML]
    (Demographics &   (Cohort SQL    (Query Outputs)   (Unsupervised Pattern   (Supervised Clinical
     Case Counts)      Analytics)                       Clustering)             Detection)
            │               │               │         │                      │
            └───────────────┴───────────────┴─────────┴──────────────────────┘
                                            │
                                            ▼
                        [Streamlit Healthcare Intelligence Dashboard]
    """, language="text")
    
    st.markdown("### 📋 Big Data & Machine Learning Stack")
    stack_df = pd.DataFrame([
        {"Tier": "Data Storage", "Component": "Apache Hadoop HDFS", "Role": "Distributed storage abstraction, block allocation (128MB), replication=3"},
        {"Tier": "Batch Computation", "Component": "Hadoop MapReduce", "Role": "Python Streaming Mapper & Reducer for demographic and case aggregation"},
        {"Tier": "Data Warehousing", "Component": "Apache Hive", "Role": "HiveQL schema tables (patient, image, audit) and multi-table analytical joins"},
        {"Tier": "Unsupervised ML", "Component": "Apache Mahout", "Role": "Vector-based K-Means clustering (Euclidean distance, K=2) pattern discovery"},
        {"Tier": "Supervised ML", "Component": "Scikit-Learn Random Forest", "Role": "Stratified cross-validated ensemble classifier on HOG feature vectors"},
        {"Tier": "Visualization", "Component": "Streamlit & Plotly", "Role": "Interactive clinical portal with live CXR inference and metric reports"}
    ])
    st.table(stack_df)

# ====================================================================
# SECTION 2: NLM DATASET & DEMOGRAPHICS
# ====================================================================
elif section == "2. NLM Dataset & Demographics":
    st.markdown('<div class="main-header">Montgomery County Chest X-ray Dataset</div>', unsafe_allow_html=True)
    st.markdown('<div class="sub-header">Official Ground Truth Demographics, Radiologist Readings & Mask Coverage</div>', unsafe_allow_html=True)
    
    if not df_meta.empty:
        c1, c2 = st.columns([1, 1])
        with c1:
            # Distribution Donut
            fig_pie = px.pie(
                df_meta, names="diagnosis", title="Disease Distribution in Cohort (N=138)",
                color="diagnosis",
                color_discrete_map={"Normal": "#10B981", "TB_Positive": "#EF4444"},
                hole=0.45
            )
            st.plotly_chart(fig_pie, use_container_width=True)
            
        with c2:
            # Gender Breakdown
            gender_counts = df_meta[df_meta["patient_gender"].isin(["Male", "Female"])]
            fig_bar = px.histogram(
                gender_counts, x="diagnosis", color="patient_gender", barmode="group",
                title="Gender Stratification by Diagnosis",
                color_discrete_map={"Male": "#3B82F6", "Female": "#EC4899"}
            )
            st.plotly_chart(fig_bar, use_container_width=True)
            
        # Age Distribution
        age_sub = df_meta[df_meta["patient_age"] > 0]
        if not age_sub.empty:
            fig_age = px.box(
                age_sub, x="diagnosis", y="patient_age", color="diagnosis",
                points="all", title="Patient Age Distribution by Diagnosis",
                color_discrete_map={"Normal": "#10B981", "TB_Positive": "#EF4444"}
            )
            st.plotly_chart(fig_age, use_container_width=True)
            
        st.markdown("### 🗃️ Ground-Truth Metadata Explorer")
        st.dataframe(df_meta, use_container_width=True)
    else:
        st.warning("Dataset metadata not yet generated. Please complete Phase 2.")

# ====================================================================
# SECTION 3: PREPROCESSING & FEATURES
# ====================================================================
elif section == "3. Image Preprocessing & Features":
    st.markdown('<div class="main-header">Image Preprocessing & Feature Engineering</div>', unsafe_allow_html=True)
    st.markdown('<div class="sub-header">Grayscale Conversion, 224x224 Lanczos Resizing, Min-Max Normalization & HOG Descriptors</div>', unsafe_allow_html=True)
    
    audit_file = config.PROCESSED_DIR / "preprocessing_audit.csv"
    if audit_file.exists():
        df_audit = pd.read_csv(audit_file)
        
        c1, c2, c3 = st.columns(3)
        with c1:
            st.metric("Total Images Processed", len(df_audit))
        with c2:
            st.metric("Standard Resolution", "224 x 224 px")
        with c3:
            st.metric("Normalization Range", "[0.00, 1.00]")
            
        # Display sample comparison
        st.markdown("### 🖼️ CXR Processing & Mask Visualizer")
        images = sorted(list(config.IMAGES_DIR.glob("*.png")))
        if images:
            selected_img_name = st.selectbox("Select CXR Sample:", [img.name for img in images[:30]])
            sel_path = config.IMAGES_DIR / selected_img_name
            stem = sel_path.stem
            
            c_raw, c_proc, c_mask = st.columns(3)
            with c_raw:
                st.caption(f"Raw CXR: {selected_img_name}")
                st.image(str(sel_path), use_container_width=True)
                
            with c_proc:
                thumb_path = config.PROCESSED_DIR / f"{stem}_thumb.png"
                if thumb_path.exists():
                    st.caption("Normalized (224x224 Grayscale)")
                    st.image(str(thumb_path), use_container_width=True)
                    
            with c_mask:
                left_m = config.MASKS_DIR / "leftMask" / selected_img_name
                right_m = config.MASKS_DIR / "rightMask" / selected_img_name
                if left_m.exists() and right_m.exists():
                    # Combine left and right mask for visualization
                    lm = Image.open(left_m).convert("L")
                    rm = Image.open(right_m).convert("L")
                    combined_mask = Image.fromarray(np.maximum(np.array(lm), np.array(rm)))
                    st.caption("Manual Lung Segmentation Mask")
                    st.image(combined_mask, use_container_width=True)
                    
        st.markdown("### 📊 Intensity Distribution Audit")
        fig_int = px.histogram(
            df_audit, x="mean_intensity", nbins=25, title="Mean Pixel Intensity Distribution Across Dataset",
            color_discrete_sequence=["#6366F1"]
        )
        st.plotly_chart(fig_int, use_container_width=True)
    else:
        st.warning("Preprocessing audit not found. Run preprocessing module.")

# ====================================================================
# SECTION 4: HADOOP HDFS & MAPREDUCE
# ====================================================================
elif section == "4. Hadoop HDFS & MapReduce":
    st.markdown('<div class="main-header">Apache Hadoop HDFS & MapReduce</div>', unsafe_allow_html=True)
    st.markdown('<div class="sub-header">Distributed Storage Namespace & Python Hadoop Streaming Pipeline</div>', unsafe_allow_html=True)
    
    t1, t2 = st.tabs(["Hadoop MapReduce Execution", "HDFS Namespace & Storage"])
    
    with t1:
        st.markdown("### ⚙️ MapReduce Job Execution Output")
        report_text = load_mapreduce_report()
        st.code(report_text, language="text")
        
        st.markdown("### 💻 Hadoop Streaming Code Architecture")
        c_map, c_red = st.columns(2)
        with c_map:
            st.markdown("**Mapper Implementation (`hadoop/mapper.py`)**")
            with open(config.HADOOP_DIR / "mapper.py", "r", encoding="utf-8") as f:
                st.code(f.read(), language="python")
        with c_red:
            st.markdown("**Reducer Implementation (`hadoop/reducer.py`)**")
            with open(config.HADOOP_DIR / "reducer.py", "r", encoding="utf-8") as f:
                st.code(f.read(), language="python")
                
    with t2:
        st.markdown("### 🗄️ HDFS Namespace & Directory Structure")
        st.code("""
        /healthcare/
        ├── raw/
        │   ├── images/       (138 Raw CXR Images in PNG format, rep=3, block=128MB)
        │   ├── masks/        (Left and Right Manual Segmentation Ground Truth)
        │   └── clinical/     (Radiologist Reports in TXT format)
        ├── processed/        (Normalized 224x224 numpy arrays)
        ├── features/         (HOG & Statistical Feature Vectors)
        └── metadata/         (Patient Metadata and Diagnostic CSV)
        """, language="text")
        
        st.markdown("### 🚀 Cluster Deployment Command")
        st.code("""
        hadoop jar $HADOOP_HOME/share/hadoop/tools/lib/hadoop-streaming-*.jar \\
          -D mapreduce.job.name='Healthcare_TB_Metadata_Analysis' \\
          -input /healthcare/data/metadata.csv \\
          -output /healthcare/output/mapreduce_analytics \\
          -mapper 'python mapper.py' \\
          -reducer 'python reducer.py' \\
          -file mapper.py \\
          -file reducer.py
        """, language="bash")

# ====================================================================
# SECTION 5: APACHE HIVE ANALYTICS
# ====================================================================
elif section == "5. Apache Hive Analytics":
    st.markdown('<div class="main-header">Apache Hive Data Warehouse Analytics</div>', unsafe_allow_html=True)
    st.markdown('<div class="sub-header">HiveQL Schema & Analytical Multi-Table Relational Queries</div>', unsafe_allow_html=True)
    
    hive_data = load_hive_summary()
    
    if hive_data:
        st.markdown("### 📊 Hive Analytical Query Results")
        
        for q_key, rows in hive_data.items():
            st.markdown(f"**{q_key.upper().replace('_', ' ')}**")
            df_q = pd.DataFrame(rows)
            st.dataframe(df_q, use_container_width=True)
    else:
        st.info("Hive query results will appear here once Module 5 runs.")
        
    st.markdown("### 📜 HiveQL Schema Definition (`hive/healthcare_schema.sql`)")
    with open(config.HIVE_DIR / "healthcare_schema.sql", "r", encoding="utf-8") as f:
        st.code(f.read(), language="sql")

# ====================================================================
# SECTION 6: APACHE MAHOUT CLUSTERING
# ====================================================================
elif section == "6. Apache Mahout Clustering":
    st.markdown('<div class="main-header">Apache Mahout K-Means Clustering</div>', unsafe_allow_html=True)
    st.markdown('<div class="sub-header">Unsupervised Vector Pattern Discovery on Numerical CXR Descriptors</div>', unsafe_allow_html=True)
    
    if mahout_data:
        c1, c2, c3 = st.columns(3)
        with c1:
            st.metric("Clusters (K)", mahout_data.get("k_clusters", 2))
        with c2:
            st.metric("Silhouette Coefficient", f"{mahout_data.get('silhouette_score', 0):.4f}")
        with c3:
            st.metric("Inertia (SSE)", f"{mahout_data.get('inertia', 0):.1f}")
            
        st.markdown("### 🎯 Cluster Purity & Class Separation Analysis")
        purity_records = []
        for cname, pstats in mahout_data.get("purity_analysis", {}).items():
            purity_records.append({
                "Cluster": cname,
                "Total Samples": pstats["total_samples"],
                "Normal Samples": pstats["normal_samples"],
                "TB-Positive Samples": pstats["tb_positive_samples"],
                "Dominant Diagnosis": pstats["dominant_pattern"],
                "Cluster Purity (%)": pstats["purity_percentage"]
            })
        st.table(pd.DataFrame(purity_records))
        
        # Patient cluster visualization
        cluster_csv = config.RESULTS_DIR / "mahout_patient_clusters.csv"
        feat_csv = config.FEATURES_DIR / "features_summary.csv"
        if cluster_csv.exists() and feat_csv.exists():
            df_c = pd.read_csv(cluster_csv)
            df_f = pd.read_csv(feat_csv)
            merged = pd.merge(df_c, df_f[["patient_id", "stat_mean", "stat_std", "stat_entropy"]], on="patient_id")
            
            fig_cl = px.scatter(
                merged, x="stat_mean", y="stat_entropy", color="assigned_cluster",
                symbol="ground_truth",
                title="Mahout Feature Space Clustering (Mean Intensity vs Entropy)",
                color_continuous_scale="Viridis",
                hover_data=["patient_id", "ground_truth"]
            )
            st.plotly_chart(fig_cl, use_container_width=True)
    else:
        st.warning("Mahout clustering results not yet generated. Please execute Module 6.")

# ====================================================================
# SECTION 7: RANDOM FOREST CLASSIFIER
# ====================================================================
elif section == "7. Random Forest Classifier":
    st.markdown('<div class="main-header">Random Forest Supervised Classification</div>', unsafe_allow_html=True)
    st.markdown('<div class="sub-header">Stratified Cross-Validation & Clinical Performance Metrics</div>', unsafe_allow_html=True)
    
    if metrics:
        c1, c2, c3, c4 = st.columns(4)
        with c1:
            st.metric("Hold-Out Accuracy", f"{metrics.get('accuracy', 0)*100:.2f}%")
        with c2:
            st.metric("Sensitivity (Recall)", f"{metrics.get('recall_sensitivity', 0)*100:.2f}%")
        with c3:
            st.metric("Specificity", f"{metrics.get('specificity', 0)*100:.2f}%")
        with c4:
            st.metric("ROC-AUC Score", f"{metrics.get('roc_auc', 0):.4f}")
            
        c_left, c_right = st.columns(2)
        with c_left:
            st.markdown("### 🧩 Confusion Matrix")
            cm_img_path = config.RESULTS_DIR / "confusion_matrix.png"
            if cm_img_path.exists():
                st.image(str(cm_img_path), use_container_width=True)
            else:
                cm = metrics.get("confusion_matrix", {})
                st.json(cm)
                
        with c_right:
            st.markdown("### 📈 ROC Curve")
            roc_img_path = config.RESULTS_DIR / "roc_curve.png"
            if roc_img_path.exists():
                st.image(str(roc_img_path), use_container_width=True)
                
        st.markdown("### 📄 Classification Report")
        cr_path = config.RESULTS_DIR / "classification_report.txt"
        if cr_path.exists():
            with open(cr_path, "r", encoding="utf-8") as f:
                st.code(f.read(), language="text")
    else:
        st.warning("Model evaluation metrics not yet available. Train model first.")

# ====================================================================
# SECTION 8: CLINICAL CXR INFERENCE
# ====================================================================
elif section == "8. Clinical CXR Inference":
    st.markdown('<div class="main-header">Clinical CXR Real-Time Inference</div>', unsafe_allow_html=True)
    st.markdown('<div class="sub-header">Select or Upload a Chest X-ray to Execute Diagnostic Classification</div>', unsafe_allow_html=True)
    
    images = sorted(list(config.IMAGES_DIR.glob("*.png")))
    if images:
        selected_option = st.selectbox("Select Test Patient from Montgomery Cohort:", [img.name for img in images])
        test_file_path = config.IMAGES_DIR / selected_option
        
        c_img, c_pred = st.columns([1, 1])
        with c_img:
            st.image(str(test_file_path), caption=f"Selected CXR: {selected_option}", use_container_width=True)
            
            # Radiologist report if available
            stem = test_file_path.stem
            txt_p = config.CLINICAL_DIR / f"{stem}.txt"
            if txt_p.exists():
                with open(txt_p, "r", encoding="utf-8", errors="ignore") as tf:
                    reading = tf.read().strip()
                st.info(f"**NLM Radiologist Finding:**\n\n{reading}")
                
        with c_pred:
            if st.button("🔬 Run Diagnostic Inference", type="primary"):
                try:
                    res = predict_cxr(test_file_path)
                    
                    is_tb = res["predicted_diagnosis"] == "TB_Positive"
                    badge_color = "#EF4444" if is_tb else "#10B981"
                    status_text = "TUBERCULOSIS DETECTED" if is_tb else "NORMAL / NO TB DETECTED"
                    
                    st.markdown(f"""
                    <div style="background-color: {badge_color}; color: white; padding: 20px; border-radius: 8px; text-align: center; margin-bottom: 20px;">
                        <h2 style="color: white; margin: 0;">{status_text}</h2>
                        <h4 style="color: white; margin: 5px 0 0 0;">Confidence: {res['confidence_score']*100:.2f}%</h4>
                    </div>
                    """, unsafe_allow_html=True)
                    
                    # Probability Bars
                    prob_df = pd.DataFrame({
                        "Diagnosis": ["Normal", "TB_Positive"],
                        "Probability": [res["probabilities"]["Normal"], res["probabilities"]["TB_Positive"]]
                    })
                    fig_prob = px.bar(
                        prob_df, x="Diagnosis", y="Probability", color="Diagnosis",
                        color_discrete_map={"Normal": "#10B981", "TB_Positive": "#EF4444"},
                        range_y=[0.0, 1.0], title="Model Prediction Probabilities"
                    )
                    st.plotly_chart(fig_prob, use_container_width=True)
                    
                except Exception as e:
                    st.error(f"Inference error: {e}")
    else:
        st.info("No CXR images available. Complete dataset acquisition in Phase 2.")
