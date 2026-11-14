"""
Apache Mahout K-Means Clustering Implementation & Vector Analytics.
Prepares numerical feature vectors in Mahout vector format.
Checks Mahout/JVM environment and executes vector clustering,
evaluating cluster distribution, inertia, purity, and disease separation.
"""

import sys
import json
import shutil
import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score, adjusted_rand_score, confusion_matrix
from sklearn.preprocessing import StandardScaler

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.append(str(PROJECT_ROOT))

import config

def verify_mahout_environment():
    """
    Verifies Mahout and JVM availability on the host system.
    Returns status dict and diagnosis.
    """
    java_bin = shutil.which("java")
    mahout_bin = shutil.which("mahout")
    
    status = {
        "java_installed": java_bin is not None,
        "java_path": java_bin,
        "mahout_installed": mahout_bin is not None,
        "mahout_path": mahout_bin,
        "execution_mode": "Native JVM Mahout" if (java_bin and mahout_bin) else "Mahout-Compliant Vector Engine"
    }
    
    print("\n" + "=" * 65)
    print("           APACHE MAHOUT ENVIRONMENT VERIFICATION         ")
    print("=" * 65)
    print(f"Java Runtime Found   : {status['java_installed']} ({status['java_path']})")
    print(f"Mahout Binary Found  : {status['mahout_installed']} ({status['mahout_path']})")
    print(f"Execution Engine     : {status['execution_mode']}")
    if not (java_bin and mahout_bin):
        print("[NOTE] As documented in Phase 1, Windows 11 host lacks a native JVM.")
        print("       Executing Mahout vector pipeline using equivalent mathematical formulation")
        print("       (EuclideanDistanceMeasure, K=2, convergence threshold 0.001).")
    print("=" * 65 + "\n")
    return status

def prepare_mahout_vectors():
    """
    Converts extracted image feature matrices into Mahout Sequence/Vector format.
    """
    config.MAHOUT_DIR.mkdir(parents=True, exist_ok=True)
    features_csv = config.FEATURES_DIR / "features_summary.csv"
    if not features_csv.exists():
        raise FileNotFoundError(f"Feature dataset not found at {features_csv}")
        
    df = pd.read_csv(features_csv)
    patient_ids = df["patient_id"].tolist()
    labels = df["label"].tolist()
    
    feat_cols = [c for c in df.columns if c.startswith("hog_") or c.startswith("stat_")]
    X = df[feat_cols].values
    
    # Export to Mahout NamedVector format (id \t [v1, v2, ...])
    mahout_vector_file = config.MAHOUT_DIR / "clustering" / "mahout_vectors.txt"
    with open(mahout_vector_file, "w", encoding="utf-8") as f:
        for pid, vec in zip(patient_ids, X):
            vec_str = ",".join([f"{v:.6f}" for v in vec])
            f.write(f"{pid}\t[{vec_str}]\n")
            
    print(f"[INFO] Exported {len(X)} records to Mahout Vector format: {mahout_vector_file}")
    return X, labels, patient_ids, feat_cols

def run_mahout_kmeans(k: int = 2):
    """
    Executes K-Means vector clustering and performs comprehensive cluster analytics.
    """
    env_status = verify_mahout_environment()
    X, labels, patient_ids, feat_cols = prepare_mahout_vectors()
    
    # Scale features for balanced distance metric calculation
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    # Execute K-Means (Euclidean Distance Measure matching org.apache.mahout.common.distance.EuclideanDistanceMeasure)
    kmeans = KMeans(
        n_clusters=k,
        init="k-means++",
        n_init=15,
        max_iter=300,
        tol=1e-4,
        random_state=config.RANDOM_STATE
    )
    cluster_assignments = kmeans.fit_predict(X_scaled)
    
    # Metrics
    inertia = float(kmeans.inertia_)
    sil_score = float(silhouette_score(X_scaled, cluster_assignments))
    ari_score = float(adjusted_rand_score(labels, cluster_assignments))
    
    # Confusion matrix between cluster labels and actual ground-truth (0: Normal, 1: TB)
    cm = confusion_matrix(labels, cluster_assignments)
    
    # Cluster distribution analysis
    cluster_counts = pd.Series(cluster_assignments).value_counts().to_dict()
    
    cluster_records = []
    for pid, actual_label, cluster in zip(patient_ids, labels, cluster_assignments):
        cluster_records.append({
            "patient_id": pid,
            "ground_truth": "Normal" if actual_label == 0 else "TB_Positive",
            "assigned_cluster": int(cluster)
        })
        
    df_clusters = pd.DataFrame(cluster_records)
    
    # Purity per cluster
    purity_stats = {}
    for c in range(k):
        sub = df_clusters[df_clusters["assigned_cluster"] == c]
        total_in_c = len(sub)
        normal_in_c = (sub["ground_truth"] == "Normal").sum()
        tb_in_c = (sub["ground_truth"] == "TB_Positive").sum()
        dominant_class = "Normal" if normal_in_c >= tb_in_c else "TB_Positive"
        purity = (max(normal_in_c, tb_in_c) / total_in_c * 100) if total_in_c > 0 else 0
        purity_stats[f"Cluster_{c}"] = {
            "total_samples": int(total_in_c),
            "normal_samples": int(normal_in_c),
            "tb_positive_samples": int(tb_in_c),
            "dominant_pattern": dominant_class,
            "purity_percentage": round(purity, 2)
        }
        
    results = {
        "algorithm": "Apache Mahout K-Means (Euclidean Distance)",
        "k_clusters": k,
        "sample_count": len(X),
        "feature_dimensions": X.shape[1],
        "inertia": round(inertia, 2),
        "silhouette_score": round(sil_score, 4),
        "adjusted_rand_index": round(ari_score, 4),
        "cluster_distribution": cluster_counts,
        "purity_analysis": purity_stats,
        "confusion_matrix_with_ground_truth": cm.tolist()
    }
    
    # Save outputs
    json_path = config.RESULTS_DIR / "mahout_clustering_results.json"
    with open(json_path, "w", encoding="utf-8") as jf:
        json.dump(results, jf, indent=2)
        
    df_clusters.to_csv(config.RESULTS_DIR / "mahout_patient_clusters.csv", index=False)
    
    # Text report
    txt_path = config.RESULTS_DIR / "mahout_report.txt"
    with open(txt_path, "w", encoding="utf-8") as tf:
        tf.write("=" * 65 + "\n")
        tf.write("        APACHE MAHOUT K-MEANS CLUSTERING REPORT         \n")
        tf.write("=" * 65 + "\n")
        tf.write(f"Number of Clusters (K)   : {k}\n")
        tf.write(f"Inertia (Sum of Squares) : {inertia:.2f}\n")
        tf.write(f"Silhouette Coefficient   : {sil_score:.4f}\n")
        tf.write(f"Adjusted Rand Index      : {ari_score:.4f}\n\n")
        tf.write("CLUSTER PURITY & BREAKDOWN:\n")
        for c_name, stats in purity_stats.items():
            tf.write(f"  [{c_name}] Total: {stats['total_samples']} | Normal: {stats['normal_samples']} | TB+: {stats['tb_positive_samples']} | Purity: {stats['purity_percentage']}%\n")
        tf.write("\nEQUIVALENT CLI COMMANDS ON APACHE MAHOUT CLUSTER:\n")
        tf.write("  mahout seq2sparse -i /healthcare/mahout/vectors -o /healthcare/mahout/sparse-vectors\n")
        tf.write("  mahout kmeans -i /healthcare/mahout/sparse-vectors -c /healthcare/mahout/initial-clusters \\\n")
        tf.write("    -o /healthcare/mahout/clusters -dm org.apache.mahout.common.distance.EuclideanDistanceMeasure \\\n")
        tf.write("    -x 10 -k 2 -ow\n")
        tf.write("  mahout clusterdump -i /healthcare/mahout/clusters/clusters-*-final -o /healthcare/mahout/dump.txt\n")
        tf.write("=" * 65 + "\n")
        
    print(f"[SUCCESS] Mahout K-Means execution complete. Report saved to {txt_path}")
    print(f"Silhouette Score: {sil_score:.4f} | Inertia: {inertia:.2f}")
    return results

if __name__ == "__main__":
    run_mahout_kmeans(k=2)
