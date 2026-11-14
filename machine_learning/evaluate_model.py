"""
Model Evaluation and Diagnostic Inspection Module.
Loads trained Random Forest model and presents full evaluation metrics.
"""

import sys
import json
from pathlib import Path
import joblib
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.append(str(PROJECT_ROOT))

import config

def display_evaluation_summary():
    metrics_path = config.RESULTS_DIR / "model_evaluation_metrics.json"
    if not metrics_path.exists():
        print("[ERROR] Evaluation metrics not found. Train model first.")
        return
        
    with open(metrics_path, "r", encoding="utf-8") as f:
        metrics = json.load(f)
        
    print("\n" + "=" * 65)
    print("      HEALTHCARE TUBERCULOSIS RANDOM FOREST AUDIT SUMMARY    ")
    print("=" * 65)
    print(f"Algorithm            : {metrics['model_name']}")
    print(f"5-Fold CV Accuracy   : {metrics['cv_5fold_accuracy_mean']*100:.2f}% (+/- {metrics['cv_5fold_accuracy_std']*100:.2f}%)")
    print(f"Hold-out Test Size   : {metrics['test_sample_size']} CXR images")
    print(f"Test Accuracy        : {metrics['accuracy']*100:.2f}%")
    print(f"Sensitivity / Recall : {metrics['recall_sensitivity']*100:.2f}%")
    print(f"Specificity          : {metrics['specificity']*100:.2f}%")
    print(f"Precision            : {metrics['precision']*100:.2f}%")
    print(f"F1-Score             : {metrics['f1_score']:.4f}")
    print(f"ROC-AUC              : {metrics['roc_auc']:.4f}")
    
    cm = metrics["confusion_matrix"]
    print("\nConfusion Matrix Breakdown:")
    print(f"  True Negatives (Normal correctly identified) : {cm['true_negatives']}")
    print(f"  False Positives (Normal misdiagnosed as TB)  : {cm['false_positives']}")
    print(f"  False Negatives (TB missed by model)        : {cm['false_negatives']}")
    print(f"  True Positives (TB correctly diagnosed)     : {cm['true_positives']}")
    print("=" * 65 + "\n")

if __name__ == "__main__":
    display_evaluation_summary()
