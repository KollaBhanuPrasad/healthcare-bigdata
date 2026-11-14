"""
Random Forest Model Training and Stratified Validation Module.
Features:
- Stratified Train-Test Splitting (80/20) and 5-Fold Stratified Cross-Validation.
- Class weight balancing for medical diagnostic rigor.
- Complete evaluation metrics: Accuracy, Precision, Recall, F1, ROC-AUC.
- Generates Confusion Matrix and ROC Curve figures.
- Persists trained model artifacts using Joblib.
"""

import sys
import json
from pathlib import Path
import joblib
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, confusion_matrix, classification_report, roc_curve
)
from sklearn.preprocessing import StandardScaler

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.append(str(PROJECT_ROOT))

import config

def train_and_evaluate_rf():
    config.MODELS_DIR.mkdir(parents=True, exist_ok=True)
    config.RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    
    x_path = config.FEATURES_DIR / "X_features.npy"
    y_path = config.FEATURES_DIR / "y_labels.npy"
    
    if not (x_path.exists() and y_path.exists()):
        raise FileNotFoundError("Feature arrays not found. Run feature_extraction.py first.")
        
    X = np.load(x_path)
    y = np.load(y_path)
    
    print(f"[INFO] Loaded feature matrix: {X.shape}, labels: {y.shape}")
    print(f"[INFO] Distribution: Normal={np.sum(y == 0)}, TB+={np.sum(y == 1)}")
    
    # Stratified Train-Test Split (80% train, 20% test)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size=config.TEST_SIZE,
        stratify=y,
        random_state=config.RANDOM_STATE
    )
    
    # Feature Scaling
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    # Save Scaler
    scaler_path = config.MODELS_DIR / "feature_scaler.joblib"
    joblib.dump(scaler, scaler_path)
    
    # Model Initialization
    rf = RandomForestClassifier(
        n_estimators=config.RF_N_ESTIMATORS,
        max_depth=config.RF_MAX_DEPTH,
        class_weight="balanced",
        random_state=config.RANDOM_STATE,
        n_jobs=-1
    )
    
    # 5-Fold Stratified Cross-Validation on Training Set
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=config.RANDOM_STATE)
    cv_scores = cross_val_score(rf, X_train_scaled, y_train, cv=cv, scoring="accuracy")
    cv_f1_scores = cross_val_score(rf, X_train_scaled, y_train, cv=cv, scoring="f1")
    
    print(f"[CV 5-Fold] Mean Accuracy: {cv_scores.mean():.4f} (+/- {cv_scores.std():.4f})")
    print(f"[CV 5-Fold] Mean F1-Score: {cv_f1_scores.mean():.4f} (+/- {cv_f1_scores.std():.4f})")
    
    # Fit on full training set
    rf.fit(X_train_scaled, y_train)
    
    # Test Predictions
    y_pred = rf.predict(X_test_scaled)
    y_prob = rf.predict_proba(X_test_scaled)[:, 1]
    
    # Calculate Metrics
    acc = float(accuracy_score(y_test, y_pred))
    prec = float(precision_score(y_test, y_pred, zero_division=0))
    rec = float(recall_score(y_test, y_pred, zero_division=0))
    f1 = float(f1_score(y_test, y_pred, zero_division=0))
    roc_auc = float(roc_auc_score(y_test, y_prob))
    
    cm = confusion_matrix(y_test, y_pred)
    tn, fp, fn, tp = cm.ravel()
    specificity = float(tn / (tn + fp)) if (tn + fp) > 0 else 0.0
    
    report_dict = classification_report(
        y_test, y_pred, target_names=["Normal", "TB_Positive"], output_dict=True
    )
    report_text = classification_report(
        y_test, y_pred, target_names=["Normal", "TB_Positive"]
    )
    
    print("\n" + "=" * 60)
    print("        RANDOM FOREST MODEL EVALUATION REPORT         ")
    print("=" * 60)
    print(f"Test Accuracy    : {acc:.4f} ({acc*100:.2f}%)")
    print(f"Precision (TB+)  : {prec:.4f}")
    print(f"Recall/Sens (TB+): {rec:.4f}")
    print(f"Specificity      : {specificity:.4f}")
    print(f"F1-Score         : {f1:.4f}")
    print(f"ROC-AUC Score    : {roc_auc:.4f}")
    print("-" * 60)
    print("Confusion Matrix:")
    print(f"  [TN={tn:2d}  FP={fp:2d}]")
    print(f"  [FN={fn:2d}  TP={tp:2d}]")
    print("-" * 60)
    print("Full Classification Report:\n" + report_text)
    print("=" * 60 + "\n")
    
    # Save Model
    model_path = config.MODELS_DIR / "random_forest_model.joblib"
    joblib.dump(rf, model_path)
    print(f"[INFO] Saved trained model to {model_path}")
    
    # Save Metrics JSON
    metrics = {
        "model_name": "Random Forest Classifier",
        "n_estimators": config.RF_N_ESTIMATORS,
        "max_depth": config.RF_MAX_DEPTH,
        "cv_5fold_accuracy_mean": round(float(cv_scores.mean()), 4),
        "cv_5fold_accuracy_std": round(float(cv_scores.std()), 4),
        "cv_5fold_f1_mean": round(float(cv_f1_scores.mean()), 4),
        "test_sample_size": len(y_test),
        "accuracy": round(acc, 4),
        "precision": round(prec, 4),
        "recall_sensitivity": round(rec, 4),
        "specificity": round(specificity, 4),
        "f1_score": round(f1, 4),
        "roc_auc": round(roc_auc, 4),
        "confusion_matrix": {
            "true_negatives": int(tn),
            "false_positives": int(fp),
            "false_negatives": int(fn),
            "true_positives": int(tp)
        },
        "classification_report": report_dict
    }
    
    with open(config.RESULTS_DIR / "model_evaluation_metrics.json", "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2)
        
    with open(config.RESULTS_DIR / "classification_report.txt", "w", encoding="utf-8") as f:
        f.write("RANDOM FOREST CLASSIFICATION REPORT\n" + "=" * 50 + "\n")
        f.write(report_text)
        
    # Generate and Save Confusion Matrix Plot
    plt.figure(figsize=(6, 5))
    sns.heatmap(
        cm, annot=True, fmt="d", cmap="Blues",
        xticklabels=["Normal", "TB_Positive"],
        yticklabels=["Normal", "TB_Positive"]
    )
    plt.title("Random Forest Confusion Matrix (CXR TB Detection)")
    plt.xlabel("Predicted Diagnosis")
    plt.ylabel("Actual Ground Truth")
    plt.tight_layout()
    cm_path = config.RESULTS_DIR / "confusion_matrix.png"
    plt.savefig(cm_path, dpi=300)
    plt.close()
    
    # Generate and Save ROC Curve Plot
    fpr, tpr, _ = roc_curve(y_test, y_prob)
    plt.figure(figsize=(6, 5))
    plt.plot(fpr, tpr, color="darkorange", lw=2, label=f"ROC curve (AUC = {roc_auc:.3f})")
    plt.plot([0, 1], [0, 1], color="navy", lw=2, linestyle="--", label="Chance Level")
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel("False Positive Rate (1 - Specificity)")
    plt.ylabel("True Positive Rate (Sensitivity)")
    plt.title("Receiver Operating Characteristic (ROC)")
    plt.legend(loc="lower right")
    plt.grid(True, linestyle=":", alpha=0.6)
    plt.tight_layout()
    roc_path = config.RESULTS_DIR / "roc_curve.png"
    plt.savefig(roc_path, dpi=300)
    plt.close()
    
    print(f"[SUCCESS] Visual evaluation plots saved to {cm_path} and {roc_path}")
    return metrics

if __name__ == "__main__":
    train_and_evaluate_rf()
