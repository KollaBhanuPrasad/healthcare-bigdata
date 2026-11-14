"""
Inference & Prediction Pipeline for Chest X-ray Images.
Accepts an image path or array, processes it through the standardized pipeline,
and outputs predicted diagnosis (Normal vs TB_Positive) with probabilities.
"""

import sys
from pathlib import Path
import joblib
import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.append(str(PROJECT_ROOT))

import config
from preprocessing.image_preprocessing import preprocess_single_image
from preprocessing.feature_extraction import extract_features_from_array

def load_prediction_pipeline():
    model_path = config.MODELS_DIR / "random_forest_model.joblib"
    scaler_path = config.MODELS_DIR / "feature_scaler.joblib"
    
    if not (model_path.exists() and scaler_path.exists()):
        raise FileNotFoundError("Model or Scaler not found. Train model first.")
        
    model = joblib.load(model_path)
    scaler = joblib.load(scaler_path)
    return model, scaler

def predict_cxr(image_path: Path):
    """
    Runs end-to-end inference on a single CXR image.
    """
    image_path = Path(image_path)
    if not image_path.exists():
        raise FileNotFoundError(f"Image not found at {image_path}")
        
    model, scaler = load_prediction_pipeline()
    
    # 1. Preprocess
    norm_arr = preprocess_single_image(image_path)
    if norm_arr is None:
        raise ValueError(f"Failed to preprocess image {image_path.name}")
        
    # 2. Extract features
    feat_vec = extract_features_from_array(norm_arr).reshape(1, -1)
    
    # 3. Scale features
    feat_scaled = scaler.transform(feat_vec)
    
    # 4. Predict
    pred_code = int(model.predict(feat_scaled)[0])
    probs = model.predict_proba(feat_scaled)[0]
    
    prob_normal = round(float(probs[0]), 4)
    prob_tb = round(float(probs[1]), 4)
    diagnosis = "Normal" if pred_code == 0 else "TB_Positive"
    
    result = {
        "filename": image_path.name,
        "prediction_code": pred_code,
        "predicted_diagnosis": diagnosis,
        "confidence_score": round(float(max(prob_normal, prob_tb)), 4),
        "probabilities": {
            "Normal": prob_normal,
            "TB_Positive": prob_tb
        }
    }
    return result

if __name__ == "__main__":
    if len(sys.argv) > 1:
        test_img = Path(sys.argv[1])
    else:
        # Default test image from images dir if present
        images = list(config.IMAGES_DIR.glob("*.png"))
        test_img = images[0] if images else None
        
    if test_img and test_img.exists():
        res = predict_cxr(test_img)
        print("\n--- CXR INFERENCE RESULT ---")
        print(f"File       : {res['filename']}")
        print(f"Diagnosis  : {res['predicted_diagnosis']}")
        print(f"Confidence : {res['confidence_score']*100:.2f}%")
        print(f"Details    : {res['probabilities']}")
    else:
        print("[INFO] Please provide an image path for prediction.")
