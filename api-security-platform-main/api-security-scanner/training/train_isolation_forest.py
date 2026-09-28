import os
import sys
from pathlib import Path
import pandas as pd
import numpy as np
import joblib
from sklearn.ensemble import IsolationForest
from sklearn.metrics import classification_report, precision_score, recall_score, f1_score

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from config.logging_config import logger
from config.settings import DATASETS_DIR, MODELS_DIR

FEATURE_COLUMNS = [
    "encoded_method",
    "path_depth",
    "url_length",
    "query_param_count",
    "query_string_length",
    "payload_length",
    "payload_entropy",
    "special_char_count",
    "header_count",
    "auth_header_present",
    "status_code",
    "response_size",
    "keyword_risk_score",
    "param_name_risk",
    "url_encoded_ratio",
    "payload_digit_ratio",
    "has_sql_structure"
]

def train_isolation_forest():
    """
    Trains an Isolation Forest machine learning model on normal/anomalous HTTP request feature vectors.
    Evaluates metrics (Precision, Recall, F1-Score) and exports the model to models/isolation_forest.pkl.
    """
    processed_dir = Path(DATASETS_DIR) / "processed"
    train_path = processed_dir / "train.csv"
    test_path = processed_dir / "test.csv"
    model_output_path = Path(MODELS_DIR) / "isolation_forest.pkl"

    if not train_path.exists() or not test_path.exists():
        logger.error(f"Dataset CSV files not found at {processed_dir}. Run training/prepare_dataset.py first.")
        print(f"[!] Error: Missing dataset files in {processed_dir}. Please run prepare_dataset.py first.")
        return

    logger.info("Loading processed training and testing dataset CSVs...")
    train_df = pd.read_csv(train_path).fillna(0.0)
    test_df = pd.read_csv(test_path).fillna(0.0)

    # Methodologically correct unsupervised anomaly detection:
    # Train Isolation Forest baseline strictly on normal HTTP traffic (label == 0)
    normal_train_df = train_df[train_df["label"] == 0]
    X_train_normal = normal_train_df[FEATURE_COLUMNS].values

    X_test = test_df[FEATURE_COLUMNS].values
    y_test = test_df["label"].values

    contamination_rate = 0.05
    logger.info(f"Training IsolationForest on {len(X_train_normal)} normal baseline samples (contamination={contamination_rate})...")
    
    model = IsolationForest(
        n_estimators=100,
        max_samples="auto",
        contamination=contamination_rate,
        random_state=42,
        n_jobs=-1
    )

    model.fit(X_train_normal)

    # Predictions: Isolation Forest returns -1 for anomalies, 1 for normal
    raw_preds = model.predict(X_test)
    y_pred = np.where(raw_preds == -1, 1, 0) # Convert -1 -> 1 (Anomaly), 1 -> 0 (Normal)

    precision = precision_score(y_test, y_pred, zero_division=0)
    recall = recall_score(y_test, y_pred, zero_division=0)
    f1 = f1_score(y_test, y_pred, zero_division=0)

    print("\n" + "="*50)
    print(" ISOLATION FOREST MODEL EVALUATION METRICS")
    print("="*50)
    print(f" Precision : {precision*100:.2f}%")
    print(f" Recall    : {recall*100:.2f}%")
    print(f" F1-Score  : {f1*100:.2f}%")
    print("\n Detailed Classification Report:\n")
    print(classification_report(y_test, y_pred, target_names=["Normal (0)", "Anomaly (1)"]))
    print("="*50)

    # Save model binary file
    os.makedirs(MODELS_DIR, exist_ok=True)
    joblib.dump(model, model_output_path)
    logger.info(f"Saved trained Isolation Forest model to {model_output_path}")
    print(f"\n[+] Model saved successfully to: {model_output_path}\n")

if __name__ == "__main__":
    train_isolation_forest()
