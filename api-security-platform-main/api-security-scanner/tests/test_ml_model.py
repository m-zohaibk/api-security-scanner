import pytest
import os
import pandas as pd
from pathlib import Path
from detection.ml_model import MLAnomalyDetector
from config.settings import MODELS_DIR

def test_ml_anomaly_detector_inference():
    model_path = Path(MODELS_DIR) / "isolation_forest.pkl"
    detector = MLAnomalyDetector(model_path=str(model_path))
    
    sample_features = {
        "encoded_method": 2,
        "path_depth": 3,
        "url_length": 50,
        "query_param_count": 1,
        "query_string_length": 10,
        "payload_length": 20,
        "payload_entropy": 2.5,
        "special_char_count": 2,
        "header_count": 5,
        "auth_header_present": 1,
        "status_code": 200,
        "response_size": 500
    }
    
    res = detector.predict(sample_features)
    assert "anomaly_score" in res
    assert "points" in res
    assert 0.0 <= res["anomaly_score"] <= 1.0
    assert 0.0 <= res["points"] <= 40.0
