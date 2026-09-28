import pytest
from detection.deep_learning import DeepLearningDetector

def test_deep_learning_detector_inference():
    detector = DeepLearningDetector()
    
    sample_payload = "' OR 1=1 --"
    sample_features = {
        "encoded_method": 2,
        "path_depth": 3,
        "url_length": 60,
        "query_param_count": 2,
        "query_string_length": 30,
        "payload_length": 45,
        "payload_entropy": 5.8,
        "special_char_count": 12,
        "header_count": 6,
        "auth_header_present": 0,
        "status_code": 200,
        "response_size": 1024
    }
    
    res = detector.analyze(sample_payload, sample_features)
    assert "lstm_probability" in res
    assert "lstm_points" in res
    assert "autoencoder_error" in res
    assert "autoencoder_points" in res
    assert 0.0 <= res["lstm_points"] <= 30.0
    assert 0.0 <= res["autoencoder_points"] <= 20.0
