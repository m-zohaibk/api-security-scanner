import pytest
from core.response_parser import ResponseParser
from detection.ml_model import MLAnomalyDetector


def test_feature_vector_length_matches_feature_keys():
    parser = ResponseParser()
    sample = {
        "method": "POST",
        "url": "http://example.com/api/v1/users?id=10",
        "payload": "user_payload!",
        "status_code": 200,
        "response_size": 256,
        "response_body": "{\"token\": \"secret_token\"}",
        "request_headers": {"Authorization": "Bearer abc.def.ghi"}
    }
    extracted = parser.extract_features(sample)

    assert "feature_vector" in extracted
    # Ensure parser feature vector aligns with detectors' FEATURE_KEYS length
    assert len(extracted["feature_vector"]) == len(MLAnomalyDetector.FEATURE_KEYS)


def test_auth_header_bearer_value_detected():
    parser = ResponseParser()
    sample = {
        "method": "GET",
        "url": "http://example.com/health",
        "payload": "",
        "status_code": 200,
        "response_size": 32,
        "response_body": "OK",
        "request_headers": {"Some-Header": "value", "Authorization": "Bearer token123"}
    }
    extracted = parser.extract_features(sample)
    assert extracted["auth_header_present"] == 1
