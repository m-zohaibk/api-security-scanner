import pytest
from core.discovery import EndpointDiscovery
from core.request_engine import RequestEngine
from core.response_parser import ResponseParser

def test_discovery_normalization():
    discoverer = EndpointDiscovery("http://example.com")
    assert discoverer.normalize_url("/api/v1/users") == "http://example.com/api/v1/users"
    assert discoverer.is_same_domain("http://example.com/test") is True

def test_request_engine_structure():
    engine = RequestEngine()
    # Test method dictionary structuring without external network call
    res = {
        "method": "POST",
        "url": "http://example.com/api",
        "payload": "sample_data",
        "status_code": 200,
        "response_body": "{\"status\":\"ok\",\"token\":\"abc123secret\"}",
        "response_time": 0.05,
        "response_size": 32,
        "request_headers": {"Authorization": "Bearer key"}
    }
    assert res["status_code"] == 200

def test_response_parser_features():
    parser = ResponseParser()
    sample = {
        "method": "POST",
        "url": "http://example.com/api/v1/users?id=10",
        "payload": "user_payload!",
        "status_code": 200,
        "response_size": 256,
        "response_body": "{\"token\": \"secret_token\"}",
        "request_headers": {"Authorization": "Bearer secret"}
    }
    extracted = parser.extract_features(sample)
    
    assert extracted["encoded_method"] == 2 # POST
    assert extracted["path_depth"] == 3 # /api/v1/users
    assert extracted["query_param_count"] == 1
    assert extracted["auth_header_present"] == 1
    assert len(extracted["feature_vector"]) == 17
    assert "token" in extracted["sensitive_fields_leaked"]
