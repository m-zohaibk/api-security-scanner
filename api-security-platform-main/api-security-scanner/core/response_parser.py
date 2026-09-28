import math
import re
from typing import Dict, Any, List
from urllib.parse import urlparse, parse_qs
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from config.logging_config import logger

class ResponseParser:
    """
    Parses request/response telemetry data, extracts 17 numerical features for ML/DL models,
    and checks response bodies for sensitive field leaks.
    """

    METHOD_ENCODING = {
        "GET": 1,
        "POST": 2,
        "PUT": 3,
        "DELETE": 4,
        "PATCH": 5,
        "OPTIONS": 6,
        "HEAD": 7
    }

    SENSITIVE_FIELD_PATTERNS = [
        r"password",
        r"passwd",
        r"secret",
        r"token",
        r"api[_-]?key",
        r"auth[_-]?token",
        r"access[_-]?token",
        r"card[_-]?number",
        r"credit[_-]?card",
        r"ssn",
        r"private[_-]?key"
    ]

    @staticmethod
    def calculate_shannon_entropy(data: str) -> float:
        """Calculates Shannon entropy of string payload."""
        if not data:
            return 0.0
        entropy = 0.0
        length = len(data)
        freq: Dict[str, int] = {}
        for char in data:
            freq[char] = freq.get(char, 0) + 1
        for count in freq.values():
            p = count / length
            entropy -= p * math.log2(p)
        return round(entropy, 4)

    @staticmethod
    def count_special_characters(text: str) -> int:
        """Counts special non-alphanumeric characters in payload string."""
        if not text:
            return 0
        return len(re.findall(r"[^\w\s]", text))

    def extract_features(self, response_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Extracts the 17 numerical features from telemetry dict:
          1.  HTTP method (encoded)
          2.  URL path depth
          3.  URL length
          4.  Query parameter count
          5.  Query string length
          6.  Payload length
          7.  Shannon entropy of payload
          8.  Special character count
          9.  Header count
          10. Auth header present (0 or 1)
          11. Response status code
          12. Response size
          13. Keyword risk score
          14. Parameter-name risk score
          15. URL-encoded character ratio
          16. Payload digit ratio
          17. SQL structure indicator
        """
        method_str = response_data.get("method", "GET").upper()
        encoded_method = self.METHOD_ENCODING.get(method_str, 0)

        url_str = response_data.get("url", "")
        parsed_url = urlparse(url_str)

        # Path depth & URL length
        path_segments = [seg for seg in parsed_url.path.split("/") if seg]
        path_depth = len(path_segments)
        url_length = len(url_str)

        # Query string analysis
        query_string = parsed_url.query
        query_params = parse_qs(query_string, keep_blank_values=True)
        query_param_count = len(query_params)
        query_string_length = len(query_string)

        # Payload analysis
        payload = response_data.get("payload", "") or ""
        payload_length = len(payload)
        payload_entropy = self.calculate_shannon_entropy(payload)
        special_char_count = self.count_special_characters(payload)

        # Header analysis
        headers = response_data.get("request_headers", {}) or {}
        header_count = len(headers)
        # Detect presence of authentication headers or bearer tokens in header values
        auth_header_present = 0
        for h, hv in headers.items():
            h_l = h.lower()
            if h_l in ["authorization", "x-api-key", "x-access-token"]:
                auth_header_present = 1
                break
            try:
                if isinstance(hv, str) and hv.strip().lower().startswith("bearer "):
                    auth_header_present = 1
                    break
            except Exception:
                # any unusual header value representation - ignore and continue
                pass

        # Response attributes
        status_code = int(response_data.get("status_code", 0))
        response_size = int(response_data.get("response_size", 0))

        # 5 New Security Risk Features
        risk_keywords = [
            'select', 'union', 'insert', 'drop',
            'delete', 'update', 'exec', 'script',
            'alert', 'onerror', 'onload', 'eval',
            'base64', 'passwd', 'shadow', 'whoami',
            'null', 'none', 'true', 'admin', 'root'
        ]
        combined_text = (str(payload) + " " + str(url_str)).lower()
        words = re.findall(r'\w+', combined_text)
        total_words = max(len(words), 1)
        keyword_matches = sum(1 for kw in risk_keywords if kw in combined_text)
        keyword_risk_score = round(min(keyword_matches / total_words, 1.0), 4)

        risky_params = [
            'id', 'user', 'uid', 'admin', 'debug',
            'test', 'token', 'key', 'pass', 'pwd',
            'cmd', 'exec', 'query', 'sql', 'file'
        ]
        param_names = [k.lower() for k in query_params.keys()]
        if payload.strip().startswith('{') and payload.strip().endswith('}'):
            try:
                import json
                j_data = json.loads(payload)
                if isinstance(j_data, dict):
                    param_names.extend([k.lower() for k in j_data.keys()])
            except Exception:
                pass
        param_name_risk = sum(1 for p in param_names if any(rp in p for rp in risky_params))

        url_encoded_ratio = round(url_str.count('%') / max(len(url_str), 1), 4)
        digit_count = sum(1 for c in payload if c.isdigit())
        payload_digit_ratio = round(digit_count / max(len(payload), 1), 4)

        sql_patterns = [
            r"'\s*(or|and)\s+'",
            r"\bunion\b.*\bselect\b",
            r"--\s*$",
            r";\s*(drop|delete|insert|update)",
            r"'\s*=\s*'"
        ]
        has_sql_structure = 1 if any(re.search(p, combined_text, re.IGNORECASE) for p in sql_patterns) else 0

        feature_vector = [
            encoded_method,
            path_depth,
            url_length,
            query_param_count,
            query_string_length,
            payload_length,
            payload_entropy,
            special_char_count,
            header_count,
            auth_header_present,
            status_code,
            response_size,
            keyword_risk_score,
            param_name_risk,
            url_encoded_ratio,
            payload_digit_ratio,
            has_sql_structure
        ]

        feature_dict = {
            "encoded_method": encoded_method,
            "path_depth": path_depth,
            "url_length": url_length,
            "query_param_count": query_param_count,
            "query_string_length": query_string_length,
            "payload_length": payload_length,
            "payload_entropy": payload_entropy,
            "special_char_count": special_char_count,
            "header_count": header_count,
            "auth_header_present": auth_header_present,
            "status_code": status_code,
            "response_size": response_size,
            "keyword_risk_score": keyword_risk_score,
            "param_name_risk": param_name_risk,
            "url_encoded_ratio": url_encoded_ratio,
            "payload_digit_ratio": payload_digit_ratio,
            "has_sql_structure": has_sql_structure,
            "feature_vector": feature_vector
        }

        # Check for sensitive fields exposure in response body
        response_body = response_data.get("response_body", "")
        sensitive_matches = self.check_sensitive_fields(response_body)
        feature_dict["sensitive_fields_leaked"] = sensitive_matches

        return feature_dict

    def check_sensitive_fields(self, response_body: str) -> List[str]:
        """Scans response body text for exposed sensitive field patterns."""
        if not response_body:
            return []
        found_matches = []
        for pattern in self.SENSITIVE_FIELD_PATTERNS:
            if re.search(pattern, response_body, re.IGNORECASE):
                found_matches.append(pattern)
        return found_matches


if __name__ == "__main__":
    parser = ResponseParser()
    sample_response = {
        "method": "POST",
        "url": "https://api.example.com/v1/users?ref=123",
        "payload": "user_input_payload_text!",
        "status_code": 200,
        "response_size": 512,
        "response_body": '{"status": "success", "token": "secret_token_123"}',
        "request_headers": {"Authorization": "Bearer token", "Content-Type": "application/json"}
    }
    extracted = parser.extract_features(sample_response)
    print("\n[+] Extracted 12 Telemetry Features:")
    for k, v in extracted.items():
        print(f"  {k}: {v}")
