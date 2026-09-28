import json
from pathlib import Path

FIXTURE = Path(__file__).parent / "fixtures" / "proof_verified_regression_cases.json"

def load_fixture():
    return json.loads(FIXTURE.read_text(encoding="utf-8"))

def test_proof_verified_fixture_is_sanitized_and_nonempty():
    fixture = load_fixture()
    assert fixture["purpose"].startswith("Regression evidence only")
    assert fixture["cases"]
