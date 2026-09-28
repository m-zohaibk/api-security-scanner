"""End-to-end operational and integration testing matrix."""
import json
import os
import sys
from pathlib import Path
import pytest

ROOT_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT_DIR))

from config.settings import MODELS_DIR, REPORTS_DIR
from database.db import (
    init_db,
    save_scan_session,
    save_endpoint,
    save_finding,
    get_session_findings,
    delete_session,
    SessionLocal,
)
from database.models import ScanSession, Finding
from detection.signature import SignatureDetector
from detection.ml_model import MLAnomalyDetector
from detection.deep_learning import DeepLearningDetector
from detection.risk_scorer import RiskScorer
from reports.pdf_generator import PDFReportGenerator
from reports.json_exporter import JSONReportExporter
from reports.html_exporter import HTMLReportExporter
from reports.sarif_exporter import SARIFReportExporter
from scripts.install_ml_bundle import install_bundle
from dashboard.app import create_app


def test_model_bundle_loading_and_fallback(tmp_path):
    """Test installing bundle artifacts and loading models with fallback support."""
    # Test installation helper
    src_bundle = ROOT_DIR / "models"
    target_models = tmp_path / "models"
    
    if src_bundle.exists():
        success = install_bundle(src_bundle, target_models)
        assert success is True
        assert (target_models / "tabular_ranker.pkl").exists()

    # Test MLAnomalyDetector loading from custom models dir
    ml_det = MLAnomalyDetector(model_path=str(target_models / "isolation_forest.pkl"))
    res = ml_det.predict({"encoded_method": 1, "path_depth": 2, "url_length": 30})
    assert "anomaly_score" in res
    assert "supervised_probability" in res

    # Test fallback behavior when model is absent
    ml_fallback = MLAnomalyDetector(model_path=str(tmp_path / "non_existent.pkl"))
    fallback_res = ml_fallback.predict({"encoded_method": 1})
    assert fallback_res["is_anomaly"] is False


def test_proof_confirmed_vs_model_suspected_distinction():
    """Verify that ML anomaly scores alone never mark findings as Confirmed."""
    sig_det = SignatureDetector()
    risk_scorer = RiskScorer()

    # Case 1: High ML anomaly but NO signature response proof
    sample_response_no_proof = {
        "url": "http://localhost:5001/api/v1/users",
        "payload": "' OR 1=1 --",
        "status_code": 200,
        "response_body": "Normal user search results",
        "response_size": 25,
        "response_time": 0.05,
    }
    sig_res = sig_det.analyze(sample_response_no_proof)
    assert sig_res["has_proof"] is False

    # Risk score calculation
    ml_res = {"is_anomaly": True, "anomaly_score": 0.85, "points": 25.0}
    dl_res = {"is_suspicious": True, "lstm_score": 15.0, "autoencoder_score": 10.0, "points": 25.0}
    scored = risk_scorer.calculate_risk(
        signature_result=sig_res,
        ml_result=ml_res,
        dl_result=dl_res,
        endpoint_url="http://localhost:5001/api/v1/users",
        http_method="GET"
    )
    # Finding must remain Suspected / Informational, NEVER Confirmed without proof
    assert scored["finding_status"] in ("Suspected", "Informational", "Suspicious")
    assert scored["finding_status"] != "Confirmed"

    # Case 2: Deterministic SQL error response proof present
    sample_response_with_proof = {
        "url": "http://localhost:5001/api/v1/login",
        "payload": "' OR 1=1 --",
        "status_code": 500,
        "response_body": "PostgreSQL ERROR: syntax error at or near 'OR' at line 1.",
        "response_size": 250,
        "response_time": 0.12,
    }
    sig_proof = sig_det.analyze(sample_response_with_proof)
    assert sig_proof["matched"] is True
    assert sig_proof["has_proof"] is True

    scored_confirmed = risk_scorer.calculate_risk(
        signature_result=sig_proof,
        ml_result=ml_res,
        dl_result=dl_res,
        endpoint_url="http://localhost:5001/api/v1/login",
        http_method="POST"
    )
    assert scored_confirmed["finding_status"] == "Confirmed"
    assert scored_confirmed["severity"] in ("HIGH", "CRITICAL", "High", "Critical")


def test_report_exporters_end_to_end(tmp_path):
    """Test generation of JSON, HTML, SARIF 2.1.0, and PDF report formats."""
    session_data = {
        "id": 999,
        "target_url": "http://localhost:5001/api",
        "scan_start_time": "2026-08-21T12:00:00",
        "overall_risk_score": 85.0,
        "overall_severity": "High",
        "total_endpoints_found": 3,
        "total_vulnerabilities_found": 1,
    }
    endpoints_list = [{"url": "http://localhost:5001/api/users", "method": "GET"}]
    findings_data = [
        {
            "id": 1,
            "url": "http://localhost:5001/api/users",
            "method": "GET",
            "attack_type": "SQL_Injection",
            "finding_status": "Confirmed",
            "severity": "High",
            "risk_score": 85.0,
            "signature_triggered": "syntax error near OR",
            "ml_score": 25.0,
            "lstm_score": 15.0,
            "autoencoder_score": 10.0,
            "recommendation": "Use parameterized prepared statements.",
            "response_status": 500,
            "response_size": 250,
            "response_time": 0.08,
        }
    ]

    # 1. JSON Export
    json_path_str = JSONReportExporter(output_dir=str(tmp_path)).export(session_data, endpoints_list, findings_data)
    json_path = Path(json_path_str)
    assert json_path.exists()
    loaded_json = json.loads(json_path.read_text(encoding="utf-8"))
    assert loaded_json["session_summary"]["target_url"] == session_data["target_url"]

    # 2. HTML Export
    html_path_str = HTMLReportExporter(output_dir=str(tmp_path)).export(session_data, findings_data)
    html_path = Path(html_path_str)
    assert html_path.exists()
    assert "<!DOCTYPE html>" in html_path.read_text(encoding="utf-8")

    # 3. SARIF 2.1.0 Export
    sarif_path = tmp_path / "report.sarif"
    SARIFReportExporter().export(session_data, findings_data, output_path=str(sarif_path))
    assert sarif_path.exists()
    sarif_data = json.loads(sarif_path.read_text(encoding="utf-8"))
    assert sarif_data["version"] == "2.1.0"
    assert "API Security" in sarif_data["runs"][0]["tool"]["driver"]["name"]

    # 4. PDF Generation
    pdf_path_str = PDFReportGenerator(output_dir=str(tmp_path)).generate(session_data, findings_data)
    pdf_path = Path(pdf_path_str)
    assert pdf_path.exists()
    assert pdf_path.stat().st_size > 100


def test_database_persistence_and_cascade(tmp_path):
    """Test full database lifecycle including findings persistence and cascade deletion."""
    init_db()
    sess = save_scan_session("http://testlab.local/api", 2, 1, 75.0, "High")
    assert sess.id is not None

    ep = save_endpoint(sess.id, "http://testlab.local/api/users", "GET")
    assert ep.id is not None

    fnd = save_finding(
        session_id=sess.id,
        endpoint_id=ep.id,
        attack_type="Reflected_XSS",
        severity="Medium",
        risk_score=50.0,
        finding_status="Suspected",
        signature_triggered="script reflection",
    )
    assert fnd.id is not None

    findings = get_session_findings(sess.id)
    assert len(findings) == 1
    assert findings[0].attack_type == "Reflected_XSS"

    # Cascade delete session
    deleted = delete_session(sess.id)
    assert deleted is True
    assert len(get_session_findings(sess.id)) == 0
