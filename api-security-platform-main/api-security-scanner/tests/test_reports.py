import os
import json
import pytest
from pathlib import Path
from reports.pdf_generator import PDFReportGenerator
from reports.json_exporter import JSONReportExporter
from reports.html_exporter import HTMLReportExporter

def test_pdf_report_generation():
    generator = PDFReportGenerator()
    dummy_session = {
        "id": 999,
        "target_url": "http://example-target.org",
        "overall_risk_score": 55.5,
        "overall_severity": "Medium",
        "total_endpoints_found": 2,
        "total_vulnerabilities_found": 1
    }
    dummy_findings = [{
        "url": "http://example-target.org/api",
        "attack_type": "Security_Misconfiguration",
        "severity": "Medium",
        "risk_score": 55.5,
        "recommendation": "Enforce strict CSP and frame options."
    }]

    file_path = generator.generate(dummy_session, dummy_findings)
    assert os.path.exists(file_path)
    assert file_path.endswith(".pdf")

def test_json_report_export():
    exporter = JSONReportExporter()
    dummy_session = {"id": 999, "target_url": "http://example-target.org", "overall_risk_score": 55.5, "overall_severity": "Medium"}
    dummy_eps = [{"url": "http://example-target.org/api", "method": "GET"}]
    dummy_findings = [{"attack_type": "Security_Misconfiguration", "risk_score": 55.5}]

    file_path = exporter.export(dummy_session, dummy_eps, dummy_findings)
    assert os.path.exists(file_path)
    assert file_path.endswith(".json")

def test_html_report_export():
    exporter = HTMLReportExporter()
    dummy_session = {"id": 999, "target_url": "http://example-target.org", "overall_risk_score": 55.5, "overall_severity": "Medium"}
    dummy_findings = [{"url": "http://example-target.org/api", "attack_type": "Security_Misconfiguration", "severity": "Medium", "risk_score": 55.5, "recommendation": "Check CSP headers."}]

    file_path = exporter.export(dummy_session, dummy_findings)
    assert os.path.exists(file_path)
    assert file_path.endswith(".html")


def test_sarif_report_export_contains_structured_proof_and_status(tmp_path):
    from reports.sarif_exporter import SARIFReportExporter

    session = {
        "id": 1001,
        "target_url": "https://example.test",
        "overall_risk_score": 40.0,
        "overall_severity": "High",
        "total_endpoints_found": 1,
        "total_vulnerabilities_found": 1,
    }
    findings = [
        {
            "url": "https://example.test/api/users/1",
            "method": "GET",
            "attack_type": "BOLA_IDOR",
            "finding_status": "Confirmed",
            "severity": "High",
            "risk_score": 40.0,
            "signature_triggered": "Unauthorized object access returned sensitive object properties",
            "request_payload": "id=1",
            "response_status": 200,
            "recommendation": "Enforce object-level authorization.",
        },
        {
            "url": "https://example.test/health",
            "method": "GET",
            "attack_type": "Security_Misconfiguration",
            "finding_status": "Informational",
            "severity": "Low",
            "risk_score": 3.5,
            "signature_triggered": "Missing headers",
        },
    ]

    path = SARIFReportExporter(output_dir=str(tmp_path)).export(session, findings)
    document = json.loads(Path(path).read_text(encoding="utf-8"))
    results = document["runs"][0]["results"]
    assert document["version"] == "2.1.0"
    assert len(results) == 2
    assert results[0]["properties"]["confirmed"] is True
    assert results[0]["level"] == "error"
    assert results[0]["locations"][0]["logicalLocations"][0]["kind"] == "endpoint"
    assert "Unauthorized object access" in results[0]["message"]["text"]
    assert results[1]["properties"]["confirmed"] is False
    assert "apiSecurityIdentity" in results[0]["partialFingerprints"]
