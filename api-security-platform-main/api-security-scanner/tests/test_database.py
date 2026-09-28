import pytest
from database.db import init_db, save_scan_session, save_endpoint, save_finding, get_session_findings, delete_session

def test_database_crud_operations():
    init_db()
    
    session_obj = save_scan_session("http://test-api.org", total_endpoints=2, overall_risk_score=45.0, overall_severity="Medium")
    assert session_obj.id is not None
    
    endpoint_obj = save_endpoint(session_obj.id, "http://test-api.org/v1/auth", "POST")
    assert endpoint_obj.id is not None
    
    finding_obj = save_finding(
        session_id=session_obj.id,
        endpoint_id=endpoint_obj.id,
        attack_type="Broken_Auth",
        severity="Medium",
        risk_score=45.0,
        recommendation="Use JWT validation."
    )
    assert finding_obj.id is not None
    
    findings_list = get_session_findings(session_obj.id)
    assert len(findings_list) == 1
    assert findings_list[0].attack_type == "Broken_Auth"
    
    deleted = delete_session(session_obj.id)
    assert deleted is True
