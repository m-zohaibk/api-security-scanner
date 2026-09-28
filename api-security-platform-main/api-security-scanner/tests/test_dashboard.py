import pytest
from dashboard.app import create_app
from database.db import save_scan_session

def test_dashboard_routes():
    app = create_app()
    app.config["TESTING"] = True
    client = app.test_client()

    response_index = client.get("/")
    assert response_index.status_code == 200
    assert b"Complete API Security Scan" in response_index.data

    response_history = client.get("/history")
    assert response_history.status_code == 200
    assert b"Inspection Scan History" in response_history.data

def test_dashboard_complete_pipeline_ui():
    app = create_app()
    app.config["TESTING"] = True
    client = app.test_client()

    response = client.get("/")
    assert response.status_code == 200
    assert b"Complete API Security Scan" in response.data
    assert b"GraphQL introspection" in response.data
    assert b"Export SARIF" not in response.data
    assert b"module_sqli" not in response.data


def test_dashboard_api_routes():
    app = create_app()
    app.config["TESTING"] = True
    client = app.test_client()

    # List sessions API
    res = client.get("/api/sessions")
    assert res.status_code == 200
    json_data = res.get_json()
    assert json_data["status"] == "success"
    assert isinstance(json_data["sessions"], list)

    # API Scan error on missing target_url
    bad_scan = client.post("/api/scan", json={})
    assert bad_scan.status_code == 400


def test_dashboard_api_scan_uses_shared_pipeline_and_returns_sarif_url(monkeypatch):
    app = create_app()
    app.config["TESTING"] = True
    client = app.test_client()
    session = save_scan_session(target_url="http://example.test", total_endpoints=1)
    calls = []

    def fake_pipeline(target_url, return_session_id=False, **kwargs):
        calls.append((target_url, return_session_id))
        return session.id

    monkeypatch.setattr("dashboard.routes.run_pipeline", fake_pipeline)
    response = client.post("/api/scan", json={"target_url": "http://example.test"})
    assert response.status_code == 200
    payload = response.get_json()
    assert calls == [("http://example.test", True)]
    assert payload["session_id"] == session.id
    assert payload["results_url"].endswith(f"/results/{session.id}")
    assert payload["sarif_url"].endswith(f"/export/{session.id}?format=sarif")


def test_dashboard_form_scan_uses_shared_pipeline(monkeypatch):
    app = create_app()
    app.config["TESTING"] = True
    client = app.test_client()
    session = save_scan_session(target_url="http://example.test", total_endpoints=1)
    calls = []

    def fake_pipeline(target_url, return_session_id=False, **kwargs):
        calls.append((target_url, return_session_id))
        return session.id

    monkeypatch.setattr("dashboard.routes.run_pipeline", fake_pipeline)
    response = client.post("/scan", data={"target_url": "http://example.test"})
    assert response.status_code == 302
    assert response.headers["Location"].endswith(f"/results/{session.id}")
    assert calls == [("http://example.test", True)]


def test_dashboard_csrf_enforcement():
    app = create_app()
    app.config["TESTING"] = True
    app.config["CSRF_ENABLED"] = True
    client = app.test_client()

    # Form post without CSRF token should fail with 400
    res_bad = client.post("/scan", data={"target_url": "http://example.test"})
    assert res_bad.status_code == 400

    # Get index page to establish session and obtain csrf token
    res_get = client.get("/")
    assert res_get.status_code == 200
    with client.session_transaction() as sess:
        token = sess.get("csrf_token")
    assert token is not None

    # Form post with valid CSRF token passes validation
    res_good = client.post("/scan", data={"target_url": "http://example.test", "csrf_token": token})
    # Will attempt to scan or redirect
    assert res_good.status_code in (302, 500)


def test_dashboard_auth_workflow():
    app = create_app()
    app.config["TESTING"] = True
    app.config["DASHBOARD_AUTH_ENABLED"] = True
    app.config["DASHBOARD_ADMIN_USER"] = "admin_test"
    app.config["DASHBOARD_ADMIN_PASSWORD"] = "pass_test_123"
    client = app.test_client()

    # Unauthenticated access redirects to /login
    res = client.get("/")
    assert res.status_code == 302
    assert "/login" in res.headers["Location"]

    # Login with wrong credentials fails
    res_bad_login = client.post("/login", data={"username": "admin_test", "password": "wrong"})
    assert res_bad_login.status_code == 200
    assert b"Invalid username or password" in res_bad_login.data

    # Login with correct credentials succeeds
    res_login = client.post("/login", data={"username": "admin_test", "password": "pass_test_123"})
    assert res_login.status_code == 302

    # Now authenticated access to / succeeds
    res_auth = client.get("/")
    assert res_auth.status_code == 200

    # Logout clears authentication
    res_logout = client.get("/logout")
    assert res_logout.status_code == 302
    res_after = client.get("/")
    assert res_after.status_code == 302
    assert "/login" in res_after.headers["Location"]

