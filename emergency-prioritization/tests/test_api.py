"""
Stage 15 — Automated tests: API integration.

Uses an isolated temporary SQLite DB (never touches data/emergency.db) and
FastAPI's TestClient. Requires the full app dependencies installed
(fastapi, sqlalchemy, python-jose, passlib) -- run after `pip install -r
requirements.txt`.

Run with: pytest tests/test_api.py -v
"""
import os
import tempfile

import pytest

pytest.importorskip("fastapi")

# Point at a throwaway DB file BEFORE importing app modules that read settings.
_tmp_db_fd, _tmp_db_path = tempfile.mkstemp(suffix=".db")
os.environ["DATABASE_URL"] = f"sqlite:///{_tmp_db_path}"
os.environ["SECRET_KEY"] = "test-secret-key-not-for-production"

from fastapi.testclient import TestClient  # noqa: E402
from app.main import app  # noqa: E402
from app.models.database import init_db, SessionLocal, User, UserRole  # noqa: E402
from app.services.auth_service import hash_password  # noqa: E402


@pytest.fixture(scope="module")
def client():
    init_db()
    db = SessionLocal()
    db.add(User(username="testadmin", hashed_password=hash_password("Test@123"), role=UserRole.admin))
    db.commit()
    db.close()
    with TestClient(app) as c:
        yield c
    os.close(_tmp_db_fd)
    os.remove(_tmp_db_path)


@pytest.fixture
def auth_headers(client):
    resp = client.post("/api/auth/login", data={"username": "testadmin", "password": "Test@123"})
    assert resp.status_code == 200
    token = resp.json()["access_token"]
    return {"Authorization": f"Bearer {token}"}


def test_health_check(client):
    resp = client.get("/health")
    assert resp.status_code == 200
    assert resp.json()["status"] == "ok"


def test_login_wrong_password_rejected(client):
    resp = client.post("/api/auth/login", data={"username": "testadmin", "password": "wrong"})
    assert resp.status_code == 401


def test_login_success_returns_token(client):
    resp = client.post("/api/auth/login", data={"username": "testadmin", "password": "Test@123"})
    assert resp.status_code == 200
    assert "access_token" in resp.json()


def test_queue_requires_auth(client):
    resp = client.get("/api/messages/queue")
    assert resp.status_code == 401


def test_submit_and_retrieve_message(client, auth_headers):
    resp = client.post("/api/messages/submit", json={"text": "Fire near Whitefield, people trapped inside"},
                        headers=auth_headers)
    # 503 is acceptable here if NLP models haven't been trained in this
    # test environment yet -- otherwise expect 200 with real analysis.
    assert resp.status_code in (200, 503)
    if resp.status_code == 200:
        data = resp.json()
        assert data["category"] is not None
        assert data["urgency"] is not None

        queue_resp = client.get("/api/messages/queue", headers=auth_headers)
        assert queue_resp.status_code == 200
        assert any(m["message_id"] == data["message_id"] for m in queue_resp.json())


def test_override_requires_valid_range(client, auth_headers):
    submit_resp = client.post("/api/messages/submit", json={"text": "Test override message flood"},
                               headers=auth_headers)
    if submit_resp.status_code != 200:
        pytest.skip("NLP models not trained in this test environment")
    message_id = submit_resp.json()["message_id"]

    bad_resp = client.post(f"/api/messages/{message_id}/override", json={"new_priority": 1.5},
                            headers=auth_headers)
    assert bad_resp.status_code == 422  # out of [0,1] range

    good_resp = client.post(f"/api/messages/{message_id}/override", json={"new_priority": 0.9},
                             headers=auth_headers)
    assert good_resp.status_code == 200
    assert good_resp.json()["final_priority_source"] == "human_override"
