import pytest
from fastapi.testclient import TestClient
from app.main import app

@pytest.fixture(scope="module")
def client():
    with TestClient(app) as c: yield c

def test_health(client):
    r = client.get("/health").json(); assert r["status"] == "ok" and r["llm_enabled"] is False and r["gemini_enabled"] is False and r["confidence_threshold"] == 0.82

def test_confident_query_auto_routes(client):
    r = client.post("/v1/route", json={"text": "my card has not arrived yet"}).json()
    assert r["decision"] == "AUTO_ROUTE" and r["intent"] == "card_arrival" and r["reply"]["mode"] == "template"
    assert len(r["top_k"]) == 3 and r["latency_ms"] < 500

def test_offtopic_goes_to_human(client):
    assert client.post("/v1/route", json={"text": "hello"}).json()["decision"] == "HUMAN_REVIEW"

def test_pii_is_redacted_and_reported(client):
    r = client.post("/v1/route", json={"text": "card 4539 1488 0343 6467 declined, mail me jo@x.com"}).json()
    assert set(r["pii_redacted"]) == {"CARD_NUMBER", "EMAIL"}

def test_draft_reply_can_be_disabled(client):
    assert client.post("/v1/route", json={"text": "why was I charged twice", "draft_reply": False}).json()["reply"] is None

@pytest.mark.parametrize("payload", [{}, {"text": ""}, {"text": "   "}, {"text": "a" * 1001}])
def test_invalid_input_rejected(client, payload):
    assert client.post("/v1/route", json=payload).status_code == 422

def test_batch(client):
    r = client.post("/v1/route/batch", json={"texts": ["my card has not arrived yet", "why is my top up pending"]}).json()
    assert r["count"] == 2 and all("decision" in x for x in r["results"])
    assert client.post("/v1/route/batch", json={"texts": ["x"] * 101}).status_code == 422

def test_ui_served(client):
    assert "Banking Query Router" in client.get("/").text
