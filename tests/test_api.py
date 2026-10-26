
from fastapi.testclient import TestClient
from unittest.mock import patch

from app.api.main import app


client = TestClient(app)


def test_health_endpoint():
    response = client.get("/health")

    assert response.status_code == 200
    assert response.json()["status"] == "ok"


def test_chat_endpoint():
    fake_result = {
        "question": "How can I cancel my order?",
        "answer": "Open your order history to cancel the order.",
        "relevant": True,
        "retrieved_faqs": [
            {
                "faq_id": "faq_test_1",
                "question": "How can I cancel my order?",
                "answer": "Open your order history.",
                "category": "ORDER",
                "intent": "cancel_order",
                "score": 0.80,
            }
        ],
    }

    with patch(
        "app.api.main.answer_question",
        return_value=fake_result,
    ):
        response = client.post(
            "/chat",
            json={"question": "How can I cancel my order?"},
        )

    assert response.status_code == 200

    data = response.json()
    assert data["answer"] == fake_result["answer"]
    assert data["relevant"] is True
    assert len(data["retrieved_faqs"]) == 1
    assert data["processing_time_ms"] >= 0


def test_chat_rejects_blank_question():
    response = client.post("/chat", json={"question": "   "})

    assert response.status_code == 422


def test_chat_rejects_missing_question():
    response = client.post("/chat", json={})

    assert response.status_code == 422
