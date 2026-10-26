
from unittest.mock import patch

from app.rag.graph import answer_question


def test_relevant_question_generates_answer():
    fake_faqs = [
        {
            "faq_id": "faq_test_1",
            "question": "How can I cancel my order?",
            "answer": "Open your order history.",
            "category": "ORDER",
            "intent": "cancel_order",
            "score": 0.80,
        }
    ]

    with (
        patch("app.rag.graph.retrieve_faqs", return_value=fake_faqs),
        patch(
            "app.rag.graph.generate_answer",
            return_value="Open your order history to cancel the order.",
        ),
    ):
        result = answer_question("How can I cancel my order?")

    assert result["relevant"] is True
    assert "order history" in result["answer"].lower()
    assert len(result["retrieved_faqs"]) == 1


def test_irrelevant_question_uses_fallback():
    fake_faqs = [
        {
            "faq_id": "faq_test_2",
            "question": "Contact customer support.",
            "answer": "Contact our support team.",
            "category": "CONTACT",
            "intent": "contact_customer_service",
            "score": 0.08,
        }
    ]

    with (
        patch("app.rag.graph.retrieve_faqs", return_value=fake_faqs),
        patch("app.rag.graph.generate_answer") as mocked_generator,
    ):
        result = answer_question("What is the weather on Mars?")

    assert result["relevant"] is False
    assert "couldn't find sufficiently relevant information" in (
        result["answer"].lower()
    )
    mocked_generator.assert_not_called()


def test_empty_question_is_rejected():
    import pytest

    with pytest.raises(ValueError):
        answer_question("   ")
