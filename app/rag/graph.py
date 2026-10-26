
from typing import TypedDict

from langgraph.graph import END, START, StateGraph

from app.rag.generator import generate_answer
from app.rag.retriever import retrieve_faqs


# Tune this threshold later using a representative evaluation set.
DEFAULT_RELEVANCE_THRESHOLD = 0.30


class SupportState(TypedDict, total=False):
    question: str
    retrieved_faqs: list[dict]
    answer: str
    relevant: bool


def retrieve_node(state: SupportState) -> dict:
    question = state["question"]
    faqs = retrieve_faqs(question, top_k=3)

    return {"retrieved_faqs": faqs}


def relevance_node(state: SupportState) -> dict:
    faqs = state.get("retrieved_faqs", [])

    relevant = bool(faqs) and max(
        float(faq.get("score", 0.0)) for faq in faqs
    ) >= DEFAULT_RELEVANCE_THRESHOLD

    return {"relevant": relevant}


def generate_node(state: SupportState) -> dict:
    answer = generate_answer(
        state["question"],
        state.get("retrieved_faqs", []),
    )

    return {"answer": answer}


def fallback_node(state: SupportState) -> dict:
    return {
        "answer": (
            "I couldn't find sufficiently relevant information in the "
            "available FAQs. Please contact customer support for help."
        )
    }


def route_after_relevance(state: SupportState) -> str:
    return "generate" if state.get("relevant", False) else "fallback"


def build_support_graph():
    builder = StateGraph(SupportState)

    builder.add_node("retrieve", retrieve_node)
    builder.add_node("check_relevance", relevance_node)
    builder.add_node("generate", generate_node)
    builder.add_node("fallback", fallback_node)

    builder.add_edge(START, "retrieve")
    builder.add_edge("retrieve", "check_relevance")

    builder.add_conditional_edges(
        "check_relevance",
        route_after_relevance,
        {
            "generate": "generate",
            "fallback": "fallback",
        },
    )

    builder.add_edge("generate", END)
    builder.add_edge("fallback", END)

    return builder.compile()


support_graph = build_support_graph()


def answer_question(question: str) -> dict:
    question = question.strip()

    if not question:
        raise ValueError("Question must not be empty.")

    result = support_graph.invoke({"question": question})

    return {
        "question": question,
        "answer": result["answer"],
        "retrieved_faqs": result.get("retrieved_faqs", []),
        "relevant": result.get("relevant", False),
    }
