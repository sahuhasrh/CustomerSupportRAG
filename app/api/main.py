
from time import perf_counter

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

from app.rag.graph import answer_question
import logging
import uuid

from fastapi import Request

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
)

logger = logging.getLogger("customer_support_rag")
app = FastAPI(
    title="Customer Support RAG API",
    description=(
        "A customer support chatbot using semantic retrieval, "
        "Qdrant, LangGraph, and Groq."
    ),
    version="1.0.0",
)



class ChatRequest(BaseModel):
    question: str = Field(
        ...,
        min_length=1,
        max_length=2000,
        description="The customer's support question.",
    )


class RetrievedFAQ(BaseModel):
    faq_id: str
    question: str
    answer: str
    category: str
    intent: str
    score: float


class ChatResponse(BaseModel):
    question: str
    answer: str
    relevant: bool
    retrieved_faqs: list[RetrievedFAQ]
    processing_time_ms: float


@app.get("/")
def root():
    return {
        "message": "Customer Support RAG API is running.",
        "docs": "/docs",
        "health": "/health",
    }


@app.get("/health")
def health():
    return {"status": "ok", "service": "customer-support-rag"}


@app.post("/chat", response_model=ChatResponse)
def chat(request: ChatRequest):
    request_id = str(uuid.uuid4())[:8]
    question = request.question.strip()

    if not question:
        logger.warning(
            "request_id=%s | Empty question received",
            request_id,
        )
        raise HTTPException(
            status_code=422,
            detail="Question must not be empty.",
        )

    start = perf_counter()

    logger.info(
        "request_id=%s | Chat request received",
        request_id,
    )

    try:
        result = answer_question(question)

    except ValueError as exc:
        logger.warning(
            "request_id=%s | Validation error: %s",
            request_id,
            str(exc),
        )
        raise HTTPException(status_code=422, detail=str(exc)) from exc

    except Exception:
        logger.exception(
            "request_id=%s | Chat processing failed",
            request_id,
        )
        raise HTTPException(
            status_code=500,
            detail="The chatbot could not process your question. Please try again.",
        )

    elapsed_ms = (perf_counter() - start) * 1000

    logger.info(
        "request_id=%s | relevant=%s | retrieved_faqs=%d | processing_time_ms=%.2f",
        request_id,
        result["relevant"],
        len(result["retrieved_faqs"]),
        elapsed_ms,
    )

    return {
        "question": result["question"],
        "answer": result["answer"],
        "relevant": result["relevant"],
        "retrieved_faqs": result["retrieved_faqs"],
        "processing_time_ms": round(elapsed_ms, 2),
    }