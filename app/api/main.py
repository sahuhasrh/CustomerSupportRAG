
from time import perf_counter

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

from app.rag.graph import answer_question


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
    question = request.question.strip()

    if not question:
        raise HTTPException(
            status_code=422,
            detail="Question must not be empty.",
        )

    start = perf_counter()

    try:
        result = answer_question(question)
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc
    except Exception as exc:
        # Avoid returning API keys or internal exception details to clients.
        raise HTTPException(
            status_code=500,
            detail="The chatbot could not process your question. Please try again.",
        ) from exc

    elapsed_ms = (perf_counter() - start) * 1000

    return {
        "question": result["question"],
        "answer": result["answer"],
        "relevant": result["relevant"],
        "retrieved_faqs": result["retrieved_faqs"],
        "processing_time_ms": round(elapsed_ms, 2),
    }
