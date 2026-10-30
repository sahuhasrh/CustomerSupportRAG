# Customer Support RAG Chatbot

An AI-powered customer support chatbot that retrieves relevant FAQ entries using semantic search and generates grounded responses with a Groq-hosted language model. The application combines **FastAPI, LangGraph, Qdrant, FastEmbed, Groq, and Streamlit** and is deployed as a cloud-hosted application.

## Live Application

- **Frontend:** Replace this line with your Streamlit Community Cloud URL.
- **Backend API:** https://customer-support-rag-api.onrender.com
- **API documentation (Swagger):** https://customer-support-rag-api.onrender.com/docs
- **Health check:** https://customer-support-rag-api.onrender.com/health
- **Source code:** https://github.com/sahuhasrh/CustomerSupportRAG

> The Render free service may enter an idle state after inactivity. The first request after an idle period can take longer while the service starts.

## Features

- Semantic retrieval of customer-support FAQs using 384-dimensional embeddings.
- Qdrant vector search with cosine similarity.
- FastEmbed (ONNX Runtime) for lightweight query-embedding inference.
- Groq LLM-based answer generation grounded in retrieved FAQ context.
- LangGraph workflow for retrieval, relevance checking, answer generation, and fallback handling.
- Out-of-domain query handling when the FAQ knowledge base does not contain relevant information.
- FastAPI backend with request validation, `/chat`, `/health`, and root endpoints.
- Streamlit chat interface with retrieved FAQ details and response-time display.
- Structured application logs with timestamps and unique request IDs.
- Automated API and graph tests using pytest.
- GitHub Actions workflow for automated testing.
- Automatic cloud deployment through Render and Streamlit Community Cloud.
- Dockerfiles and Docker Compose configuration for local containerized execution.

## Architecture

```text
                         User
                          |
                          v
                 Streamlit Frontend
              (Streamlit Community Cloud)
                          |
                          v
                    FastAPI Backend
                       (Render)
                          |
                          v
                  LangGraph Workflow
                          |
             +------------+------------+
             |                         |
             v                         v
       FAQ Retrieval              Relevance Check
             |                         |
             v                         v
       FastEmbed                  Relevant context?
   (query embedding)               /         \
             |                   Yes          No
             v                    |            |
       Qdrant Cloud                v            v
     (vector search)         Groq LLM       Fallback
                                  |
                                  v
                         Grounded Response
                                  |
                                  v
                            User Response
```

## RAG Workflow

1. The user submits a customer-support question through the Streamlit interface.
2. The frontend sends the question to the FastAPI `/chat` endpoint.
3. FastEmbed converts the question into a 384-dimensional query vector.
4. Qdrant searches the FAQ collection using cosine similarity and returns the most relevant entries.
5. LangGraph evaluates whether the retrieved context is relevant.
6. If relevant, the retrieved FAQ context is passed to the Groq LLM to generate a grounded response.
7. If the context is not relevant, the workflow returns a fallback response rather than attempting to answer an unrelated question.
8. The API returns the answer, relevance status, retrieved FAQs, and processing time.

## Technology Stack

| Component | Technology |
|---|---|
| Language | Python 3.11 |
| Query embeddings | FastEmbed / ONNX Runtime |
| Embedding model | `sentence-transformers/all-MiniLM-L6-v2` compatible 384-dimensional vectors |
| Vector database | Qdrant |
| Cloud vector database | Qdrant Cloud |
| LLM provider | Groq |
| LLM integration | LangChain |
| Workflow orchestration | LangGraph |
| Backend | FastAPI |
| Frontend | Streamlit |
| API testing | pytest |
| Containerization | Docker, Docker Compose |
| Version control | Git, GitHub |
| CI | GitHub Actions |
| Backend hosting | Render |
| Frontend hosting | Streamlit Community Cloud |

### Embedding optimization

The initial retrieval implementation used Sentence Transformers. For deployment within a memory-constrained environment, query embedding inference was migrated to FastEmbed, which uses ONNX Runtime. The existing Qdrant collection was retained rather than re-embedding all FAQ records. A compatibility check confirmed matching 384-dimensional output and cosine similarity of `1.0` for the tested query.

## Dataset

The project uses the [Bitext Customer Support LLM Chatbot Training Dataset](https://github.com/bitext/customer-support-llm-chatbot-training-dataset).

- Original dataset: 26,872 records.
- Intents: 27.
- Categories: 11.
- Current retrieval collection: 1,000 FAQ records selected as a smaller subset while retaining coverage across the intents.

The raw dataset, processed data, generated embeddings, local environment files, and credentials are not committed to this repository.

## Repository Structure

```text
CustomerSupportRAG/
├── app/
│   ├── api/
│   │   └── main.py
│   └── rag/
│       ├── retriever.py
│       ├── generator.py
│       └── graph.py
├── frontend/
│   └── streamlit_app.py
├── scripts/
│   ├── inspect_dataset.py
│   ├── prepare_dataset.py
│   ├── create_subset.py
│   ├── generate_subset_embeddings.py
│   ├── setup_qdrant_collection.py
│   └── migrate_qdrant_to_cloud.py
├── tests/
│   ├── test_api.py
│   └── test_graph.py
├── .github/
│   └── workflows/
│       └── ci.yml
├── Dockerfile.api
├── Dockerfile.frontend
├── docker-compose.yml
├── requirements.txt
├── .env.example
├── .gitignore
└── README.md
```

## API Endpoints

| Method | Endpoint | Description |
|---|---|---|
| `GET` | `/` | Returns API information and available endpoint paths. |
| `GET` | `/health` | Returns the service health status. |
| `POST` | `/chat` | Accepts a question and returns a generated answer with retrieval details. |
| `GET` | `/docs` | Interactive Swagger API documentation provided by FastAPI. |

### Example chat request

```json
{
  "question": "How can I recover my password?"
}
```

### Response fields

The `/chat` endpoint returns:

- `question`: submitted question.
- `answer`: generated answer or fallback response.
- `relevant`: whether the retrieved FAQ context was considered relevant.
- `retrieved_faqs`: retrieved FAQ entries, including their category, intent, and similarity score.
- `processing_time_ms`: total backend processing time in milliseconds.

## Deployment

The application is deployed using separate hosting services for the frontend and backend.

| Service | Hosting platform | Purpose |
|---|---|---|
| Streamlit frontend | Streamlit Community Cloud | Public chatbot interface |
| FastAPI backend | Render | Chat API and RAG orchestration |
| Vector database | Qdrant Cloud | Hosted FAQ vectors and payloads |
| LLM inference | Groq | Response generation |

The backend uses environment variables for configuration and secrets. The cloud deployment is configured with the Qdrant Cloud URL and API key, collection name, Groq API key, and Groq model. The Streamlit frontend receives the deployed backend URL through its `API_URL` setting.

Do not commit `.env`, API keys, database credentials, or Streamlit secrets to GitHub.

## CI/CD

### Continuous Integration — GitHub Actions

The workflow at `.github/workflows/ci.yml` runs on pushes to `main` and pull requests targeting `main`. It checks out the repository, sets up Python 3.11, installs dependencies, and runs:

```bash
python -m pytest tests/ -v
```

**Latest validation:** 7 tests passed (4 API tests and 3 graph tests).

### Continuous Deployment — Hosting platforms

- **Render:** Configured to automatically build and deploy the FastAPI backend when changes are pushed to the connected `main` branch.
- **Streamlit Community Cloud:** Connected to the GitHub repository for frontend deployment and updates.

> CI and deployment are currently configured as separate triggers. GitHub Actions reports test results, while Render's automatic deployment is triggered by GitHub commits independently; deployment is not yet gated on CI success.

## Monitoring, Logging, and Reliability

- Structured logs include timestamps, log levels, and unique request IDs.
- Chat request logs record whether retrieved FAQs were relevant, the number of FAQs retrieved, and processing time.
- Customer question text is not included in the application logs.
- The API validates question input and returns appropriate HTTP errors for invalid requests and processing failures.
- The `/health` endpoint provides a basic service health check.
- The frontend displays retrieved FAQ details and backend processing time.

## Local Setup

### Prerequisites

- Python 3.11
- Git
- Docker Desktop (for local Qdrant / Docker Compose)
- Groq API key
- Qdrant URL and API key if using Qdrant Cloud

### 1. Clone the repository

```bash
git clone https://github.com/sahuhasrh/CustomerSupportRAG.git
cd CustomerSupportRAG
```

### 2. Create and activate a virtual environment

Windows PowerShell:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
```

### 3. Install dependencies

```powershell
python -m pip install --upgrade pip
pip install -r requirements.txt
pip install pytest
```

### 4. Configure environment variables

```powershell
Copy-Item .env.example .env
```

Set the required values in `.env`. Typical settings include:

```env
GROQ_API_KEY=your_groq_api_key
GROQ_MODEL=your_groq_model

QDRANT_URL=http://127.0.0.1:6333
QDRANT_API_KEY=
QDRANT_COLLECTION=customer_support_faq_1000
```

For Qdrant Cloud, use the cluster URL and API key instead of the local Qdrant URL. Never commit the populated `.env` file.

### 5. Start Qdrant locally (if using local Qdrant)

```powershell
docker compose up -d qdrant
```

A populated collection is required before retrieval can work. If starting from scratch, obtain the dataset from the Bitext source and follow the repository's preprocessing, subset, embedding, and Qdrant setup scripts. If using the already populated Qdrant Cloud collection, configure its URL, API key, and collection name instead.

### 6. Start the FastAPI backend

```powershell
$env:PYTHONPATH = (Get-Location).Path
uvicorn app.api.main:app --reload --host 127.0.0.1 --port 8000
```

- API: http://127.0.0.1:8000
- Swagger UI: http://127.0.0.1:8000/docs
- Health check: http://127.0.0.1:8000/health

### 7. Start the Streamlit frontend

Open another terminal, activate the virtual environment, and run:

```powershell
python -m streamlit run frontend/streamlit_app.py
```

Open http://localhost:8501.

If the backend is hosted remotely, set `API_URL` to the backend URL before starting Streamlit.

### 8. Run tests

```powershell
$env:PYTHONPATH = (Get-Location).Path
python -m pytest tests/ -v
```

## Docker

The repository includes `Dockerfile.api`, `Dockerfile.frontend`, and `docker-compose.yml` for containerized local execution.

```powershell
docker compose up --build
```

The Qdrant collection must be populated and the required environment variables must be configured before chat requests can be processed successfully.

## Current Limitations

- Answer quality depends on the coverage and quality of the FAQ dataset and retrieved context.
- Some source FAQ answers contain template placeholders, which may appear in generated responses.
- Similarity thresholds and retrieval quality can be improved through a broader evaluation set.
- The current analytics are based on application logs; a persistent analytics dashboard and user-feedback store have not been implemented.
- Render's free service can have cold starts after inactivity.
- CI test execution and automatic deployment are not yet connected through a test-gated deployment workflow.

## Future Improvements

- Add persistent query analytics and a visual admin dashboard.
- Add thumbs-up / thumbs-down feedback and evaluate answer quality.
- Improve placeholder detection and answer post-processing.
- Expand retrieval evaluation with precision, recall, and relevance metrics.
- Add CI-gated deployment so production deployment occurs only after tests pass.
- Add distributed tracing and deeper observability.

## Dataset Attribution

This project uses the [Bitext Customer Support LLM Chatbot Training Dataset](https://github.com/bitext/customer-support-llm-chatbot-training-dataset). Refer to the dataset repository for its license and terms. The raw dataset and generated embeddings are not included in this repository.
