
# Customer Support RAG Chatbot

An AI-powered customer support chatbot that retrieves relevant FAQ entries using semantic search and generates grounded responses with a large language model.

## Features

- Semantic FAQ retrieval using Sentence Transformers.
- Vector search using Qdrant.
- Grounded answer generation using GroqCloud.
- LangGraph workflow with relevance checking and fallback responses.
- FastAPI backend with health and chat endpoints.
- Streamlit chat interface with retrieved FAQ details.
- Automated tests using pytest.
- Docker Compose configuration for the application services.

## Architecture

```text
                Customer
                   |
                   v
          Streamlit Frontend
               :8501
                   |
                   v
             FastAPI API
               :8000
                   |
                   v
            LangGraph Workflow
                   |
           +-------+-------+
           |               |
           v               v
     FAQ Retrieval    Relevance Check
           |               |
           v               v
      Qdrant DB       GroqCloud LLM
        :6333               |
           |                |
           +-------+--------+
                   |
                   v
             Chat Response
```

## Technology Stack

| Component | Technology |
|---|---|
| Programming language | Python |
| Embeddings | sentence-transformers/all-MiniLM-L6-v2 |
| Vector database | Qdrant |
| LLM provider | GroqCloud |
| LLM integration | LangChain |
| Workflow orchestration | LangGraph |
| Backend | FastAPI |
| Frontend | Streamlit |
| Testing | pytest |
| Containerization | Docker and Docker Compose |

## Dataset

The project uses the Bitext customer support dataset.

The original dataset contains 26,872 records, 27 intents, and 11 categories. A smaller 1,000-record subset is used for the current local vector-search setup.

Dataset files, generated embeddings, and local database storage are excluded from Git to keep the repository lightweight.

Download the dataset from the official source:

https://github.com/bitext/customer-support-llm-chatbot-training-dataset

Follow the project's preprocessing and embedding scripts to prepare the local data before indexing.

## Project Structure

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
│   └── setup_qdrant_collection.py
├── tests/
│   ├── test_api.py
│   └── test_graph.py
├── Dockerfile.api
├── Dockerfile.frontend
├── docker-compose.yml
├── requirements.txt
├── .env.example
├── .gitignore
└── README.md
```

## Prerequisites

- Python 3.11 recommended
- Docker Desktop
- A GroqCloud API key
- Git

## Local Setup

### 1. Clone the repository

```bash
git clone <YOUR_GITHUB_REPOSITORY_URL>
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
python -m pip install -r requirements.txt
```

### 4. Configure environment variables

Copy the example environment file:

```powershell
Copy-Item .env.example .env
```

Set your Groq API key and an available Groq model in `.env`.

Never commit `.env` or expose API keys publicly.

### 5. Start Qdrant

```powershell
docker compose up -d qdrant
```

### 6. Prepare the dataset and vector collection

Place the downloaded dataset at the input path expected by `scripts/inspect_dataset.py`.

Run the preprocessing and subset scripts in order:

```powershell
python scripts/inspect_dataset.py
python scripts/prepare_dataset.py
python scripts/create_subset.py
python scripts/generate_subset_embeddings.py
python scripts/setup_qdrant_collection.py
```

The embedding model is downloaded on first use. The indexing script expects Qdrant to be running locally.

### 7. Start the FastAPI backend

```powershell
$env:PYTHONPATH = (Get-Location).Path
uvicorn app.api.main:app --reload --host 127.0.0.1 --port 8000
```

API documentation:

http://127.0.0.1:8000/docs

Health endpoint:

http://127.0.0.1:8000/health

### 8. Start the Streamlit frontend

Open another terminal, activate the virtual environment, and run:

```powershell
python -m streamlit run frontend/streamlit_app.py
```

Open:

http://localhost:8501

## Running Tests

```powershell
$env:PYTHONPATH = (Get-Location).Path
python -m pytest tests -v
```

## Docker

Build and start the services:

```powershell
docker compose up --build
```

The Docker setup requires the Qdrant collection to be populated and the Groq API key to be configured. On a fresh machine, prepare the collection before expecting chat requests to succeed.

Local service addresses:

- Streamlit: http://localhost:8501
- FastAPI: http://localhost:8000
- FastAPI Swagger UI: http://localhost:8000/docs
- Qdrant: http://localhost:6333

## Limitations

- Responses depend on the relevance and quality of retrieved FAQ records.
- The dataset contains template placeholders that may appear in source answers.
- Similarity thresholds are initial settings and require further evaluation.
- The application is currently configured for local development.
- Public deployment, production monitoring, and tracing require additional setup.

## Future Improvements

- Improve relevance evaluation and threshold calibration.
- Add conversation feedback and retrieval analytics.
- Add LangSmith tracing.
- Improve answer validation and placeholder handling.
- Deploy the application publicly.
- Expand automated integration testing.

## License and Dataset Attribution

Review the dataset's original license and terms before redistribution. This repository does not include the raw dataset or generated embeddings.
