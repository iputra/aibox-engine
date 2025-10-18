# AIBox Engine

FastAPI-based service for LLM inference, document indexing, and authentication system with GPU acceleration.

## Overview

AIBox Engine adalah API service yang menyediakan:
- **Basic Authentication System**: User management dan JWT-based authentication
- **LLM Service**: Inference dengan llama.cpp menggunakan GPU Vulkan acceleration
- **Document Indexing**: PDF processing dan vector database dengan pgvector
- **Vector Search**: Semantic search pada indexed documents

## Architecture

```
┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐
│   FastAPI       │    │  PostgreSQL     │    │  llama.cpp      │
│   Application   │◄──►│  + pgvector     │    │  (Vulkan GPU)   │
│                 │    │                 │    │                 │
└─────────────────┘    └─────────────────┘    └─────────────────┘
         │                       │                       │
         ▼                       ▼                       ▼
┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐
│   Authentication│    │  Vector Storage │    │  LLM Inference  │
│   (JWT)         │    │  (Embeddings)   │    │  (Text Gen)     │
└─────────────────┘    └─────────────────┘    └─────────────────┘
```

## Features

### Authentication System
- User registration dan login
- JWT token-based authentication
- Role-based access control (RBAC)
- Password hashing dengan bcrypt

### LLM Service
- Integration dengan llama.cpp
- GPU acceleration menggunakan Vulkan
- Model loading dan management
- Text generation dengan streaming support
- Customizable generation parameters

### Document Processing
- PDF text extraction
- Text chunking dan preprocessing
- Embedding generation (dengan sentence-transformers)
- Vector storage di PostgreSQL dengan pgvector

### Vector Search
- Semantic search functionality
- Similarity scoring
- Hybrid search (semantic + keyword)
- RAG (Retrieval-Augmented Generation) support

## Requirements

### System Requirements
- Python 3.13+
- PostgreSQL 13+ dengan pgvector extension
- Vulkan-compatible GPU
- Sufficient RAM untuk model loading

### Python Dependencies
```python
# Core Framework
fastapi>=0.104.0
uvicorn[standard]>=0.24.0
pydantic>=2.5.0
python-multipart>=0.0.6

# Database
sqlalchemy>=2.0.0
asyncpg>=0.29.0
alembic>=1.13.0
pgvector>=0.2.0

# Authentication
python-jose[cryptography]>=3.3.0
passlib[bcrypt]>=1.7.4
python-decouple>=3.8

# Document Processing
PyPDF2>=3.0.1
python-docx>=1.1.0
sentence-transformers>=2.2.0
numpy>=1.24.0

# LLM Integration
llama-cpp-python>=0.2.0
```

## Installation

### 1. Clone Repository
```bash
git clone <repository-url>
cd aibox-engine
```

### 2. Setup Environment
```bash
# Create virtual environment
python3.13 -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

### 3. Database Setup
```bash
# Create PostgreSQL database
createdb aibox_engine

# Enable pgvector extension
psql aibox_engine -c "CREATE EXTENSION IF NOT EXISTS vector;"

# Run migrations
alembic upgrade head
```

### 4. Configuration
```bash
# Copy environment template
cp .env.example .env

# Edit .env with your configuration
DATABASE_URL=postgresql+asyncpg://user:password@localhost/aibox_engine
SECRET_KEY=your-secret-key-here
JWT_ALGORITHM=HS256
ACCESS_TOKEN_EXPIRE_MINUTES=30

# LLM Configuration
LLM_MODEL_PATH=/path/to/your/model.gguf
LLM_CONTEXT_SIZE=2048
LLM_GPU_LAYERS=-1  # -1 for all layers on GPU
```

### 5. Start Service
```bash
# Development server
uvicorn app.main:app --reload --host 0.0.0.0 --port 8000

# Production server
uvicorn app.main:app --host 0.0.0.0 --port 8000 --workers 4
```

## API Endpoints

### Authentication
- `POST /auth/register` - Register new user
- `POST /auth/login` - User login
- `POST /auth/refresh` - Refresh JWT token
- `GET /auth/me` - Get current user info

### LLM Service
- `POST /llm/generate` - Generate text response
- `GET /llm/models` - List available models
- `POST /llm/load` - Load specific model
- `GET /llm/status` - Get model status

### Document Management
- `POST /documents/upload` - Upload PDF document
- `GET /documents/` - List user documents
- `GET /documents/{doc_id}` - Get document details
- `DELETE /documents/{doc_id}` - Delete document
- `POST /documents/{doc_id}/index` - Index document for search

### Search & RAG
- `POST /search/semantic` - Semantic search in documents
- `POST /search/hybrid` - Hybrid semantic + keyword search
- `POST /search/rag` - RAG query with LLM context

## Usage Examples

### Authentication
```python
import requests

# Register
response = requests.post("http://localhost:8000/auth/register", json={
    "username": "john_doe",
    "email": "john@example.com",
    "password": "secure_password"
})

# Login
response = requests.post("http://localhost:8000/auth/login", json={
    "username": "john_doe",
    "password": "secure_password"
})
token = response.json()["access_token"]
```

### LLM Generation
```python
headers = {"Authorization": f"Bearer {token}"}

response = requests.post(
    "http://localhost:8000/llm/generate",
    json={
        "prompt": "Explain artificial intelligence",
        "max_tokens": 500,
        "temperature": 0.7
    },
    headers=headers
)
print(response.json()["generated_text"])
```

### Document Upload & Indexing
```python
# Upload PDF
with open("document.pdf", "rb") as f:
    response = requests.post(
        "http://localhost:8000/documents/upload",
        files={"file": f},
        headers=headers
    )
    doc_id = response.json()["id"]

# Index document
response = requests.post(
    f"http://localhost:8000/documents/{doc_id}/index",
    headers=headers
)
```

### Semantic Search
```python
response = requests.post(
    "http://localhost:8000/search/semantic",
    json={
        "query": "machine learning algorithms",
        "limit": 5,
        "threshold": 0.7
    },
    headers=headers
)

for result in response.json()["results"]:
    print(f"Document: {result['document_title']}")
    print(f"Score: {result['similarity_score']}")
    print(f"Content: {result['text_chunk'][:100]}...")
```

## Development

### Project Structure
```
aibox-engine/
├── app/
│   ├── api/
│   │   ├── auth.py
│   │   ├── llm.py
│   │   ├── documents.py
│   │   └── search.py
│   ├── core/
│   │   ├── auth.py
│   │   ├── database.py
│   │   └── security.py
│   ├── models/
│   │   ├── user.py
│   │   ├── document.py
│   │   └── vector.py
│   ├── services/
│   │   ├── llm_service.py
│   │   ├── document_service.py
│   │   └── search_service.py
│   └── main.py
├── alembic/
├── tests/
├── requirements.txt
├── .env.example
└── README.md
```

### Running Tests
```bash
# Install test dependencies
pip install pytest pytest-asyncio httpx

# Run tests
pytest

# Run with coverage
pytest --cov=app
```

### Database Migrations
```bash
# Create new migration
alembic revision --autogenerate -m "Add new feature"

# Apply migrations
alembic upgrade head

# Rollback migration
alembic downgrade -1
```

## Configuration Options

### LLM Configuration
- `LLM_MODEL_PATH`: Path to GGUF model file
- `LLM_CONTEXT_SIZE`: Maximum context window size
- `LLM_GPU_LAYERS`: Number of layers to offload to GPU
- `LLM_BATCH_SIZE`: Batch size for inference
- `LLM_TEMPERATURE`: Default generation temperature

### Database Configuration
- `DATABASE_URL`: PostgreSQL connection string
- `MAX_CONNECTIONS`: Maximum database connections
- `VECTOR_DIMENSION`: Embedding dimension size
- `CHUNK_SIZE`: Document chunking size
- `CHUNK_OVERLAP`: Overlap between chunks

### Authentication Configuration
- `SECRET_KEY`: JWT signing key
- `JWT_ALGORITHM`: Token signing algorithm
- `ACCESS_TOKEN_EXPIRE_MINUTES`: Token expiration time
- `REFRESH_TOKEN_EXPIRE_DAYS`: Refresh token expiration

## Monitoring & Logging

### Health Check Endpoints
- `GET /health` - Basic health check
- `GET /health/llm` - LLM service status
- `GET /health/database` - Database connectivity

### Logging
- Structured logging dengan JSON format
- Request/response logging
- Error tracking dan alerting
- Performance metrics

## Security Considerations

- Input validation dan sanitization
- SQL injection prevention
- Rate limiting pada API endpoints
- File upload security checks
- Secure password handling
- Token-based authentication
- CORS configuration

## Performance Optimization

- Database connection pooling
- Async/await patterns
- Vector indexing strategies
- Model caching
- Batch processing for documents
- GPU optimization for LLM

## Troubleshooting

### Common Issues
1. **Vulkan GPU not detected**: Ensure proper GPU drivers
2. **Model loading fails**: Check model format and paths
3. **Database connection errors**: Verify PostgreSQL setup
4. **Memory issues**: Adjust model context size

### Debug Mode
```bash
# Enable debug logging
export LOG_LEVEL=DEBUG
uvicorn app.main:app --reload --log-level debug
```

## Contributing

1. Fork repository
2. Create feature branch
3. Make changes with tests
4. Submit pull request

## License

[Add your license here]

## Support

[Add contact information or support links]