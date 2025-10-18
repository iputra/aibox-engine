"""
AIBox Engine - FastAPI Application

FastAPI-based service for LLM inference, document indexing, and authentication system with GPU acceleration.
"""

from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.responses import HTMLResponse
from scalar_fastapi import get_scalar_api_reference

from app.api import auth, documents

# Core modules
from app.core.database import init_db, test_db_connection


@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    Manage application lifecycle events.

    This function handles startup and shutdown events for the FastAPI application.
    """
    # Startup
    print("🚀 AIBox Engine is starting up...")

    # Test database connection
    if await test_db_connection():
        # Initialize database tables
        await init_db()
        print("✅ Database initialized successfully")
    else:
        print("❌ Failed to connect to database")

    yield

    # Shutdown
    print("🛑 AIBox Engine is shutting down...")


# Create FastAPI application instance
app = FastAPI(
    title="AIBox Engine",
    description="FastAPI-based service for LLM inference, document indexing, and authentication system with GPU acceleration",
    version="1.0.0",
    docs_url=None,  # Disable classic Swagger UI
    redoc_url=None,  # Disable ReDoc
    lifespan=lifespan,
)


@app.get("/", response_class=HTMLResponse)
async def root():
    """
    Root endpoint with welcome message.

    Returns:
        HTMLResponse: Welcome message with navigation
    """
    return """
    <!DOCTYPE html>
    <html>
    <head>
        <title>AIBox Engine</title>
        <style>
            body { font-family: Arial, sans-serif; max-width: 800px; margin: 50px auto; padding: 20px; }
            .header { text-align: center; color: #333; }
            .links { margin-top: 30px; }
            .link { display: block; margin: 10px 0; padding: 10px; background: #f5f5f5; text-decoration: none; color: #333; border-radius: 5px; }
            .link:hover { background: #e0e0e0; }
        </style>
    </head>
    <body>
        <div class="header">
            <h1>🚀 AIBox Engine</h1>
            <p>FastAPI-based service for LLM inference, document indexing, and authentication system with GPU acceleration</p>
        </div>
        <div class="links">
            <a href="/docs" class="link">📚 API Documentation (Scalar)</a>
            <a href="/health" class="link">❤️ Health Check</a>
            <a href="/info" class="link">ℹ️ App Info</a>
        </div>
    </body>
    </html>
    """


@app.get("/docs", response_class=HTMLResponse)
async def scalar_docs():
    """
    Scalar API documentation endpoint.

    Returns:
        HTMLResponse: Scalar API documentation interface
    """
    return get_scalar_api_reference(
        openapi_url=app.openapi_url,
        title=app.title,
    )


@app.get("/health")
async def health_check():
    """
    Basic health check endpoint.

    Returns:
        dict: Health status of the service
    """
    return {"status": "healthy", "service": "AIBox Engine", "version": "1.0.0"}


@app.get("/info")
async def app_info():
    """
    Application information endpoint.

    Returns:
        dict: Basic information about the application
    """
    return {
        "name": app.title,
        "description": app.description,
        "version": "1.0.0",
        "docs_url": "/docs",
        "docs_type": "Scalar API Documentation",
        "health_check": "/health",
        "welcome_page": "/",
    }


# Include API routes
app.include_router(auth.router, prefix="/api/v1/auth")
app.include_router(documents.router, prefix="/api/v1")

# Placeholder for future route includes
# app.include_router(llm.router, prefix="/api/v1/llm", tags=["LLM Service"])
# app.include_router(search.router, prefix="/api/v1/search", tags=["Search & RAG"])


if __name__ == "__main__":
    import uvicorn

    # Development server configuration
    uvicorn.run(
        "app.main:app", host="0.0.0.0", port=8000, reload=True, log_level="info"
    )
