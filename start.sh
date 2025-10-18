#!/bin/bash

# AIBox Engine Startup Script
# This script starts the FastAPI application with scalar-fastapi integration

echo "🚀 Starting AIBox Engine..."

# Activate virtual environment
source venv/bin/activate

# Start the FastAPI application
echo "📋 Starting FastAPI server on http://localhost:8000"
echo "🏠 Welcome page: http://localhost:8000"
echo "📚 API Documentation (Scalar): http://localhost:8000/docs"
echo "❤️ Health Check: http://localhost:8000/health"
echo "ℹ️ App Info: http://localhost:8000/info"
echo ""

# Run the application
uvicorn app.main:app --host 0.0.0.0 --port 8000 --reload