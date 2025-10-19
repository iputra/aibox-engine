"""
Document API endpoints for AIBox Engine.
"""

import logging
from typing import List

from fastapi import APIRouter, Depends, HTTPException, UploadFile, File, Form, Query
from sqlalchemy.ext.asyncio import AsyncSession

from app.api.auth import get_current_user
from app.core.database import get_db
from app.models.schemas import (
    DocumentResponse,
    DocumentUpdate,
    FileUploadResponse,
    SearchQuery,
    SearchResponse,
    UserResponse,
)
from app.services.document_service import DocumentService

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/documents", tags=["documents"])

# Initialize document service
document_service = DocumentService()


@router.post("/upload", response_model=FileUploadResponse)
async def upload_document(
    file: UploadFile = File(...),
    description: str = Form(None),
    current_user: UserResponse = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
):
    """
    Upload a new document for processing and indexing.

    Supported file types: PDF, DOCX, TXT, CSV, Markdown, HTML
    Maximum file size: 50MB
    """
    try:
        # Validate file
        if not file.filename:
            raise HTTPException(status_code=400, detail="No filename provided")

        # Check file size (50MB limit)
        if file.size and file.size > 50 * 1024 * 1024:
            raise HTTPException(status_code=400, detail="File too large. Maximum size is 50MB")

        # Check file type
        file_ext = file.filename.split('.')[-1].lower()
        if not document_service.text_extractor.is_supported_file_type(file_ext):
            supported_types = ", ".join(document_service.text_extractor.get_supported_file_types())
            raise HTTPException(
                status_code=400,
                detail=f"Unsupported file type. Supported types: {supported_types}",
            )

        # Read file content
        file_content = await file.read()
        if not file_content:
            raise HTTPException(status_code=400, detail="Empty file")

        # Create and process document
        document = await document_service.create_document(
            db=db,
            file_content=file_content,
            filename=file.filename,
            content_type=file.content_type or "application/octet-stream",
            user_id=current_user.id,
            description=description,
        )

        if not document:
            raise HTTPException(status_code=500, detail="Failed to process document")

        return FileUploadResponse(
            message="Document uploaded and processed successfully",
            document_id=document.id,
            filename=document.original_filename,
            file_size=document.file_size,
            file_type=document.file_type,
            processing_status="completed",
        )

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error uploading document: {str(e)}")
        raise HTTPException(status_code=500, detail="Internal server error")


@router.get("/", response_model=List[DocumentResponse])
async def get_documents(
    skip: int = Query(0, ge=0),
    limit: int = Query(100, ge=1, le=1000),
    current_user: UserResponse = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
):
    """Get all documents for the current user."""
    try:
        documents = await document_service.get_user_documents(
            db=db,
            user_id=current_user.id,
            skip=skip,
            limit=limit,
        )
        return documents

    except Exception as e:
        logger.error(f"Error getting documents: {str(e)}")
        raise HTTPException(status_code=500, detail="Internal server error")


@router.get("/{document_id}", response_model=DocumentResponse)
async def get_document(
    document_id: int,
    current_user: UserResponse = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
):
    """Get a specific document by ID."""
    try:
        document = await document_service.get_document(
            db=db,
            document_id=document_id,
            user_id=current_user.id,
        )

        if not document:
            raise HTTPException(status_code=404, detail="Document not found")

        return document

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error getting document {document_id}: {str(e)}")
        raise HTTPException(status_code=500, detail="Internal server error")


@router.put("/{document_id}", response_model=DocumentResponse)
async def update_document(
    document_id: int,
    update_data: DocumentUpdate,
    current_user: UserResponse = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
):
    """Update document metadata."""
    try:
        document = await document_service.update_document(
            db=db,
            document_id=document_id,
            user_id=current_user.id,
            update_data=update_data,
        )

        if not document:
            raise HTTPException(status_code=404, detail="Document not found")

        return document

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error updating document {document_id}: {str(e)}")
        raise HTTPException(status_code=500, detail="Internal server error")


@router.delete("/{document_id}")
async def delete_document(
    document_id: int,
    current_user: UserResponse = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
):
    """Delete a document."""
    try:
        success = await document_service.delete_document(
            db=db,
            document_id=document_id,
            user_id=current_user.id,
        )

        if not success:
            raise HTTPException(status_code=404, detail="Document not found")

        return {"message": "Document deleted successfully"}

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error deleting document {document_id}: {str(e)}")
        raise HTTPException(status_code=500, detail="Internal server error")


@router.post("/search", response_model=SearchResponse)
async def search_documents(
    search_query: SearchQuery,
    current_user: UserResponse = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
):
    """
    Search documents using semantic similarity.

    This endpoint uses vector embeddings to find the most relevant
    document chunks based on semantic similarity to the query.
    """
    try:
        # Check if embedding service is available
        if not document_service.embedding_service.is_model_loaded():
            raise HTTPException(
                status_code=503,
                detail="Search service unavailable. Embedding model not loaded.",
            )

        results = await document_service.search_documents(
            db=db,
            user_id=current_user.id,
            search_query=search_query,
        )

        if results is None:
            raise HTTPException(status_code=500, detail="Search failed")

        return results

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error searching documents: {str(e)}")
        raise HTTPException(status_code=500, detail="Internal server error")


@router.get("/search/supported-formats")
async def get_supported_formats():
    """Get list of supported file formats for upload."""
    return {
        "supported_formats": document_service.text_extractor.get_supported_file_types(),
        "max_file_size": "50MB",
        "chunk_size": document_service.chunk_size,
        "embedding_dimension": document_service.embedding_service.get_embedding_dimension(),
    }


@router.get("/health/status")
async def get_service_health():
    """Get health status of document processing services."""
    return {
        "text_extraction": "available",
        "text_chunking": "available",
        "embedding_service": (
            "available" if document_service.embedding_service.is_model_loaded() else "unavailable"
        ),
        "model_path": document_service.embedding_service.model_path,
        "embedding_dimension": document_service.embedding_service.get_embedding_dimension(),
    }