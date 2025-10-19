"""
Document API endpoints for AIBox Engine.
"""

import logging
from typing import List

from fastapi import APIRouter, Depends, File, Form, HTTPException, Query, UploadFile
from sqlalchemy.ext.asyncio import AsyncSession

from app.api.auth import get_current_user
from app.core.database import get_db
from app.models.schemas import (
    AdvancedSearchQuery,
    AdvancedSearchResponse,
    DocumentResponse,
    DocumentUpdate,
    FileUploadResponse,
    KeywordSearchQuery,
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


# Advanced Search Endpoints
@router.post("/search/advanced", response_model=AdvancedSearchResponse)
async def search_documents_advanced(
    search_query: AdvancedSearchQuery,
    current_user: UserResponse = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
):
    """
    Advanced search documents with semantic, keyword, or hybrid approach.

    Features:
    - Semantic search: Vector-based similarity search
    - Keyword search: Text-based search with fuzzy matching
    - Hybrid search: Combines semantic and keyword search with weighted scores
    - Advanced filtering: Date range, file types, file size, filename patterns
    - Multiple sort options: Relevance, date, filename, file size
    - Pagination support with offset/limit
    """
    try:
        results = await document_service.search_documents_advanced(
            db=db,
            user_id=current_user.id,
            search_query=search_query,
        )

        if results is None:
            raise HTTPException(status_code=500, detail="Advanced search failed")

        return results

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error performing advanced search: {str(e)}")
        raise HTTPException(status_code=500, detail="Internal server error")


@router.post("/search/keyword", response_model=AdvancedSearchResponse)
async def search_documents_keyword(
    search_query: KeywordSearchQuery,
    current_user: UserResponse = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
):
    """
    Search documents using keyword-based search with PostgreSQL text search.

    Features:
    - Case-insensitive text matching
    - Fuzzy search with trigram similarity (optional)
    - Advanced filtering: Date range, file types, file size, filename patterns
    - Multiple sort options: Relevance, date, filename, file size
    - Pagination support with offset/limit
    """
    try:
        results = await document_service.search_documents_keyword(
            db=db,
            user_id=current_user.id,
            search_query=search_query,
        )

        if results is None:
            raise HTTPException(status_code=500, detail="Keyword search failed")

        return results

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error performing keyword search: {str(e)}")
        raise HTTPException(status_code=500, detail="Internal server error")


@router.get("/search/filters/info")
async def get_search_filters_info():
    """Get information about available search filters and options."""
    return {
        "search_types": {
            "semantic": {
                "description": "Vector-based semantic similarity search",
                "requires_embedding_model": True,
                "best_for": "Concept-based queries, finding related content"
            },
            "keyword": {
                "description": "Text-based keyword search with optional fuzzy matching",
                "requires_embedding_model": False,
                "best_for": "Exact term matching, specific phrase searches"
            },
            "hybrid": {
                "description": "Combines semantic and keyword search with weighted scoring",
                "requires_embedding_model": True,
                "best_for": "Balanced search results, leveraging both approaches"
            }
        },
        "filters": {
            "file_types": {
                "description": "Filter by document file type",
                "supported_types": ["pdf", "docx", "txt", "csv", "md", "html"]
            },
            "date_range": {
                "description": "Filter documents by creation date",
                "format": "ISO 8601 datetime (YYYY-MM-DDTHH:MM:SS)"
            },
            "file_size": {
                "description": "Filter by file size in bytes",
                "min_max_filter": True
            },
            "filename_pattern": {
                "description": "Filter by filename using SQL LIKE pattern",
                "examples": ["%.pdf", "report_%", "2023%"]
            }
        },
        "sort_options": {
            "relevance": "Sort by search relevance score",
            "date_asc": "Sort by creation date (oldest first)",
            "date_desc": "Sort by creation date (newest first)",
            "filename_asc": "Sort by filename (A-Z)",
            "filename_desc": "Sort by filename (Z-A)",
            "file_size_asc": "Sort by file size (smallest first)",
            "file_size_desc": "Sort by file size (largest first)"
        },
        "pagination": {
            "limit": "Number of results to return (1-100, default: 10)",
            "offset": "Number of results to skip (default: 0)",
            "has_more": "Indicates if more results are available"
        },
        "scoring": {
            "similarity_score": "Semantic similarity score (0.0-1.0)",
            "keyword_score": "Keyword match score (0.0-1.0)",
            "hybrid_score": "Combined weighted score (0.0-1.0)",
            "similarity_weight": "Weight for semantic search in hybrid mode (0.0-1.0, default: 0.7)"
        }
    }
