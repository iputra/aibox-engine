"""
Document processing and indexing service.
"""

import logging
import os
import uuid
from datetime import datetime
from typing import List, Optional

from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select, delete, func

from app.models.document import Document, DocumentChunk
from app.models.schemas import (
    DocumentCreate,
    DocumentUpdate,
    DocumentResponse,
    DocumentChunkCreate,
    SearchQuery,
    SearchResult,
    SearchResponse,
)
from app.services.text_extraction import TextExtractionService
from app.services.text_chunking import TextChunkingService
from app.services.embedding_service import EmbeddingService

logger = logging.getLogger(__name__)


class DocumentService:
    """Service for managing document processing and indexing."""

    def __init__(
        self,
        upload_dir: str = "uploads",
        chunk_size: int = 1000,
        chunk_overlap: int = 200,
    ):
        """
        Initialize document service.

        Args:
            upload_dir: Directory to store uploaded files
            chunk_size: Maximum size of text chunks
            chunk_overlap: Overlap between chunks
        """
        self.upload_dir = upload_dir
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap

        # Initialize sub-services
        self.text_extractor = TextExtractionService()
        self.text_chunker = TextChunkingService(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
        )
        self.embedding_service = EmbeddingService()

        # Create upload directory if it doesn't exist
        os.makedirs(upload_dir, exist_ok=True)

    async def create_document(
        self,
        db: AsyncSession,
        file_content: bytes,
        filename: str,
        content_type: str,
        user_id: int,
        description: Optional[str] = None,
    ) -> Optional[DocumentResponse]:
        """
        Create and process a new document.

        Args:
            db: Database session
            file_content: File content as bytes
            filename: Original filename
            content_type: MIME type of the file
            user_id: ID of the user uploading the file
            description: Optional document description

        Returns:
            Created document response or None if creation fails
        """
        try:
            # Generate unique filename
            file_ext = os.path.splitext(filename)[1].lower()
            unique_filename = f"{uuid.uuid4()}{file_ext}"
            file_path = os.path.join(self.upload_dir, unique_filename)

            # Save file
            with open(file_path, 'wb') as f:
                f.write(file_content)

            # Extract file type from extension
            file_type = file_ext.lstrip('.')

            # Create document record
            document_create = DocumentCreate(
                filename=unique_filename,
                original_filename=filename,
                file_type=file_type,
                file_size=len(file_content),
                content_type=content_type,
                file_path=file_path,
                user_id=user_id,
                description=description,
            )

            document = Document(**document_create.model_dump())
            db.add(document)
            await db.commit()
            await db.refresh(document)

            logger.info(f"Created document record: {document.id}")

            # Process document asynchronously (extract text, chunk, embed)
            await self._process_document(db, document)

            return DocumentResponse.from_orm(document)

        except Exception as e:
            logger.error(f"Error creating document: {str(e)}")
            await db.rollback()
            # Clean up file if document creation failed
            if os.path.exists(file_path):
                os.remove(file_path)
            return None

    async def _process_document(self, db: AsyncSession, document: Document):
        """
        Process document: extract text, chunk, and generate embeddings.

        Args:
            db: Database session
            document: Document to process
        """
        try:
            logger.info(f"Processing document {document.id}: {document.filename}")

            # Extract text from document
            text = self.text_extractor.extract_text(
                document.file_path,
                document.file_type,
            )

            if not text:
                logger.warning(f"No text extracted from document {document.id}")
                return

            logger.info(f"Extracted {len(text)} characters from document {document.id}")

            # Chunk text
            chunks_data = self.text_chunker.chunk_text_with_metadata(
                text,
                document.id,
            )

            logger.info(f"Created {len(chunks_data)} chunks from document {document.id}")

            # Generate embeddings for chunks
            if not self.embedding_service.is_model_loaded():
                logger.warning("Embedding model not loaded, skipping embedding generation")
                # Still save chunks with zero embeddings for later processing
                zero_embedding = [0.0] * self.embedding_service.get_embedding_dimension()
                chunks_without_embeddings = []
                for chunk_data in chunks_data:
                    chunk_data_copy = chunk_data.copy()
                    chunk_data_copy["embedding"] = zero_embedding
                    chunks_without_embeddings.append(chunk_data_copy)
                chunks_data = chunks_without_embeddings
            else:
                chunks_data = self.embedding_service.generate_embeddings_for_chunks(chunks_data)

            # Save chunks to database
            for chunk_data in chunks_data:
                chunk = DocumentChunk(**chunk_data)
                db.add(chunk)

            await db.commit()
            logger.info(f"Successfully processed document {document.id} with {len(chunks_data)} chunks")

        except Exception as e:
            logger.error(f"Error processing document {document.id}: {str(e)}")
            await db.rollback()

    async def get_user_documents(
        self,
        db: AsyncSession,
        user_id: int,
        skip: int = 0,
        limit: int = 100,
    ) -> List[DocumentResponse]:
        """
        Get documents for a user.

        Args:
            db: Database session
            user_id: User ID
            skip: Number of documents to skip
            limit: Maximum number of documents to return

        Returns:
            List of document responses
        """
        try:
            result = await db.execute(
                select(Document)
                .where(Document.user_id == user_id, Document.is_active == True)
                .order_by(Document.created_at.desc())
                .offset(skip)
                .limit(limit)
            )
            documents = result.scalars().all()
            return [DocumentResponse.from_orm(doc) for doc in documents]

        except Exception as e:
            logger.error(f"Error getting user documents: {str(e)}")
            return []

    async def get_document(
        self,
        db: AsyncSession,
        document_id: int,
        user_id: int,
    ) -> Optional[DocumentResponse]:
        """
        Get a specific document by ID.

        Args:
            db: Database session
            document_id: Document ID
            user_id: User ID (for authorization)

        Returns:
            Document response or None if not found
        """
        try:
            result = await db.execute(
                select(Document).where(
                    Document.id == document_id,
                    Document.user_id == user_id,
                    Document.is_active == True,
                )
            )
            document = result.scalar_one_or_none()
            return DocumentResponse.from_orm(document) if document else None

        except Exception as e:
            logger.error(f"Error getting document {document_id}: {str(e)}")
            return None

    async def update_document(
        self,
        db: AsyncSession,
        document_id: int,
        user_id: int,
        update_data: DocumentUpdate,
    ) -> Optional[DocumentResponse]:
        """
        Update document metadata.

        Args:
            db: Database session
            document_id: Document ID
            user_id: User ID (for authorization)
            update_data: Data to update

        Returns:
            Updated document response or None if not found
        """
        try:
            result = await db.execute(
                select(Document).where(
                    Document.id == document_id,
                    Document.user_id == user_id,
                    Document.is_active == True,
                )
            )
            document = result.scalar_one_or_none()

            if not document:
                return None

            # Update fields
            update_dict = update_data.model_dump(exclude_unset=True)
            for field, value in update_dict.items():
                setattr(document, field, value)

            document.updated_at = datetime.utcnow()
            await db.commit()
            await db.refresh(document)

            return DocumentResponse.from_orm(document)

        except Exception as e:
            logger.error(f"Error updating document {document_id}: {str(e)}")
            await db.rollback()
            return None

    async def delete_document(
        self,
        db: AsyncSession,
        document_id: int,
        user_id: int,
    ) -> bool:
        """
        Delete a document (soft delete).

        Args:
            db: Database session
            document_id: Document ID
            user_id: User ID (for authorization)

        Returns:
            True if deleted, False if not found
        """
        try:
            result = await db.execute(
                select(Document).where(
                    Document.id == document_id,
                    Document.user_id == user_id,
                    Document.is_active == True,
                )
            )
            document = result.scalar_one_or_none()

            if not document:
                return False

            # Soft delete
            document.is_active = False
            document.updated_at = datetime.utcnow()

            # Delete associated chunks
            await db.execute(
                delete(DocumentChunk).where(DocumentChunk.document_id == document_id)
            )

            await db.commit()
            logger.info(f"Deleted document {document_id}")

            # Optionally delete the file
            if os.path.exists(document.file_path):
                os.remove(document.file_path)

            return True

        except Exception as e:
            logger.error(f"Error deleting document {document_id}: {str(e)}")
            await db.rollback()
            return False

    async def search_documents(
        self,
        db: AsyncSession,
        user_id: int,
        search_query: SearchQuery,
    ) -> Optional[SearchResponse]:
        """
        Search documents using semantic similarity.

        Args:
            db: Database session
            user_id: User ID
            search_query: Search query parameters

        Returns:
            Search response or None if search fails
        """
        try:
            if not self.embedding_service.is_model_loaded():
                logger.error("Embedding model not loaded, cannot perform search")
                return None

            # Generate embedding for search query
            query_embedding = self.embedding_service.generate_embedding(search_query.query)
            if not query_embedding:
                logger.error("Failed to generate embedding for search query")
                return None

            import time
            from sqlalchemy import text
            start_time = time.time()

            # Perform vector similarity search using pgvector
            # Convert embedding to PostgreSQL vector string format
            embedding_str = "[" + ",".join(map(str, query_embedding)) + "]"

            # Use raw SQL with properly formatted vector string
            query = f"""
            SELECT
                dc.id as chunk_id,
                dc.content as chunk_content,
                dc.chunk_index,
                d.id as document_id,
                d.filename as document_filename,
                d.original_filename,
                1 - (dc.embedding <=> '{embedding_str}'::vector) as similarity_score
            FROM document_chunks dc
            JOIN documents d ON dc.document_id = d.id
            WHERE d.user_id = {user_id}
            AND d.is_active = true
            AND 1 - (dc.embedding <=> '{embedding_str}'::vector) > {search_query.threshold}
            ORDER BY similarity_score DESC
            LIMIT {search_query.limit}
            """

            result = await db.execute(text(query))

            rows = result.fetchall()

            # Convert to search results
            results = []
            for row in rows:
                results.append(SearchResult(
                    document_id=row.document_id,
                    document_filename=row.original_filename or row.document_filename,
                    document_title=None,  # Could be added later
                    chunk_id=row.chunk_id,
                    chunk_content=row.chunk_content,
                    similarity_score=float(row.similarity_score),
                    chunk_index=row.chunk_index,
                ))

            search_time = time.time() - start_time

            return SearchResponse(
                query=search_query.query,
                total_results=len(results),
                results=results,
                search_time=search_time,
            )

        except Exception as e:
            logger.error(f"Error searching documents: {str(e)}")
            return None