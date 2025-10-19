"""
Document processing and indexing service.
"""

import logging
import os
import uuid
from datetime import datetime
from typing import List, Optional

from sqlalchemy import delete, select
from sqlalchemy.ext.asyncio import AsyncSession

from app.models.document import Document, DocumentChunk
from app.models.schemas import (
    AdvancedSearchQuery,
    AdvancedSearchResponse,
    AdvancedSearchResult,
    DocumentCreate,
    DocumentResponse,
    DocumentUpdate,
    KeywordSearchQuery,
    SearchFilters,
    SearchQuery,
    SearchResponse,
    SearchResult,
    SearchType,
    SortOrder,
)
from app.services.embedding_service import EmbeddingService
from app.services.text_chunking import TextChunkingService
from app.services.text_extraction import TextExtractionService


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

    def _build_filter_conditions(self, filters: Optional[SearchFilters]) -> List[str]:
        """Build SQL filter conditions based on filters."""
        conditions = []

        if not filters:
            return conditions

        if filters.file_types:
            file_type_list = "', '".join(filters.file_types)
            conditions.append(f"d.file_type IN ('{file_type_list}')")

        if filters.date_from:
            date_from_str = filters.date_from.strftime("%Y-%m-%d %H:%M:%S")
            conditions.append(f"d.created_at >= '{date_from_str}'")

        if filters.date_to:
            date_to_str = filters.date_to.strftime("%Y-%m-%d %H:%M:%S")
            conditions.append(f"d.created_at <= '{date_to_str}'")

        if filters.min_file_size:
            conditions.append(f"d.file_size >= {filters.min_file_size}")

        if filters.max_file_size:
            conditions.append(f"d.file_size <= {filters.max_file_size}")

        if filters.filename_pattern:
            conditions.append(f"d.original_filename LIKE '{filters.filename_pattern}'")

        return conditions

    def _build_sort_clause(self, sort_by: SortOrder, search_type: SearchType) -> str:
        """Build SQL sort clause based on sort order and search type."""
        if sort_by == SortOrder.RELEVANCE:
            if search_type == SearchType.SEMANTIC:
                return "ORDER BY similarity_score DESC"
            elif search_type == SearchType.KEYWORD:
                return "ORDER BY keyword_score DESC"
            else:  # HYBRID
                return "ORDER BY hybrid_score DESC"
        elif sort_by == SortOrder.DATE_ASC:
            return "ORDER BY d.created_at ASC"
        elif sort_by == SortOrder.DATE_DESC:
            return "ORDER BY d.created_at DESC"
        elif sort_by == SortOrder.FILENAME_ASC:
            return "ORDER BY d.original_filename ASC"
        elif sort_by == SortOrder.FILENAME_DESC:
            return "ORDER BY d.original_filename DESC"
        elif sort_by == SortOrder.FILE_SIZE_ASC:
            return "ORDER BY d.file_size ASC"
        elif sort_by == SortOrder.FILE_SIZE_DESC:
            return "ORDER BY d.file_size DESC"
        else:
            return "ORDER BY similarity_score DESC"

    async def search_documents_keyword(
        self,
        db: AsyncSession,
        user_id: int,
        search_query: KeywordSearchQuery,
    ) -> Optional[AdvancedSearchResponse]:
        """
        Search documents using keyword-based search with PostgreSQL full-text search.

        Args:
            db: Database session
            user_id: User ID
            search_query: Keyword search query parameters

        Returns:
            Advanced search response or None if search fails
        """
        try:
            import time

            from sqlalchemy import text
            start_time = time.time()

            # Prepare query terms
            query_terms = search_query.query.strip()
            if search_query.use_fuzzy_search:
                # Use fuzzy matching with trigram similarity
                query_conditions = [
                    f"(similarity(dc.content, '{query_terms}') > 0.2 OR similarity(d.original_filename, '{query_terms}') > 0.3)"
                ]
            else:
                # Use simple text matching
                query_conditions = [f"dc.content ILIKE '%{query_terms}%' OR d.original_filename ILIKE '%{query_terms}%'"]

            # Build filter conditions
            filter_conditions = self._build_filter_conditions(search_query.filters)
            all_conditions = query_conditions + filter_conditions

            # Build WHERE clause
            where_clause = " AND ".join([
                f"d.user_id = {user_id}",
                "d.is_active = true"
            ] + all_conditions)

            # Build sort clause
            sort_clause = self._build_sort_clause(search_query.sort_by, SearchType.KEYWORD)

            # Build full query
            query = f"""
            SELECT
                dc.id as chunk_id,
                dc.content as chunk_content,
                dc.chunk_index,
                d.id as document_id,
                d.filename as document_filename,
                d.original_filename,
                d.file_type,
                d.file_size,
                d.created_at as document_created_at,
                CASE
                    WHEN dc.content ILIKE '%{query_terms}%' THEN 1.0
                    WHEN similarity(dc.content, '{query_terms}') > 0.2 THEN similarity(dc.content, '{query_terms}')
                    ELSE 0.0
                END as keyword_score
            FROM document_chunks dc
            JOIN documents d ON dc.document_id = d.id
            WHERE {where_clause}
            {sort_clause}
            LIMIT {search_query.limit}
            OFFSET {search_query.offset}
            """

            result = await db.execute(text(query))
            rows = result.fetchall()

            # Convert to advanced search results
            results = []
            for row in rows:
                results.append(AdvancedSearchResult(
                    document_id=row.document_id,
                    document_filename=row.original_filename or row.document_filename,
                    document_title=None,
                    document_file_type=row.file_type,
                    document_file_size=row.file_size,
                    document_created_at=row.document_created_at,
                    chunk_id=row.chunk_id,
                    chunk_content=row.chunk_content if search_query.include_content else None,
                    chunk_index=row.chunk_index,
                    similarity_score=None,
                    keyword_score=float(row.keyword_score),
                    hybrid_score=None,
                    match_type=SearchType.KEYWORD,
                ))

            search_time = time.time() - start_time

            return AdvancedSearchResponse(
                query=search_query.query,
                search_type=SearchType.KEYWORD,
                total_results=len(results),
                results=results,
                search_time=search_time,
                has_more=len(results) == search_query.limit,
                limit=search_query.limit,
                offset=search_query.offset,
            )

        except Exception as e:
            logger.error(f"Error performing keyword search: {str(e)}")
            return None

    async def search_documents_advanced(
        self,
        db: AsyncSession,
        user_id: int,
        search_query: AdvancedSearchQuery,
    ) -> Optional[AdvancedSearchResponse]:
        """
        Search documents using advanced search with semantic, keyword, or hybrid approach.

        Args:
            db: Database session
            user_id: User ID
            search_query: Advanced search query parameters

        Returns:
            Advanced search response or None if search fails
        """
        try:
            import time

            start_time = time.time()

            if search_query.search_type in [SearchType.SEMANTIC, SearchType.HYBRID]:
                if not self.embedding_service.is_model_loaded():
                    logger.error("Embedding model not loaded, cannot perform semantic search")
                    return None

            results = []

            if search_query.search_type == SearchType.SEMANTIC:
                # Perform semantic search only
                results = await self._perform_semantic_search(db, user_id, search_query, SearchType.SEMANTIC)

            elif search_query.search_type == SearchType.KEYWORD:
                # Perform keyword search only
                results = await self._perform_keyword_search(db, user_id, search_query, SearchType.KEYWORD)

            elif search_query.search_type == SearchType.HYBRID:
                # Perform both semantic and keyword search, then combine results
                semantic_results = await self._perform_semantic_search(db, user_id, search_query, SearchType.SEMANTIC)
                keyword_results = await self._perform_keyword_search(db, user_id, search_query, SearchType.KEYWORD)

                # Combine and deduplicate results
                results = self._combine_hybrid_results(
                    semantic_results,
                    keyword_results,
                    search_query.similarity_weight
                )

            # Apply sorting
            if search_query.sort_by != SortOrder.RELEVANCE:
                results = self._sort_results(results, search_query.sort_by)

            # Apply pagination
            total_results = len(results)
            paginated_results = results[search_query.offset:search_query.offset + search_query.limit]

            search_time = time.time() - start_time

            return AdvancedSearchResponse(
                query=search_query.query,
                search_type=search_query.search_type,
                total_results=total_results,
                results=paginated_results,
                search_time=search_time,
                has_more=total_results > search_query.offset + search_query.limit,
                limit=search_query.limit,
                offset=search_query.offset,
            )

        except Exception as e:
            logger.error(f"Error performing advanced search: {str(e)}")
            return None

    async def _perform_semantic_search(
        self,
        db: AsyncSession,
        user_id: int,
        search_query: AdvancedSearchQuery,
        search_type: SearchType,
    ) -> List[AdvancedSearchResult]:
        """Perform semantic search and return advanced results."""
        try:
            from sqlalchemy import text

            # Generate embedding for search query
            query_embedding = self.embedding_service.generate_embedding(search_query.query)
            if not query_embedding:
                logger.error("Failed to generate embedding for search query")
                return []

            # Perform vector similarity search
            embedding_str = "[" + ",".join(map(str, query_embedding)) + "]"

            # Build filter conditions
            filter_conditions = self._build_filter_conditions(search_query.filters)

            # Build WHERE clause
            where_clause = " AND ".join([
                f"d.user_id = {user_id}",
                "d.is_active = true",
                f"1 - (dc.embedding <=> '{embedding_str}'::vector) > {search_query.threshold}"
            ] + filter_conditions)

            # Build sort clause (default to relevance for semantic)
            if search_query.sort_by == SortOrder.RELEVANCE:
                sort_clause = "ORDER BY similarity_score DESC"
            else:
                sort_clause = self._build_sort_clause(search_query.sort_by, search_type)

            query = f"""
            SELECT
                dc.id as chunk_id,
                dc.content as chunk_content,
                dc.chunk_index,
                d.id as document_id,
                d.filename as document_filename,
                d.original_filename,
                d.file_type,
                d.file_size,
                d.created_at as document_created_at,
                1 - (dc.embedding <=> '{embedding_str}'::vector) as similarity_score
            FROM document_chunks dc
            JOIN documents d ON dc.document_id = d.id
            WHERE {where_clause}
            {sort_clause}
            LIMIT 100  -- Get more results for combination in hybrid search
            """

            result = await db.execute(text(query))
            rows = result.fetchall()

            # Convert to advanced search results
            results = []
            for row in rows:
                results.append(AdvancedSearchResult(
                    document_id=row.document_id,
                    document_filename=row.original_filename or row.document_filename,
                    document_title=None,
                    document_file_type=row.file_type,
                    document_file_size=row.file_size,
                    document_created_at=row.document_created_at,
                    chunk_id=row.chunk_id,
                    chunk_content=row.chunk_content if search_query.include_content else None,
                    chunk_index=row.chunk_index,
                    similarity_score=float(row.similarity_score),
                    keyword_score=None,
                    hybrid_score=None,
                    match_type=search_type,
                ))

            return results

        except Exception as e:
            logger.error(f"Error performing semantic search: {str(e)}")
            return []

    async def _perform_keyword_search(
        self,
        db: AsyncSession,
        user_id: int,
        search_query: AdvancedSearchQuery,
        search_type: SearchType,
    ) -> List[AdvancedSearchResult]:
        """Perform keyword search and return advanced results."""
        try:
            from sqlalchemy import text

            # Prepare query terms
            query_terms = search_query.query.strip()
            query_conditions = [f"dc.content ILIKE '%{query_terms}%' OR d.original_filename ILIKE '%{query_terms}%'"]

            # Build filter conditions
            filter_conditions = self._build_filter_conditions(search_query.filters)

            # Build WHERE clause
            where_clause = " AND ".join([
                f"d.user_id = {user_id}",
                "d.is_active = true"
            ] + query_conditions + filter_conditions)

            # Build sort clause (default to relevance for keyword)
            if search_query.sort_by == SortOrder.RELEVANCE:
                sort_clause = "ORDER BY keyword_score DESC"
            else:
                sort_clause = self._build_sort_clause(search_query.sort_by, search_type)

            query = f"""
            SELECT
                dc.id as chunk_id,
                dc.content as chunk_content,
                dc.chunk_index,
                d.id as document_id,
                d.filename as document_filename,
                d.original_filename,
                d.file_type,
                d.file_size,
                d.created_at as document_created_at,
                CASE
                    WHEN dc.content ILIKE '%{query_terms}%' THEN 1.0
                    ELSE 0.0
                END as keyword_score
            FROM document_chunks dc
            JOIN documents d ON dc.document_id = d.id
            WHERE {where_clause}
            {sort_clause}
            LIMIT 100  -- Get more results for combination in hybrid search
            """

            result = await db.execute(text(query))
            rows = result.fetchall()

            # Convert to advanced search results
            results = []
            for row in rows:
                results.append(AdvancedSearchResult(
                    document_id=row.document_id,
                    document_filename=row.original_filename or row.document_filename,
                    document_title=None,
                    document_file_type=row.file_type,
                    document_file_size=row.file_size,
                    document_created_at=row.document_created_at,
                    chunk_id=row.chunk_id,
                    chunk_content=row.chunk_content if search_query.include_content else None,
                    chunk_index=row.chunk_index,
                    similarity_score=None,
                    keyword_score=float(row.keyword_score),
                    hybrid_score=None,
                    match_type=search_type,
                ))

            return results

        except Exception as e:
            logger.error(f"Error performing keyword search: {str(e)}")
            return []

    def _combine_hybrid_results(
        self,
        semantic_results: List[AdvancedSearchResult],
        keyword_results: List[AdvancedSearchResult],
        similarity_weight: float,
    ) -> List[AdvancedSearchResult]:
        """Combine semantic and keyword search results for hybrid search."""
        from collections import defaultdict

        # Create a dictionary to store results by document-chunk combination
        combined_results = defaultdict(lambda: {
            'semantic_score': 0.0,
            'keyword_score': 0.0,
            'result': None
        })

        # Process semantic results
        for result in semantic_results:
            key = (result.document_id, result.chunk_id)
            combined_results[key]['semantic_score'] = result.similarity_score or 0.0
            if combined_results[key]['result'] is None:
                combined_results[key]['result'] = result

        # Process keyword results
        for result in keyword_results:
            key = (result.document_id, result.chunk_id)
            combined_results[key]['keyword_score'] = result.keyword_score or 0.0
            if combined_results[key]['result'] is None:
                combined_results[key]['result'] = result

        # Calculate hybrid scores and create final results
        final_results = []
        keyword_weight = 1.0 - similarity_weight

        for key, data in combined_results.items():
            result = data['result']
            semantic_score = data['semantic_score']
            keyword_score = data['keyword_score']

            # Calculate hybrid score (weighted average)
            hybrid_score = (semantic_score * similarity_weight + keyword_score * keyword_weight)

            # Only include results with at least one match
            if semantic_score > 0 or keyword_score > 0:
                result.hybrid_score = hybrid_score
                result.similarity_score = semantic_score if semantic_score > 0 else None
                result.keyword_score = keyword_score if keyword_score > 0 else None
                result.match_type = SearchType.HYBRID
                final_results.append(result)

        # Sort by hybrid score descending
        final_results.sort(key=lambda x: x.hybrid_score or 0, reverse=True)

        return final_results

    def _sort_results(self, results: List[AdvancedSearchResult], sort_by: SortOrder) -> List[AdvancedSearchResult]:
        """Sort results based on the specified sort order."""
        if sort_by == SortOrder.DATE_ASC:
            results.sort(key=lambda x: x.document_created_at)
        elif sort_by == SortOrder.DATE_DESC:
            results.sort(key=lambda x: x.document_created_at, reverse=True)
        elif sort_by == SortOrder.FILENAME_ASC:
            results.sort(key=lambda x: x.document_filename.lower())
        elif sort_by == SortOrder.FILENAME_DESC:
            results.sort(key=lambda x: x.document_filename.lower(), reverse=True)
        elif sort_by == SortOrder.FILE_SIZE_ASC:
            results.sort(key=lambda x: x.document_file_size)
        elif sort_by == SortOrder.FILE_SIZE_DESC:
            results.sort(key=lambda x: x.document_file_size, reverse=True)
        # For RELEVANCE, results are already sorted by score

        return results
