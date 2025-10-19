"""
Chat service for AIBox Engine conversation system with document context.
"""

import logging
import secrets
import time
from datetime import datetime
from typing import Any, List, Optional

from sqlalchemy import delete, func, select, update
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.orm import selectinload

from app.models.chat import ChatFolder, ChatFolderMembership, ChatMessage, ChatSession
from app.models.schemas import (
    ChatFolderCreate,
    ChatFolderResponse,
    ChatMessageResponse,
    ChatSearchQuery,
    ChatSearchResponse,
    ChatSessionCreate,
    ChatSessionResponse,
    ChatSessionUpdate,
    ChatSessionWithMessages,
    DocumentCitation,
    MessageRole,
    SendMessageRequest,
    SendMessageResponse,
)
from app.services.document_service import DocumentService
from app.services.llm_service import LLMService


logger = logging.getLogger(__name__)


class ChatService:
    """Service for managing chat sessions and conversations with document context."""

    def __init__(self, document_service: Optional[DocumentService] = None):
        """
        Initialize chat service.

        Args:
            document_service: Service for document search operations
        """
        self.document_service = document_service or DocumentService()
        self.llm_service = LLMService()

    async def create_chat_session(
        self,
        db: AsyncSession,
        session_data: ChatSessionCreate,
        user_id: int,
    ) -> Optional[ChatSessionResponse]:
        """
        Create a new chat session.

        Args:
            db: Database session
            session_data: Chat session creation data
            user_id: User ID from authenticated request

        Returns:
            Created chat session or None if creation fails
        """
        try:
            # Auto-generate title if not provided
            title = session_data.title or f"Chat {datetime.now().strftime('%Y-%m-%d %H:%M')}"

            # Generate share token if public
            share_token = None
            if session_data.is_public:
                share_token = secrets.token_urlsafe(32)

            chat_session = ChatSession(
                user_id=user_id,
                title=title,
                persona_id=session_data.persona_id,
                temperature=session_data.temperature,
                max_tokens=session_data.max_tokens,
                system_prompt=session_data.system_prompt,
                include_document_context=session_data.include_document_context,
                max_document_references=session_data.max_document_references,
                document_search_threshold=session_data.document_search_threshold,
                is_public=session_data.is_public,
                share_token=share_token,
            )

            db.add(chat_session)
            await db.commit()
            await db.refresh(chat_session)

            return ChatSessionResponse.from_orm(chat_session)

        except Exception as e:
            logger.error(f"Error creating chat session: {str(e)}")
            await db.rollback()
            return None

    async def get_chat_session(
        self,
        db: AsyncSession,
        session_id: int,
        user_id: Optional[int] = None,
    ) -> Optional[ChatSessionResponse]:
        """
        Get chat session by ID.

        Args:
            db: Database session
            session_id: Chat session ID
            user_id: User ID for permission check

        Returns:
            Chat session or None if not found
        """
        try:
            query = select(ChatSession).where(ChatSession.id == session_id)

            if user_id:
                query = query.where(ChatSession.user_id == user_id)

            result = await db.execute(query)
            chat_session = result.scalar_one_or_none()

            if chat_session:
                return ChatSessionResponse.from_orm(chat_session)
            return None

        except Exception as e:
            logger.error(f"Error getting chat session {session_id}: {str(e)}")
            return None

    async def get_chat_session_with_messages(
        self,
        db: AsyncSession,
        session_id: int,
        user_id: Optional[int] = None,
    ) -> Optional[ChatSessionWithMessages]:
        """
        Get chat session with all its messages.

        Args:
            db: Database session
            session_id: Chat session ID
            user_id: User ID for permission check

        Returns:
            Chat session with messages or None if not found
        """
        try:
            query = select(ChatSession).options(
                selectinload(ChatSession.messages)
            ).where(ChatSession.id == session_id)

            if user_id:
                query = query.where(ChatSession.user_id == user_id)

            result = await db.execute(query)
            chat_session = result.scalar_one_or_none()

            if chat_session:
                messages = [
                    ChatMessageResponse.from_orm(msg)
                    for msg in sorted(chat_session.messages, key=lambda x: x.created_at)
                ]

                session_data = ChatSessionResponse.from_orm(chat_session)
                return ChatSessionWithMessages(
                    **session_data.dict(),
                    messages=messages
                )
            return None

        except Exception as e:
            logger.error(f"Error getting chat session with messages {session_id}: {str(e)}")
            return None

    async def update_chat_session(
        self,
        db: AsyncSession,
        session_id: int,
        update_data: ChatSessionUpdate,
        user_id: Optional[int] = None,
    ) -> Optional[ChatSessionResponse]:
        """
        Update chat session.

        Args:
            db: Database session
            session_id: Chat session ID
            update_data: Update data
            user_id: User ID for permission check

        Returns:
            Updated chat session or None if update fails
        """
        try:
            # Build update data dict, excluding None values
            update_dict = {
                k: v for k, v in update_data.dict().items()
                if v is not None
            }

            if update_dict:
                query = update(ChatSession).where(ChatSession.id == session_id)

                if user_id:
                    query = query.where(ChatSession.user_id == user_id)

                query = query.values(**update_dict, updated_at=datetime.utcnow())

                await db.execute(query)
                await db.commit()

            return await self.get_chat_session(db, session_id, user_id)

        except Exception as e:
            logger.error(f"Error updating chat session {session_id}: {str(e)}")
            await db.rollback()
            return None

    async def delete_chat_session(
        self,
        db: AsyncSession,
        session_id: int,
        user_id: Optional[int] = None,
    ) -> bool:
        """
        Delete chat session.

        Args:
            db: Database session
            session_id: Chat session ID
            user_id: User ID for permission check

        Returns:
            True if deleted successfully, False otherwise
        """
        try:
            query = delete(ChatSession).where(ChatSession.id == session_id)

            if user_id:
                query = query.where(ChatSession.user_id == user_id)

            result = await db.execute(query)
            await db.commit()

            return result.rowcount > 0

        except Exception as e:
            logger.error(f"Error deleting chat session {session_id}: {str(e)}")
            await db.rollback()
            return False

    async def get_user_chat_sessions(
        self,
        db: AsyncSession,
        user_id: int,
        limit: int = 20,
        offset: int = 0,
    ) -> List[ChatSessionResponse]:
        """
        Get user's chat sessions.

        Args:
            db: Database session
            user_id: User ID
            limit: Maximum number of sessions to return
            offset: Number of sessions to skip

        Returns:
            List of chat sessions
        """
        try:
            query = (
                select(ChatSession)
                .where(ChatSession.user_id == user_id, ChatSession.is_active == True)
                .order_by(ChatSession.last_message_at.desc().nulls_last())
                .limit(limit)
                .offset(offset)
            )

            result = await db.execute(query)
            sessions = result.scalars().all()

            return [ChatSessionResponse.from_orm(session) for session in sessions]

        except Exception as e:
            logger.error(f"Error getting user chat sessions: {str(e)}")
            return []

    async def search_chat_sessions(
        self,
        db: AsyncSession,
        user_id: int,
        search_query: ChatSearchQuery,
    ) -> Optional[ChatSearchResponse]:
        """
        Search user's chat sessions.

        Args:
            db: Database session
            user_id: User ID
            search_query: Search parameters

        Returns:
            Search response or None if search fails
        """
        try:
            # Build base query
            query = select(ChatSession).where(ChatSession.user_id == user_id, ChatSession.is_active == True)

            # Apply filters
            if search_query.query:
                query = query.where(ChatSession.title.ilike(f"%{search_query.query}%"))

            if search_query.date_from:
                query = query.where(ChatSession.created_at >= search_query.date_from)

            if search_query.date_to:
                query = query.where(ChatSession.created_at <= search_query.date_to)

            if search_query.is_public is not None:
                query = query.where(ChatSession.is_public == search_query.is_public)

            # Get total count
            count_query = select(func.count()).select_from(query.subquery())
            total_result = await db.execute(count_query)
            total_results = total_result.scalar()

            # Apply ordering and pagination
            query = query.order_by(ChatSession.last_message_at.desc().nulls_last())
            query = query.limit(search_query.limit).offset(search_query.offset)

            result = await db.execute(query)
            sessions = result.scalars().all()

            session_responses = [ChatSessionResponse.from_orm(session) for session in sessions]
            has_more = total_results > search_query.offset + search_query.limit

            return ChatSearchResponse(
                total_results=total_results,
                results=session_responses,
                has_more=has_more
            )

        except Exception as e:
            logger.error(f"Error searching chat sessions: {str(e)}")
            return None

    async def send_message(
        self,
        db: AsyncSession,
        user_id: int,
        message_request: SendMessageRequest,
    ) -> Optional[SendMessageResponse]:
        """
        Send a message and get AI response with document context.

        Args:
            db: Database session
            user_id: User ID
            message_request: Message request data

        Returns:
            Response with AI message and citations or None if fails
        """
        try:
            start_time = time.time()

            # Get or create chat session
            session = None
            if message_request.session_id:
                session = await self.get_chat_session(db, message_request.session_id, user_id)

            if not session:
                # Create new session
                session_data = ChatSessionCreate(
                    user_id=user_id,
                    title=self._generate_session_title(message_request.message),
                    include_document_context=message_request.search_documents,
                )
                session = await self.create_chat_session(db, session_data)

            if not session:
                logger.error("Failed to get or create chat session")
                return None

            # Save user message
            user_message = ChatMessage(
                session_id=session.id,
                role=MessageRole.USER,
                content=message_request.message,
            )
            db.add(user_message)
            await db.flush()

            # Search documents for context if enabled
            document_citations = []
            search_time = None

            if message_request.search_documents and session.include_document_context:
                search_start = time.time()
                search_results = await self._search_documents_for_context(
                    db, user_id, message_request.message, session
                )
                search_time = time.time() - search_start

                if search_results:
                    document_citations = self._extract_document_citations(search_results)

            # Generate AI response
            generation_start = time.time()
            ai_response = await self._generate_ai_response(
                message_request.message, document_citations, session
            )
            generation_time = time.time() - generation_start

            # Save AI response
            ai_message = ChatMessage(
                session_id=session.id,
                role=MessageRole.ASSISTANT,
                content=ai_response,
                document_references=[citation.dict() for citation in document_citations],
                processing_time=f"{generation_time:.2f}",
            )
            db.add(ai_message)

            # Update session
            await db.execute(
                update(ChatSession)
                .where(ChatSession.id == session.id)
                .values(
                    last_message_at=datetime.utcnow(),
                    message_count=ChatSession.message_count + 1,
                    updated_at=datetime.utcnow()
                )
            )

            await db.commit()

            total_time = time.time() - start_time

            return SendMessageResponse(
                session_id=session.id,
                message_id=ai_message.id,
                response=ai_response,
                document_citations=document_citations,
                search_time=search_time,
                generation_time=generation_time,
                total_time=total_time,
            )

        except Exception as e:
            logger.error(f"Error sending message: {str(e)}")
            await db.rollback()
            return None

    async def _search_documents_for_context(
        self,
        db: AsyncSession,
        user_id: int,
        query: str,
        session: ChatSessionResponse,
    ) -> Optional[List[Any]]:
        """Search documents for chat context (simplified for testing)."""
        try:
            # For now, return None to test LLM without document context
            # TODO: Integrate with basic document search
            logger.info(f"Document search temporarily disabled for query: {query}")
            return None

        except Exception as e:
            logger.error(f"Error searching documents for context: {str(e)}")
            return None

    def _extract_document_citations(self, search_results: List[Any]) -> List[DocumentCitation]:
        """Extract document citations from search results."""
        citations = []
        for result in search_results:
            citation = DocumentCitation(
                document_id=result.document_id,
                document_filename=result.document_filename,
                document_title=result.document_title,
                chunk_id=result.chunk_id,
                chunk_content=result.chunk_content,
                similarity_score=result.hybrid_score or result.similarity_score,
            )
            citations.append(citation)
        return citations

    async def _generate_ai_response(
        self,
        user_message: str,
        document_citations: List[DocumentCitation],
        session: ChatSessionResponse,
    ) -> str:
        """
        Generate AI response based on message and document context using LLM API.
        """
        try:
            # Build context from document citations
            context_text = ""
            if document_citations:
                context_text = "\n\nDocument Context:\n"
                for i, citation in enumerate(document_citations[:3], 1):
                    context_text += f"[{i}] From '{citation.document_filename}': {citation.chunk_content[:200]}...\n"

            # Build system prompt
            system_prompt = session.system_prompt or (
                "You are a helpful AI assistant. Use the provided document context to answer questions accurately. "
                "Cite your sources using the reference numbers in square brackets."
            )

            # Build context from document citations
            context_text = ""
            if document_citations:
                context_text = "Document Context:\n"
                for i, citation in enumerate(document_citations[:5], 1):
                    source_info = f"[{i}] From '{citation.document_filename}'"
                    if citation.similarity_score:
                        source_info += f" (relevance: {citation.similarity_score:.2f})"
                    context_text += f"{source_info}:\n{citation.chunk_content[:300]}...\n\n"

            # Build messages for LLM
            system_prompt = session.system_prompt or (
                "You are a helpful AI assistant with access to document context. "
                "Use the provided document information to answer questions accurately. "
                "Cite your sources using reference numbers in square brackets [1], [2], etc. "
                "If the documents don't contain relevant information, say so politely and provide general knowledge."
            )

            messages = self.llm_service.build_context_messages(
                user_message=user_message,
                document_context=context_text if document_citations else None,
                system_prompt=system_prompt,
            )

            # Generate response using LLM service
            response_data = await self.llm_service.generate_response(
                messages=messages,
                temperature=float(session.temperature),
                max_tokens=session.max_tokens,
                stream=False,
            )

            if response_data:
                response_text = self.llm_service.extract_response_text(response_data)
                if response_text:
                    return response_text

            # Fallback to placeholder if LLM fails
            logger.warning("LLM generation failed, using fallback response")
            if document_citations:
                return f"Based on the {len(document_citations)} documents I found, I can provide some information about '{user_message}'. However, I'm experiencing technical difficulties with the AI service. Please try again later."
                # Removed - replaced with direct return
                response += "The documents suggest relevant information that addresses your question.\n\n"
                response += "Please note: This is a placeholder response. In a production environment, this would connect to an actual AI model."
            else:
                response = f"I understand you're asking about '{user_message}'. "
                response += "I don't have specific document context for this question, but I can help discuss the topic generally. "
                response += "In a production environment, this would connect to an actual AI model for more detailed assistance."

            return response

        except Exception as e:
            logger.error(f"Error generating AI response: {str(e)}")
            return "I apologize, but I encountered an error while generating a response. Please try again."

    def _generate_session_title(self, first_message: str) -> str:
        """Generate a session title from the first message."""
        # Truncate and clean the first message
        title = first_message.strip()
        if len(title) > 50:
            title = title[:47] + "..."
        return title or "New Chat"

    async def create_folder(
        self,
        db: AsyncSession,
        folder_data: ChatFolderCreate,
    ) -> Optional[ChatFolderResponse]:
        """Create a new chat folder."""
        try:
            folder = ChatFolder(
                user_id=folder_data.user_id,
                name=folder_data.name,
                description=folder_data.description,
                color=folder_data.color,
                parent_id=folder_data.parent_id,
            )

            db.add(folder)
            await db.commit()
            await db.refresh(folder)

            return ChatFolderResponse.from_orm(folder)

        except Exception as e:
            logger.error(f"Error creating chat folder: {str(e)}")
            await db.rollback()
            return None

    async def add_chat_to_folder(
        self,
        db: AsyncSession,
        session_id: int,
        folder_id: int,
        user_id: int,
    ) -> bool:
        """Add chat session to folder."""
        try:
            # Verify ownership
            session_query = select(ChatSession).where(
                ChatSession.id == session_id, ChatSession.user_id == user_id
            )
            session_result = await db.execute(session_query)
            if not session_result.scalar_one_or_none():
                return False

            folder_query = select(ChatFolder).where(
                ChatFolder.id == folder_id, ChatFolder.user_id == user_id
            )
            folder_result = await db.execute(folder_query)
            if not folder_result.scalar_one_or_none():
                return False

            # Check if already in folder
            existing_query = select(ChatFolderMembership).where(
                ChatFolderMembership.session_id == session_id,
                ChatFolderMembership.folder_id == folder_id
            )
            existing_result = await db.execute(existing_query)
            if existing_result.scalar_one_or_none():
                return True  # Already in folder

            # Add membership
            membership = ChatFolderMembership(
                folder_id=folder_id,
                session_id=session_id,
            )
            db.add(membership)

            # Update folder chat count
            await db.execute(
                update(ChatFolder)
                .where(ChatFolder.id == folder_id)
                .values(chat_count=ChatFolder.chat_count + 1)
            )

            await db.commit()
            return True

        except Exception as e:
            logger.error(f"Error adding chat to folder: {str(e)}")
            await db.rollback()
            return False
