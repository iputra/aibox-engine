"""
Chat API endpoints for AIBox Engine conversation system.
"""

import logging
from typing import List

from fastapi import APIRouter, Depends, HTTPException, Query
from fastapi.responses import StreamingResponse
from sqlalchemy.ext.asyncio import AsyncSession

from app.api.auth import get_current_user
from app.core.database import get_db
from app.models.schemas import (
    AddChatToFolderRequest,
    ChatFolderCreate,
    ChatFolderResponse,
    ChatSearchQuery,
    ChatSearchResponse,
    ChatSessionCreate,
    ChatSessionResponse,
    ChatSessionUpdate,
    ChatSessionWithMessages,
    SendMessageRequest,
    SendMessageResponse,
    StreamingChunk,
    UserResponse,
)
from app.services.chat_service import ChatService
from app.services.llm_service import LLMService


logger = logging.getLogger(__name__)

router = APIRouter(prefix="/chat", tags=["chat"])

# Initialize chat service
chat_service = ChatService()


@router.post("/create-chat-session", response_model=ChatSessionResponse)
async def create_chat_session(
    session_data: ChatSessionCreate,
    current_user: UserResponse = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
):
    """
    Create a new chat session.

    Features:
    - Auto-generate title if not provided
    - Configure AI parameters (temperature, max tokens)
    - Enable/disable document context search
    - Set custom system prompts
    - Public sharing with share tokens
    """
    try:
        # Pass user_id directly to service method (ChatSessionCreate doesn't have user_id field)
        session = await chat_service.create_chat_session(db, session_data, current_user.id)
        if not session:
            raise HTTPException(status_code=500, detail="Failed to create chat session")

        return session

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error creating chat session: {str(e)}")
        raise HTTPException(status_code=500, detail="Internal server error")


@router.get("/chat-session/{session_id}", response_model=ChatSessionWithMessages)
async def get_chat_session(
    session_id: int,
    current_user: UserResponse = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
):
    """
    Get chat session with all messages.

    Returns complete conversation history with document citations
    and AI response metadata including processing times.
    """
    try:
        session = await chat_service.get_chat_session_with_messages(
            db, session_id, current_user.id
        )
        if not session:
            raise HTTPException(status_code=404, detail="Chat session not found")

        return session

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error getting chat session {session_id}: {str(e)}")
        raise HTTPException(status_code=500, detail="Internal server error")


@router.patch("/chat-session/{session_id}", response_model=ChatSessionResponse)
async def update_chat_session(
    session_id: int,
    update_data: ChatSessionUpdate,
    current_user: UserResponse = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
):
    """
    Update chat session settings.

    Can update:
    - Title and description
    - AI parameters (temperature, max tokens)
    - Document search settings
    - Public sharing status
    - Custom system prompts
    """
    try:
        session = await chat_service.update_chat_session(
            db, session_id, update_data, current_user.id
        )
        if not session:
            raise HTTPException(status_code=404, detail="Chat session not found")

        return session

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error updating chat session {session_id}: {str(e)}")
        raise HTTPException(status_code=500, detail="Internal server error")


@router.delete("/chat-session/{session_id}")
async def delete_chat_session(
    session_id: int,
    current_user: UserResponse = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
):
    """
    Delete chat session permanently.

    Deletes all messages and associated data. This action cannot be undone.
    """
    try:
        success = await chat_service.delete_chat_session(
            db, session_id, current_user.id
        )
        if not success:
            raise HTTPException(status_code=404, detail="Chat session not found")

        return {"message": "Chat session deleted successfully"}

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error deleting chat session {session_id}: {str(e)}")
        raise HTTPException(status_code=500, detail="Internal server error")


@router.get("/user-sessions", response_model=List[ChatSessionResponse])
async def get_user_chat_sessions(
    limit: int = Query(20, ge=1, le=100, description="Maximum number of sessions"),
    offset: int = Query(0, ge=0, description="Number of sessions to skip"),
    current_user: UserResponse = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
):
    """
    Get user's chat sessions.

    Returns paginated list of user's chat sessions ordered by last message time.
    Includes session metadata like message count and last activity.
    """
    try:
        sessions = await chat_service.get_user_chat_sessions(
            db, current_user.id, limit, offset
        )
        return sessions

    except Exception as e:
        logger.error(f"Error getting user chat sessions: {str(e)}")
        raise HTTPException(status_code=500, detail="Internal server error")


@router.post("/send-message")
async def send_message(
    message_request: SendMessageRequest,
    current_user: UserResponse = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
):
    """
    Send a message and get AI response with document context.

    Features:
    - Automatic session creation if session_id not provided
    - Document search for context (configurable)
    - AI response with citations from referenced documents
    - Processing time tracking (search, generation, total)
    - Multi-turn conversation support
    - Streaming support (configurable)

    Streaming Behavior:
    - If stream=true: Returns StreamingResponse with real-time chunks
    - If stream=false: Returns regular SendMessageResponse

    Document Integration:
    - Automatically searches user's documents for relevant context
    - Provides citations with similarity scores
    - Limits references based on session settings
    """
    try:
        # Check if streaming is requested
        if message_request.stream:
            async def generate_stream():
                """Generator function for streaming response."""
                async for chunk in chat_service.send_message_stream(
                    db, current_user.id, message_request
                ):
                    if chunk is None:
                        # Send error chunk
                        error_chunk = StreamingChunk(
                            content="",
                            session_id=0,
                            finished=True,
                        ).model_dump_json()
                        yield f"data: {error_chunk}\n\n"
                        break

                    # Send chunk as Server-Sent Events
                    chunk_json = chunk.model_dump_json()
                    yield f"data: {chunk_json}\n\n"

                # Send final message
                yield "data: [DONE]\n\n"

            return StreamingResponse(
                generate_stream(),
                media_type="text/plain",
                headers={
                    "Cache-Control": "no-cache",
                    "Connection": "keep-alive",
                    "Access-Control-Allow-Origin": "*",
                    "Access-Control-Allow-Headers": "Cache-Control",
                }
            )
        else:
            # Regular non-streaming response
            response = await chat_service.send_message(
                db, current_user.id, message_request
            )
            if not response:
                raise HTTPException(status_code=500, detail="Failed to send message")

            return response

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error sending message: {str(e)}")
        raise HTTPException(status_code=500, detail="Internal server error")


@router.post("/search-chats", response_model=ChatSearchResponse)
async def search_chat_sessions(
    search_query: ChatSearchQuery,
    current_user: UserResponse = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
):
    """
    Search chat sessions with filters.

    Search options:
    - Text search in session titles
    - Date range filtering
    - Public/private session filtering
    - Pagination support

    Useful for finding specific conversations or organizing chat history.
    """
    try:
        result = await chat_service.search_chat_sessions(
            db, current_user.id, search_query
        )
        if not result:
            raise HTTPException(status_code=500, detail="Search failed")

        return result

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error searching chat sessions: {str(e)}")
        raise HTTPException(status_code=500, detail="Internal server error")


# Folder Management Endpoints
@router.post("/create-folder", response_model=ChatFolderResponse)
async def create_folder(
    folder_data: ChatFolderCreate,
    current_user: UserResponse = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
):
    """
    Create a new folder for organizing chat sessions.

    Features:
    - Hierarchical folder structure (parent/child folders)
    - Custom colors and descriptions
    - Automatic chat count tracking
    """
    try:
        # Set user_id from authenticated user
        folder_data.user_id = current_user.id

        folder = await chat_service.create_folder(db, folder_data)
        if not folder:
            raise HTTPException(status_code=500, detail="Failed to create folder")

        return folder

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error creating folder: {str(e)}")
        raise HTTPException(status_code=500, detail="Internal server error")


@router.post("/add-chat-to-folder")
async def add_chat_to_folder(
    request: AddChatToFolderRequest,
    current_user: UserResponse = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
):
    """
    Add chat session to a folder.

    Organize conversations by adding them to folders.
    Supports moving chats between folders.
    """
    try:
        success = await chat_service.add_chat_to_folder(
            db, request.session_id, request.folder_id, current_user.id
        )
        if not success:
            raise HTTPException(status_code=400, detail="Failed to add chat to folder")

        return {"message": "Chat added to folder successfully"}

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error adding chat to folder: {str(e)}")
        raise HTTPException(status_code=500, detail="Internal server error")


@router.get("/llm-status")
async def get_llm_status():
    """
    Get LLM service status and configuration.

    Returns information about LLM service availability,
    configuration, and connection status.
    """
    try:
        llm_service = LLMService()
        model_info = llm_service.get_model_info()

        # Test connection if configured
        connection_status = "not_configured"
        if llm_service.is_configured:
            connection_test = await llm_service.test_connection()
            connection_status = "connected" if connection_test else "connection_failed"

        return {
            "service_status": connection_status,
            "model_info": model_info,
            "features": {
                "chat_completions": True,
                "streaming": True,
                "document_context": True,
                "citations": True
            }
        }

    except Exception as e:
        logger.error(f"Error getting LLM status: {str(e)}")
        raise HTTPException(status_code=500, detail="Failed to get LLM status")


@router.get("/session-info")
async def get_session_info():
    """
    Get information about chat session features and capabilities.

    Returns details about:
    - Available AI parameters
    - Document search capabilities
    - Citation system
    - Folder organization features
    - API limits and recommendations
    """
    return {
        "ai_parameters": {
            "temperature": {
                "description": "Controls randomness in AI responses",
                "range": "0.0-1.0",
                "default": "0.7",
                "recommended": {"creative": "0.8", "balanced": "0.7", "focused": "0.3"}
            },
            "max_tokens": {
                "description": "Maximum response length",
                "range": "100-4000",
                "default": 1000,
                "recommended": {"short": 500, "medium": 1000, "long": 2000}
            },
            "system_prompt": {
                "description": "Custom instructions for AI behavior",
                "examples": [
                    "You are a helpful assistant that provides concise answers.",
                    "You are an expert researcher that provides detailed, cited responses.",
                    "You are a creative writer that engages in storytelling."
                ]
            }
        },
        "document_context": {
            "enabled_by_default": True,
            "max_references": 5,
            "search_threshold": 0.7,
            "search_types": ["semantic", "keyword", "hybrid"],
            "citation_format": "References with similarity scores and document metadata"
        },
        "features": {
            "multi_turn_conversations": True,
            "document_search": True,
            "citations": True,
            "public_sharing": True,
            "folder_organization": True,
            "session_search": True,
            "message_history": True,
            "processing_metrics": True
        },
        "limits": {
            "message_length": 4000,
            "max_document_references": 10,
            "session_title_length": 255,
            "folder_name_length": 100
        },
        "response_format": {
            "includes_citations": True,
            "includes_processing_times": True,
            "includes_similarity_scores": True,
            "supports_streaming": True
        }
    }
