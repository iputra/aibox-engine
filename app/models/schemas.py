"""
Pydantic schemas for data validation and serialization.
"""

from datetime import datetime
from enum import Enum
from typing import List, Optional

from pydantic import BaseModel, EmailStr, Field, field_validator


class UserRole(str, Enum):
    """User role enumeration."""

    USER = "user"
    ADMIN = "admin"


class UserBase(BaseModel):
    """Base user schema."""

    username: str = Field(..., min_length=3, max_length=50, pattern="^[a-zA-Z0-9_-]+$")
    email: EmailStr
    full_name: Optional[str] = Field(None, max_length=100)
    bio: Optional[str] = None
    avatar_url: Optional[str] = None

    @field_validator("username")
    @classmethod
    def validate_username(cls, v):
        """Validate username format."""
        if not v.strip():
            raise ValueError("Username cannot be empty")
        if v.lower() in ["admin", "root", "system", "api"]:
            raise ValueError(f'Username "{v}" is reserved')
        return v.lower()


class UserCreate(UserBase):
    """Schema for user registration."""

    password: str = Field(..., min_length=8, max_length=100)
    role: UserRole = UserRole.USER

    @field_validator("password")
    @classmethod
    def validate_password(cls, v):
        """Validate password strength."""
        if len(v) < 8:
            raise ValueError("Password must be at least 8 characters long")
        if not any(c.isupper() for c in v):
            raise ValueError("Password must contain at least one uppercase letter")
        if not any(c.islower() for c in v):
            raise ValueError("Password must contain at least one lowercase letter")
        if not any(c.isdigit() for c in v):
            raise ValueError("Password must contain at least one digit")
        return v


class UserUpdate(BaseModel):
    """Schema for updating user profile."""

    full_name: Optional[str] = Field(None, max_length=100)
    bio: Optional[str] = None
    avatar_url: Optional[str] = None
    email: Optional[EmailStr] = None


class UserLogin(BaseModel):
    """Schema for user login."""

    username: str  # Can be username or email
    password: str


class UserResponse(UserBase):
    """Schema for user response (safe data)."""

    id: int
    is_active: bool
    is_verified: bool
    role: UserRole
    created_at: datetime
    updated_at: Optional[datetime] = None
    last_login: Optional[datetime] = None

    class Config:
        from_attributes = True


class UserResponseSafe(BaseModel):
    """Schema for user response with limited fields."""

    id: int
    username: str
    full_name: Optional[str] = None
    bio: Optional[str] = None
    avatar_url: Optional[str] = None
    role: UserRole
    created_at: datetime

    class Config:
        from_attributes = True


class Token(BaseModel):
    """Schema for JWT token response."""

    access_token: str
    token_type: str = "bearer"
    expires_in: int
    refresh_token: Optional[str] = None


class TokenData(BaseModel):
    """Schema for token payload."""

    username: Optional[str] = None
    user_id: Optional[int] = None
    role: Optional[UserRole] = None


class PasswordChange(BaseModel):
    """Schema for password change."""

    current_password: str
    new_password: str = Field(..., min_length=8, max_length=100)

    @field_validator("new_password")
    @classmethod
    def validate_new_password(cls, v):
        """Validate new password strength."""
        if len(v) < 8:
            raise ValueError("Password must be at least 8 characters long")
        if not any(c.isupper() for c in v):
            raise ValueError("Password must contain at least one uppercase letter")
        if not any(c.islower() for c in v):
            raise ValueError("Password must contain at least one lowercase letter")
        if not any(c.isdigit() for c in v):
            raise ValueError("Password must contain at least one digit")
        return v


class PasswordReset(BaseModel):
    """Schema for password reset request."""

    email: EmailStr


class PasswordResetConfirm(BaseModel):
    """Schema for password reset confirmation."""

    token: str
    new_password: str = Field(..., min_length=8, max_length=100)

    @field_validator("new_password")
    @classmethod
    def validate_new_password(cls, v):
        """Validate new password strength."""
        if len(v) < 8:
            raise ValueError("Password must be at least 8 characters long")
        if not any(c.isupper() for c in v):
            raise ValueError("Password must contain at least one uppercase letter")
        if not any(c.islower() for c in v):
            raise ValueError("Password must contain at least one lowercase letter")
        if not any(c.isdigit() for c in v):
            raise ValueError("Password must contain at least one digit")
        return v


# Document schemas
class DocumentBase(BaseModel):
    """Base document schema."""

    description: Optional[str] = None


class DocumentCreate(DocumentBase):
    """Schema for document creation."""

    filename: str
    original_filename: str
    file_type: str
    file_size: int
    content_type: str
    file_path: str
    user_id: int


class DocumentUpdate(BaseModel):
    """Schema for updating document metadata."""

    description: Optional[str] = None
    is_active: Optional[bool] = None


class DocumentResponse(DocumentBase):
    """Schema for document response."""

    id: int
    filename: str
    original_filename: str
    file_type: str
    file_size: int
    content_type: str
    file_path: str
    is_active: bool
    created_at: datetime
    updated_at: datetime
    user_id: int

    class Config:
        from_attributes = True


class DocumentChunkBase(BaseModel):
    """Base document chunk schema."""

    content: str
    token_count: Optional[int] = None


class DocumentChunkCreate(DocumentChunkBase):
    """Schema for creating document chunk."""

    document_id: int
    chunk_index: int
    embedding: List[float]


class DocumentChunkResponse(DocumentChunkBase):
    """Schema for document chunk response."""

    id: int
    document_id: int
    chunk_index: int
    embedding: List[float]
    created_at: datetime

    class Config:
        from_attributes = True


class DocumentWithChunks(DocumentResponse):
    """Schema for document with its chunks."""

    chunks: List[DocumentChunkResponse] = []


# Search schemas
class SearchQuery(BaseModel):
    """Schema for search query."""

    query: str = Field(..., min_length=1, max_length=1000)
    limit: int = Field(10, ge=1, le=100)
    threshold: float = Field(0.7, ge=0.0, le=1.0)


class SearchResult(BaseModel):
    """Schema for individual search result."""

    document_id: int
    document_filename: str
    document_title: Optional[str] = None
    chunk_id: int
    chunk_content: str
    similarity_score: float
    chunk_index: int


class SearchResponse(BaseModel):
    """Schema for search response."""

    query: str
    total_results: int
    results: List[SearchResult]
    search_time: float  # in seconds


# File upload schemas
class FileUploadResponse(BaseModel):
    """Schema for file upload response."""

    message: str
    document_id: int
    filename: str
    file_size: int
    file_type: str
    processing_status: str


# Chat Schemas
class MessageRole(str, Enum):
    """Message role enumeration."""
    USER = "user"
    ASSISTANT = "assistant"
    SYSTEM = "system"


class ChatSessionBase(BaseModel):
    """Base chat session schema."""

    title: Optional[str] = None
    persona_id: Optional[str] = None
    temperature: str = "0.7"
    max_tokens: int = 1000
    system_prompt: Optional[str] = None
    include_document_context: bool = True
    max_document_references: int = 5
    document_search_threshold: str = "0.7"
    is_public: bool = False


class ChatSessionCreate(ChatSessionBase):
    """Schema for creating chat session."""
    pass  # user_id will be set from authenticated user


class ChatSessionUpdate(BaseModel):
    """Schema for updating chat session."""

    title: Optional[str] = None
    persona_id: Optional[str] = None
    temperature: Optional[str] = None
    max_tokens: Optional[int] = None
    system_prompt: Optional[str] = None
    include_document_context: Optional[bool] = None
    max_document_references: Optional[int] = None
    document_search_threshold: Optional[str] = None
    is_public: Optional[bool] = None


class ChatSessionResponse(ChatSessionBase):
    """Schema for chat session response."""

    id: int
    user_id: int
    share_token: Optional[str] = None
    is_active: bool
    last_message_at: Optional[datetime] = None
    message_count: int
    created_at: datetime
    updated_at: Optional[datetime] = None

    class Config:
        from_attributes = True


class DocumentCitation(BaseModel):
    """Schema for document citation in chat messages."""

    document_id: int
    document_filename: str
    document_title: Optional[str] = None
    chunk_id: Optional[int] = None
    chunk_content: Optional[str] = None
    similarity_score: Optional[float] = None
    page_number: Optional[int] = None


class ChatMessageBase(BaseModel):
    """Base chat message schema."""

    role: MessageRole
    content: str


class ChatMessageCreate(ChatMessageBase):
    """Schema for creating chat message."""

    session_id: int


class ChatMessageResponse(ChatMessageBase):
    """Schema for chat message response."""

    id: int
    session_id: int
    token_count: Optional[int] = None
    model_used: Optional[str] = None
    processing_time: Optional[str] = None
    document_references: Optional[List[DocumentCitation]] = None
    search_results: Optional[dict] = None
    is_edited: bool
    edited_at: Optional[datetime] = None
    created_at: datetime
    updated_at: Optional[datetime] = None

    class Config:
        from_attributes = True


class ChatSessionWithMessages(ChatSessionResponse):
    """Schema for chat session with its messages."""

    messages: List[ChatMessageResponse] = []


class SendMessageRequest(BaseModel):
    """Schema for sending a message."""

    message: str = Field(..., min_length=1, max_length=4000)
    session_id: Optional[int] = None  # If None, creates new session
    stream: bool = True  # Whether to stream response
    search_documents: bool = True  # Whether to search documents for context


class SendMessageResponse(BaseModel):
    """Schema for send message response."""

    session_id: int
    message_id: int
    response: str
    document_citations: List[DocumentCitation] = []
    search_time: Optional[float] = None
    generation_time: Optional[float] = None
    total_time: Optional[float] = None


class ChatFolderBase(BaseModel):
    """Base chat folder schema."""

    name: str = Field(..., min_length=1, max_length=100)
    description: Optional[str] = None
    color: Optional[str] = None
    parent_id: Optional[int] = None


class ChatFolderCreate(ChatFolderBase):
    """Schema for creating chat folder."""

    user_id: int


class ChatFolderResponse(ChatFolderBase):
    """Schema for chat folder response."""

    id: int
    user_id: int
    is_active: bool
    chat_count: int
    created_at: datetime
    updated_at: Optional[datetime] = None

    class Config:
        from_attributes = True


class AddChatToFolderRequest(BaseModel):
    """Schema for adding chat to folder."""

    session_id: int
    folder_id: int


class ChatSearchQuery(BaseModel):
    """Schema for searching chat sessions."""

    query: Optional[str] = None
    folder_id: Optional[int] = None
    date_from: Optional[datetime] = None
    date_to: Optional[datetime] = None
    is_public: Optional[bool] = None
    limit: int = Field(20, ge=1, le=100)
    offset: int = Field(0, ge=0)


class ChatSearchResponse(BaseModel):
    """Schema for chat search response."""

    total_results: int
    results: List[ChatSessionResponse]
    has_more: bool
