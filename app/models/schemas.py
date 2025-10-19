"""
Pydantic schemas for data validation and serialization.
"""

from datetime import datetime
from enum import Enum
from typing import List, Optional

from pydantic import BaseModel, EmailStr, Field, validator


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

    @validator("username")
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

    @validator("password")
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

    @validator("new_password")
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

    @validator("new_password")
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


# Advanced Search Schemas
class SearchType(str, Enum):
    """Search type enumeration."""
    SEMANTIC = "semantic"
    KEYWORD = "keyword"
    HYBRID = "hybrid"


class SortOrder(str, Enum):
    """Sort order enumeration."""
    RELEVANCE = "relevance"
    DATE_ASC = "date_asc"
    DATE_DESC = "date_desc"
    FILENAME_ASC = "filename_asc"
    FILENAME_DESC = "filename_desc"
    FILE_SIZE_ASC = "file_size_asc"
    FILE_SIZE_DESC = "file_size_desc"


class SearchFilters(BaseModel):
    """Schema for search filters."""

    file_types: Optional[List[str]] = None
    date_from: Optional[datetime] = None
    date_to: Optional[datetime] = None
    min_file_size: Optional[int] = None  # in bytes
    max_file_size: Optional[int] = None  # in bytes
    filename_pattern: Optional[str] = None  # SQL LIKE pattern


class AdvancedSearchQuery(BaseModel):
    """Schema for advanced search query."""

    query: str = Field(..., min_length=1, max_length=1000)
    search_type: SearchType = SearchType.HYBRID
    limit: int = Field(10, ge=1, le=100)
    offset: int = Field(0, ge=0)
    threshold: float = Field(0.7, ge=0.0, le=1.0)
    filters: Optional[SearchFilters] = None
    sort_by: SortOrder = SortOrder.RELEVANCE
    include_content: bool = True  # include chunk content in results
    similarity_weight: float = Field(0.7, ge=0.0, le=1.0)  # weight for hybrid search


class AdvancedSearchResult(BaseModel):
    """Schema for advanced search result."""

    document_id: int
    document_filename: str
    document_title: Optional[str] = None
    document_file_type: str
    document_file_size: int
    document_created_at: datetime
    chunk_id: Optional[int] = None  # None for document-level results
    chunk_content: Optional[str] = None
    chunk_index: Optional[int] = None
    similarity_score: Optional[float] = None  # For semantic search
    keyword_score: Optional[float] = None  # For keyword search
    hybrid_score: Optional[float] = None  # Combined score for hybrid search
    match_type: SearchType  # What type of search found this result


class AdvancedSearchResponse(BaseModel):
    """Schema for advanced search response."""

    query: str
    search_type: SearchType
    total_results: int
    results: List[AdvancedSearchResult]
    search_time: float  # in seconds
    has_more: bool  # for pagination
    limit: int
    offset: int


class KeywordSearchQuery(BaseModel):
    """Schema for keyword-based search query."""

    query: str = Field(..., min_length=1, max_length=1000)
    limit: int = Field(10, ge=1, le=100)
    offset: int = Field(0, ge=0)
    filters: Optional[SearchFilters] = None
    sort_by: SortOrder = SortOrder.RELEVANCE
    include_content: bool = True
    use_fuzzy_search: bool = True
    min_word_match: int = Field(1, ge=1, le=10)  # minimum words to match


# File upload schemas
class FileUploadResponse(BaseModel):
    """Schema for file upload response."""

    message: str
    document_id: int
    filename: str
    file_size: int
    file_type: str
    processing_status: str
