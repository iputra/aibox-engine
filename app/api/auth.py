"""
Authentication API endpoints for AIBox Engine.
"""

from datetime import datetime
from typing import Optional
from fastapi import APIRouter, Depends, HTTPException, status, Query
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from sqlalchemy.ext.asyncio import AsyncSession
from pydantic import BaseModel, Field

from app.core.database import get_db
from app.services.auth_service import AuthService
from app.models.schemas import (
    UserCreate, UserLogin, UserResponse, UserResponseSafe,
    Token, UserUpdate, PasswordChange, PasswordReset,
    PasswordResetConfirm, UserRole
)
from app.models.user import User
from app.core.security import AuthenticationError, AuthorizationError

# Router configuration
router = APIRouter(tags=["Authentication"])
security = HTTPBearer()


class MessageResponse(BaseModel):
    """Standard message response."""
    message: str
    success: bool = True


class UserListResponse(BaseModel):
    """Response for user list endpoint."""
    users: list[UserResponseSafe]
    total: int
    page: int
    per_page: int


# Helper functions
async def get_current_user(
    credentials: HTTPAuthorizationCredentials = Depends(security),
    db: AsyncSession = Depends(get_db)
) -> User:
    """
    Get current authenticated user.

    Args:
        credentials: HTTP Bearer credentials
        db: Database session

    Returns:
        User: Current authenticated user

    Raises:
        HTTPException: If authentication fails
    """
    auth_service = AuthService(db)
    user = await auth_service.verify_user_token(credentials.credentials)

    if not user:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Could not validate credentials",
            headers={"WWW-Authenticate": "Bearer"},
        )

    return user


async def get_current_active_user(
    current_user: User = Depends(get_current_user)
) -> User:
    """
    Get current active user.

    Args:
        current_user: Current authenticated user

    Returns:
        User: Current active user

    Raises:
        HTTPException: If user is inactive
    """
    if not current_user.is_active:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Inactive user"
        )
    return current_user


async def get_current_admin_user(
    current_user: User = Depends(get_current_active_user)
) -> User:
    """
    Get current admin user.

    Args:
        current_user: Current authenticated user

    Returns:
        User: Current admin user

    Raises:
        HTTPException: If user is not admin
    """
    if current_user.role != UserRole.ADMIN:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Not enough permissions"
        )
    return current_user


# Public endpoints (no authentication required)
@router.post("/register", response_model=UserResponse, status_code=status.HTTP_201_CREATED)
async def register(
    user_data: UserCreate,
    db: AsyncSession = Depends(get_db)
):
    """
    Register a new user.

    Args:
        user_data: User registration data
        db: Database session

    Returns:
        UserResponse: Created user data
    """
    auth_service = AuthService(db)
    user = await auth_service.create_user(user_data)
    return UserResponse.from_orm(user)


@router.post("/login", response_model=Token)
async def login(
    login_data: UserLogin,
    db: AsyncSession = Depends(get_db)
):
    """
    Authenticate user and return tokens.

    Args:
        login_data: Login credentials
        db: Database session

    Returns:
        Token: Access and refresh tokens
    """
    auth_service = AuthService(db)
    user = await auth_service.authenticate_user(login_data)

    if not user:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Incorrect username or password",
            headers={"WWW-Authenticate": "Bearer"},
        )

    tokens = await auth_service.create_user_tokens(user)
    return tokens


class RefreshTokenRequest(BaseModel):
    """Request model for token refresh."""
    refresh_token: str = Field(..., description="Refresh token")


@router.post("/refresh", response_model=Token)
async def refresh_token(
    request: RefreshTokenRequest,
    db: AsyncSession = Depends(get_db)
):
    """
    Refresh access token.

    Args:
        request: Refresh token request
        db: Database session

    Returns:
        Token: New access and refresh tokens
    """
    auth_service = AuthService(db)
    try:
        tokens = await auth_service.refresh_access_token(request.refresh_token)
        return tokens
    except AuthenticationError as e:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail=str(e)
        )


@router.post("/forgot-password", response_model=MessageResponse)
async def forgot_password(
    email_data: PasswordReset,
    db: AsyncSession = Depends(get_db)
):
    """
    Request password reset.

    Args:
        email_data: Email for password reset
        db: Database session

    Returns:
        MessageResponse: Success message
    """
    auth_service = AuthService(db)
    reset_token = await auth_service.create_password_reset_token(email_data.email)

    # In a real application, you would send an email with the reset token
    # For now, we'll just return a success message
    # In production, integrate with email service like SendGrid, AWS SES, etc.

    return MessageResponse(
        message="If the email exists in our system, a password reset link has been sent."
    )


@router.post("/reset-password", response_model=MessageResponse)
async def reset_password(
    reset_data: PasswordResetConfirm,
    db: AsyncSession = Depends(get_db)
):
    """
    Reset password using token.

    Args:
        reset_data: Reset token and new password
        db: Database session

    Returns:
        MessageResponse: Success message
    """
    auth_service = AuthService(db)
    try:
        await auth_service.reset_password(reset_data.token, reset_data.new_password)
        return MessageResponse(message="Password has been reset successfully")
    except HTTPException as e:
        raise e


# Protected endpoints (authentication required)
@router.get("/me", response_model=UserResponse)
async def get_current_user_info(
    current_user: User = Depends(get_current_active_user)
):
    """
    Get current user information.

    Args:
        current_user: Current authenticated user

    Returns:
        UserResponse: Current user data
    """
    return UserResponse.from_orm(current_user)


@router.put("/me", response_model=UserResponse)
async def update_current_user(
    user_data: UserUpdate,
    current_user: User = Depends(get_current_active_user),
    db: AsyncSession = Depends(get_db)
):
    """
    Update current user profile.

    Args:
        user_data: User update data
        current_user: Current authenticated user
        db: Database session

    Returns:
        UserResponse: Updated user data
    """
    auth_service = AuthService(db)
    updated_user = await auth_service.update_user(current_user.id, user_data)

    if not updated_user:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="User not found"
        )

    return UserResponse.from_orm(updated_user)


@router.post("/change-password", response_model=MessageResponse)
async def change_password(
    password_data: PasswordChange,
    current_user: User = Depends(get_current_active_user),
    db: AsyncSession = Depends(get_db)
):
    """
    Change current user password.

    Args:
        password_data: Password change data
        current_user: Current authenticated user
        db: Database session

    Returns:
        MessageResponse: Success message
    """
    auth_service = AuthService(db)
    try:
        await auth_service.change_password(current_user.id, password_data)
        return MessageResponse(message="Password changed successfully")
    except HTTPException as e:
        raise e


# Admin endpoints (admin role required)
@router.get("/users", response_model=UserListResponse)
async def list_users(
    page: int = Query(1, ge=1, description="Page number"),
    per_page: int = Query(10, ge=1, le=100, description="Items per page"),
    current_user: User = Depends(get_current_admin_user),
    db: AsyncSession = Depends(get_db)
):
    """
    List all users (admin only).

    Args:
        page: Page number
        per_page: Items per page
        current_user: Current admin user
        db: Database session

    Returns:
        UserListResponse: List of users
    """
    # This is a simple implementation - in production, you might want to add
    # filtering, sorting, and more sophisticated pagination
    from sqlalchemy import select, func

    # Get total count
    count_result = await db.execute(select(func.count(User.id)))
    total = count_result.scalar()

    # Get users with pagination
    offset = (page - 1) * per_page
    result = await db.execute(
        select(User)
        .offset(offset)
        .limit(per_page)
        .order_by(User.created_at.desc())
    )
    users = result.scalars().all()

    user_responses = [UserResponseSafe.from_orm(user) for user in users]

    return UserListResponse(
        users=user_responses,
        total=total,
        page=page,
        per_page=per_page
    )


@router.get("/users/{user_id}", response_model=UserResponse)
async def get_user_by_id(
    user_id: int,
    current_user: User = Depends(get_current_admin_user),
    db: AsyncSession = Depends(get_db)
):
    """
    Get user by ID (admin only).

    Args:
        user_id: User ID
        current_user: Current admin user
        db: Database session

    Returns:
        UserResponse: User data
    """
    auth_service = AuthService(db)
    user = await auth_service.get_user_by_id(user_id)

    if not user:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="User not found"
        )

    return UserResponse.from_orm(user)


@router.put("/users/{user_id}/role", response_model=MessageResponse)
async def update_user_role(
    user_id: int,
    role: UserRole,
    current_user: User = Depends(get_current_admin_user),
    db: AsyncSession = Depends(get_db)
):
    """
    Update user role (admin only).

    Args:
        user_id: User ID
        role: New role
        current_user: Current admin user
        db: Database session

    Returns:
        MessageResponse: Success message
    """
    auth_service = AuthService(db)
    user = await auth_service.get_user_by_id(user_id)

    if not user:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="User not found"
        )

    # Prevent admin from changing their own role
    if user.id == current_user.id:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Cannot change your own role"
        )

    user.role = role
    user.updated_at = datetime.utcnow()
    await db.commit()

    return MessageResponse(message=f"User role updated to {role.value} successfully")


@router.delete("/users/{user_id}/deactivate", response_model=MessageResponse)
async def deactivate_user(
    user_id: int,
    current_user: User = Depends(get_current_admin_user),
    db: AsyncSession = Depends(get_db)
):
    """
    Deactivate user account (admin only).

    Args:
        user_id: User ID
        current_user: Current admin user
        db: Database session

    Returns:
        MessageResponse: Success message
    """
    auth_service = AuthService(db)
    user = await auth_service.get_user_by_id(user_id)

    if not user:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="User not found"
        )

    # Prevent admin from deactivating themselves
    if user.id == current_user.id:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Cannot deactivate your own account"
        )

    user.is_active = False
    user.updated_at = datetime.utcnow()
    await db.commit()

    return MessageResponse(message="User account deactivated successfully")


@router.delete("/users/{user_id}/delete", response_model=MessageResponse)
async def delete_user(
    user_id: int,
    current_user: User = Depends(get_current_admin_user),
    db: AsyncSession = Depends(get_db)
):
    """
    Fully delete user from database (admin only).

    Args:
        user_id: User ID
        current_user: Current admin user
        db: Database session

    Returns:
        MessageResponse: Success message
    """
    # Prevent admin from deleting themselves
    if user_id == current_user.id:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Cannot delete your own account"
        )

    auth_service = AuthService(db)
    deleted = await auth_service.delete_user(user_id)

    if not deleted:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="User not found"
        )

    return MessageResponse(message="User account fully deleted successfully")
