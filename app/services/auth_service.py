"""
Authentication service for AIBox Engine.
"""

from datetime import datetime, timedelta
from typing import Optional, Tuple
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select
from sqlalchemy.orm import selectinload
from fastapi import HTTPException, status

from app.models.user import User
from app.models.schemas import (
    UserCreate, UserLogin, UserUpdate, Token, TokenData,
    UserRole, PasswordChange
)
from app.core.security import (
    verify_password, get_password_hash,
    create_access_token, create_refresh_token,
    verify_token, verify_refresh_token,
    create_password_reset_token, verify_password_reset_token,
    AuthenticationError, AuthorizationError
)
from app.core.database import get_db


class AuthService:
    """Authentication service class."""

    def __init__(self, db: AsyncSession):
        self.db = db

    async def get_user_by_id(self, user_id: int) -> Optional[User]:
        """
        Get user by ID.

        Args:
            user_id: User ID

        Returns:
            User: User object or None if not found
        """
        result = await self.db.execute(
            select(User).where(User.id == user_id)
        )
        return result.scalar_one_or_none()

    async def get_user_by_username(self, username: str) -> Optional[User]:
        """
        Get user by username.

        Args:
            username: Username

        Returns:
            User: User object or None if not found
        """
        result = await self.db.execute(
            select(User).where(User.username == username)
        )
        return result.scalar_one_or_none()

    async def get_user_by_email(self, email: str) -> Optional[User]:
        """
        Get user by email.

        Args:
            email: User email

        Returns:
            User: User object or None if not found
        """
        result = await self.db.execute(
            select(User).where(User.email == email)
        )
        return result.scalar_one_or_none()

    async def get_user_by_username_or_email(self, identifier: str) -> Optional[User]:
        """
        Get user by username or email.

        Args:
            identifier: Username or email

        Returns:
            User: User object or None if not found
        """
        result = await self.db.execute(
            select(User).where(
                (User.username == identifier) | (User.email == identifier)
            )
        )
        return result.scalar_one_or_none()

    async def create_user(self, user_data: UserCreate) -> User:
        """
        Create a new user.

        Args:
            user_data: User creation data

        Returns:
            User: Created user object

        Raises:
            HTTPException: If username or email already exists
        """
        # Check if username already exists
        existing_user = await self.get_user_by_username(user_data.username)
        if existing_user:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Username already registered"
            )

        # Check if email already exists
        existing_email = await self.get_user_by_email(user_data.email)
        if existing_email:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Email already registered"
            )

        # Create new user
        hashed_password = get_password_hash(user_data.password)
        db_user = User(
            username=user_data.username.lower(),
            email=user_data.email.lower(),
            hashed_password=hashed_password,
            full_name=user_data.full_name,
            bio=user_data.bio,
            avatar_url=user_data.avatar_url,
            role=user_data.role,
            is_active=True,
            is_verified=False
        )

        self.db.add(db_user)
        await self.db.commit()
        await self.db.refresh(db_user)

        return db_user

    async def authenticate_user(self, login_data: UserLogin) -> Optional[User]:
        """
        Authenticate user with username/email and password.

        Args:
            login_data: Login credentials

        Returns:
            User: Authenticated user or None if authentication fails
        """
        user = await self.get_user_by_username_or_email(login_data.username)

        if not user:
            return None

        if not verify_password(login_data.password, user.hashed_password):
            return None

        if not user.is_active:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Account is inactive"
            )

        # Update last login
        user.last_login = datetime.utcnow()
        await self.db.commit()

        return user

    async def create_user_tokens(self, user: User) -> Token:
        """
        Create access and refresh tokens for user.

        Args:
            user: User object

        Returns:
            Token: Access and refresh tokens
        """
        # Create access token
        access_token_expires = timedelta(minutes=30)
        access_token = create_access_token(
            data={
                "sub": user.username,
                "user_id": user.id,
                "role": user.role
            },
            expires_delta=access_token_expires
        )

        # Create refresh token
        refresh_token_expires = timedelta(days=7)
        refresh_token = create_refresh_token(
            data={
                "sub": user.username,
                "user_id": user.id
            },
            expires_delta=refresh_token_expires
        )

        return Token(
            access_token=access_token,
            token_type="bearer",
            expires_in=int(access_token_expires.total_seconds()),
            refresh_token=refresh_token
        )

    async def refresh_access_token(self, refresh_token: str) -> Token:
        """
        Refresh access token using refresh token.

        Args:
            refresh_token: Refresh token

        Returns:
            Token: New access and refresh tokens

        Raises:
            AuthenticationError: If refresh token is invalid
        """
        token_data = verify_refresh_token(refresh_token)

        if not token_data:
            raise AuthenticationError("Invalid refresh token")

        user = await self.get_user_by_id(token_data.user_id)

        if not user or not user.is_active:
            raise AuthenticationError("User not found or inactive")

        # Create new tokens
        return await self.create_user_tokens(user)

    async def update_user(self, user_id: int, user_data: UserUpdate) -> Optional[User]:
        """
        Update user profile.

        Args:
            user_id: User ID
            user_data: Update data

        Returns:
            User: Updated user object or None if not found
        """
        user = await self.get_user_by_id(user_id)

        if not user:
            return None

        # Check if email is being updated and if it's already taken
        if user_data.email and user_data.email != user.email:
            existing_email = await self.get_user_by_email(user_data.email)
            if existing_email:
                raise HTTPException(
                    status_code=status.HTTP_400_BAD_REQUEST,
                    detail="Email already registered"
                )
            user.email = user_data.email.lower()

        # Update other fields
        if user_data.full_name is not None:
            user.full_name = user_data.full_name
        if user_data.bio is not None:
            user.bio = user_data.bio
        if user_data.avatar_url is not None:
            user.avatar_url = user_data.avatar_url

        user.updated_at = datetime.utcnow()
        await self.db.commit()
        await self.db.refresh(user)

        return user

    async def change_password(self, user_id: int, password_data: PasswordChange) -> bool:
        """
        Change user password.

        Args:
            user_id: User ID
            password_data: Password change data

        Returns:
            bool: True if password changed successfully

        Raises:
            HTTPException: If current password is incorrect
        """
        user = await self.get_user_by_id(user_id)

        if not user:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="User not found"
            )

        if not verify_password(password_data.current_password, user.hashed_password):
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Current password is incorrect"
            )

        user.hashed_password = get_password_hash(password_data.new_password)
        user.updated_at = datetime.utcnow()
        await self.db.commit()

        return True

    async def create_password_reset_token(self, email: str) -> Optional[str]:
        """
        Create password reset token for user.

        Args:
            email: User email

        Returns:
            str: Password reset token or None if user not found
        """
        user = await self.get_user_by_email(email)

        if not user:
            return None

        return create_password_reset_token(email)

    async def reset_password(self, token: str, new_password: str) -> bool:
        """
        Reset user password using token.

        Args:
            token: Password reset token
            new_password: New password

        Returns:
            bool: True if password reset successfully

        Raises:
            HTTPException: If token is invalid
        """
        email = verify_password_reset_token(token)

        if not email:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Invalid or expired reset token"
            )

        user = await self.get_user_by_email(email)

        if not user:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="User not found"
            )

        user.hashed_password = get_password_hash(new_password)
        user.updated_at = datetime.utcnow()
        await self.db.commit()

        return True

    async def verify_user_token(self, token: str) -> Optional[User]:
        """
        Verify JWT token and return user.

        Args:
            token: JWT access token

        Returns:
            User: User object or None if token is invalid
        """
        token_data = verify_token(token)

        if not token_data:
            return None

        user = await self.get_user_by_id(token_data.user_id)

        if not user or not user.is_active:
            return None

        return user

    async def check_user_permission(
        self, user: User, required_role: UserRole
    ) -> bool:
        """
        Check if user has required role.

        Args:
            user: User object
            required_role: Required role

        Returns:
            bool: True if user has permission
        """
        if user.role == UserRole.ADMIN:
            return True

        if required_role == UserRole.USER:
            return True

        return False

    async def delete_user(self, user_id: int) -> bool:
        """
        Fully delete user from database.

        Args:
            user_id: User ID

        Returns:
            bool: True if user was deleted, False if user not found

        Raises:
            HTTPException: If trying to delete oneself
        """
        user = await self.get_user_by_id(user_id)

        if not user:
            return False

        # Delete the user from database
        await self.db.delete(user)
        await self.db.commit()

        return True
