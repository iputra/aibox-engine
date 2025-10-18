"""
Database configuration and connection management for AIBox Engine.
"""

from decouple import config
from sqlalchemy import MetaData
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker, create_async_engine
from sqlalchemy.ext.declarative import declarative_base


# Database configuration
DATABASE_URL = config(
    "DATABASE_URL",
    default="postgresql+asyncpg://neondb_owner:npg_BTE5vSu4KWtO@ep-aged-unit-advtc9a5-pooler.c-2.us-east-1.aws.neon.tech/neondb",
)

# Create async engine
engine = create_async_engine(
    DATABASE_URL,
    echo=config("DEBUG", default=False, cast=bool),
    future=True,
    pool_pre_ping=True,
    pool_recycle=300,
    connect_args={"ssl": True, "server_settings": {"application_name": "aibox_engine"}},
)

# Create async session factory
AsyncSessionLocal = async_sessionmaker(
    engine,
    class_=AsyncSession,
    expire_on_commit=False,
    autoflush=True,
)

# Create base class for models
Base = declarative_base()

# Metadata for migrations
metadata = MetaData()


async def get_db() -> AsyncSession:
    """
    Dependency function to get database session.

    Yields:
        AsyncSession: Database session
    """
    async with AsyncSessionLocal() as session:
        try:
            yield session
        finally:
            await session.close()


async def init_db():
    """
    Initialize database tables.
    """
    async with engine.begin() as conn:
        # Import all models here to ensure they are registered with Base

        # Create all tables
        await conn.run_sync(Base.metadata.create_all)
        print("✅ Database tables created successfully")


async def close_db():
    """
    Close database connection.
    """
    await engine.dispose()
    print("✅ Database connection closed")


async def test_db_connection():
    """
    Test database connection.

    Returns:
        bool: True if connection is successful
    """
    try:
        async with engine.connect() as conn:
            # Use text() for raw SQL queries
            from sqlalchemy import text

            result = await conn.execute(text("SELECT 1"))
            print("✅ Database connection successful")
            return True
    except Exception as e:
        print(f"❌ Database connection failed: {e}")
        return False
