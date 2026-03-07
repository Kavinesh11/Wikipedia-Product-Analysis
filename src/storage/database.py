"""Database Connection Utilities

Provides connection pooling and session management for PostgreSQL.
"""
from typing import Generator, Optional
from contextlib import contextmanager
from sqlalchemy import create_engine, event, Engine
from sqlalchemy.orm import sessionmaker, Session, declarative_base
from sqlalchemy.pool import QueuePool

from src.utils.logging_config import get_logger
from src.utils.config import get_config

logger = get_logger(__name__)

# Base class for SQLAlchemy models
Base = declarative_base()


class Database:
    """Database connection manager with connection pooling"""
    
    def __init__(self, database_url: Optional[str] = None, pool_size: int = 10):
        """Initialize database connection
        
        Args:
            database_url: PostgreSQL connection URL
            pool_size: Maximum number of connections in pool
        """
        if database_url is None:
            config = get_config()
            database_url = config.database_url
        
        self.database_url = database_url
        self.engine: Optional[Engine] = None
        self.SessionLocal: Optional[sessionmaker] = None
        self.pool_size = pool_size
        
        self._initialize_engine()

    def _initialize_engine(self) -> None:
        """Initialize SQLAlchemy engine with connection pooling"""
        try:
            self.engine = create_engine(
                self.database_url,
                poolclass=QueuePool,
                pool_size=self.pool_size,
                max_overflow=20,
                pool_pre_ping=True,  # Verify connections before using
                echo=False,
            )
            
            # Set up event listeners
            @event.listens_for(self.engine, "connect")
            def receive_connect(dbapi_conn, connection_record):
                logger.debug("Database connection established")
            
            @event.listens_for(self.engine, "close")
            def receive_close(dbapi_conn, connection_record):
                logger.debug("Database connection closed")
            
            # Create session factory
            self.SessionLocal = sessionmaker(
                autocommit=False,
                autoflush=False,
                bind=self.engine
            )
            
            logger.info("Database engine initialized successfully")
            
        except Exception as e:
            logger.error(f"Failed to initialize database engine: {e}", exc_info=True)
            raise

    @contextmanager
    def get_session(self) -> Generator[Session, None, None]:
        """Get database session with automatic cleanup
        
        Yields:
            SQLAlchemy session
        """
        if self.SessionLocal is None:
            raise RuntimeError("Database not initialized")
        
        session = self.SessionLocal()
        try:
            yield session
            session.commit()
        except Exception as e:
            session.rollback()
            logger.error(f"Database session error: {e}", exc_info=True)
            raise
        finally:
            session.close()
    
    def create_tables(self) -> None:
        """Create all tables defined in models"""
        if self.engine is None:
            raise RuntimeError("Database engine not initialized")
        
        try:
            Base.metadata.create_all(bind=self.engine)
            logger.info("Database tables created successfully")
        except Exception as e:
            logger.error(f"Failed to create tables: {e}", exc_info=True)
            raise
    
    def drop_tables(self) -> None:
        """Drop all tables (use with caution!)"""
        if self.engine is None:
            raise RuntimeError("Database engine not initialized")
        
        try:
            Base.metadata.drop_all(bind=self.engine)
            logger.warning("All database tables dropped")
        except Exception as e:
            logger.error(f"Failed to drop tables: {e}", exc_info=True)
            raise
    
    def close(self) -> None:
        """Close database connections"""
        if self.engine:
            self.engine.dispose()
            logger.info("Database connections closed")


# Global database instance
_db_instance: Optional[Database] = None


def get_database() -> Database:
    """Get global database instance
    
    Returns:
        Database instance
    """
    global _db_instance
    
    if _db_instance is None:
        _db_instance = Database()
    
    return _db_instance
