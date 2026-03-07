"""
Create tables directly using SQLAlchemy without Alembic

This bypasses Alembic migration issues and creates tables directly.
"""
import os
import sys
from pathlib import Path

project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

# Set environment
os.environ['ENVIRONMENT'] = 'development'

from sqlalchemy import create_engine
from src.storage.database import Base
from src.storage import models  # Import models to register them
from src.utils.logging_config import get_logger

logger = get_logger(__name__)


def create_tables():
    """Create all tables directly"""
    
    # Use absolute path for SQLite
    test_db_path = project_root / "data" / "test_wikipedia_intelligence.db"
    
    # Remove existing database
    if test_db_path.exists():
        test_db_path.unlink()
        logger.info("Removed existing test database")
    
    db_url = f"sqlite:///{test_db_path.absolute()}"
    logger.info(f"Creating tables in: {db_url}")
    
    try:
        # Create engine
        engine = create_engine(db_url, echo=True)
        
        # Create all tables
        Base.metadata.create_all(bind=engine)
        
        logger.info("✓ All tables created successfully")
        
        # Verify tables
        from sqlalchemy import inspect
        inspector = inspect(engine)
        tables = inspector.get_table_names()
        logger.info(f"✓ Created {len(tables)} tables: {tables}")
        
        engine.dispose()
        return True
        
    except Exception as e:
        logger.error(f"✗ Failed to create tables: {e}", exc_info=True)
        return False


if __name__ == "__main__":
    success = create_tables()
    sys.exit(0 if success else 1)
