"""
Setup Test Database

This script:
1. Creates a test database configuration
2. Runs Alembic migrations
3. Verifies the setup
"""
import os
import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from sqlalchemy import create_engine, text
from alembic.config import Config as AlembicConfig
from alembic import command
from src.utils.logging_config import get_logger

logger = get_logger(__name__)


def setup_test_database():
    """Set up test database and run migrations"""
    
    # Use SQLite for testing if PostgreSQL is not available
    use_sqlite = os.getenv('USE_SQLITE_FOR_TESTS', 'true').lower() == 'true'
    
    if use_sqlite:
        logger.info("Using SQLite for test database")
        
        # Create data directory if it doesn't exist
        data_dir = project_root / "data"
        data_dir.mkdir(exist_ok=True)
        
        # Remove existing test database
        test_db_path = data_dir / "test_wikipedia_intelligence.db"
        if test_db_path.exists():
            test_db_path.unlink()
            logger.info("Removed existing test database")
        
        # Use absolute path for SQLite
        db_url = f"sqlite:///{test_db_path.absolute()}"
        logger.info(f"Database path: {test_db_path.absolute()}")
    else:
        logger.info("Using PostgreSQL for test database")
        # Get PostgreSQL connection details from environment
        host = os.getenv('POSTGRES_HOST', 'localhost')
        port = os.getenv('POSTGRES_PORT', '5432')
        user = os.getenv('POSTGRES_USER', 'dev_user')
        password = os.getenv('POSTGRES_PASSWORD', 'dev_password')
        db_name = os.getenv('POSTGRES_DB', 'wikipedia_intelligence_test')
        
        db_url = f"postgresql://{user}:{password}@{host}:{port}/{db_name}"
        
        # Create database if it doesn't exist
        try:
            admin_url = f"postgresql://{user}:{password}@{host}:{port}/postgres"
            admin_engine = create_engine(admin_url, isolation_level="AUTOCOMMIT")
            
            with admin_engine.connect() as conn:
                # Check if database exists
                result = conn.execute(
                    text(f"SELECT 1 FROM pg_database WHERE datname = '{db_name}'")
                )
                if not result.fetchone():
                    conn.execute(text(f"CREATE DATABASE {db_name}"))
                    logger.info(f"Created test database: {db_name}")
                else:
                    logger.info(f"Test database already exists: {db_name}")
            
            admin_engine.dispose()
        except Exception as e:
            logger.error(f"Failed to create test database: {e}")
            logger.info("Falling back to SQLite")
            use_sqlite = True
            db_url = "sqlite:///data/test_wikipedia_intelligence.db"
            data_dir = project_root / "data"
            data_dir.mkdir(exist_ok=True)
    
    # Set database URL in environment for Alembic
    os.environ['DATABASE_URL'] = db_url
    
    # Configure Alembic
    alembic_cfg = AlembicConfig(str(project_root / "alembic.ini"))
    alembic_cfg.set_main_option("script_location", str(project_root / "alembic"))
    alembic_cfg.set_main_option("sqlalchemy.url", db_url)
    
    try:
        # Run migrations
        logger.info("Running Alembic migrations...")
        command.upgrade(alembic_cfg, "head")
        logger.info("Migrations completed successfully")
        
        # Verify setup
        engine = create_engine(db_url)
        with engine.connect() as conn:
            # Check alembic version
            result = conn.execute(text("SELECT version_num FROM alembic_version"))
            version = result.scalar()
            logger.info(f"Current migration version: {version}")
            
            # Count tables
            if use_sqlite:
                result = conn.execute(
                    text("SELECT COUNT(*) FROM sqlite_master WHERE type='table'")
                )
            else:
                result = conn.execute(
                    text("""
                        SELECT COUNT(*) 
                        FROM information_schema.tables 
                        WHERE table_schema = 'public'
                    """)
                )
            table_count = result.scalar()
            logger.info(f"Total tables created: {table_count}")
        
        engine.dispose()
        
        logger.info("Test database setup completed successfully")
        logger.info(f"Database URL: {db_url}")
        return True
        
    except Exception as e:
        logger.error(f"Failed to set up test database: {e}", exc_info=True)
        return False


if __name__ == "__main__":
    success = setup_test_database()
    sys.exit(0 if success else 1)
