#!/usr/bin/env python3
"""
Data Migration Script
Handles data migration between different versions of the schema
"""

import sys
import os
import logging
from datetime import datetime
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from sqlalchemy import create_engine, text
from sqlalchemy.orm import sessionmaker
import yaml

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def load_config(environment: str = None) -> dict:
    """Load configuration from config file"""
    config_path = Path(__file__).parent.parent / "config" / "config.yaml"
    
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    
    # Get environment from env var or parameter
    env = environment or os.getenv('ENVIRONMENT', 'development')
    
    if env not in config:
        raise ValueError(f"Environment '{env}' not found in config")
    
    return config[env]


def get_database_url(config: dict) -> str:
    """Construct database URL from config"""
    if config['database'].get('use_sqlite', False):
        db_path = config['database']['sqlite_path']
        return f"sqlite:///{db_path}"
    else:
        host = os.getenv('POSTGRES_HOST', config['database']['postgres_host'])
        port = config['database']['postgres_port']
        db = os.getenv('POSTGRES_DB', config['database']['postgres_db'])
        user = os.getenv('POSTGRES_USER', config['database']['postgres_user'])
        password = os.getenv('POSTGRES_PASSWORD', config['database']['postgres_password'])
        
        return f"postgresql://{user}:{password}@{host}:{port}/{db}"


def create_backup(engine, backup_dir: Path):
    """Create database backup before migration"""
    backup_dir.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    backup_file = backup_dir / f"backup_{timestamp}.sql"
    
    logger.info(f"Creating backup at {backup_file}")
    
    # For PostgreSQL, use pg_dump
    if 'postgresql' in str(engine.url):
        import subprocess
        
        host = engine.url.host
        port = engine.url.port
        database = engine.url.database
        user = engine.url.username
        
        env = os.environ.copy()
        env['PGPASSWORD'] = engine.url.password
        
        cmd = [
            'pg_dump',
            '-h', host,
            '-p', str(port),
            '-U', user,
            '-d', database,
            '-f', str(backup_file)
        ]
        
        try:
            subprocess.run(cmd, env=env, check=True)
            logger.info("Backup created successfully")
        except subprocess.CalledProcessError as e:
            logger.error(f"Backup failed: {e}")
            raise
    else:
        logger.warning("Backup not implemented for SQLite, skipping")


def migrate_v1_to_v2(session):
    """Example migration: Add new columns or tables"""
    logger.info("Running migration v1 to v2")
    
    # Example: Add a new column to existing table
    try:
        session.execute(text("""
            ALTER TABLE dim_articles 
            ADD COLUMN IF NOT EXISTS last_crawled TIMESTAMP
        """))
        session.commit()
        logger.info("Added last_crawled column to dim_articles")
    except Exception as e:
        logger.error(f"Migration failed: {e}")
        session.rollback()
        raise


def migrate_v2_to_v3(session):
    """Example migration: Create new indexes"""
    logger.info("Running migration v2 to v3")
    
    try:
        session.execute(text("""
            CREATE INDEX IF NOT EXISTS idx_pageviews_device_type 
            ON fact_pageviews(device_type)
        """))
        session.commit()
        logger.info("Created index on device_type")
    except Exception as e:
        logger.error(f"Migration failed: {e}")
        session.rollback()
        raise


def get_current_version(session) -> int:
    """Get current schema version"""
    try:
        result = session.execute(text("""
            SELECT value FROM runtime_config WHERE key = 'schema_version'
        """))
        row = result.fetchone()
        return int(row[0]) if row else 0
    except Exception:
        # Table doesn't exist yet, version 0
        return 0


def set_schema_version(session, version: int):
    """Update schema version"""
    session.execute(text("""
        INSERT INTO runtime_config (key, value, value_type, description)
        VALUES ('schema_version', :version, 'integer', 'Current database schema version')
        ON CONFLICT (key) DO UPDATE SET value = :version, updated_at = CURRENT_TIMESTAMP
    """), {'version': str(version)})
    session.commit()


def run_migrations(environment: str = None, target_version: int = None, skip_backup: bool = False):
    """Run all pending migrations"""
    logger.info("Starting data migration")
    
    # Load config and create engine
    config = load_config(environment)
    db_url = get_database_url(config)
    engine = create_engine(db_url)
    Session = sessionmaker(bind=engine)
    session = Session()
    
    try:
        # Create backup unless skipped
        if not skip_backup:
            backup_dir = Path(__file__).parent.parent / "backups"
            create_backup(engine, backup_dir)
        
        # Get current version
        current_version = get_current_version(session)
        logger.info(f"Current schema version: {current_version}")
        
        # Define migration functions
        migrations = {
            1: migrate_v1_to_v2,
            2: migrate_v2_to_v3,
            # Add more migrations here as needed
        }
        
        # Determine target version
        if target_version is None:
            target_version = max(migrations.keys()) if migrations else 0
        
        logger.info(f"Target schema version: {target_version}")
        
        # Run migrations
        for version in range(current_version + 1, target_version + 1):
            if version in migrations:
                logger.info(f"Applying migration to version {version}")
                migrations[version](session)
                set_schema_version(session, version)
                logger.info(f"Migration to version {version} completed")
            else:
                logger.warning(f"No migration defined for version {version}")
        
        logger.info("All migrations completed successfully")
        
    except Exception as e:
        logger.error(f"Migration failed: {e}")
        session.rollback()
        raise
    finally:
        session.close()


def main():
    """Main entry point"""
    import argparse
    
    parser = argparse.ArgumentParser(description='Run database migrations')
    parser.add_argument(
        '--environment',
        choices=['development', 'staging', 'production'],
        help='Environment to migrate'
    )
    parser.add_argument(
        '--target-version',
        type=int,
        help='Target schema version (default: latest)'
    )
    parser.add_argument(
        '--skip-backup',
        action='store_true',
        help='Skip database backup'
    )
    
    args = parser.parse_args()
    
    try:
        run_migrations(
            environment=args.environment,
            target_version=args.target_version,
            skip_backup=args.skip_backup
        )
        sys.exit(0)
    except Exception as e:
        logger.error(f"Migration failed: {e}")
        sys.exit(1)


if __name__ == '__main__':
    main()
