#!/usr/bin/env python3
"""
Application Startup Script
Handles initialization, health checks, and application launch
"""

import sys
import os
import time
import logging
from pathlib import Path
from typing import Dict, Tuple

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

import yaml
from sqlalchemy import create_engine, text
import redis

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def load_config(environment: str = None) -> dict:
    """Load configuration from config file"""
    config_path = Path(__file__).parent.parent / "config" / "config.yaml"
    
    if not config_path.exists():
        logger.error(f"Configuration file not found: {config_path}")
        sys.exit(1)
    
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    
    # Get environment from env var or parameter
    env = environment or os.getenv('ENVIRONMENT', 'development')
    
    if env not in config:
        logger.error(f"Environment '{env}' not found in config")
        sys.exit(1)
    
    logger.info(f"Loaded configuration for environment: {env}")
    return config[env]


def validate_config(config: dict) -> Tuple[bool, list]:
    """Validate configuration parameters"""
    errors = []
    
    # Check required sections
    required_sections = ['api', 'database', 'cache', 'logging']
    for section in required_sections:
        if section not in config:
            errors.append(f"Missing required section: {section}")
    
    # Validate API config
    if 'api' in config:
        if 'wikimedia_base_url' not in config['api']:
            errors.append("Missing api.wikimedia_base_url")
        if 'rate_limit' not in config['api']:
            errors.append("Missing api.rate_limit")
        elif not (1 <= config['api']['rate_limit'] <= 200):
            errors.append("api.rate_limit must be between 1 and 200")
    
    # Validate database config
    if 'database' in config:
        if not config['database'].get('use_sqlite', False):
            required_db_fields = ['postgres_host', 'postgres_db', 'postgres_user', 'postgres_password']
            for field in required_db_fields:
                value = config['database'].get(field, '')
                # Check if it's an environment variable reference
                if value.startswith('${') and value.endswith('}'):
                    env_var = value[2:-1]
                    if not os.getenv(env_var):
                        errors.append(f"Environment variable {env_var} not set for database.{field}")
                elif not value:
                    errors.append(f"Missing database.{field}")
    
    # Validate cache config
    if 'cache' in config:
        required_cache_fields = ['redis_host', 'redis_port']
        for field in required_cache_fields:
            value = config['cache'].get(field, '')
            if isinstance(value, str) and value.startswith('${') and value.endswith('}'):
                env_var = value[2:-1]
                if not os.getenv(env_var):
                    errors.append(f"Environment variable {env_var} not set for cache.{field}")
            elif not value:
                errors.append(f"Missing cache.{field}")
    
    return len(errors) == 0, errors


def resolve_env_vars(value):
    """Resolve environment variable references in config values"""
    if isinstance(value, str) and value.startswith('${') and value.endswith('}'):
        env_var = value[2:-1]
        return os.getenv(env_var, value)
    return value


def check_database_health(config: dict) -> Tuple[bool, str]:
    """Check database connectivity"""
    logger.info("Checking database health...")
    
    try:
        if config['database'].get('use_sqlite', False):
            db_path = config['database']['sqlite_path']
            db_url = f"sqlite:///{db_path}"
        else:
            host = resolve_env_vars(config['database']['postgres_host'])
            port = config['database']['postgres_port']
            db = resolve_env_vars(config['database']['postgres_db'])
            user = resolve_env_vars(config['database']['postgres_user'])
            password = resolve_env_vars(config['database']['postgres_password'])
            
            db_url = f"postgresql://{user}:{password}@{host}:{port}/{db}"
        
        engine = create_engine(db_url, pool_pre_ping=True)
        
        # Test connection
        with engine.connect() as conn:
            result = conn.execute(text("SELECT 1"))
            result.fetchone()
        
        logger.info("Database health check: OK")
        return True, "Database connection successful"
        
    except Exception as e:
        logger.error(f"Database health check failed: {e}")
        return False, f"Database connection failed: {str(e)}"


def check_redis_health(config: dict) -> Tuple[bool, str]:
    """Check Redis connectivity"""
    logger.info("Checking Redis health...")
    
    try:
        host = resolve_env_vars(config['cache']['redis_host'])
        port = config['cache']['redis_port']
        db = config['cache']['redis_db']
        
        client = redis.Redis(
            host=host,
            port=port,
            db=db,
            socket_timeout=5,
            socket_connect_timeout=5
        )
        
        # Test connection
        client.ping()
        
        logger.info("Redis health check: OK")
        return True, "Redis connection successful"
        
    except Exception as e:
        logger.error(f"Redis health check failed: {e}")
        return False, f"Redis connection failed: {str(e)}"


def check_api_health(config: dict) -> Tuple[bool, str]:
    """Check Wikimedia API availability"""
    logger.info("Checking API health...")
    
    try:
        import requests
        
        base_url = config['api']['wikimedia_base_url']
        timeout = config['api'].get('timeout', 30)
        
        # Test API endpoint
        response = requests.get(
            f"{base_url}/metrics/pageviews/",
            timeout=timeout
        )
        
        if response.status_code in [200, 404]:  # 404 is OK, means API is responding
            logger.info("API health check: OK")
            return True, "API is accessible"
        else:
            return False, f"API returned status code: {response.status_code}"
            
    except Exception as e:
        logger.error(f"API health check failed: {e}")
        return False, f"API check failed: {str(e)}"


def run_health_checks(config: dict, max_retries: int = 3, retry_delay: int = 5) -> bool:
    """Run all health checks with retries"""
    checks = {
        'database': check_database_health,
        'redis': check_redis_health,
        'api': check_api_health
    }
    
    results = {}
    
    for check_name, check_func in checks.items():
        success = False
        message = ""
        
        for attempt in range(max_retries):
            if attempt > 0:
                logger.info(f"Retrying {check_name} health check (attempt {attempt + 1}/{max_retries})")
                time.sleep(retry_delay)
            
            success, message = check_func(config)
            
            if success:
                break
        
        results[check_name] = (success, message)
        
        if not success:
            logger.error(f"{check_name} health check failed after {max_retries} attempts: {message}")
    
    # Check if all critical checks passed
    critical_checks = ['database', 'redis']
    all_critical_passed = all(results[check][0] for check in critical_checks if check in results)
    
    if not all_critical_passed:
        logger.error("Critical health checks failed. Cannot start application.")
        return False
    
    # Warn about non-critical failures
    if not results.get('api', (True, ''))[0]:
        logger.warning("API health check failed, but continuing startup")
    
    logger.info("All critical health checks passed")
    return True


def initialize_directories():
    """Create necessary directories"""
    directories = ['logs', 'data', 'output', 'backups']
    
    for directory in directories:
        path = Path(__file__).parent.parent / directory
        path.mkdir(parents=True, exist_ok=True)
        logger.info(f"Ensured directory exists: {path}")


def run_database_migrations():
    """Run database migrations if needed"""
    logger.info("Checking for pending database migrations...")
    
    try:
        from alembic.config import Config
        from alembic import command
        
        alembic_cfg = Config(str(Path(__file__).parent.parent / "alembic.ini"))
        command.upgrade(alembic_cfg, "head")
        
        logger.info("Database migrations completed")
        return True
        
    except Exception as e:
        logger.error(f"Database migration failed: {e}")
        return False


def start_application(config: dict):
    """Start the main application"""
    logger.info("Starting Wikipedia Intelligence System...")
    
    # Import and start the dashboard
    try:
        import subprocess
        
        dashboard_port = config.get('dashboard', {}).get('port', 8501)
        
        # Start Streamlit dashboard
        cmd = [
            'streamlit', 'run',
            str(Path(__file__).parent.parent / 'src' / 'visualization' / 'dashboard.py'),
            '--server.port', str(dashboard_port),
            '--server.address', '0.0.0.0'
        ]
        
        logger.info(f"Starting dashboard on port {dashboard_port}")
        subprocess.run(cmd)
        
    except Exception as e:
        logger.error(f"Failed to start application: {e}")
        sys.exit(1)


def main():
    """Main startup sequence"""
    logger.info("=" * 60)
    logger.info("Wikipedia Intelligence System - Startup")
    logger.info("=" * 60)
    
    # Load configuration
    config = load_config()
    
    # Validate configuration
    logger.info("Validating configuration...")
    valid, errors = validate_config(config)
    if not valid:
        logger.error("Configuration validation failed:")
        for error in errors:
            logger.error(f"  - {error}")
        sys.exit(1)
    logger.info("Configuration validation: OK")
    
    # Initialize directories
    initialize_directories()
    
    # Run health checks
    if not run_health_checks(config):
        logger.error("Health checks failed. Exiting.")
        sys.exit(1)
    
    # Run database migrations
    if not run_database_migrations():
        logger.warning("Database migrations failed, but continuing startup")
    
    # Start application
    logger.info("All startup checks passed. Starting application...")
    start_application(config)


if __name__ == '__main__':
    try:
        main()
    except KeyboardInterrupt:
        logger.info("Shutdown requested by user")
        sys.exit(0)
    except Exception as e:
        logger.error(f"Startup failed: {e}", exc_info=True)
        sys.exit(1)
