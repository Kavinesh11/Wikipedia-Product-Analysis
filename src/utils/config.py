"""Configuration Management Module

Loads configuration from environment variables and config files.
Supports multiple profiles (development, staging, production).
"""
import os
import yaml
from typing import Any, Dict, Optional
from pathlib import Path
from dotenv import load_dotenv


class ConfigurationError(Exception):
    """Raised when configuration is invalid or missing"""
    pass


class Config:
    """Configuration manager for the Wikipedia Intelligence System"""
    
    def __init__(self, profile: str = "development", config_path: Optional[Path] = None):
        """Initialize configuration
        
        Args:
            profile: Configuration profile (development, staging, production)
            config_path: Path to config file (defaults to config/config.yaml)
        """
        self.profile = profile
        self.config_path = config_path or Path("config/config.yaml")
        self._config: Dict[str, Any] = {}
        
        # Load environment variables
        load_dotenv()
        
        # Load configuration
        self._load_config()
        self._validate_config()

    def _load_config(self) -> None:
        """Load configuration from file and environment variables"""
        # Load from YAML file if exists
        if self.config_path.exists():
            with open(self.config_path, 'r') as f:
                file_config = yaml.safe_load(f) or {}
                
            # Get profile-specific config
            if self.profile in file_config:
                self._config = file_config[self.profile]
            else:
                self._config = file_config.get('default', {})
        
        # Override with environment variables
        self._load_from_env()
    
    def _load_from_env(self) -> None:
        """Load configuration from environment variables"""
        env_mappings = {
            'WIKIMEDIA_API_BASE_URL': ('api', 'wikimedia_base_url'),
            'WIKIMEDIA_RATE_LIMIT': ('api', 'rate_limit'),
            'POSTGRES_HOST': ('database', 'postgres_host'),
            'POSTGRES_PORT': ('database', 'postgres_port'),
            'POSTGRES_DB': ('database', 'postgres_db'),
            'POSTGRES_USER': ('database', 'postgres_user'),
            'POSTGRES_PASSWORD': ('database', 'postgres_password'),
            'REDIS_HOST': ('cache', 'redis_host'),
            'REDIS_PORT': ('cache', 'redis_port'),
            'REDIS_DB': ('cache', 'redis_db'),
            'LOG_LEVEL': ('logging', 'level'),
        }
        
        for env_var, (section, key) in env_mappings.items():
            value = os.getenv(env_var)
            if value is not None:
                if section not in self._config:
                    self._config[section] = {}
                self._config[section][key] = value

    def _validate_config(self) -> None:
        """Validate required configuration parameters"""
        required_sections = ['database', 'cache', 'api']
        
        for section in required_sections:
            if section not in self._config:
                raise ConfigurationError(
                    f"Missing required configuration section: {section}"
                )
        
        # Validate database config
        db_required = ['postgres_host', 'postgres_db', 'postgres_user', 'postgres_password']
        for key in db_required:
            if key not in self._config['database']:
                raise ConfigurationError(
                    f"Missing required database configuration: {key}"
                )
        
        # Validate cache config
        cache_required = ['redis_host', 'redis_port']
        for key in cache_required:
            if key not in self._config['cache']:
                raise ConfigurationError(
                    f"Missing required cache configuration: {key}"
                )
    
    def get(self, section: str, key: str, default: Any = None) -> Any:
        """Get configuration value
        
        Args:
            section: Configuration section
            key: Configuration key
            default: Default value if not found
            
        Returns:
            Configuration value
        """
        return self._config.get(section, {}).get(key, default)
    
    def get_section(self, section: str) -> Dict[str, Any]:
        """Get entire configuration section
        
        Args:
            section: Configuration section name
            
        Returns:
            Dictionary of configuration values
        """
        return self._config.get(section, {})

    @property
    def database_url(self) -> str:
        """Get database URL (PostgreSQL or SQLite)"""
        db_config = self._config['database']
        
        # Check if SQLite is configured for development
        if db_config.get('use_sqlite', False):
            sqlite_path = db_config.get('sqlite_path', 'data/wikipedia_intelligence.db')
            return f"sqlite:///{sqlite_path}"
        
        # Otherwise use PostgreSQL
        host = db_config['postgres_host']
        port = db_config.get('postgres_port', 5432)
        db = db_config['postgres_db']
        user = db_config['postgres_user']
        password = db_config['postgres_password']
        return f"postgresql://{user}:{password}@{host}:{port}/{db}"
    
    @property
    def redis_url(self) -> str:
        """Get Redis connection URL"""
        cache_config = self._config['cache']
        host = cache_config['redis_host']
        port = cache_config.get('redis_port', 6379)
        db = cache_config.get('redis_db', 0)
        return f"redis://{host}:{port}/{db}"


# Global config instance
_config_instance: Optional[Config] = None


def get_config(profile: Optional[str] = None) -> Config:
    """Get global configuration instance
    
    Args:
        profile: Configuration profile (defaults to ENVIRONMENT env var or 'development')
        
    Returns:
        Config instance
    """
    global _config_instance
    
    if _config_instance is None:
        if profile is None:
            profile = os.getenv('ENVIRONMENT', 'development')
        _config_instance = Config(profile=profile)
    
    return _config_instance
