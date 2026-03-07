"""Unit tests for configuration management"""
import pytest
import os
from pathlib import Path
from src.utils.config import Config, ConfigurationError, get_config


def test_config_loads_from_file(tmp_path):
    """Test configuration loads from YAML file"""
    # Create temporary config file
    config_file = tmp_path / "config.yaml"
    config_file.write_text("""
development:
  database:
    postgres_host: test_host
    postgres_port: 5432
    postgres_db: test_db
    postgres_user: test_user
    postgres_password: test_pass
  cache:
    redis_host: test_redis
    redis_port: 6379
  api:
    wikimedia_base_url: https://test.api
""")
    
    config = Config(profile="development", config_path=config_file)
    
    assert config.get("database", "postgres_host") == "test_host"
    assert config.get("cache", "redis_host") == "test_redis"


def test_config_validates_required_sections(tmp_path):
    """Test configuration validation fails with missing sections"""
    config_file = tmp_path / "config.yaml"
    config_file.write_text("""
development:
  database:
    postgres_host: test_host
""")
    
    with pytest.raises(ConfigurationError):
        Config(profile="development", config_path=config_file)


def test_config_database_url():
    """Test database URL generation"""
    config_file = Path("config/config.yaml")
    if config_file.exists():
        config = Config(profile="development", config_path=config_file)
        db_url = config.database_url
        assert db_url.startswith("postgresql://")
        assert "wikipedia_intelligence_dev" in db_url
