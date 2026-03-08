"""Property-based tests for configuration management

Feature: wikipedia-intelligence-system
"""
import pytest
import os
import yaml
import tempfile
from pathlib import Path
from hypothesis import given, strategies as st, assume, settings, HealthCheck
from src.utils.config import Config, ConfigurationError


# Feature: wikipedia-intelligence-system, Property 67: Configuration Loading
@given(
    config_value=st.text(min_size=1, max_size=50, alphabet=st.characters(blacklist_categories=('Cs', 'Cc'))),
    env_value=st.text(min_size=1, max_size=50, alphabet=st.characters(blacklist_categories=('Cs', 'Cc')))
)
@settings(suppress_health_check=[HealthCheck.function_scoped_fixture], max_examples=5)
def test_property_67_env_vars_override_config_file(config_value, env_value):
    """Property 67: Environment variables should override config file values
    
    For any configuration parameter, the System should load it from environment 
    variables first, then config files, with environment variables taking precedence.
    
    Validates: Requirements 15.1
    """
    # Assume values are different to test override
    assume(config_value != env_value)
    assume(config_value.strip() and env_value.strip())
    
    with tempfile.TemporaryDirectory() as tmp_dir:
        # Create config file with one value
        config_file = Path(tmp_dir) / "config.yaml"
        config_data = {
            "development": {
                "database": {
                    "postgres_host": config_value,
                    "postgres_port": 5432,
                    "postgres_db": "test_db",
                    "postgres_user": "test_user",
                    "postgres_password": "test_pass"
                },
                "cache": {
                    "redis_host": "localhost",
                    "redis_port": 6379
                },
                "api": {
                    "wikimedia_base_url": "https://test.api"
                }
            }
        }
        config_file.write_text(yaml.dump(config_data))

        # Set environment variable with different value
        os.environ["POSTGRES_HOST"] = env_value
        
        try:
            config = Config(profile="development", config_path=config_file)
            
            # Environment variable should take precedence
            assert config.get("database", "postgres_host") == env_value
        finally:
            # Clean up
            if "POSTGRES_HOST" in os.environ:
                del os.environ["POSTGRES_HOST"]


# Feature: wikipedia-intelligence-system, Property 68: Configuration Profile Support
@given(profile=st.sampled_from(["development", "staging", "production"]))
@settings(max_examples=5)
def test_property_68_profile_specific_settings(profile):
    """Property 68: Loading a profile should apply profile-specific settings
    
    For any configuration profile (development, staging, production), loading that 
    profile should apply the profile-specific settings.
    
    Validates: Requirements 15.2
    """
    with tempfile.TemporaryDirectory() as tmp_dir:
        # Create config file with profile-specific values
        config_file = Path(tmp_dir) / "config.yaml"
        config_data = {
            "development": {
                "database": {
                    "postgres_host": "dev_host",
                    "postgres_port": 5432,
                    "postgres_db": "dev_db",
                    "postgres_user": "dev_user",
                    "postgres_password": "dev_pass"
                },
                "cache": {
                    "redis_host": "dev_redis",
                    "redis_port": 6379
                },
                "api": {
                    "wikimedia_base_url": "https://dev.api"
                }
            },
            "staging": {
                "database": {
                    "postgres_host": "staging_host",
                    "postgres_port": 5432,
                    "postgres_db": "staging_db",
                    "postgres_user": "staging_user",
                    "postgres_password": "staging_pass"
                },
                "cache": {
                    "redis_host": "staging_redis",
                    "redis_port": 6379
                },
                "api": {
                    "wikimedia_base_url": "https://staging.api"
                }
            },

            "production": {
                "database": {
                    "postgres_host": "prod_host",
                    "postgres_port": 5432,
                    "postgres_db": "prod_db",
                    "postgres_user": "prod_user",
                    "postgres_password": "prod_pass"
                },
                "cache": {
                    "redis_host": "prod_redis",
                    "redis_port": 6379
                },
                "api": {
                    "wikimedia_base_url": "https://prod.api"
                }
            }
        }
        config_file.write_text(yaml.dump(config_data))
        
        config = Config(profile=profile, config_path=config_file)
        
        # Verify profile-specific settings are loaded
        if profile == "development":
            expected_host = "dev_host"
        elif profile == "staging":
            expected_host = "staging_host"
        else:  # production
            expected_host = "prod_host"
        
        assert config.get("database", "postgres_host") == expected_host


# Feature: wikipedia-intelligence-system, Property 69: Configuration Validation on Startup
@given(
    missing_section=st.sampled_from(["database", "cache", "api"])
)
@settings(max_examples=5)
def test_property_69_validation_fails_on_missing_required_section(missing_section):
    """Property 69: System should fail startup with clear error on invalid configuration
    
    For any invalid configuration (missing required parameter, invalid value type, 
    out-of-range value), the System should fail startup with a clear error message.
    
    Validates: Requirements 15.3
    """
    with tempfile.TemporaryDirectory() as tmp_dir:
        # Create config with missing required section
        config_file = Path(tmp_dir) / "config.yaml"
        config_data = {
            "development": {
                "database": {
                    "postgres_host": "localhost",
                    "postgres_port": 5432,
                    "postgres_db": "test_db",
                    "postgres_user": "test_user",
                    "postgres_password": "test_pass"
                },
                "cache": {
                    "redis_host": "localhost",
                    "redis_port": 6379
                },
                "api": {
                    "wikimedia_base_url": "https://test.api"
                }
            }
        }
        
        # Remove the specified section
        del config_data["development"][missing_section]
        config_file.write_text(yaml.dump(config_data))
        
        # Should raise ConfigurationError
        with pytest.raises(ConfigurationError) as exc_info:
            Config(profile="development", config_path=config_file)
        
        # Error message should mention the missing section
        assert missing_section in str(exc_info.value).lower()



@given(
    missing_key=st.sampled_from(["postgres_host", "postgres_db", "postgres_user", "postgres_password"])
)
@settings(max_examples=5)
def test_property_69_validation_fails_on_missing_required_key(missing_key):
    """Property 69: System should fail startup on missing required database keys
    
    Validates: Requirements 15.3
    """
    with tempfile.TemporaryDirectory() as tmp_dir:
        # Create config with missing required key
        config_file = Path(tmp_dir) / "config.yaml"
        config_data = {
            "development": {
                "database": {
                    "postgres_host": "localhost",
                    "postgres_port": 5432,
                    "postgres_db": "test_db",
                    "postgres_user": "test_user",
                    "postgres_password": "test_pass"
                },
                "cache": {
                    "redis_host": "localhost",
                    "redis_port": 6379
                },
                "api": {
                    "wikimedia_base_url": "https://test.api"
                }
            }
        }
        
        # Remove the specified key
        del config_data["development"]["database"][missing_key]
        config_file.write_text(yaml.dump(config_data))
        
        # Should raise ConfigurationError
        with pytest.raises(ConfigurationError) as exc_info:
            Config(profile="development", config_path=config_file)
        
        # Error message should mention the missing key
        assert missing_key in str(exc_info.value).lower()
