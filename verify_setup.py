"""Verify project setup is complete."""

import sys
from pathlib import Path


def verify_structure():
    """Verify project structure."""
    print("Verifying project structure...")
    
    required_dirs = [
        "wikipedia_health",
        "wikipedia_health/data_acquisition",
        "wikipedia_health/time_series",
        "wikipedia_health/statistical_validation",
        "wikipedia_health/causal_inference",
        "wikipedia_health/evidence_framework",
        "wikipedia_health/visualization",
        "wikipedia_health/models",
        "wikipedia_health/utils",
        "wikipedia_health/config",
        "tests",
    ]
    
    required_files = [
        "pyproject.toml",
        "pytest.ini",
        "config.yaml",
        "README.md",
        ".gitignore",
        "wikipedia_health/__init__.py",
        "wikipedia_health/config/config.py",
        "wikipedia_health/config/logging_config.py",
        "tests/conftest.py",
        "tests/test_config.py",
        "tests/test_logging.py",
    ]
    
    missing_dirs = []
    missing_files = []
    
    for dir_path in required_dirs:
        if not Path(dir_path).is_dir():
            missing_dirs.append(dir_path)
    
    for file_path in required_files:
        if not Path(file_path).is_file():
            missing_files.append(file_path)
    
    if missing_dirs:
        print(f"\n❌ Missing directories: {', '.join(missing_dirs)}")
        return False
    
    if missing_files:
        print(f"\n❌ Missing files: {', '.join(missing_files)}")
        return False
    
    print("✓ All required directories and files present")
    return True


def verify_config():
    """Verify configuration can be loaded."""
    print("\nVerifying configuration...")
    
    try:
        from wikipedia_health.config import Config, load_config
        
        # Test default config
        config = Config()
        assert config.api.pageviews_endpoint is not None
        assert config.statistical.significance_level == 0.05
        
        # Test loading from file
        config = load_config(Path("config.yaml"))
        assert config.api.timeout == 30
        
        print("✓ Configuration loading works")
        return True
    except Exception as e:
        print(f"❌ Configuration error: {e}")
        return False


def verify_logging():
    """Verify logging configuration."""
    print("\nVerifying logging...")
    
    try:
        from wikipedia_health.config import setup_logging, get_logger
        
        setup_logging(log_level="INFO")
        logger = get_logger("test")
        logger.info("Test message")
        
        print("✓ Logging configuration works")
        return True
    except Exception as e:
        print(f"❌ Logging error: {e}")
        return False


def main():
    """Run all verification checks."""
    print("=" * 60)
    print("Wikipedia Health Analysis - Setup Verification")
    print("=" * 60)
    
    checks = [
        verify_structure(),
        verify_config(),
        verify_logging(),
    ]
    
    print("\n" + "=" * 60)
    if all(checks):
        print("✓ All verification checks passed!")
        print("\nNext steps:")
        print("1. Install dependencies: python setup_dev.py")
        print("2. Run tests: pytest tests/")
        print("3. Start implementing Task 2")
        return 0
    else:
        print("❌ Some verification checks failed")
        return 1


if __name__ == "__main__":
    sys.exit(main())
