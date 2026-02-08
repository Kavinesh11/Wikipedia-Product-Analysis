"""Tests for logging configuration."""

import logging
from pathlib import Path
import tempfile
from wikipedia_health.config import setup_logging, get_logger


def test_setup_logging_console_only():
    """Test logging setup with console output only."""
    setup_logging(log_level="INFO", log_to_console=True)
    logger = get_logger("test_logger")
    
    assert logger.level <= logging.INFO
    assert isinstance(logger, logging.Logger)


def test_setup_logging_with_file():
    """Test logging setup with file output."""
    with tempfile.TemporaryDirectory() as tmpdir:
        log_file = Path(tmpdir) / "test.log"
        setup_logging(log_level="DEBUG", log_file=log_file, log_to_console=False)
        
        logger = get_logger("test_file_logger")
        logger.info("Test message")
        
        # Verify file was created
        file_exists = log_file.exists()
        
        # Close all handlers to release file locks on Windows
        root_logger = logging.getLogger()
        for handler in root_logger.handlers[:]:
            handler.close()
            root_logger.removeHandler(handler)
        
        for handler in logger.handlers[:]:
            handler.close()
            logger.removeHandler(handler)
        
        assert file_exists


def test_get_logger():
    """Test getting a logger instance."""
    logger = get_logger("wikipedia_health.test")
    
    assert isinstance(logger, logging.Logger)
    assert logger.name == "wikipedia_health.test"
