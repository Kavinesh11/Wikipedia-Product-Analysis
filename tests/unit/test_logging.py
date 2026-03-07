"""Unit tests for logging configuration"""
import logging
import json
from src.utils.logging_config import setup_logging, get_logger


def test_setup_logging_creates_logger():
    """Test logging setup creates configured logger"""
    logger = setup_logging(level="INFO", json_format=False)
    
    assert logger is not None
    assert logger.level == logging.INFO
    assert len(logger.handlers) > 0


def test_get_logger_returns_logger():
    """Test get_logger returns a logger instance"""
    logger = get_logger("test_module")
    
    assert logger is not None
    assert isinstance(logger, logging.Logger)
    assert logger.name == "test_module"


def test_logger_logs_messages(caplog):
    """Test logger can log messages"""
    logger = get_logger("test")
    
    with caplog.at_level(logging.INFO):
        logger.info("Test message")
    
    assert "Test message" in caplog.text


def test_error_logging_with_stack_traces(caplog):
    """Test error logging includes stack traces"""
    logger = get_logger("test_errors")
    
    with caplog.at_level(logging.ERROR):
        try:
            raise ValueError("Test error")
        except ValueError as e:
            logger.error("An error occurred", exc_info=True)
    
    assert "An error occurred" in caplog.text
    assert "ValueError" in caplog.text
    assert "Test error" in caplog.text


def test_structured_json_log_format(tmp_path):
    """Test structured JSON log format"""
    log_file = tmp_path / "test.log"
    logger = setup_logging(level="INFO", log_file=str(log_file), json_format=True)
    
    test_logger = get_logger("json_test")
    test_logger.info("Test JSON message", extra={"custom_field": "value"})
    
    # Read log file and verify JSON format
    with open(log_file, 'r') as f:
        log_line = f.readline()
        log_data = json.loads(log_line)
        
        assert "message" in log_data
        assert "level" in log_data
        assert log_data["level"] == "INFO"
        assert "timestamp" in log_data


def test_log_level_filtering():
    """Test log level filtering"""
    # Test that logger respects level setting
    logger = setup_logging(level="WARNING", json_format=False)
    
    assert logger.level == logging.WARNING
    
    # Test that a logger at WARNING level filters correctly
    test_logger = logging.getLogger("level_test")
    test_logger.setLevel(logging.WARNING)
    
    # Verify level is set correctly
    assert test_logger.isEnabledFor(logging.WARNING)
    assert test_logger.isEnabledFor(logging.ERROR)
    assert not test_logger.isEnabledFor(logging.INFO)
    assert not test_logger.isEnabledFor(logging.DEBUG)
