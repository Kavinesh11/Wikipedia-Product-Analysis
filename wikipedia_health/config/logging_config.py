"""Logging configuration for Wikipedia Health Analysis System."""

import logging
import logging.config
from pathlib import Path
from typing import Optional


def setup_logging(
    log_level: str = "INFO",
    log_file: Optional[Path] = None,
    log_to_console: bool = True
) -> None:
    """Configure logging for the application.
    
    Args:
        log_level: Logging level (DEBUG, INFO, WARNING, ERROR, CRITICAL)
        log_file: Optional path to log file
        log_to_console: Whether to log to console
    """
    handlers = {}
    
    if log_to_console:
        handlers['console'] = {
            'class': 'logging.StreamHandler',
            'level': log_level,
            'formatter': 'standard',
            'stream': 'ext://sys.stdout'
        }
    
    if log_file:
        log_file.parent.mkdir(parents=True, exist_ok=True)
        handlers['file'] = {
            'class': 'logging.handlers.RotatingFileHandler',
            'level': log_level,
            'formatter': 'detailed',
            'filename': str(log_file),
            'maxBytes': 10485760,  # 10MB
            'backupCount': 5
        }
    
    config = {
        'version': 1,
        'disable_existing_loggers': False,
        'formatters': {
            'standard': {
                'format': '%(asctime)s - %(name)s - %(levelname)s - %(message)s',
                'datefmt': '%Y-%m-%d %H:%M:%S'
            },
            'detailed': {
                'format': '%(asctime)s - %(name)s - %(levelname)s - %(funcName)s:%(lineno)d - %(message)s',
                'datefmt': '%Y-%m-%d %H:%M:%S'
            }
        },
        'handlers': handlers,
        'root': {
            'level': log_level,
            'handlers': list(handlers.keys())
        },
        'loggers': {
            'wikipedia_health': {
                'level': log_level,
                'handlers': list(handlers.keys()),
                'propagate': False
            }
        }
    }
    
    logging.config.dictConfig(config)


def get_logger(name: str) -> logging.Logger:
    """Get a logger instance.
    
    Args:
        name: Logger name (typically __name__)
        
    Returns:
        Logger instance
    """
    return logging.getLogger(name)
