"""Comprehensive error handling for Fortune 500 Knowledge Graph Analytics.

Implements error handling patterns for:
- Data ingestion (Requirements 1.5, 15.4)
- Analytics and ML (Requirements 8.1, 16.5)
- Visualization and export (Requirements 6.1, 17.3, 17.4)
"""

import logging
import time
from typing import Callable, Any, Optional, List, Dict
from functools import wraps

logger = logging.getLogger(__name__)

# Maximum retry attempts for network operations
MAX_RETRIES = 3
# Failure rate threshold for alerts (10%)
FAILURE_RATE_THRESHOLD = 0.10


def retry_with_backoff(
    max_retries: int = MAX_RETRIES,
    initial_delay: float = 1.0,
    backoff_factor: float = 2.0,
    exceptions: tuple = (Exception,),
):
    """
    Decorator for retrying functions with exponential backoff.

    Args:
        max_retries: Maximum number of retry attempts
        initial_delay: Initial delay in seconds
        backoff_factor: Multiplier for each subsequent delay
        exceptions: Tuple of exception types to catch and retry

    Validates: Requirements 1.5, 17.3, 17.4
    """
    def decorator(func: Callable) -> Callable:
        @wraps(func)
        def wrapper(*args, **kwargs):
            delay = initial_delay
            last_exception = None
            for attempt in range(max_retries + 1):
                try:
                    return func(*args, **kwargs)
                except exceptions as e:
                    last_exception = e
                    if attempt < max_retries:
                        logger.warning(
                            "Attempt %d/%d failed for %s: %s. Retrying in %.1fs...",
                            attempt + 1,
                            max_retries,
                            func.__name__,
                            e,
                            delay,
                        )
                        time.sleep(delay)
                        delay *= backoff_factor
                    else:
                        logger.error(
                            "All %d attempts failed for %s: %s",
                            max_retries,
                            func.__name__,
                            e,
                        )
            raise last_exception
        return wrapper
    return decorator


class ErrorHandler:
    """
    Centralized error handling for all system components.

    Tracks failure rates and generates alerts when thresholds are exceeded.
    """

    def __init__(self) -> None:
        self._operation_counts: Dict[str, int] = {}
        self._failure_counts: Dict[str, int] = {}
        self._error_log: List[Dict[str, Any]] = []

    def record_operation(self, operation_name: str, success: bool, error: Optional[Exception] = None) -> None:
        """Record an operation result for failure rate tracking."""
        self._operation_counts[operation_name] = self._operation_counts.get(operation_name, 0) + 1
        if not success:
            self._failure_counts[operation_name] = self._failure_counts.get(operation_name, 0) + 1
            if error:
                self._error_log.append({
                    "operation": operation_name,
                    "error": str(error),
                    "error_type": type(error).__name__,
                })

    def get_failure_rate(self, operation_name: str) -> float:
        """
        Calculate failure rate for an operation.

        Returns:
            Failure rate as a fraction [0.0, 1.0]

        Validates: Requirement 1.5 (alert if failure rate > 10%)
        """
        total = self._operation_counts.get(operation_name, 0)
        failures = self._failure_counts.get(operation_name, 0)
        if total == 0:
            return 0.0
        return failures / total

    def check_failure_rate_alert(self, operation_name: str) -> bool:
        """
        Check if failure rate exceeds the alert threshold (10%).

        Returns:
            True if alert should be generated, False otherwise

        Validates: Requirement 1.5
        """
        rate = self.get_failure_rate(operation_name)
        if rate > FAILURE_RATE_THRESHOLD:
            logger.warning(
                "ALERT: Failure rate for '%s' is %.1f%% (threshold: %.1f%%)",
                operation_name,
                rate * 100,
                FAILURE_RATE_THRESHOLD * 100,
            )
            return True
        return False

    def handle_invalid_metric_calculation(
        self,
        company_id: str,
        metric_name: str,
        error: Exception,
        default_value: float = 0.0,
    ) -> float:
        """
        Handle invalid metric calculations (division by zero, null values).

        Logs the error and returns a default value.

        Args:
            company_id: Company identifier
            metric_name: Name of the metric being calculated
            error: The exception that occurred
            default_value: Value to return when calculation fails

        Returns:
            Default value for the metric

        Validates: Requirement 8.1 (handle invalid calculations)
        """
        logger.warning(
            "Invalid metric calculation for company '%s', metric '%s': %s. "
            "Using default value %.4f",
            company_id,
            metric_name,
            error,
            default_value,
        )
        self.record_operation(f"metric_calculation_{metric_name}", success=False, error=error)
        return default_value

    def handle_visualization_error(
        self,
        visualization_type: str,
        error: Exception,
    ) -> Dict[str, Any]:
        """
        Handle visualization rendering failures.

        Returns an error placeholder visualization.

        Args:
            visualization_type: Type of visualization that failed
            error: The exception that occurred

        Returns:
            Error placeholder dict

        Validates: Requirement 6.1 (handle Bloom connection failures)
        """
        logger.error(
            "Visualization rendering failed for '%s': %s",
            visualization_type,
            error,
        )
        self.record_operation(f"visualization_{visualization_type}", success=False, error=error)
        return {
            "error": True,
            "visualization_type": visualization_type,
            "error_message": str(error),
            "placeholder": True,
        }

    def handle_export_error(
        self,
        export_format: str,
        error: Exception,
        fallback_path: Optional[str] = None,
    ) -> Dict[str, Any]:
        """
        Handle file system and export errors.

        Args:
            export_format: Export format that failed ('csv', 'json', 'tableau', 'power_bi')
            error: The exception that occurred
            fallback_path: Optional alternative path to retry

        Returns:
            Error result dict

        Validates: Requirements 17.3, 17.4
        """
        logger.error(
            "Export failed for format '%s': %s",
            export_format,
            error,
        )
        self.record_operation(f"export_{export_format}", success=False, error=error)
        return {
            "status": "failed",
            "format": export_format,
            "error": str(error),
            "fallback_path": fallback_path,
        }
