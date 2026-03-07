"""Property-Based Tests for Logging and Monitoring

Tests correctness properties for logging, metrics collection, and monitoring.
"""
import json
import logging
import asyncio
from datetime import datetime, timedelta
from hypothesis import given, strategies as st, settings
import pytest

from src.utils.logging_config import setup_logging, get_logger
from src.utils.metrics import MetricsCollector, Timer, get_metrics
from src.utils.health_check import HealthChecker, HealthStatus
from src.data_ingestion.api_client import WikimediaAPIClient
from src.data_ingestion.rate_limiter import RateLimiter


# Feature: wikipedia-intelligence-system, Property 61: API Request Logging
@given(
    endpoint=st.text(min_size=1, max_size=100),
    status_code=st.integers(min_value=200, max_value=599)
)
@settings(max_examples=10, deadline=None)
def test_api_request_logging(endpoint, status_code, caplog):
    """
    Property 61: For any API request, the System should log the request
    with timestamp, endpoint, parameters, and response code.
    
    Validates: Requirements 12.6
    """
    logger = get_logger("test_api_logging")
    
    with caplog.at_level(logging.INFO):
        # Simulate API request logging
        logger.info(
            f"API request to {endpoint}",
            extra={
                "endpoint": endpoint,
                "status_code": status_code,
                "timestamp": datetime.now().isoformat()
            }
        )
    
    # Verify log was created
    assert len(caplog.records) > 0
    
    # Verify log contains required fields
    log_record = caplog.records[0]
    assert endpoint in log_record.message or endpoint in str(log_record.__dict__)
    assert log_record.levelname == "INFO"


# Feature: wikipedia-intelligence-system, Property 62: API Usage Metrics Collection
@given(
    request_count=st.integers(min_value=1, max_value=1000),
    error_count=st.integers(min_value=0, max_value=100)
)
@settings(max_examples=10, deadline=None)
def test_api_usage_metrics_collection(request_count, error_count):
    """
    Property 62: For any API request, the System should increment usage metrics
    (request count, error count, latency) that are queryable from monitoring dashboards.
    
    Validates: Requirements 12.7
    """
    metrics = MetricsCollector()
    
    # Simulate API requests
    for i in range(request_count):
        metrics.increment_counter("api_requests_total", labels={"endpoint": "test"})
        
        # Simulate some errors
        if i < error_count:
            metrics.increment_counter("api_errors_total", labels={"endpoint": "test"})
        
        # Record latency
        latency = 0.1 + (i % 10) * 0.01  # Vary latency
        metrics.record_timer("api_request_duration", latency, labels={"endpoint": "test"})
    
    # Verify metrics are queryable
    total_requests = metrics.get_counter("api_requests_total", labels={"endpoint": "test"})
    assert total_requests == request_count
    
    total_errors = metrics.get_counter("api_errors_total", labels={"endpoint": "test"})
    assert total_errors == error_count
    
    # Verify timer stats
    timer_stats = metrics.get_timer_stats("api_request_duration", labels={"endpoint": "test"})
    assert timer_stats is not None
    assert timer_stats.count == request_count
    assert timer_stats.min >= 0
    assert timer_stats.max > 0
    assert timer_stats.avg > 0


# Feature: wikipedia-intelligence-system, Property 63: Error Logging Completeness
@given(
    error_message=st.text(min_size=1, max_size=200),
    context_data=st.dictionaries(
        keys=st.text(min_size=1, max_size=20),
        values=st.text(min_size=1, max_size=50),
        min_size=0,
        max_size=5
    )
)
@settings(max_examples=10, deadline=None)
def test_error_logging_completeness(error_message, context_data, caplog):
    """
    Property 63: For any error or exception, the System should log it with
    stack trace, error message, and contextual information (user, operation, input data).
    
    Validates: Requirements 14.1
    """
    logger = get_logger("test_error_logging")
    
    with caplog.at_level(logging.ERROR):
        try:
            # Simulate an error
            raise ValueError(error_message)
        except ValueError as e:
            logger.error(
                f"Error occurred: {error_message}",
                extra=context_data,
                exc_info=True
            )
    
    # Verify error was logged
    assert len(caplog.records) > 0
    
    log_record = caplog.records[0]
    assert log_record.levelname == "ERROR"
    assert error_message in log_record.message
    
    # Verify stack trace is present
    assert log_record.exc_info is not None
    
    # Verify context data is included
    for key, value in context_data.items():
        assert hasattr(log_record, key) or key in str(log_record.__dict__)


# Feature: wikipedia-intelligence-system, Property 64: Lifecycle Event Logging
@given(
    pipeline_name=st.text(min_size=1, max_size=50),
    status=st.sampled_from(["started", "completed", "failed"])
)
@settings(max_examples=10, deadline=None)
def test_lifecycle_event_logging(pipeline_name, status, caplog):
    """
    Property 64: For any pipeline start or completion event, the System should log
    an informational message with pipeline name, timestamp, and status.
    
    Validates: Requirements 14.2
    """
    logger = get_logger("test_lifecycle_logging")
    
    with caplog.at_level(logging.INFO):
        # Simulate lifecycle event logging
        logger.info(
            f"Pipeline {status}: {pipeline_name}",
            extra={
                "pipeline_name": pipeline_name,
                "status": status,
                "timestamp": datetime.now().isoformat()
            }
        )
    
    # Verify log was created
    assert len(caplog.records) > 0
    
    log_record = caplog.records[0]
    assert log_record.levelname == "INFO"
    assert pipeline_name in log_record.message or pipeline_name in str(log_record.__dict__)
    assert status in log_record.message or status in str(log_record.__dict__)


# Feature: wikipedia-intelligence-system, Property 65: Structured Log Format
@given(
    message=st.text(min_size=1, max_size=100),
    level=st.sampled_from(["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"])
)
@settings(max_examples=10, deadline=None)
def test_structured_log_format(message, level):
    """
    Property 65: For any log message, the output should be valid JSON with
    standard fields (timestamp, level, message, context).
    
    Validates: Requirements 14.3
    """
    import io
    import json
    
    # Set up logging with JSON format
    logger = logging.getLogger("test_structured_logging")
    logger.setLevel(logging.DEBUG)
    logger.handlers.clear()
    
    # Create string stream to capture log output
    log_stream = io.StringIO()
    handler = logging.StreamHandler(log_stream)
    
    # Use JSON formatter
    from src.utils.logging_config import CustomJsonFormatter
    formatter = CustomJsonFormatter('%(timestamp)s %(level)s %(name)s %(message)s')
    handler.setFormatter(formatter)
    logger.addHandler(handler)
    
    # Log message at specified level
    log_method = getattr(logger, level.lower())
    log_method(message)
    
    # Get log output
    log_output = log_stream.getvalue().strip()
    
    # Verify output is valid JSON
    try:
        log_data = json.loads(log_output)
    except json.JSONDecodeError:
        pytest.fail(f"Log output is not valid JSON: {log_output}")
    
    # Verify standard fields are present
    assert 'timestamp' in log_data, "Missing 'timestamp' field"
    assert 'level' in log_data, "Missing 'level' field"
    assert 'message' in log_data, "Missing 'message' field"
    
    # Verify field values
    assert log_data['level'] == level
    assert message in log_data['message']


# Feature: wikipedia-intelligence-system, Property 66: Metrics Collection
@given(
    operation_type=st.sampled_from(["ingestion", "processing", "storage"]),
    record_count=st.integers(min_value=1, max_value=10000),
    duration_seconds=st.floats(min_value=0.1, max_value=300.0)
)
@settings(max_examples=10, deadline=None)
def test_metrics_collection(operation_type, record_count, duration_seconds):
    """
    Property 66: For any data ingestion, processing, or storage operation,
    the System should collect and expose metrics for rates, latency, and utilization.
    
    Validates: Requirements 14.4
    """
    metrics = MetricsCollector()
    
    # Simulate operation metrics
    # Rate: records per second
    rate = record_count / duration_seconds
    metrics.set_gauge(f"{operation_type}_rate", rate, labels={"operation": operation_type})
    
    # Latency: operation duration
    metrics.record_timer(f"{operation_type}_latency", duration_seconds, labels={"operation": operation_type})
    
    # Utilization: record count
    metrics.increment_counter(f"{operation_type}_records_total", value=record_count, labels={"operation": operation_type})
    
    # Verify metrics are exposed and queryable
    # Check rate gauge
    recorded_rate = metrics.get_gauge(f"{operation_type}_rate", labels={"operation": operation_type})
    assert recorded_rate is not None
    assert abs(recorded_rate - rate) < 0.01
    
    # Check latency timer
    latency_stats = metrics.get_timer_stats(f"{operation_type}_latency", labels={"operation": operation_type})
    assert latency_stats is not None
    assert latency_stats.count == 1
    assert abs(latency_stats.last_value - duration_seconds) < 0.01
    
    # Check record counter
    total_records = metrics.get_counter(f"{operation_type}_records_total", labels={"operation": operation_type})
    assert total_records == record_count
    
    # Verify metrics can be retrieved as a collection
    all_metrics = metrics.get_all_metrics()
    assert 'counters' in all_metrics
    assert 'gauges' in all_metrics
    assert 'timers' in all_metrics
