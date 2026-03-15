"""Unit tests for comprehensive error handling (Task 24).

Covers:
- ErrorHandler.record_operation and failure rate tracking (Requirement 1.5)
- ErrorHandler.handle_invalid_metric_calculation (Requirement 8.1)
- ErrorHandler.handle_visualization_error (Requirement 6.1)
- ErrorHandler.handle_export_error (Requirements 17.3, 17.4)
- retry_with_backoff decorator (Requirements 1.5, 17.3, 17.4)
- AnalyticsEngine.execute_custom_query timeout (Requirement 16.5)
"""

import pytest
from unittest.mock import patch, MagicMock

from fortune500_kg.error_handler import ErrorHandler, retry_with_backoff, FAILURE_RATE_THRESHOLD
from fortune500_kg.analytics_engine import AnalyticsEngine, MetricsRepository
from fortune500_kg.exceptions import QuerySyntaxError, QueryTimeoutError


@pytest.fixture
def handler():
    return ErrorHandler()


# ---------------------------------------------------------------------------
# Failure rate tracking
# ---------------------------------------------------------------------------

class TestFailureRateTracking:

    def test_zero_failure_rate_initially(self, handler):
        assert handler.get_failure_rate("test_op") == pytest.approx(0.0)

    def test_failure_rate_calculation(self, handler):
        handler.record_operation("op", success=True)
        handler.record_operation("op", success=True)
        handler.record_operation("op", success=False)
        # 1 failure out of 3 = 33.3%
        assert handler.get_failure_rate("op") == pytest.approx(1/3)

    def test_alert_triggered_above_threshold(self, handler):
        # 2 failures out of 10 = 20% > 10% threshold
        for _ in range(8):
            handler.record_operation("op", success=True)
        for _ in range(2):
            handler.record_operation("op", success=False)
        assert handler.check_failure_rate_alert("op") is True

    def test_no_alert_below_threshold(self, handler):
        # 1 failure out of 20 = 5% < 10% threshold
        for _ in range(19):
            handler.record_operation("op", success=True)
        handler.record_operation("op", success=False)
        assert handler.check_failure_rate_alert("op") is False

    def test_no_alert_for_zero_operations(self, handler):
        assert handler.check_failure_rate_alert("nonexistent") is False


# ---------------------------------------------------------------------------
# Invalid metric calculation handling
# ---------------------------------------------------------------------------

class TestHandleInvalidMetricCalculation:

    def test_returns_default_value(self, handler):
        result = handler.handle_invalid_metric_calculation(
            "C1", "innovation_score", ZeroDivisionError("division by zero")
        )
        assert result == pytest.approx(0.0)

    def test_custom_default_value(self, handler):
        result = handler.handle_invalid_metric_calculation(
            "C1", "digital_maturity", ValueError("null value"), default_value=-1.0
        )
        assert result == pytest.approx(-1.0)

    def test_records_failure(self, handler):
        handler.handle_invalid_metric_calculation(
            "C1", "innovation_score", ZeroDivisionError("division by zero")
        )
        assert handler.get_failure_rate("metric_calculation_innovation_score") > 0


# ---------------------------------------------------------------------------
# Visualization error handling
# ---------------------------------------------------------------------------

class TestHandleVisualizationError:

    def test_returns_error_placeholder(self, handler):
        result = handler.handle_visualization_error(
            "network_graph", ConnectionError("Bloom connection failed")
        )
        assert result["error"] is True
        assert result["placeholder"] is True

    def test_error_message_included(self, handler):
        result = handler.handle_visualization_error(
            "leaderboard", RuntimeError("Rendering failed")
        )
        assert "error_message" in result
        assert result["error_message"]

    def test_visualization_type_included(self, handler):
        result = handler.handle_visualization_error(
            "heatmap", Exception("Failed")
        )
        assert result["visualization_type"] == "heatmap"


# ---------------------------------------------------------------------------
# Export error handling
# ---------------------------------------------------------------------------

class TestHandleExportError:

    def test_returns_failed_status(self, handler):
        result = handler.handle_export_error("csv", IOError("Disk full"))
        assert result["status"] == "failed"

    def test_format_included(self, handler):
        result = handler.handle_export_error("tableau", ConnectionError("API error"))
        assert result["format"] == "tableau"

    def test_fallback_path_included(self, handler):
        result = handler.handle_export_error(
            "json", IOError("Write failed"), fallback_path="/tmp/backup.json"
        )
        assert result["fallback_path"] == "/tmp/backup.json"


# ---------------------------------------------------------------------------
# retry_with_backoff decorator
# ---------------------------------------------------------------------------

class TestRetryWithBackoff:

    def test_succeeds_on_first_try(self):
        call_count = [0]

        @retry_with_backoff(max_retries=3, initial_delay=0.001)
        def always_succeeds():
            call_count[0] += 1
            return "success"

        result = always_succeeds()
        assert result == "success"
        assert call_count[0] == 1

    def test_retries_on_failure(self):
        call_count = [0]

        @retry_with_backoff(max_retries=2, initial_delay=0.001)
        def fails_twice_then_succeeds():
            call_count[0] += 1
            if call_count[0] < 3:
                raise ConnectionError("Network error")
            return "success"

        result = fails_twice_then_succeeds()
        assert result == "success"
        assert call_count[0] == 3

    def test_raises_after_max_retries(self):
        @retry_with_backoff(max_retries=2, initial_delay=0.001)
        def always_fails():
            raise ConnectionError("Always fails")

        with pytest.raises(ConnectionError):
            always_fails()

    def test_only_catches_specified_exceptions(self):
        @retry_with_backoff(max_retries=3, initial_delay=0.001, exceptions=(ConnectionError,))
        def raises_value_error():
            raise ValueError("Not a connection error")

        with pytest.raises(ValueError):
            raises_value_error()


# ---------------------------------------------------------------------------
# Query timeout enforcement
# ---------------------------------------------------------------------------

class TestQueryTimeoutEnforcement:

    def test_timeout_raises_error(self):
        engine = AnalyticsEngine(metrics_repo=MetricsRepository())

        def slow_executor(q):
            return ([], [], 31000.0)  # 31 seconds

        with pytest.raises(QueryTimeoutError):
            engine.execute_custom_query(
                "MATCH (n) RETURN n",
                timeout_seconds=30.0,
                mock_executor=slow_executor,
            )

    def test_fast_query_does_not_timeout(self):
        engine = AnalyticsEngine(metrics_repo=MetricsRepository())

        def fast_executor(q):
            return (["id"], [{"id": "C1"}], 100.0)  # 100ms

        result = engine.execute_custom_query(
            "MATCH (n) RETURN n",
            timeout_seconds=30.0,
            mock_executor=fast_executor,
        )
        assert result is not None
