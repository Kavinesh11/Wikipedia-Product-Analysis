"""Unit tests for performance monitoring (Tasks 23.1, 23.3).

Covers:
- PerformanceMonitor.log_algorithm_execution (Requirements 18.1, 18.2)
- PerformanceMonitor.check_performance_alert (Requirement 18.3)
- PerformanceMonitor.calculate_ingestion_throughput (Requirement 18.4)
- PerformanceMonitor.get_system_health_dashboard (Requirement 18.5)
"""

import pytest

from fortune500_kg.performance_monitor import (
    PerformanceMonitor,
    AlgorithmPerformanceLog,
    PerformanceAlert,
    IngestionThroughput,
    SystemHealthDashboard,
)


@pytest.fixture
def monitor():
    return PerformanceMonitor()


# ---------------------------------------------------------------------------
# log_algorithm_execution
# ---------------------------------------------------------------------------

class TestLogAlgorithmExecution:

    def test_returns_log_entry(self, monitor):
        log = monitor.log_algorithm_execution("pagerank", 150.0, 256.0)
        assert isinstance(log, AlgorithmPerformanceLog)

    def test_log_has_algorithm_name(self, monitor):
        log = monitor.log_algorithm_execution("louvain", 200.0, 128.0)
        assert log.algorithm_name == "louvain"

    def test_log_has_execution_time(self, monitor):
        log = monitor.log_algorithm_execution("pagerank", 150.0, 256.0)
        assert log.execution_time_ms == pytest.approx(150.0)

    def test_log_has_memory_consumption(self, monitor):
        log = monitor.log_algorithm_execution("pagerank", 150.0, 256.0)
        assert log.peak_memory_mb == pytest.approx(256.0)

    def test_log_has_timestamp(self, monitor):
        from datetime import datetime
        log = monitor.log_algorithm_execution("pagerank", 150.0, 256.0)
        assert isinstance(log.timestamp, datetime)

    def test_multiple_logs_stored(self, monitor):
        monitor.log_algorithm_execution("pagerank", 100.0, 100.0)
        monitor.log_algorithm_execution("louvain", 200.0, 200.0)
        dashboard = monitor.get_system_health_dashboard()
        assert len(dashboard.algorithm_logs) == 2


# ---------------------------------------------------------------------------
# check_performance_alert
# ---------------------------------------------------------------------------

class TestCheckPerformanceAlert:

    def test_no_alert_when_within_threshold(self, monitor):
        monitor.set_baseline("pagerank", 100.0)
        alert = monitor.check_performance_alert("pagerank", 140.0, baseline_time_ms=100.0)
        assert alert is None

    def test_alert_when_exceeds_threshold(self, monitor):
        alert = monitor.check_performance_alert("pagerank", 200.0, baseline_time_ms=100.0)
        assert isinstance(alert, PerformanceAlert)

    def test_alert_threshold_is_1_5x(self, monitor):
        # 150ms = exactly 1.5x baseline of 100ms; should NOT trigger (not strictly greater)
        alert = monitor.check_performance_alert("pagerank", 150.0, baseline_time_ms=100.0)
        assert alert is None

    def test_alert_above_1_5x_triggers(self, monitor):
        # 151ms > 1.5x baseline of 100ms; should trigger
        alert = monitor.check_performance_alert("pagerank", 151.0, baseline_time_ms=100.0)
        assert alert is not None

    def test_alert_has_algorithm_name(self, monitor):
        alert = monitor.check_performance_alert("louvain", 300.0, baseline_time_ms=100.0)
        assert alert.algorithm_name == "louvain"

    def test_alert_has_message(self, monitor):
        alert = monitor.check_performance_alert("pagerank", 200.0, baseline_time_ms=100.0)
        assert alert.message

    def test_no_baseline_returns_none(self, monitor):
        alert = monitor.check_performance_alert("new_algorithm", 100.0)
        assert alert is None


# ---------------------------------------------------------------------------
# calculate_ingestion_throughput
# ---------------------------------------------------------------------------

class TestCalculateIngestionThroughput:

    def test_returns_throughput_object(self, monitor):
        result = monitor.calculate_ingestion_throughput(1000, 10.0)
        assert isinstance(result, IngestionThroughput)

    def test_throughput_formula(self, monitor):
        result = monitor.calculate_ingestion_throughput(500, 5.0)
        assert result.throughput_rps == pytest.approx(100.0)

    def test_zero_elapsed_returns_zero_throughput(self, monitor):
        result = monitor.calculate_ingestion_throughput(100, 0.0)
        assert result.throughput_rps == pytest.approx(0.0)

    def test_records_and_elapsed_stored(self, monitor):
        result = monitor.calculate_ingestion_throughput(200, 4.0)
        assert result.records_processed == 200
        assert result.elapsed_seconds == pytest.approx(4.0)


# ---------------------------------------------------------------------------
# get_system_health_dashboard
# ---------------------------------------------------------------------------

class TestGetSystemHealthDashboard:

    def test_returns_dashboard(self, monitor):
        dashboard = monitor.get_system_health_dashboard()
        assert isinstance(dashboard, SystemHealthDashboard)

    def test_dashboard_has_algorithm_logs(self, monitor):
        monitor.log_algorithm_execution("pagerank", 100.0, 100.0)
        dashboard = monitor.get_system_health_dashboard()
        assert len(dashboard.algorithm_logs) == 1

    def test_dashboard_has_performance_alerts(self, monitor):
        monitor.check_performance_alert("pagerank", 200.0, baseline_time_ms=100.0)
        dashboard = monitor.get_system_health_dashboard()
        assert len(dashboard.performance_alerts) == 1

    def test_dashboard_has_resource_utilization(self, monitor):
        resources = {"cpu_percent": 45.0, "memory_percent": 60.0, "disk_percent": 30.0}
        dashboard = monitor.get_system_health_dashboard(resource_utilization=resources)
        assert dashboard.resource_utilization["cpu_percent"] == pytest.approx(45.0)
        assert dashboard.resource_utilization["memory_percent"] == pytest.approx(60.0)
        assert dashboard.resource_utilization["disk_percent"] == pytest.approx(30.0)

    def test_dashboard_has_ingestion_throughputs(self, monitor):
        monitor.calculate_ingestion_throughput(500, 5.0)
        dashboard = monitor.get_system_health_dashboard()
        assert len(dashboard.ingestion_throughputs) == 1

    def test_dashboard_has_all_required_sections(self, monitor):
        dashboard = monitor.get_system_health_dashboard()
        assert hasattr(dashboard, "algorithm_logs")
        assert hasattr(dashboard, "performance_alerts")
        assert hasattr(dashboard, "ingestion_throughputs")
        assert hasattr(dashboard, "resource_utilization")
