"""Property-based tests for performance monitoring (Tasks 23.2, 23.4).

Properties:
- Property 86: Algorithm Execution Time Logging
- Property 87: Algorithm Memory Consumption Logging
- Property 88: Performance Alert Threshold Detection
- Property 89: Ingestion Throughput Calculation
- Property 90: System Health Dashboard Metrics Coverage

Validates: Requirements 18.1, 18.2, 18.3, 18.4, 18.5
"""

from hypothesis import given, settings, assume
import hypothesis.strategies as st
import pytest

from fortune500_kg.performance_monitor import PerformanceMonitor


# ---------------------------------------------------------------------------
# Strategies
# ---------------------------------------------------------------------------

_algorithm_name = st.sampled_from([
    "pagerank", "louvain", "betweenness_centrality", "innovation_score",
])

_exec_time_ms = st.floats(
    min_value=0.1, max_value=100000.0, allow_nan=False, allow_infinity=False
)

_memory_mb = st.floats(
    min_value=0.1, max_value=10000.0, allow_nan=False, allow_infinity=False
)

_records = st.integers(min_value=1, max_value=100000)

_elapsed_seconds = st.floats(
    min_value=0.001, max_value=3600.0, allow_nan=False, allow_infinity=False
)


# ---------------------------------------------------------------------------
# Property 86: Algorithm Execution Time Logging
# Feature: fortune500-kg-analytics, Property 86
# Validates: Requirements 18.1
# ---------------------------------------------------------------------------

@settings(max_examples=100)
@given(_algorithm_name, _exec_time_ms, _memory_mb)
def test_property_86_algorithm_execution_time_logging(algorithm_name, exec_time_ms, memory_mb):
    """
    For any graph algorithm execution, the performance log should contain an
    entry with the algorithm name and execution time in milliseconds.

    **Validates: Requirements 18.1**
    """
    monitor = PerformanceMonitor()
    log = monitor.log_algorithm_execution(algorithm_name, exec_time_ms, memory_mb)

    assert log.algorithm_name == algorithm_name
    assert log.execution_time_ms == pytest.approx(exec_time_ms)

    # Verify it's stored in the dashboard
    dashboard = monitor.get_system_health_dashboard()
    assert len(dashboard.algorithm_logs) == 1
    assert dashboard.algorithm_logs[0].algorithm_name == algorithm_name
    assert dashboard.algorithm_logs[0].execution_time_ms == pytest.approx(exec_time_ms)


# ---------------------------------------------------------------------------
# Property 87: Algorithm Memory Consumption Logging
# Feature: fortune500-kg-analytics, Property 87
# Validates: Requirements 18.2
# ---------------------------------------------------------------------------

@settings(max_examples=100)
@given(_algorithm_name, _exec_time_ms, _memory_mb)
def test_property_87_algorithm_memory_consumption_logging(algorithm_name, exec_time_ms, memory_mb):
    """
    For any graph algorithm execution, the performance log should contain an
    entry with peak memory consumption during execution.

    **Validates: Requirements 18.2**
    """
    monitor = PerformanceMonitor()
    log = monitor.log_algorithm_execution(algorithm_name, exec_time_ms, memory_mb)

    assert log.peak_memory_mb == pytest.approx(memory_mb)

    dashboard = monitor.get_system_health_dashboard()
    assert dashboard.algorithm_logs[0].peak_memory_mb == pytest.approx(memory_mb)


# ---------------------------------------------------------------------------
# Property 88: Performance Alert Threshold Detection
# Feature: fortune500-kg-analytics, Property 88
# Validates: Requirements 18.3
# ---------------------------------------------------------------------------

@settings(max_examples=100)
@given(
    _algorithm_name,
    st.floats(min_value=0.1, max_value=10000.0, allow_nan=False, allow_infinity=False),  # baseline
    st.floats(min_value=0.1, max_value=100000.0, allow_nan=False, allow_infinity=False),  # current
)
def test_property_88_performance_alert_threshold(algorithm_name, baseline_ms, current_ms):
    """
    For any algorithm execution with time T_current, if T_current > 1.5 × T_baseline,
    a performance alert should be generated.

    **Validates: Requirements 18.3**
    """
    monitor = PerformanceMonitor()
    alert = monitor.check_performance_alert(
        algorithm_name,
        current_ms,
        baseline_time_ms=baseline_ms,
        threshold_multiplier=1.5,
    )

    if current_ms > 1.5 * baseline_ms:
        assert alert is not None, (
            f"Alert should be generated when {current_ms:.1f}ms > 1.5 × {baseline_ms:.1f}ms"
        )
        assert alert.algorithm_name == algorithm_name
        assert alert.current_time_ms == pytest.approx(current_ms)
        assert alert.baseline_time_ms == pytest.approx(baseline_ms)
    else:
        assert alert is None, (
            f"No alert should be generated when {current_ms:.1f}ms <= 1.5 × {baseline_ms:.1f}ms"
        )


# ---------------------------------------------------------------------------
# Property 89: Ingestion Throughput Calculation
# Feature: fortune500-kg-analytics, Property 89
# Validates: Requirements 18.4
# ---------------------------------------------------------------------------

@settings(max_examples=100)
@given(_records, _elapsed_seconds)
def test_property_89_ingestion_throughput_calculation(records, elapsed_seconds):
    """
    For any data ingestion operation processing N records in T seconds, the
    throughput should be calculated as N / T records per second.

    **Validates: Requirements 18.4**
    """
    monitor = PerformanceMonitor()
    result = monitor.calculate_ingestion_throughput(records, elapsed_seconds)

    expected_throughput = records / elapsed_seconds
    assert result.throughput_rps == pytest.approx(expected_throughput, rel=1e-6)
    assert result.records_processed == records
    assert result.elapsed_seconds == pytest.approx(elapsed_seconds)


# ---------------------------------------------------------------------------
# Property 90: System Health Dashboard Metrics Coverage
# Feature: fortune500-kg-analytics, Property 90
# Validates: Requirements 18.5
# ---------------------------------------------------------------------------

@settings(max_examples=100)
@given(
    st.lists(
        st.tuples(_algorithm_name, _exec_time_ms, _memory_mb),
        min_size=0,
        max_size=10,
    ),
    st.floats(min_value=0.0, max_value=100.0, allow_nan=False, allow_infinity=False),
    st.floats(min_value=0.0, max_value=100.0, allow_nan=False, allow_infinity=False),
    st.floats(min_value=0.0, max_value=100.0, allow_nan=False, allow_infinity=False),
)
def test_property_90_system_health_dashboard_coverage(
    algorithm_runs, cpu_percent, memory_percent, disk_percent
):
    """
    For any rendered system health dashboard, the dashboard should display both
    algorithm performance metrics (execution time, memory) and resource
    utilization metrics (CPU, memory, disk).

    **Validates: Requirements 18.5**
    """
    monitor = PerformanceMonitor()

    # Log algorithm executions
    for algo_name, exec_time, mem_mb in algorithm_runs:
        monitor.log_algorithm_execution(algo_name, exec_time, mem_mb)

    resources = {
        "cpu_percent": cpu_percent,
        "memory_percent": memory_percent,
        "disk_percent": disk_percent,
    }
    dashboard = monitor.get_system_health_dashboard(resource_utilization=resources)

    # Must have algorithm performance metrics
    assert hasattr(dashboard, "algorithm_logs")
    assert len(dashboard.algorithm_logs) == len(algorithm_runs)

    # Each log must have execution time and memory
    for log in dashboard.algorithm_logs:
        assert log.execution_time_ms >= 0
        assert log.peak_memory_mb >= 0

    # Must have resource utilization metrics
    assert "cpu_percent" in dashboard.resource_utilization
    assert "memory_percent" in dashboard.resource_utilization
    assert "disk_percent" in dashboard.resource_utilization

    assert dashboard.resource_utilization["cpu_percent"] == pytest.approx(cpu_percent)
    assert dashboard.resource_utilization["memory_percent"] == pytest.approx(memory_percent)
    assert dashboard.resource_utilization["disk_percent"] == pytest.approx(disk_percent)
