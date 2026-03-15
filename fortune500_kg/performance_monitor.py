"""Performance Monitoring for Fortune 500 Knowledge Graph Analytics.

Implements:
- Algorithm execution time logging (Requirement 18.1)
- Memory consumption logging (Requirement 18.2)
- Performance alert generation (Requirement 18.3)
- Data ingestion throughput calculation (Requirement 18.4)
- System health dashboard (Requirement 18.5)
"""

import time
import logging
from datetime import datetime
from typing import Dict, List, Optional, Any
from dataclasses import dataclass, field

logger = logging.getLogger(__name__)


@dataclass
class AlgorithmPerformanceLog:
    """Performance log entry for a graph algorithm execution."""
    algorithm_name: str
    execution_time_ms: float
    peak_memory_mb: float
    timestamp: datetime = field(default_factory=datetime.now)
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class PerformanceAlert:
    """Alert generated when algorithm performance exceeds baseline."""
    algorithm_name: str
    current_time_ms: float
    baseline_time_ms: float
    threshold_multiplier: float
    message: str
    timestamp: datetime = field(default_factory=datetime.now)


@dataclass
class IngestionThroughput:
    """Data ingestion throughput measurement."""
    records_processed: int
    elapsed_seconds: float
    throughput_rps: float  # records per second
    timestamp: datetime = field(default_factory=datetime.now)


@dataclass
class SystemHealthDashboard:
    """System health dashboard data."""
    algorithm_logs: List[AlgorithmPerformanceLog]
    performance_alerts: List[PerformanceAlert]
    ingestion_throughputs: List[IngestionThroughput]
    resource_utilization: Dict[str, float]  # cpu_percent, memory_percent, disk_percent
    generated_at: datetime = field(default_factory=datetime.now)


class PerformanceMonitor:
    """
    Performance monitoring for graph algorithms and data ingestion.

    Responsibilities:
    - log_algorithm_execution(): Record execution time and memory (Req 18.1, 18.2)
    - check_performance_alert(): Generate alerts when time exceeds baseline (Req 18.3)
    - calculate_ingestion_throughput(): Compute records/second (Req 18.4)
    - get_system_health_dashboard(): Aggregate health metrics (Req 18.5)
    """

    def __init__(self) -> None:
        self._algorithm_logs: List[AlgorithmPerformanceLog] = []
        self._performance_alerts: List[PerformanceAlert] = []
        self._ingestion_throughputs: List[IngestionThroughput] = []
        self._baselines: Dict[str, float] = {}  # algorithm_name -> baseline_ms

    def log_algorithm_execution(
        self,
        algorithm_name: str,
        execution_time_ms: float,
        peak_memory_mb: float,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> AlgorithmPerformanceLog:
        """
        Log execution time and memory consumption for a graph algorithm.

        Args:
            algorithm_name: Name of the algorithm (e.g. 'pagerank', 'louvain')
            execution_time_ms: Execution time in milliseconds
            peak_memory_mb: Peak memory consumption in megabytes
            metadata: Optional additional metadata

        Returns:
            AlgorithmPerformanceLog entry

        Validates: Requirements 18.1, 18.2
        """
        log_entry = AlgorithmPerformanceLog(
            algorithm_name=algorithm_name,
            execution_time_ms=execution_time_ms,
            peak_memory_mb=peak_memory_mb,
            metadata=metadata or {},
        )
        self._algorithm_logs.append(log_entry)

        logger.info(
            "Algorithm '%s' completed: %.1fms, %.1fMB peak memory",
            algorithm_name,
            execution_time_ms,
            peak_memory_mb,
        )
        return log_entry

    def check_performance_alert(
        self,
        algorithm_name: str,
        current_time_ms: float,
        baseline_time_ms: Optional[float] = None,
        threshold_multiplier: float = 1.5,
    ) -> Optional[PerformanceAlert]:
        """
        Generate a performance alert when execution time exceeds baseline by threshold.

        Alert is generated when: current_time > threshold_multiplier * baseline_time

        Args:
            algorithm_name: Name of the algorithm
            current_time_ms: Current execution time in milliseconds
            baseline_time_ms: Baseline execution time (uses stored baseline if None)
            threshold_multiplier: Multiplier for baseline (default 1.5 = 50% over baseline)

        Returns:
            PerformanceAlert if threshold exceeded, None otherwise

        Validates: Requirement 18.3
        """
        # Use provided baseline or stored baseline
        if baseline_time_ms is None:
            baseline_time_ms = self._baselines.get(algorithm_name)

        if baseline_time_ms is None or baseline_time_ms <= 0:
            # No baseline available; store current as baseline
            self._baselines[algorithm_name] = current_time_ms
            return None

        if current_time_ms > threshold_multiplier * baseline_time_ms:
            alert = PerformanceAlert(
                algorithm_name=algorithm_name,
                current_time_ms=current_time_ms,
                baseline_time_ms=baseline_time_ms,
                threshold_multiplier=threshold_multiplier,
                message=(
                    f"Algorithm '{algorithm_name}' exceeded performance threshold: "
                    f"{current_time_ms:.1f}ms > {threshold_multiplier}x baseline "
                    f"({baseline_time_ms:.1f}ms)"
                ),
            )
            self._performance_alerts.append(alert)
            logger.warning(alert.message)
            return alert

        return None

    def set_baseline(self, algorithm_name: str, baseline_time_ms: float) -> None:
        """Set the baseline execution time for an algorithm."""
        self._baselines[algorithm_name] = baseline_time_ms

    def calculate_ingestion_throughput(
        self,
        records_processed: int,
        elapsed_seconds: float,
    ) -> IngestionThroughput:
        """
        Calculate data ingestion throughput in records per second.

        throughput = records_processed / elapsed_seconds

        Args:
            records_processed: Number of records ingested
            elapsed_seconds: Time taken for ingestion in seconds

        Returns:
            IngestionThroughput with throughput_rps

        Validates: Requirement 18.4
        """
        if elapsed_seconds <= 0:
            throughput_rps = 0.0
        else:
            throughput_rps = records_processed / elapsed_seconds

        measurement = IngestionThroughput(
            records_processed=records_processed,
            elapsed_seconds=elapsed_seconds,
            throughput_rps=throughput_rps,
        )
        self._ingestion_throughputs.append(measurement)

        logger.info(
            "Ingestion throughput: %d records in %.2fs = %.1f records/sec",
            records_processed,
            elapsed_seconds,
            throughput_rps,
        )
        return measurement

    def get_system_health_dashboard(
        self,
        resource_utilization: Optional[Dict[str, float]] = None,
    ) -> SystemHealthDashboard:
        """
        Generate a system health dashboard with algorithm performance and resource metrics.

        Args:
            resource_utilization: Optional dict with 'cpu_percent', 'memory_percent',
                                  'disk_percent' values. Uses zeros if not provided.

        Returns:
            SystemHealthDashboard with all performance metrics and resource utilization

        Validates: Requirement 18.5
        """
        default_resources = {
            "cpu_percent": 0.0,
            "memory_percent": 0.0,
            "disk_percent": 0.0,
        }
        resources = {**default_resources, **(resource_utilization or {})}

        dashboard = SystemHealthDashboard(
            algorithm_logs=list(self._algorithm_logs),
            performance_alerts=list(self._performance_alerts),
            ingestion_throughputs=list(self._ingestion_throughputs),
            resource_utilization=resources,
        )

        logger.info(
            "System health dashboard generated: %d algorithm logs, %d alerts",
            len(dashboard.algorithm_logs),
            len(dashboard.performance_alerts),
        )
        return dashboard
