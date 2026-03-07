"""Metrics Collection System

Collects and exposes metrics for monitoring data ingestion rates,
processing latency, storage utilization, and API usage.
"""
import time
import logging
from typing import Dict, Optional, List
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from collections import defaultdict
from threading import Lock


logger = logging.getLogger(__name__)


@dataclass
class MetricValue:
    """Single metric value with timestamp"""
    value: float
    timestamp: datetime = field(default_factory=datetime.now)


@dataclass
class MetricStats:
    """Statistical summary of a metric"""
    count: int
    sum: float
    min: float
    max: float
    avg: float
    last_value: float
    last_updated: datetime


class MetricsCollector:
    """
    Collects and aggregates system metrics.
    
    Supports counters, gauges, histograms, and timers.
    Thread-safe for concurrent metric collection.
    
    Requirements: 12.7, 14.4
    """
    
    def __init__(self, retention_minutes: int = 60):
        """Initialize metrics collector
        
        Args:
            retention_minutes: How long to retain metric history
        """
        self.retention_minutes = retention_minutes
        
        # Metric storage
        # counters: monotonically increasing values
        self._counters: Dict[str, float] = defaultdict(float)
        
        # gauges: point-in-time values
        self._gauges: Dict[str, float] = {}
        
        # histograms: list of values for statistical analysis
        self._histograms: Dict[str, List[MetricValue]] = defaultdict(list)
        
        # timers: track operation durations
        self._timers: Dict[str, List[float]] = defaultdict(list)
        
        # Thread safety
        self._lock = Lock()
        
        logger.info(f"MetricsCollector initialized with retention={retention_minutes}min")
    
    def increment_counter(self, name: str, value: float = 1.0, labels: Optional[Dict[str, str]] = None) -> None:
        """Increment a counter metric
        
        Args:
            name: Metric name
            value: Amount to increment (default 1.0)
            labels: Optional labels for metric dimensions
        """
        metric_key = self._build_metric_key(name, labels)
        
        with self._lock:
            self._counters[metric_key] += value
            
        logger.debug(f"Counter incremented: {metric_key} += {value}")
    
    def set_gauge(self, name: str, value: float, labels: Optional[Dict[str, str]] = None) -> None:
        """Set a gauge metric to a specific value
        
        Args:
            name: Metric name
            value: Gauge value
            labels: Optional labels for metric dimensions
        """
        metric_key = self._build_metric_key(name, labels)
        
        with self._lock:
            self._gauges[metric_key] = value
            
        logger.debug(f"Gauge set: {metric_key} = {value}")
    
    def record_value(self, name: str, value: float, labels: Optional[Dict[str, str]] = None) -> None:
        """Record a value in a histogram
        
        Args:
            name: Metric name
            value: Value to record
            labels: Optional labels for metric dimensions
        """
        metric_key = self._build_metric_key(name, labels)
        
        with self._lock:
            self._histograms[metric_key].append(MetricValue(value=value))
            
        # Clean up old values
        self._cleanup_histogram(metric_key)
        
        logger.debug(f"Value recorded: {metric_key} = {value}")
    
    def record_timer(self, name: str, duration_seconds: float, labels: Optional[Dict[str, str]] = None) -> None:
        """Record a timer duration
        
        Args:
            name: Metric name
            duration_seconds: Duration in seconds
            labels: Optional labels for metric dimensions
        """
        metric_key = self._build_metric_key(name, labels)
        
        with self._lock:
            self._timers[metric_key].append(duration_seconds)
            
        logger.debug(f"Timer recorded: {metric_key} = {duration_seconds:.3f}s")
    
    def get_counter(self, name: str, labels: Optional[Dict[str, str]] = None) -> float:
        """Get current counter value
        
        Args:
            name: Metric name
            labels: Optional labels for metric dimensions
            
        Returns:
            Counter value
        """
        metric_key = self._build_metric_key(name, labels)
        
        with self._lock:
            return self._counters.get(metric_key, 0.0)
    
    def get_gauge(self, name: str, labels: Optional[Dict[str, str]] = None) -> Optional[float]:
        """Get current gauge value
        
        Args:
            name: Metric name
            labels: Optional labels for metric dimensions
            
        Returns:
            Gauge value or None if not set
        """
        metric_key = self._build_metric_key(name, labels)
        
        with self._lock:
            return self._gauges.get(metric_key)
    
    def get_histogram_stats(self, name: str, labels: Optional[Dict[str, str]] = None) -> Optional[MetricStats]:
        """Get statistical summary of histogram
        
        Args:
            name: Metric name
            labels: Optional labels for metric dimensions
            
        Returns:
            MetricStats or None if no data
        """
        metric_key = self._build_metric_key(name, labels)
        
        with self._lock:
            values = self._histograms.get(metric_key, [])
            
            if not values:
                return None
            
            # Clean up old values first
            self._cleanup_histogram(metric_key)
            values = self._histograms.get(metric_key, [])
            
            if not values:
                return None
            
            nums = [v.value for v in values]
            return MetricStats(
                count=len(nums),
                sum=sum(nums),
                min=min(nums),
                max=max(nums),
                avg=sum(nums) / len(nums),
                last_value=nums[-1],
                last_updated=values[-1].timestamp
            )
    
    def get_timer_stats(self, name: str, labels: Optional[Dict[str, str]] = None) -> Optional[MetricStats]:
        """Get statistical summary of timer
        
        Args:
            name: Metric name
            labels: Optional labels for metric dimensions
            
        Returns:
            MetricStats or None if no data
        """
        metric_key = self._build_metric_key(name, labels)
        
        with self._lock:
            durations = self._timers.get(metric_key, [])
            
            if not durations:
                return None
            
            return MetricStats(
                count=len(durations),
                sum=sum(durations),
                min=min(durations),
                max=max(durations),
                avg=sum(durations) / len(durations),
                last_value=durations[-1],
                last_updated=datetime.now()
            )
    
    def get_all_metrics(self) -> Dict[str, Dict]:
        """Get all metrics as a dictionary
        
        Returns:
            Dictionary with all metric types and their values
        """
        with self._lock:
            metrics = {
                'counters': dict(self._counters),
                'gauges': dict(self._gauges),
                'histograms': {},
                'timers': {}
            }
            
            # Add histogram stats
            for name in self._histograms.keys():
                stats = self.get_histogram_stats(name.split('|')[0], self._parse_labels(name))
                if stats:
                    metrics['histograms'][name] = {
                        'count': stats.count,
                        'sum': stats.sum,
                        'min': stats.min,
                        'max': stats.max,
                        'avg': stats.avg,
                        'last_value': stats.last_value
                    }
            
            # Add timer stats
            for name in self._timers.keys():
                stats = self.get_timer_stats(name.split('|')[0], self._parse_labels(name))
                if stats:
                    metrics['timers'][name] = {
                        'count': stats.count,
                        'sum': stats.sum,
                        'min': stats.min,
                        'max': stats.max,
                        'avg': stats.avg,
                        'last_value': stats.last_value
                    }
            
            return metrics
    
    def reset_counter(self, name: str, labels: Optional[Dict[str, str]] = None) -> None:
        """Reset a counter to zero
        
        Args:
            name: Metric name
            labels: Optional labels for metric dimensions
        """
        metric_key = self._build_metric_key(name, labels)
        
        with self._lock:
            self._counters[metric_key] = 0.0
            
        logger.debug(f"Counter reset: {metric_key}")
    
    def clear_all_metrics(self) -> None:
        """Clear all metrics (useful for testing)"""
        with self._lock:
            self._counters.clear()
            self._gauges.clear()
            self._histograms.clear()
            self._timers.clear()
            
        logger.info("All metrics cleared")
    
    def _build_metric_key(self, name: str, labels: Optional[Dict[str, str]]) -> str:
        """Build metric key with labels
        
        Args:
            name: Metric name
            labels: Optional labels
            
        Returns:
            Metric key string
        """
        if not labels:
            return name
        
        # Sort labels for consistent keys
        label_str = ','.join(f"{k}={v}" for k, v in sorted(labels.items()))
        return f"{name}|{label_str}"
    
    def _parse_labels(self, metric_key: str) -> Optional[Dict[str, str]]:
        """Parse labels from metric key
        
        Args:
            metric_key: Metric key string
            
        Returns:
            Labels dictionary or None
        """
        if '|' not in metric_key:
            return None
        
        _, label_str = metric_key.split('|', 1)
        labels = {}
        for pair in label_str.split(','):
            k, v = pair.split('=', 1)
            labels[k] = v
        return labels
    
    def _cleanup_histogram(self, metric_key: str) -> None:
        """Remove old values from histogram
        
        Args:
            metric_key: Metric key
        """
        cutoff_time = datetime.now() - timedelta(minutes=self.retention_minutes)
        
        if metric_key in self._histograms:
            self._histograms[metric_key] = [
                v for v in self._histograms[metric_key]
                if v.timestamp > cutoff_time
            ]


class Timer:
    """Context manager for timing operations
    
    Usage:
        with Timer(metrics, "operation_name"):
            # do work
            pass
    """
    
    def __init__(
        self,
        metrics: MetricsCollector,
        name: str,
        labels: Optional[Dict[str, str]] = None
    ):
        """Initialize timer
        
        Args:
            metrics: MetricsCollector instance
            name: Metric name
            labels: Optional labels
        """
        self.metrics = metrics
        self.name = name
        self.labels = labels
        self.start_time: Optional[float] = None
    
    def __enter__(self):
        """Start timer"""
        self.start_time = time.monotonic()
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        """Stop timer and record duration"""
        if self.start_time is not None:
            duration = time.monotonic() - self.start_time
            self.metrics.record_timer(self.name, duration, self.labels)


# Global metrics instance
_metrics_instance: Optional[MetricsCollector] = None


def get_metrics() -> MetricsCollector:
    """Get global metrics collector instance
    
    Returns:
        MetricsCollector instance
    """
    global _metrics_instance
    
    if _metrics_instance is None:
        _metrics_instance = MetricsCollector()
    
    return _metrics_instance
