"""Cross-validation module for multi-source validation.

This module implements cross-validation across multiple data sources, platforms,
and regions to ensure findings are robust and not artifacts of a single data source.
"""

from dataclasses import dataclass, field
from typing import List, Dict, Any, Optional
import pandas as pd
from pandas import Series, DataFrame
import numpy as np
from scipy import stats

from wikipedia_health.models.data_models import Finding, TimeSeriesData


@dataclass
class ValidationResult:
    """Result of cross-validation analysis.
    
    Attributes:
        is_consistent: Whether finding is consistent across sources
        consistency_score: Score from 0-1 indicating consistency level
        supporting_sources: List of sources that support the finding
        contradicting_sources: List of sources that contradict the finding
        details: Additional validation details
    """
    is_consistent: bool
    consistency_score: float
    supporting_sources: List[str]
    contradicting_sources: List[str] = field(default_factory=list)
    details: Dict[str, Any] = field(default_factory=dict)
    
    def summary(self) -> str:
        """Generate human-readable summary.
        
        Returns:
            String summary of validation result
        """
        lines = [
            f"Consistency: {'PASS' if self.is_consistent else 'FAIL'}",
            f"Consistency Score: {self.consistency_score:.2%}",
            f"Supporting Sources: {', '.join(self.supporting_sources)}",
        ]
        
        if self.contradicting_sources:
            lines.append(f"Contradicting Sources: {', '.join(self.contradicting_sources)}")
        
        return "\n".join(lines)


@dataclass
class ComparisonResult:
    """Result of benchmark comparison.
    
    Attributes:
        correlation: Correlation coefficient with benchmark
        relative_performance: Performance relative to benchmark (ratio)
        is_aligned: Whether metric aligns with benchmark trends
        p_value: Statistical significance of comparison
        details: Additional comparison details
    """
    correlation: float
    relative_performance: float
    is_aligned: bool
    p_value: float
    details: Dict[str, Any] = field(default_factory=dict)


class CrossValidator:
    """Cross-validator for multi-source validation.
    
    This class implements validation across multiple data sources, platforms,
    and regions to ensure findings are robust and reproducible.
    """
    
    def __init__(self, significance_level: float = 0.05):
        """Initialize cross-validator.
        
        Args:
            significance_level: Alpha level for statistical tests
        """
        self.significance_level = significance_level
    
    def validate_across_sources(
        self,
        finding: Finding,
        data_sources: Dict[str, TimeSeriesData]
    ) -> ValidationResult:
        """Validate finding across multiple data sources.
        
        This method checks whether a finding (e.g., a trend change) is consistent
        across pageviews, editors, and edits data sources.
        
        Args:
            finding: Finding to validate
            data_sources: Dictionary mapping source names to TimeSeriesData
                         (e.g., {'pageviews': data1, 'editors': data2, 'edits': data3})
        
        Returns:
            ValidationResult with consistency scores and supporting sources
        """
        if not data_sources:
            return ValidationResult(
                is_consistent=False,
                consistency_score=0.0,
                supporting_sources=[],
                details={'error': 'No data sources provided'}
            )
        
        supporting_sources = []
        contradicting_sources = []
        source_scores = {}
        
        # For each data source, check if the pattern is present
        for source_name, data in data_sources.items():
            # Calculate correlation with the finding's primary evidence
            # We'll use a simple heuristic: check if trends align
            if len(finding.evidence) > 0:
                # Get the effect direction from the first test result
                primary_effect = finding.evidence[0].effect_size
                
                # Calculate trend in this data source
                values = data.values
                if len(values) >= 2:
                    # Simple linear trend
                    x = np.arange(len(values))
                    slope, _, _, p_value, _ = stats.linregress(x, values)
                    
                    # Check if trend direction matches finding
                    same_direction = (slope * primary_effect) > 0
                    is_significant = p_value < self.significance_level
                    
                    if same_direction and is_significant:
                        supporting_sources.append(source_name)
                        source_scores[source_name] = 1.0
                    elif same_direction:
                        supporting_sources.append(source_name)
                        source_scores[source_name] = 0.5  # Same direction but not significant
                    else:
                        contradicting_sources.append(source_name)
                        source_scores[source_name] = 0.0
                else:
                    source_scores[source_name] = 0.0
            else:
                # No evidence to validate against
                source_scores[source_name] = 0.5
        
        # Calculate overall consistency score
        if source_scores:
            consistency_score = np.mean(list(source_scores.values()))
        else:
            consistency_score = 0.0
        
        # Finding is consistent if majority of sources support it
        is_consistent = len(supporting_sources) > len(contradicting_sources)
        
        return ValidationResult(
            is_consistent=is_consistent,
            consistency_score=consistency_score,
            supporting_sources=supporting_sources,
            contradicting_sources=contradicting_sources,
            details={'source_scores': source_scores}
        )
    
    def validate_across_platforms(
        self,
        finding: Finding,
        platform_data: Dict[str, TimeSeriesData]
    ) -> ValidationResult:
        """Validate finding across multiple platforms.
        
        This method checks whether a finding is consistent across desktop,
        mobile web, and mobile app platforms.
        
        Args:
            finding: Finding to validate
            platform_data: Dictionary mapping platform names to TimeSeriesData
                          (e.g., {'desktop': data1, 'mobile-web': data2, 'mobile-app': data3})
        
        Returns:
            ValidationResult with consistency scores and supporting platforms
        """
        if not platform_data:
            return ValidationResult(
                is_consistent=False,
                consistency_score=0.0,
                supporting_sources=[],
                details={'error': 'No platform data provided'}
            )
        
        supporting_platforms = []
        contradicting_platforms = []
        platform_scores = {}
        
        # For each platform, check if the pattern is present
        for platform_name, data in platform_data.items():
            if len(finding.evidence) > 0:
                primary_effect = finding.evidence[0].effect_size
                
                # Calculate trend in this platform
                values = data.values
                if len(values) >= 2:
                    x = np.arange(len(values))
                    slope, _, _, p_value, _ = stats.linregress(x, values)
                    
                    # Check if trend direction matches finding
                    same_direction = (slope * primary_effect) > 0
                    is_significant = p_value < self.significance_level
                    
                    if same_direction and is_significant:
                        supporting_platforms.append(platform_name)
                        platform_scores[platform_name] = 1.0
                    elif same_direction:
                        supporting_platforms.append(platform_name)
                        platform_scores[platform_name] = 0.5
                    else:
                        contradicting_platforms.append(platform_name)
                        platform_scores[platform_name] = 0.0
                else:
                    platform_scores[platform_name] = 0.0
            else:
                platform_scores[platform_name] = 0.5
        
        # Calculate overall consistency score
        if platform_scores:
            consistency_score = np.mean(list(platform_scores.values()))
        else:
            consistency_score = 0.0
        
        # Finding is consistent if majority of platforms support it
        is_consistent = len(supporting_platforms) > len(contradicting_platforms)
        
        return ValidationResult(
            is_consistent=is_consistent,
            consistency_score=consistency_score,
            supporting_sources=supporting_platforms,
            contradicting_sources=contradicting_platforms,
            details={'platform_scores': platform_scores}
        )
    
    def validate_across_regions(
        self,
        finding: Finding,
        region_data: Dict[str, TimeSeriesData]
    ) -> ValidationResult:
        """Validate finding across multiple geographic regions.
        
        This method checks whether a finding is consistent across different
        geographic regions or language editions.
        
        Args:
            finding: Finding to validate
            region_data: Dictionary mapping region names to TimeSeriesData
                        (e.g., {'en': data1, 'es': data2, 'fr': data3})
        
        Returns:
            ValidationResult with consistency scores and supporting regions
        """
        if not region_data:
            return ValidationResult(
                is_consistent=False,
                consistency_score=0.0,
                supporting_sources=[],
                details={'error': 'No region data provided'}
            )
        
        supporting_regions = []
        contradicting_regions = []
        region_scores = {}
        
        # For each region, check if the pattern is present
        for region_name, data in region_data.items():
            if len(finding.evidence) > 0:
                primary_effect = finding.evidence[0].effect_size
                
                # Calculate trend in this region
                values = data.values
                if len(values) >= 2:
                    x = np.arange(len(values))
                    slope, _, _, p_value, _ = stats.linregress(x, values)
                    
                    # Check if trend direction matches finding
                    same_direction = (slope * primary_effect) > 0
                    is_significant = p_value < self.significance_level
                    
                    if same_direction and is_significant:
                        supporting_regions.append(region_name)
                        region_scores[region_name] = 1.0
                    elif same_direction:
                        supporting_regions.append(region_name)
                        region_scores[region_name] = 0.5
                    else:
                        contradicting_regions.append(region_name)
                        region_scores[region_name] = 0.0
                else:
                    region_scores[region_name] = 0.0
            else:
                region_scores[region_name] = 0.5
        
        # Calculate overall consistency score
        if region_scores:
            consistency_score = np.mean(list(region_scores.values()))
        else:
            consistency_score = 0.0
        
        # Finding is consistent if majority of regions support it
        is_consistent = len(supporting_regions) > len(contradicting_regions)
        
        return ValidationResult(
            is_consistent=is_consistent,
            consistency_score=consistency_score,
            supporting_sources=supporting_regions,
            contradicting_sources=contradicting_regions,
            details={'region_scores': region_scores}
        )
    
    def compare_to_benchmark(
        self,
        metric: Series,
        benchmark: Series
    ) -> ComparisonResult:
        """Compare metric to external benchmark.
        
        This method compares Wikipedia metrics against external reference
        platforms or industry benchmarks.
        
        Args:
            metric: Wikipedia metric time series
            benchmark: External benchmark time series
        
        Returns:
            ComparisonResult with correlation and relative performance
        """
        if len(metric) == 0 or len(benchmark) == 0:
            return ComparisonResult(
                correlation=0.0,
                relative_performance=0.0,
                is_aligned=False,
                p_value=1.0,
                details={'error': 'Empty metric or benchmark'}
            )
        
        # Align series by index if they have different lengths
        if len(metric) != len(benchmark):
            # Use the shorter length
            min_len = min(len(metric), len(benchmark))
            metric = metric.iloc[:min_len]
            benchmark = benchmark.iloc[:min_len]
        
        # Calculate correlation
        if len(metric) >= 2:
            correlation, p_value = stats.pearsonr(metric, benchmark)
        else:
            correlation = 0.0
            p_value = 1.0
        
        # Calculate relative performance (ratio of means)
        metric_mean = metric.mean()
        benchmark_mean = benchmark.mean()
        
        if benchmark_mean != 0:
            relative_performance = metric_mean / benchmark_mean
        else:
            relative_performance = 0.0
        
        # Check if trends are aligned (both increasing or both decreasing)
        if len(metric) >= 2:
            metric_trend = stats.linregress(np.arange(len(metric)), metric).slope
            benchmark_trend = stats.linregress(np.arange(len(benchmark)), benchmark).slope
            is_aligned = (metric_trend * benchmark_trend) > 0
        else:
            is_aligned = False
        
        return ComparisonResult(
            correlation=correlation,
            relative_performance=relative_performance,
            is_aligned=is_aligned,
            p_value=p_value,
            details={
                'metric_mean': metric_mean,
                'benchmark_mean': benchmark_mean,
                'metric_std': metric.std(),
                'benchmark_std': benchmark.std()
            }
        )
