"""Robustness checking module for sensitivity analysis.

This module implements robustness checks including sensitivity analysis,
outlier sensitivity testing, method comparison, and subsample stability.
"""

from dataclasses import dataclass, field
from typing import List, Dict, Any, Callable, Optional
import pandas as pd
from pandas import Series
import numpy as np
from scipy import stats


@dataclass
class SensitivityResult:
    """Result of sensitivity analysis.
    
    Attributes:
        base_result: Result with base parameters
        parameter_results: Dictionary mapping parameter values to results
        is_stable: Whether results are stable across parameter variations
        stability_score: Score from 0-1 indicating stability
        details: Additional sensitivity details
    """
    base_result: Any
    parameter_results: Dict[str, Any]
    is_stable: bool
    stability_score: float
    details: Dict[str, Any] = field(default_factory=dict)
    
    def summary(self) -> str:
        """Generate human-readable summary.
        
        Returns:
            String summary of sensitivity analysis
        """
        lines = [
            f"Stability: {'STABLE' if self.is_stable else 'UNSTABLE'}",
            f"Stability Score: {self.stability_score:.2%}",
            f"Parameter Variations Tested: {len(self.parameter_results)}",
        ]
        return "\n".join(lines)


@dataclass
class OutlierSensitivityResult:
    """Result of outlier sensitivity analysis.
    
    Attributes:
        with_outliers: Result with outliers included
        without_outliers: Result with outliers removed
        outliers_detected: Number of outliers detected
        impact_score: Score from 0-1 indicating outlier impact
        is_robust: Whether results are robust to outlier removal
        details: Additional outlier details
    """
    with_outliers: Any
    without_outliers: Any
    outliers_detected: int
    impact_score: float
    is_robust: bool
    details: Dict[str, Any] = field(default_factory=dict)


@dataclass
class MethodComparisonResult:
    """Result of method comparison analysis.
    
    Attributes:
        method_results: Dictionary mapping method names to results
        is_consistent: Whether methods produce consistent results
        consistency_score: Score from 0-1 indicating consistency
        recommended_method: Name of recommended method
        details: Additional comparison details
    """
    method_results: Dict[str, Any]
    is_consistent: bool
    consistency_score: float
    recommended_method: str
    details: Dict[str, Any] = field(default_factory=dict)


@dataclass
class StabilityResult:
    """Result of subsample stability analysis.
    
    Attributes:
        subsample_results: List of results from subsamples
        mean_result: Mean result across subsamples
        std_result: Standard deviation of results
        is_stable: Whether results are stable across subsamples
        stability_score: Score from 0-1 indicating stability
        details: Additional stability details
    """
    subsample_results: List[Any]
    mean_result: Any
    std_result: float
    is_stable: bool
    stability_score: float
    details: Dict[str, Any] = field(default_factory=dict)


class RobustnessChecker:
    """Robustness checker for sensitivity and stability analysis.
    
    This class implements various robustness checks to ensure findings
    are not artifacts of specific parameter choices or data peculiarities.
    """
    
    def __init__(self, significance_level: float = 0.05):
        """Initialize robustness checker.
        
        Args:
            significance_level: Alpha level for statistical tests
        """
        self.significance_level = significance_level
    
    def sensitivity_analysis(
        self,
        analysis_function: Callable,
        parameters: Dict[str, List[Any]],
        base_result: Any
    ) -> SensitivityResult:
        """Perform sensitivity analysis by varying parameters.
        
        This method tests how results change when analysis parameters
        are varied, helping identify whether conclusions are robust
        to parameter choices.
        
        Args:
            analysis_function: Function to analyze (takes **kwargs)
            parameters: Dictionary mapping parameter names to lists of values to test
            base_result: Result with base/default parameters
        
        Returns:
            SensitivityResult showing how results vary with parameters
        """
        parameter_results = {}
        result_values = []
        
        # Test each parameter variation
        for param_name, param_values in parameters.items():
            for param_value in param_values:
                # Create parameter dict with this variation
                param_key = f"{param_name}={param_value}"
                
                try:
                    # Call analysis function with this parameter
                    result = analysis_function(**{param_name: param_value})
                    parameter_results[param_key] = result
                    
                    # Extract numeric value for stability calculation
                    if hasattr(result, 'effect_size'):
                        result_values.append(result.effect_size)
                    elif hasattr(result, 'statistic'):
                        result_values.append(result.statistic)
                    elif isinstance(result, (int, float)):
                        result_values.append(result)
                except Exception as e:
                    parameter_results[param_key] = {'error': str(e)}
        
        # Calculate stability score
        if len(result_values) >= 2:
            # Use coefficient of variation as stability metric
            mean_val = np.mean(result_values)
            std_val = np.std(result_values)
            
            if mean_val != 0:
                cv = std_val / abs(mean_val)
                # Stability score: high when CV is low
                stability_score = max(0.0, 1.0 - cv)
            else:
                stability_score = 0.0
            
            # Results are stable if CV < 0.2 (20% variation)
            is_stable = cv < 0.2
        else:
            stability_score = 0.0
            is_stable = False
        
        return SensitivityResult(
            base_result=base_result,
            parameter_results=parameter_results,
            is_stable=is_stable,
            stability_score=stability_score,
            details={
                'result_values': result_values,
                'mean': np.mean(result_values) if result_values else None,
                'std': np.std(result_values) if result_values else None
            }
        )
    
    def outlier_sensitivity(
        self,
        series: Series,
        analysis_function: Callable
    ) -> OutlierSensitivityResult:
        """Test sensitivity to outliers.
        
        This method compares analysis results with and without outliers
        to determine if conclusions are robust to extreme values.
        
        Args:
            series: Time series data
            analysis_function: Function to analyze (takes Series as input)
        
        Returns:
            OutlierSensitivityResult comparing with/without outliers
        """
        # Detect outliers using z-score method
        z_scores = np.abs(stats.zscore(series, nan_policy='omit'))
        outlier_threshold = 3.0
        outlier_mask = z_scores > outlier_threshold
        outliers_detected = int(outlier_mask.sum())
        
        # Run analysis with outliers
        try:
            with_outliers = analysis_function(series)
        except Exception as e:
            with_outliers = {'error': str(e)}
        
        # Run analysis without outliers
        series_no_outliers = series[~outlier_mask]
        try:
            without_outliers = analysis_function(series_no_outliers)
        except Exception as e:
            without_outliers = {'error': str(e)}
        
        # Calculate impact score
        impact_score = 0.0
        is_robust = True
        
        # Extract numeric values for comparison
        val_with = None
        val_without = None
        
        if hasattr(with_outliers, 'effect_size'):
            val_with = with_outliers.effect_size
        elif hasattr(with_outliers, 'statistic'):
            val_with = with_outliers.statistic
        elif isinstance(with_outliers, (int, float)):
            val_with = with_outliers
        
        if hasattr(without_outliers, 'effect_size'):
            val_without = without_outliers.effect_size
        elif hasattr(without_outliers, 'statistic'):
            val_without = without_outliers.statistic
        elif isinstance(without_outliers, (int, float)):
            val_without = without_outliers
        
        # Calculate impact
        if val_with is not None and val_without is not None:
            if val_with != 0:
                relative_change = abs((val_without - val_with) / val_with)
                impact_score = min(1.0, relative_change)
                # Results are robust if change < 10%
                is_robust = relative_change < 0.1
            else:
                impact_score = 0.0 if val_without == 0 else 1.0
                is_robust = val_without == 0
        
        return OutlierSensitivityResult(
            with_outliers=with_outliers,
            without_outliers=without_outliers,
            outliers_detected=outliers_detected,
            impact_score=impact_score,
            is_robust=is_robust,
            details={
                'outlier_indices': np.where(outlier_mask)[0].tolist(),
                'outlier_values': series[outlier_mask].tolist(),
                'val_with': val_with,
                'val_without': val_without
            }
        )
    
    def method_comparison(
        self,
        series: Series,
        methods: Dict[str, Callable]
    ) -> MethodComparisonResult:
        """Compare results across different analysis methods.
        
        This method applies multiple analysis methods to the same data
        and checks for consistency, helping validate that conclusions
        are not method-specific artifacts.
        
        Args:
            series: Time series data
            methods: Dictionary mapping method names to analysis functions
        
        Returns:
            MethodComparisonResult showing consistency across methods
        """
        method_results = {}
        result_values = []
        
        # Apply each method
        for method_name, method_func in methods.items():
            try:
                result = method_func(series)
                method_results[method_name] = result
                
                # Extract numeric value
                if hasattr(result, 'effect_size'):
                    result_values.append(result.effect_size)
                elif hasattr(result, 'statistic'):
                    result_values.append(result.statistic)
                elif isinstance(result, (int, float)):
                    result_values.append(result)
            except Exception as e:
                method_results[method_name] = {'error': str(e)}
        
        # Calculate consistency score
        if len(result_values) >= 2:
            # Use coefficient of variation
            mean_val = np.mean(result_values)
            std_val = np.std(result_values)
            
            if mean_val != 0:
                cv = std_val / abs(mean_val)
                consistency_score = max(0.0, 1.0 - cv)
            else:
                consistency_score = 1.0 if std_val == 0 else 0.0
            
            # Methods are consistent if CV < 0.15 (15% variation)
            is_consistent = cv < 0.15
        else:
            consistency_score = 0.0
            is_consistent = False
        
        # Recommend method with median result (most robust)
        if result_values:
            median_idx = np.argsort(result_values)[len(result_values) // 2]
            recommended_method = list(methods.keys())[median_idx]
        else:
            recommended_method = list(methods.keys())[0] if methods else 'none'
        
        return MethodComparisonResult(
            method_results=method_results,
            is_consistent=is_consistent,
            consistency_score=consistency_score,
            recommended_method=recommended_method,
            details={
                'result_values': result_values,
                'mean': np.mean(result_values) if result_values else None,
                'std': np.std(result_values) if result_values else None,
                'cv': std_val / abs(mean_val) if (result_values and mean_val != 0) else None
            }
        )
    
    def subsample_stability(
        self,
        series: Series,
        analysis_function: Callable,
        n_subsamples: int = 100,
        subsample_fraction: float = 0.8
    ) -> StabilityResult:
        """Test stability using bootstrap subsampling.
        
        This method repeatedly analyzes random subsamples of the data
        to check if results are stable across different data subsets.
        
        Args:
            series: Time series data
            analysis_function: Function to analyze (takes Series as input)
            n_subsamples: Number of subsamples to generate
            subsample_fraction: Fraction of data to include in each subsample
        
        Returns:
            StabilityResult showing stability across subsamples
        """
        subsample_results = []
        result_values = []
        
        subsample_size = int(len(series) * subsample_fraction)
        
        # Generate and analyze subsamples
        for i in range(n_subsamples):
            # Random subsample with replacement (bootstrap)
            subsample_indices = np.random.choice(
                len(series),
                size=subsample_size,
                replace=True
            )
            subsample = series.iloc[subsample_indices]
            
            try:
                result = analysis_function(subsample)
                subsample_results.append(result)
                
                # Extract numeric value
                if hasattr(result, 'effect_size'):
                    result_values.append(result.effect_size)
                elif hasattr(result, 'statistic'):
                    result_values.append(result.statistic)
                elif isinstance(result, (int, float)):
                    result_values.append(result)
            except Exception as e:
                subsample_results.append({'error': str(e)})
        
        # Calculate stability metrics
        if len(result_values) >= 2:
            mean_result = np.mean(result_values)
            std_result = np.std(result_values)
            
            if mean_result != 0:
                cv = std_result / abs(mean_result)
                stability_score = max(0.0, 1.0 - cv)
            else:
                stability_score = 1.0 if std_result == 0 else 0.0
            
            # Results are stable if CV < 0.2 (20% variation)
            is_stable = cv < 0.2
        else:
            mean_result = None
            std_result = 0.0
            stability_score = 0.0
            is_stable = False
        
        return StabilityResult(
            subsample_results=subsample_results,
            mean_result=mean_result,
            std_result=std_result,
            is_stable=is_stable,
            stability_score=stability_score,
            details={
                'result_values': result_values,
                'n_subsamples': n_subsamples,
                'subsample_fraction': subsample_fraction,
                'cv': std_result / abs(mean_result) if (mean_result and mean_result != 0) else None
            }
        )
