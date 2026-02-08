"""Synthetic Control Method for causal inference.

This module implements synthetic control methodology to estimate causal effects
by constructing a weighted combination of control units that best matches the
treated unit in the pre-intervention period.
"""

from datetime import date
from typing import List, Tuple, Dict
import numpy as np
import pandas as pd
from pandas import Series
from scipy.optimize import minimize
from scipy import stats
from wikipedia_health.models import CausalEffect, TestResult


class SyntheticControl:
    """Synthetic control constructed from donor pool.
    
    Attributes:
        weights: Weights for each donor unit
        donor_names: Names/identifiers for donor units
        fitted_values: Synthetic control values for pre-period
        r_squared: Goodness of fit in pre-period
    """
    
    def __init__(
        self,
        weights: np.ndarray,
        donor_names: List[str],
        fitted_values: Series,
        r_squared: float
    ):
        self.weights = weights
        self.donor_names = donor_names
        self.fitted_values = fitted_values
        self.r_squared = r_squared
    
    def predict(self, donor_pool: List[Series]) -> Series:
        """Generate synthetic control predictions.
        
        Args:
            donor_pool: List of donor unit time series
            
        Returns:
            Weighted combination of donor units
        """
        # Stack donor series into matrix
        donor_matrix = np.column_stack([donor.values for donor in donor_pool])
        
        # Apply weights
        synthetic = donor_matrix @ self.weights
        
        # Use index from first donor (assuming all aligned)
        return Series(synthetic, index=donor_pool[0].index, name='synthetic_control')


class PlaceboResult:
    """Result from placebo test on a donor unit.
    
    Attributes:
        unit_name: Name of the placebo unit
        effect_size: Estimated effect for placebo unit
        pre_period_fit: R² in pre-period for placebo
    """
    
    def __init__(self, unit_name: str, effect_size: float, pre_period_fit: float):
        self.unit_name = unit_name
        self.effect_size = effect_size
        self.pre_period_fit = pre_period_fit


class SyntheticControlBuilder:
    """Synthetic Control Method for causal inference.
    
    Constructs a synthetic control unit as a weighted combination of donor
    units that best matches the treated unit in the pre-intervention period.
    """
    
    def __init__(self, min_r_squared: float = 0.7):
        """Initialize synthetic control builder.
        
        Args:
            min_r_squared: Minimum required R² for synthetic control fit
        """
        self.min_r_squared = min_r_squared
    
    def construct_synthetic_control(
        self,
        treated_unit: Series,
        donor_pool: List[Series],
        pre_period: Tuple[date, date]
    ) -> SyntheticControl:
        """Construct synthetic control from donor pool.
        
        Finds optimal weights for donor units to minimize pre-period
        prediction error for the treated unit.
        
        Args:
            treated_unit: Time series for treated unit (with DatetimeIndex)
            donor_pool: List of time series for donor units
            pre_period: Tuple of (start_date, end_date) for pre-intervention period
            
        Returns:
            SyntheticControl with optimal weights
            
        Raises:
            ValueError: If insufficient donors or poor fit
        """
        if len(donor_pool) < 2:
            raise ValueError("Need at least 2 donor units for synthetic control")
        
        # Ensure all series have DatetimeIndex
        if not isinstance(treated_unit.index, pd.DatetimeIndex):
            raise ValueError("Treated unit must have DatetimeIndex")
        
        for i, donor in enumerate(donor_pool):
            if not isinstance(donor.index, pd.DatetimeIndex):
                raise ValueError(f"Donor {i} must have DatetimeIndex")
        
        # Extract pre-period data
        start_ts = pd.Timestamp(pre_period[0])
        end_ts = pd.Timestamp(pre_period[1])
        
        treated_pre = treated_unit[
            (treated_unit.index >= start_ts) & (treated_unit.index <= end_ts)
        ]
        
        donors_pre = []
        for donor in donor_pool:
            donor_pre = donor[
                (donor.index >= start_ts) & (donor.index <= end_ts)
            ]
            donors_pre.append(donor_pre)
        
        # Validate data availability
        if len(treated_pre) < 30:
            raise ValueError("Insufficient pre-period data for treated unit (need at least 30 points)")
        
        for i, donor_pre in enumerate(donors_pre):
            if len(donor_pre) < 30:
                raise ValueError(f"Insufficient pre-period data for donor {i} (need at least 30 points)")
        
        # Stack donor data into matrix (rows = time, columns = donors)
        donor_matrix = np.column_stack([donor.values for donor in donors_pre])
        treated_values = treated_pre.values
        
        # Optimize weights to minimize squared prediction error
        # Constraints: weights sum to 1, all weights >= 0
        n_donors = len(donor_pool)
        
        def objective(weights):
            """Squared prediction error."""
            predicted = donor_matrix @ weights
            return np.sum((treated_values - predicted) ** 2)
        
        # Constraints
        constraints = [
            {'type': 'eq', 'fun': lambda w: np.sum(w) - 1}  # Sum to 1
        ]
        
        # Bounds: all weights between 0 and 1
        bounds = [(0, 1) for _ in range(n_donors)]
        
        # Initial guess: equal weights
        initial_weights = np.ones(n_donors) / n_donors
        
        # Optimize
        result = minimize(
            objective,
            initial_weights,
            method='SLSQP',
            bounds=bounds,
            constraints=constraints,
            options={'maxiter': 1000}
        )
        
        if not result.success:
            raise ValueError(f"Optimization failed: {result.message}")
        
        optimal_weights = result.x
        
        # Calculate fitted values and R²
        fitted_values = donor_matrix @ optimal_weights
        fitted_series = Series(fitted_values, index=treated_pre.index, name='synthetic_control')
        
        # Calculate R²
        ss_res = np.sum((treated_values - fitted_values) ** 2)
        ss_tot = np.sum((treated_values - np.mean(treated_values)) ** 2)
        r_squared = 1 - (ss_res / ss_tot) if ss_tot > 0 else 0.0
        
        # Check fit quality
        if r_squared < self.min_r_squared:
            raise ValueError(
                f"Synthetic control fit is poor (R²={r_squared:.3f} < {self.min_r_squared}). "
                f"Consider expanding donor pool or using alternative methods."
            )
        
        # Generate donor names
        donor_names = [f"donor_{i}" for i in range(n_donors)]
        
        return SyntheticControl(
            weights=optimal_weights,
            donor_names=donor_names,
            fitted_values=fitted_series,
            r_squared=r_squared
        )
    
    def estimate_effect(
        self,
        treated: Series,
        synthetic: SyntheticControl,
        donor_pool: List[Series],
        post_period: Tuple[date, date],
        confidence_level: float = 0.95
    ) -> CausalEffect:
        """Estimate treatment effect by comparing treated to synthetic control.
        
        Args:
            treated: Time series for treated unit
            synthetic: Constructed synthetic control
            donor_pool: List of donor unit time series
            post_period: Tuple of (start_date, end_date) for post-intervention period
            confidence_level: Confidence level for intervals
            
        Returns:
            CausalEffect with treatment effect estimate
        """
        start_ts = pd.Timestamp(post_period[0])
        end_ts = pd.Timestamp(post_period[1])
        
        # Extract post-period data
        treated_post = treated[
            (treated.index >= start_ts) & (treated.index <= end_ts)
        ]
        
        # Generate synthetic control for post-period
        donors_post = []
        for donor in donor_pool:
            donor_post = donor[
                (donor.index >= start_ts) & (donor.index <= end_ts)
            ]
            donors_post.append(donor_post)
        
        synthetic_post = synthetic.predict(donors_post)
        
        # Calculate treatment effect (treated - synthetic)
        treatment_effects = treated_post.values - synthetic_post.values
        
        # Average treatment effect
        ate = np.mean(treatment_effects)
        
        # Calculate confidence interval using bootstrap
        n_bootstrap = 1000
        bootstrap_ates = []
        
        for _ in range(n_bootstrap):
            # Resample treatment effects with replacement
            resampled = np.random.choice(
                treatment_effects,
                size=len(treatment_effects),
                replace=True
            )
            bootstrap_ates.append(np.mean(resampled))
        
        # Calculate CI from bootstrap distribution
        alpha = 1 - confidence_level
        lower_percentile = (alpha / 2) * 100
        upper_percentile = (1 - alpha / 2) * 100
        ci_lower = np.percentile(bootstrap_ates, lower_percentile)
        ci_upper = np.percentile(bootstrap_ates, upper_percentile)
        
        # Calculate p-value using t-test
        t_stat, p_value = stats.ttest_1samp(treatment_effects, 0)
        
        return CausalEffect(
            effect_size=ate,
            confidence_interval=(ci_lower, ci_upper),
            p_value=p_value,
            method='SyntheticControl',
            counterfactual=synthetic_post,
            observed=treated_post,
            treatment_period=post_period
        )
    
    def placebo_test(
        self,
        donor_pool: List[Series],
        intervention_date: date,
        pre_period: Tuple[date, date],
        post_period: Tuple[date, date]
    ) -> List[PlaceboResult]:
        """Perform placebo tests on untreated units.
        
        Applies synthetic control method to each donor unit (treating it as
        "treated") to generate a distribution of placebo effects.
        
        Args:
            donor_pool: List of donor unit time series
            intervention_date: Date of intervention
            pre_period: Tuple of (start_date, end_date) for pre-period
            post_period: Tuple of (start_date, end_date) for post-period
            
        Returns:
            List of PlaceboResult for each donor unit
        """
        if len(donor_pool) < 3:
            raise ValueError("Need at least 3 donor units for placebo test")
        
        placebo_results = []
        
        # For each donor, treat it as the "treated" unit
        for i, placebo_treated in enumerate(donor_pool):
            # Create donor pool excluding current unit
            placebo_donors = [donor for j, donor in enumerate(donor_pool) if j != i]
            
            try:
                # Construct synthetic control for placebo unit
                synthetic = self.construct_synthetic_control(
                    placebo_treated,
                    placebo_donors,
                    pre_period
                )
                
                # Estimate "effect" for placebo unit
                effect = self.estimate_effect(
                    placebo_treated,
                    synthetic,
                    placebo_donors,
                    post_period
                )
                
                placebo_results.append(PlaceboResult(
                    unit_name=f"donor_{i}",
                    effect_size=effect.effect_size,
                    pre_period_fit=synthetic.r_squared
                ))
                
            except ValueError:
                # If synthetic control fails for this placebo, skip it
                continue
        
        if len(placebo_results) == 0:
            raise ValueError("All placebo tests failed - cannot perform inference")
        
        return placebo_results
    
    def inference(
        self,
        effect: CausalEffect,
        placebo_results: List[PlaceboResult],
        alpha: float = 0.05
    ) -> TestResult:
        """Calculate p-value from placebo distribution.
        
        Tests whether the actual treatment effect is unusually large compared
        to the distribution of placebo effects.
        
        Args:
            effect: CausalEffect for actual treated unit
            placebo_results: List of PlaceboResult from placebo tests
            alpha: Significance level
            
        Returns:
            TestResult with inference results
        """
        if len(placebo_results) == 0:
            raise ValueError("No placebo results available for inference")
        
        # Extract placebo effect sizes
        placebo_effects = np.array([pr.effect_size for pr in placebo_results])
        
        # Calculate p-value as proportion of placebo effects >= actual effect
        # (in absolute value)
        actual_effect_abs = abs(effect.effect_size)
        n_extreme = np.sum(np.abs(placebo_effects) >= actual_effect_abs)
        
        # Add 1 to numerator and denominator (include actual treated unit)
        p_value = (n_extreme + 1) / (len(placebo_effects) + 1)
        
        is_significant = p_value < alpha
        
        # Calculate effect size relative to placebo distribution
        placebo_mean = np.mean(placebo_effects)
        placebo_std = np.std(placebo_effects, ddof=1)
        
        if placebo_std > 0:
            standardized_effect = (effect.effect_size - placebo_mean) / placebo_std
        else:
            standardized_effect = 0.0
        
        if is_significant:
            interpretation = (
                f"Treatment effect is statistically significant (p={p_value:.4f}). "
                f"Effect is larger than {(1-p_value)*100:.1f}% of placebo effects. "
                f"Standardized effect: {standardized_effect:.2f} standard deviations."
            )
        else:
            interpretation = (
                f"Treatment effect is NOT statistically significant (p={p_value:.4f}). "
                f"Effect is not unusual compared to placebo distribution. "
                f"Standardized effect: {standardized_effect:.2f} standard deviations."
            )
        
        return TestResult(
            test_name='Synthetic Control Inference (Placebo Test)',
            statistic=standardized_effect,
            p_value=p_value,
            effect_size=effect.effect_size,
            confidence_interval=effect.confidence_interval,
            is_significant=is_significant,
            alpha=alpha,
            interpretation=interpretation
        )
