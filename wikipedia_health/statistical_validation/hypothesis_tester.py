"""Hypothesis testing for statistical validation.

This module implements various statistical hypothesis tests for validating
trends, changes, and patterns in Wikipedia product health data.
"""

from typing import List, Literal
import numpy as np
from pandas import Series
from scipy import stats
from dataclasses import dataclass

from wikipedia_health.models.data_models import TestResult


@dataclass
class HypothesisTester:
    """Performs statistical hypothesis testing.
    
    Implements various parametric and non-parametric tests for comparing
    samples and testing hypotheses about data distributions.
    """
    
    def t_test(
        self,
        sample1: Series,
        sample2: Series,
        alternative: Literal['two-sided', 'less', 'greater'] = 'two-sided'
    ) -> TestResult:
        """Perform independent samples t-test.
        
        Tests whether two independent samples have different means.
        Assumes normal distributions and equal variances.
        
        Args:
            sample1: First sample data
            sample2: Second sample data
            alternative: Type of alternative hypothesis
                - 'two-sided': means are different
                - 'less': mean of sample1 < mean of sample2
                - 'greater': mean of sample1 > mean of sample2
        
        Returns:
            TestResult with t-statistic, p-value, and effect size (Cohen's d)
        """
        # Remove NaN values
        s1 = sample1.dropna()
        s2 = sample2.dropna()
        
        # Check for zero variance (all values identical)
        if s1.std() == 0 and s2.std() == 0:
            # Both samples have zero variance
            if s1.mean() == s2.mean():
                # Identical samples - no difference
                statistic = 0.0
                p_value = 1.0
                effect_size = 0.0
                mean_diff = 0.0
                ci_lower = 0.0
                ci_upper = 0.0
            else:
                # Different means but zero variance - infinite t-statistic
                statistic = np.inf if s1.mean() > s2.mean() else -np.inf
                p_value = 0.0
                effect_size = np.inf if s1.mean() > s2.mean() else -np.inf
                mean_diff = s1.mean() - s2.mean()
                ci_lower = mean_diff
                ci_upper = mean_diff
        else:
            # Perform t-test
            statistic, p_value = stats.ttest_ind(s1, s2, alternative=alternative)
            
            # Handle NaN p-values (can occur with zero variance in one sample)
            if np.isnan(p_value):
                p_value = 1.0 if np.isnan(statistic) or statistic == 0 else 0.0
            if np.isnan(statistic):
                statistic = 0.0
            
            # Calculate Cohen's d effect size
            pooled_std = np.sqrt(((len(s1) - 1) * s1.std() ** 2 + 
                                  (len(s2) - 1) * s2.std() ** 2) / 
                                 (len(s1) + len(s2) - 2))
            effect_size = (s1.mean() - s2.mean()) / pooled_std if pooled_std > 0 else 0.0
            
            # Calculate 95% confidence interval for mean difference
            mean_diff = s1.mean() - s2.mean()
            se_diff = pooled_std * np.sqrt(1/len(s1) + 1/len(s2)) if pooled_std > 0 else 0.0
            ci_lower = mean_diff - 1.96 * se_diff
            ci_upper = mean_diff + 1.96 * se_diff
        
        # Determine significance
        alpha = 0.05
        is_significant = bool(p_value < alpha)
        
        # Generate interpretation
        interpretation = self._interpret_t_test(
            s1.mean(), s2.mean(), p_value, effect_size, is_significant
        )
        
        return TestResult(
            test_name='Independent Samples t-test',
            statistic=float(statistic),
            p_value=float(p_value),
            effect_size=float(effect_size),
            confidence_interval=(float(ci_lower), float(ci_upper)),
            is_significant=is_significant,
            alpha=alpha,
            interpretation=interpretation
        )
    
    def anova(
        self,
        groups: List[Series]
    ) -> TestResult:
        """Perform one-way ANOVA test.
        
        Tests whether multiple groups have different means.
        Assumes normal distributions and equal variances across groups.
        
        Args:
            groups: List of sample data for each group
        
        Returns:
            TestResult with F-statistic, p-value, and effect size (eta-squared)
        """
        # Remove NaN values from each group
        clean_groups = [g.dropna() for g in groups]
        
        # Check for zero variance across all groups
        all_stds = [g.std() for g in clean_groups]
        if all(std == 0 for std in all_stds):
            # All groups have zero variance
            all_means = [g.mean() for g in clean_groups]
            if len(set(all_means)) == 1:
                # All groups identical - no difference
                statistic = 0.0
                p_value = 1.0
                effect_size = 0.0
            else:
                # Different means but zero variance - infinite F-statistic
                statistic = np.inf
                p_value = 0.0
                effect_size = 1.0
        else:
            # Perform one-way ANOVA
            statistic, p_value = stats.f_oneway(*clean_groups)
            
            # Handle NaN values
            if np.isnan(p_value):
                p_value = 1.0
            if np.isnan(statistic):
                statistic = 0.0
            
            # Calculate eta-squared effect size
            grand_mean = np.mean([g.mean() for g in clean_groups])
            ss_between = sum(len(g) * (g.mean() - grand_mean) ** 2 for g in clean_groups)
            ss_total = sum(((g - grand_mean) ** 2).sum() for g in clean_groups)
            effect_size = ss_between / ss_total if ss_total > 0 else 0.0
        
        # Calculate confidence interval (approximate using F-distribution)
        alpha = 0.05
        
        # CI for effect size (eta-squared) - simplified approximation
        ci_lower = max(0.0, effect_size - 0.1)
        ci_upper = min(1.0, effect_size + 0.1)
        
        # Determine significance
        is_significant = bool(p_value < alpha)
        
        # Generate interpretation
        group_means = [g.mean() for g in clean_groups]
        interpretation = self._interpret_anova(
            group_means, p_value, effect_size, is_significant
        )
        
        return TestResult(
            test_name='One-Way ANOVA',
            statistic=float(statistic),
            p_value=float(p_value),
            effect_size=float(effect_size),
            confidence_interval=(float(ci_lower), float(ci_upper)),
            is_significant=is_significant,
            alpha=alpha,
            interpretation=interpretation
        )
    
    def mann_whitney(
        self,
        sample1: Series,
        sample2: Series
    ) -> TestResult:
        """Perform Mann-Whitney U test (non-parametric alternative to t-test).
        
        Tests whether two independent samples have different distributions.
        Does not assume normal distributions.
        
        Args:
            sample1: First sample data
            sample2: Second sample data
        
        Returns:
            TestResult with U-statistic, p-value, and effect size (rank-biserial correlation)
        """
        # Remove NaN values
        s1 = sample1.dropna()
        s2 = sample2.dropna()
        
        # Perform Mann-Whitney U test
        statistic, p_value = stats.mannwhitneyu(s1, s2, alternative='two-sided')
        
        # Calculate rank-biserial correlation as effect size
        n1, n2 = len(s1), len(s2)
        effect_size = 1 - (2 * statistic) / (n1 * n2)
        
        # Calculate confidence interval for median difference (bootstrap approximation)
        median_diff = s1.median() - s2.median()
        # Simplified CI - in practice would use bootstrap
        se_approx = np.sqrt((n1 + n2) / (n1 * n2)) * np.std(np.concatenate([s1, s2]))
        ci_lower = median_diff - 1.96 * se_approx
        ci_upper = median_diff + 1.96 * se_approx
        
        # Determine significance
        alpha = 0.05
        is_significant = bool(p_value < alpha)
        
        # Generate interpretation
        interpretation = self._interpret_mann_whitney(
            s1.median(), s2.median(), p_value, effect_size, is_significant
        )
        
        return TestResult(
            test_name='Mann-Whitney U Test',
            statistic=float(statistic),
            p_value=float(p_value),
            effect_size=float(effect_size),
            confidence_interval=(float(ci_lower), float(ci_upper)),
            is_significant=is_significant,
            alpha=alpha,
            interpretation=interpretation
        )
    
    def kruskal_wallis(
        self,
        groups: List[Series]
    ) -> TestResult:
        """Perform Kruskal-Wallis H test (non-parametric alternative to ANOVA).
        
        Tests whether multiple groups have different distributions.
        Does not assume normal distributions.
        
        Args:
            groups: List of sample data for each group
        
        Returns:
            TestResult with H-statistic, p-value, and effect size (epsilon-squared)
        """
        # Remove NaN values from each group
        clean_groups = [g.dropna() for g in groups]
        
        # Check if all values are identical across all groups
        all_values = np.concatenate([g.values for g in clean_groups])
        if len(np.unique(all_values)) == 1:
            # All values identical - no difference between groups
            statistic = 0.0
            p_value = 1.0
            effect_size = 0.0
        else:
            try:
                # Perform Kruskal-Wallis test
                statistic, p_value = stats.kruskal(*clean_groups)
                
                # Handle NaN values
                if np.isnan(p_value):
                    p_value = 1.0
                if np.isnan(statistic):
                    statistic = 0.0
                
                # Calculate epsilon-squared effect size
                n_total = sum(len(g) for g in clean_groups)
                k = len(groups)
                effect_size = (statistic - k + 1) / (n_total - k) if n_total > k else 0.0
                effect_size = max(0.0, min(1.0, effect_size))  # Bound between 0 and 1
                
            except ValueError as e:
                # Handle ValueError when all values are identical (scipy raises this)
                if "All numbers are identical" in str(e):
                    statistic = 0.0
                    p_value = 1.0
                    effect_size = 0.0
                else:
                    raise
        
        # Calculate confidence interval (approximate)
        ci_lower = max(0.0, effect_size - 0.1)
        ci_upper = min(1.0, effect_size + 0.1)
        
        # Determine significance
        alpha = 0.05
        is_significant = bool(p_value < alpha)
        
        # Generate interpretation
        group_medians = [g.median() for g in clean_groups]
        interpretation = self._interpret_kruskal_wallis(
            group_medians, p_value, effect_size, is_significant
        )
        
        return TestResult(
            test_name='Kruskal-Wallis H Test',
            statistic=float(statistic),
            p_value=float(p_value),
            effect_size=float(effect_size),
            confidence_interval=(float(ci_lower), float(ci_upper)),
            is_significant=is_significant,
            alpha=alpha,
            interpretation=interpretation
        )
    
    def permutation_test(
        self,
        sample1: Series,
        sample2: Series,
        n_permutations: int = 10000
    ) -> TestResult:
        """Perform permutation test for difference in means.
        
        Non-parametric test that makes no distributional assumptions.
        Tests whether two samples have different means by randomly
        permuting group labels.
        
        Args:
            sample1: First sample data
            sample2: Second sample data
            n_permutations: Number of random permutations to perform
        
        Returns:
            TestResult with observed difference, p-value, and effect size
        """
        # Remove NaN values
        s1 = sample1.dropna()
        s2 = sample2.dropna()
        
        # Calculate observed difference in means
        observed_diff = s1.mean() - s2.mean()
        
        # Combine samples
        combined = np.concatenate([s1, s2])
        n1 = len(s1)
        
        # Perform permutations
        perm_diffs = []
        rng = np.random.RandomState(42)  # For reproducibility
        
        for _ in range(n_permutations):
            # Randomly shuffle and split
            shuffled = rng.permutation(combined)
            perm_s1 = shuffled[:n1]
            perm_s2 = shuffled[n1:]
            perm_diffs.append(perm_s1.mean() - perm_s2.mean())
        
        perm_diffs = np.array(perm_diffs)
        
        # Calculate p-value (two-sided)
        p_value = np.mean(np.abs(perm_diffs) >= np.abs(observed_diff))
        
        # Calculate effect size (standardized mean difference)
        pooled_std = np.std(combined)
        effect_size = observed_diff / pooled_std if pooled_std > 0 else 0.0
        
        # Calculate confidence interval from permutation distribution
        ci_lower = np.percentile(perm_diffs, 2.5)
        ci_upper = np.percentile(perm_diffs, 97.5)
        
        # Determine significance
        alpha = 0.05
        is_significant = bool(p_value < alpha)
        
        # Generate interpretation
        interpretation = self._interpret_permutation_test(
            observed_diff, p_value, effect_size, is_significant, n_permutations
        )
        
        return TestResult(
            test_name='Permutation Test',
            statistic=float(observed_diff),
            p_value=float(p_value),
            effect_size=float(effect_size),
            confidence_interval=(float(ci_lower), float(ci_upper)),
            is_significant=is_significant,
            alpha=alpha,
            interpretation=interpretation
        )
    
    # Helper methods for generating interpretations
    
    def _interpret_t_test(
        self,
        mean1: float,
        mean2: float,
        p_value: float,
        effect_size: float,
        is_significant: bool
    ) -> str:
        """Generate human-readable interpretation of t-test results."""
        sig_text = "significant" if is_significant else "not significant"
        effect_magnitude = self._effect_size_magnitude(abs(effect_size))
        
        return (
            f"The difference between group means ({mean1:.2f} vs {mean2:.2f}) "
            f"is {sig_text} (p={p_value:.4f}). "
            f"Effect size (Cohen's d={effect_size:.3f}) is {effect_magnitude}."
        )
    
    def _interpret_anova(
        self,
        group_means: List[float],
        p_value: float,
        effect_size: float,
        is_significant: bool
    ) -> str:
        """Generate human-readable interpretation of ANOVA results."""
        sig_text = "significant" if is_significant else "not significant"
        effect_magnitude = self._effect_size_magnitude(effect_size)
        means_str = ", ".join(f"{m:.2f}" for m in group_means)
        
        return (
            f"The difference among group means ({means_str}) "
            f"is {sig_text} (p={p_value:.4f}). "
            f"Effect size (eta-squared={effect_size:.3f}) is {effect_magnitude}."
        )
    
    def _interpret_mann_whitney(
        self,
        median1: float,
        median2: float,
        p_value: float,
        effect_size: float,
        is_significant: bool
    ) -> str:
        """Generate human-readable interpretation of Mann-Whitney results."""
        sig_text = "significant" if is_significant else "not significant"
        effect_magnitude = self._effect_size_magnitude(abs(effect_size))
        
        return (
            f"The difference between group medians ({median1:.2f} vs {median2:.2f}) "
            f"is {sig_text} (p={p_value:.4f}). "
            f"Effect size (rank-biserial={effect_size:.3f}) is {effect_magnitude}."
        )
    
    def _interpret_kruskal_wallis(
        self,
        group_medians: List[float],
        p_value: float,
        effect_size: float,
        is_significant: bool
    ) -> str:
        """Generate human-readable interpretation of Kruskal-Wallis results."""
        sig_text = "significant" if is_significant else "not significant"
        effect_magnitude = self._effect_size_magnitude(effect_size)
        medians_str = ", ".join(f"{m:.2f}" for m in group_medians)
        
        return (
            f"The difference among group medians ({medians_str}) "
            f"is {sig_text} (p={p_value:.4f}). "
            f"Effect size (epsilon-squared={effect_size:.3f}) is {effect_magnitude}."
        )
    
    def _interpret_permutation_test(
        self,
        observed_diff: float,
        p_value: float,
        effect_size: float,
        is_significant: bool,
        n_permutations: int
    ) -> str:
        """Generate human-readable interpretation of permutation test results."""
        sig_text = "significant" if is_significant else "not significant"
        effect_magnitude = self._effect_size_magnitude(abs(effect_size))
        
        return (
            f"The observed mean difference ({observed_diff:.2f}) "
            f"is {sig_text} based on {n_permutations} permutations (p={p_value:.4f}). "
            f"Effect size (standardized difference={effect_size:.3f}) is {effect_magnitude}."
        )
    
    def _effect_size_magnitude(self, effect_size: float) -> str:
        """Classify effect size magnitude using Cohen's conventions."""
        if effect_size < 0.2:
            return "negligible"
        elif effect_size < 0.5:
            return "small"
        elif effect_size < 0.8:
            return "medium"
        else:
            return "large"
