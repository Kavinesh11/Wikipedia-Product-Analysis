# Statistical Output Interpretation Guide

This guide explains how to interpret the statistical outputs from the Wikipedia Product Health Analysis System.

## Table of Contents

1. [Hypothesis Test Results](#hypothesis-test-results)
2. [Confidence Intervals](#confidence-intervals)
3. [Effect Sizes](#effect-sizes)
4. [Changepoint Detection](#changepoint-detection)
5. [Causal Effects](#causal-effects)
6. [Forecast Results](#forecast-results)
7. [Validation Reports](#validation-reports)
8. [Common Pitfalls](#common-pitfalls)

## Hypothesis Test Results

### TestResult Object

```python
TestResult(
    test_name='t_test',
    statistic=3.45,
    p_value=0.0006,
    effect_size=0.52,
    confidence_interval=(0.23, 0.81),
    is_significant=True,
    alpha=0.05,
    interpretation='Significant difference detected'
)
```

### Interpretation

**P-value (0.0006)**:
- Very strong evidence against null hypothesis
- Probability of observing this result by chance is 0.06%
- Conclusion: The difference is statistically significant

**Statistic (3.45)**:
- Test statistic value
- Larger absolute values indicate stronger evidence
- For t-test: |t| > 2 typically indicates significance

**Effect Size (0.52)**:
- Medium effect (Cohen's d ≈ 0.5)
- Indicates practical significance
- Independent of sample size

**Confidence Interval (0.23, 0.81)**:
- 95% confident true effect is between 0.23 and 0.81
- Doesn't include zero → significant effect
- Width indicates precision (narrower = more precise)

**Is Significant (True)**:
- p-value < alpha (0.05)
- Reject null hypothesis
- Effect is statistically significant

### Decision Matrix

| P-value | Interpretation | Action |
|---------|---------------|--------|
| < 0.001 | Very strong evidence | Highly confident in finding |
| 0.001 - 0.01 | Strong evidence | Confident in finding |
| 0.01 - 0.05 | Moderate evidence | Finding is significant |
| 0.05 - 0.10 | Weak evidence | Marginally significant |
| > 0.10 | Insufficient evidence | Cannot conclude effect exists |

## Confidence Intervals

### Example Output

```python
{
    'mean': 1500000,
    'confidence_interval': (1450000, 1550000),
    'confidence_level': 0.95
}
```

### Interpretation

**Mean (1,500,000)**:
- Point estimate (best guess)
- Most likely value

**Confidence Interval (1,450,000 - 1,550,000)**:
- Range containing true value with 95% probability
- Uncertainty quantification
- Width = 100,000 (±3.3% of mean)

**Confidence Level (0.95)**:
- 95% of such intervals contain true value
- Higher level → wider interval
- Standard: 95% (sometimes 90% or 99%)

### Width Interpretation

| Relative Width | Interpretation | Confidence |
|----------------|----------------|------------|
| < 10% of mean | Very precise | High |
| 10-20% of mean | Moderately precise | Medium |
| 20-50% of mean | Imprecise | Low |
| > 50% of mean | Very imprecise | Very low |

### Using CIs for Comparison

**Scenario 1: Non-overlapping CIs**
```
Group A: [100, 120]
Group B: [130, 150]
```
→ Groups are significantly different

**Scenario 2: Overlapping CIs**
```
Group A: [100, 130]
Group B: [120, 150]
```
→ May or may not be significantly different (need formal test)

**Scenario 3: One CI contains other's mean**
```
Group A: [100, 150]  (mean = 125)
Group B: [120, 140]  (mean = 130)
```
→ Likely not significantly different

## Effect Sizes

### Cohen's d

```python
{
    'cohens_d': 0.65,
    'interpretation': 'medium effect'
}
```

**Interpretation Scale**:

| Cohen's d | Interpretation | Practical Meaning |
|-----------|----------------|-------------------|
| < 0.2 | Negligible | Barely noticeable |
| 0.2 - 0.5 | Small | Noticeable to experts |
| 0.5 - 0.8 | Medium | Noticeable to informed observers |
| > 0.8 | Large | Obvious to anyone |

**Example**: d = 0.65
- Medium-to-large effect
- Difference is noticeable
- Practically significant

### Percentage Change

```python
{
    'percentage_change': -15.3,
    'baseline': 1000000,
    'new_value': 847000
}
```

**Interpretation**:
- 15.3% decrease from baseline
- Absolute change: -153,000
- Context matters: 15% of what?

**Practical Significance**:

| Change | Interpretation |
|--------|----------------|
| < 5% | Small change |
| 5-15% | Moderate change |
| 15-30% | Large change |
| > 30% | Very large change |

## Changepoint Detection

### Changepoint Object

```python
Changepoint(
    date=date(2022, 11, 30),
    index=2890,
    confidence=0.92,
    magnitude=-0.18,
    direction='decrease',
    pre_mean=1500000,
    post_mean=1230000
)
```

### Interpretation

**Date (2022-11-30)**:
- When the change occurred
- Check for external events on this date

**Confidence (0.92)**:
- 92% confident this is a real changepoint
- High confidence (> 0.8)
- Not due to random variation

**Magnitude (-0.18)**:
- 18% decrease after changepoint
- Negative = decrease, positive = increase
- Large magnitude (> 10%)

**Pre/Post Means**:
- Before: 1,500,000 daily pageviews
- After: 1,230,000 daily pageviews
- Absolute change: -270,000 (-18%)

### Confidence Interpretation

| Confidence | Interpretation | Action |
|------------|----------------|--------|
| > 0.9 | Very high | Strong evidence for changepoint |
| 0.8 - 0.9 | High | Likely real changepoint |
| 0.6 - 0.8 | Moderate | Possible changepoint |
| < 0.6 | Low | May be noise |

## Causal Effects

### CausalEffect Object

```python
CausalEffect(
    effect_size=0.12,
    confidence_interval=(0.05, 0.19),
    p_value=0.002,
    method='interrupted_time_series',
    counterfactual=<Series>,
    observed=<Series>,
    treatment_period=DateRange(...)
)
```

### Interpretation

**Effect Size (0.12)**:
- 12% increase due to intervention
- Causal impact (not just correlation)
- Positive = beneficial effect

**Confidence Interval (0.05, 0.19)**:
- True effect between 5% and 19%
- Doesn't include zero → significant
- Relatively narrow → precise estimate

**P-value (0.002)**:
- Strong evidence for causal effect
- Very unlikely due to chance (0.2%)
- Statistically significant

**Method (interrupted_time_series)**:
- Causal inference method used
- Different methods have different assumptions
- Check assumption tests

### Causal vs Correlation

**Correlation**: Variables move together
- Example: Ice cream sales and drowning deaths

**Causation**: One causes the other
- Example: Campaign causes traffic increase

**Evidence for Causation**:
1. ✓ Temporal precedence (cause before effect)
2. ✓ Statistical association (correlation)
3. ✓ Counterfactual comparison (what would have happened)
4. ✓ Assumption tests pass
5. ✓ Robustness checks confirm

## Forecast Results

### ForecastResult Object

```python
ForecastResult(
    point_forecast=<Series>,  # [1.5M, 1.52M, 1.54M, ...]
    lower_bound=<Series>,     # [1.3M, 1.31M, 1.32M, ...]
    upper_bound=<Series>,     # [1.7M, 1.73M, 1.76M, ...]
    confidence_level=0.95,
    model_type='ensemble',
    horizon=90
)
```

### Interpretation

**Point Forecast**:
- Most likely outcome
- Best single estimate
- Not guaranteed to occur

**Prediction Intervals**:
- 95% PI: [1.3M, 1.7M]
- True value will be in this range 95% of the time
- Wider intervals = more uncertainty

**Horizon (90 days)**:
- Forecasting 90 days ahead
- Uncertainty increases with horizon
- Longer horizons → wider intervals

### Forecast Uncertainty

| Horizon | Typical Uncertainty | Reliability |
|---------|---------------------|-------------|
| 1-7 days | ±5-10% | High |
| 8-30 days | ±10-20% | Medium |
| 31-90 days | ±20-40% | Low |
| > 90 days | ±40%+ | Very low |

### Using Forecasts

**Scenario Planning**:
```python
{
    'optimistic': 1.8M,  # 90th percentile
    'baseline': 1.5M,    # 50th percentile (median)
    'pessimistic': 1.2M  # 10th percentile
}
```

**Interpretation**:
- Plan for baseline (most likely)
- Prepare for pessimistic (risk management)
- Hope for optimistic (upside potential)

## Validation Reports

### ValidationReport Object

```python
ValidationReport(
    is_valid=True,
    completeness_score=0.98,
    missing_dates=[],
    anomalies=[Anomaly(...)],
    quality_metrics={
        'mean': 1500000,
        'std': 150000,
        'cv': 0.10
    },
    recommendations=['Check anomaly on 2023-01-15']
)
```

### Interpretation

**Is Valid (True)**:
- Data passes quality checks
- Safe to proceed with analysis

**Completeness Score (0.98)**:
- 98% of expected data present
- High completeness (> 0.95)
- Missing 2% of dates

**Anomalies**:
- Unusual data points detected
- May be real events or data errors
- Investigate before analysis

**Quality Metrics**:
- Mean: Average value
- Std: Variability
- CV (Coefficient of Variation): Std/Mean
  - CV < 0.15: Low variability
  - CV 0.15-0.30: Moderate variability
  - CV > 0.30: High variability

## Common Pitfalls

### 1. Confusing Statistical and Practical Significance

**Wrong**: "p < 0.05, so the effect is important"

**Right**: "p < 0.05 (statistically significant) and effect size = 0.8 (large), so the effect is both real and important"

**Lesson**: Always check effect size, not just p-value.

### 2. Ignoring Confidence Intervals

**Wrong**: "The effect is 12%"

**Right**: "The effect is 12% (95% CI: 5-19%)"

**Lesson**: Report uncertainty, not just point estimates.

### 3. Assuming Correlation Implies Causation

**Wrong**: "Traffic and editors are correlated, so traffic causes editors"

**Right**: "Traffic and editors are correlated (r=0.7, p<0.001). Causal analysis needed to determine direction."

**Lesson**: Use causal inference methods for causal claims.

### 4. Cherry-Picking Significant Results

**Wrong**: Run 20 tests, report only the 1 significant result

**Right**: Report all tests, adjust for multiple testing

**Lesson**: Correct for multiple comparisons (Bonferroni, FDR).

### 5. Ignoring Assumptions

**Wrong**: Use t-test without checking normality

**Right**: Check assumptions, use non-parametric alternative if violated

**Lesson**: Test assumptions before applying methods.

### 6. Over-Interpreting Non-Significant Results

**Wrong**: "p = 0.08, so there's no effect"

**Right**: "p = 0.08, so we cannot conclude there's an effect with 95% confidence. The effect may exist but we lack power to detect it."

**Lesson**: Absence of evidence ≠ evidence of absence.

### 7. Ignoring Context

**Wrong**: "15% increase is large"

**Right**: "15% increase (225,000 daily pageviews) is large in absolute terms and represents a meaningful business impact"

**Lesson**: Interpret statistics in domain context.

## Quick Reference

### P-value Interpretation

| P-value | Strength of Evidence |
|---------|---------------------|
| < 0.001 | Very strong |
| 0.001-0.01 | Strong |
| 0.01-0.05 | Moderate |
| 0.05-0.10 | Weak |
| > 0.10 | Insufficient |

### Effect Size Interpretation (Cohen's d)

| Effect Size | Interpretation |
|-------------|----------------|
| < 0.2 | Negligible |
| 0.2-0.5 | Small |
| 0.5-0.8 | Medium |
| > 0.8 | Large |

### Confidence Level

| Level | Use Case |
|-------|----------|
| 90% | Exploratory analysis |
| 95% | Standard (default) |
| 99% | High-stakes decisions |

### Sample Size Guidelines

| Analysis Type | Minimum Sample Size |
|---------------|---------------------|
| Trend analysis | 90 days |
| Causal inference | 30 days pre + 30 days post |
| Seasonality | 2 full cycles (e.g., 2 years for annual) |
| Forecasting | 10× forecast horizon |

## Further Reading

- [Methodology Documentation](methodology.rst) - Detailed statistical methods
- [User Guides](user_guides.rst) - Step-by-step analysis guides
- [API Reference](api_reference.rst) - Complete API documentation
