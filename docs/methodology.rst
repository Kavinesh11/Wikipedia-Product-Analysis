Statistical Methodology
=======================

This document explains the statistical methods, causal inference approaches, and interpretation guidelines used in the Wikipedia Product Health Analysis System.

.. contents:: Table of Contents
   :local:
   :depth: 3

Overview
--------

The system implements rigorous statistical validation and causal inference methodologies to ensure all findings are evidence-backed. Every conclusion is supported by:

* **Hypothesis testing** with p-values
* **Confidence intervals** for uncertainty quantification
* **Effect size** calculations for practical significance
* **Cross-validation** across multiple data sources
* **Robustness checks** to test sensitivity to assumptions

Statistical Foundations
-----------------------

Hypothesis Testing
~~~~~~~~~~~~~~~~~~

**Purpose**: Determine if observed patterns are statistically significant or due to random chance.

**Null Hypothesis (H₀)**: No effect or no difference exists.

**Alternative Hypothesis (H₁)**: An effect or difference exists.

**P-value**: Probability of observing the data (or more extreme) if H₀ is true.

**Decision Rule**:

* If p-value < α (typically 0.05), reject H₀ (significant)
* If p-value ≥ α, fail to reject H₀ (not significant)

**Interpretation**:

* p < 0.001: Very strong evidence against H₀
* p < 0.01: Strong evidence against H₀
* p < 0.05: Moderate evidence against H₀
* p ≥ 0.05: Insufficient evidence against H₀

Confidence Intervals
~~~~~~~~~~~~~~~~~~~~

**Purpose**: Quantify uncertainty in estimates.

**95% Confidence Interval**: Range that contains the true parameter value with 95% probability.

**Interpretation**:

* Narrow CI: Precise estimate
* Wide CI: Uncertain estimate
* CI excluding zero: Significant effect

**Calculation Methods**:

1. **Parametric**: Assumes normal distribution

   .. math::
      CI = \\bar{x} \\pm t_{\\alpha/2} \\cdot \\frac{s}{\\sqrt{n}}

2. **Bootstrap**: Non-parametric resampling (10,000 iterations)

   * Resample data with replacement
   * Calculate statistic for each resample
   * Use percentiles as CI bounds

Effect Sizes
~~~~~~~~~~~~

**Purpose**: Quantify the magnitude of an effect (practical significance).

**Cohen's d**: Standardized mean difference

.. math::
   d = \\frac{\\bar{x}_1 - \\bar{x}_2}{s_{pooled}}

**Interpretation**:

* |d| < 0.2: Small effect
* |d| ≈ 0.5: Medium effect
* |d| > 0.8: Large effect

**Percentage Change**: Relative difference

.. math::
   \\text{Change} = \\frac{x_{new} - x_{baseline}}{x_{baseline}} \\times 100\\%

**Hedges' g**: Bias-corrected Cohen's d for small samples

.. math::
   g = d \\times \\left(1 - \\frac{3}{4n - 9}\\right)

Time Series Analysis
--------------------

Seasonal Decomposition
~~~~~~~~~~~~~~~~~~~~~~

**Purpose**: Separate time series into trend, seasonal, and residual components.

**STL Decomposition** (Seasonal and Trend decomposition using Loess):

.. math::
   Y_t = T_t + S_t + R_t

Where:

* :math:`Y_t`: Observed value at time t
* :math:`T_t`: Trend component
* :math:`S_t`: Seasonal component
* :math:`R_t`: Residual component

**X-13-ARIMA-SEATS**: U.S. Census Bureau method for seasonal adjustment.

**Seasonal Strength**: Measure of seasonality importance

.. math::
   F_S = \\max\\left(0, 1 - \\frac{\\text{Var}(R_t)}{\\text{Var}(S_t + R_t)}\\right)

**Interpretation**:

* F_S > 0.6: Strong seasonality
* 0.3 < F_S ≤ 0.6: Moderate seasonality
* F_S ≤ 0.3: Weak seasonality

Changepoint Detection
~~~~~~~~~~~~~~~~~~~~~~

**Purpose**: Identify dates where time series behavior changes significantly.

**PELT Algorithm** (Pruned Exact Linear Time):

* Detects multiple changepoints efficiently
* Minimizes cost function with penalty for complexity
* Optimal for long time series

**Binary Segmentation**:

* Recursive splitting of time series
* Finds changepoints sequentially
* Faster but less accurate than PELT

**Bayesian Changepoint Detection**:

* Probabilistic approach
* Provides posterior probabilities for changepoints
* Handles uncertainty naturally

**Significance Testing**:

* **Chow Test**: Tests for structural break at known date
* **CUSUM**: Cumulative sum test for detecting shifts

**Consensus Approach**:

* Require agreement from multiple methods
* Changepoints within 7-day window are considered the same
* All consensus changepoints must pass significance test (p < 0.05)

Forecasting
~~~~~~~~~~~

**Purpose**: Predict future values with uncertainty quantification.

**ARIMA** (AutoRegressive Integrated Moving Average):

.. math::
   \\phi(B)(1-B)^d Y_t = \\theta(B)\\epsilon_t

Where:

* :math:`\\phi(B)`: AR polynomial
* :math:`\\theta(B)`: MA polynomial
* :math:`d`: Differencing order
* :math:`\\epsilon_t`: White noise

**Prophet** (Facebook's forecasting tool):

.. math::
   y(t) = g(t) + s(t) + h(t) + \\epsilon_t

Where:

* :math:`g(t)`: Trend (piecewise linear or logistic)
* :math:`s(t)`: Seasonality (Fourier series)
* :math:`h(t)`: Holiday effects
* :math:`\\epsilon_t`: Error term

**Exponential Smoothing**:

.. math::
   \\hat{y}_{t+1} = \\alpha y_t + (1-\\alpha)\\hat{y}_t

**Ensemble Forecasting**:

* Combine multiple methods
* Weighted average based on historical accuracy
* Reduces model-specific bias

**Prediction Intervals**:

* 50% PI: Range containing true value with 50% probability
* 80% PI: Range containing true value with 80% probability
* 95% PI: Range containing true value with 95% probability

**Forecast Accuracy Metrics**:

* **MAPE** (Mean Absolute Percentage Error):

  .. math::
     MAPE = \\frac{100\\%}{n}\\sum_{t=1}^{n}\\left|\\frac{y_t - \\hat{y}_t}{y_t}\\right|

* **RMSE** (Root Mean Squared Error):

  .. math::
     RMSE = \\sqrt{\\frac{1}{n}\\sum_{t=1}^{n}(y_t - \\hat{y}_t)^2}

* **MAE** (Mean Absolute Error):

  .. math::
     MAE = \\frac{1}{n}\\sum_{t=1}^{n}|y_t - \\hat{y}_t|

* **MASE** (Mean Absolute Scaled Error):

  .. math::
     MASE = \\frac{MAE}{\\frac{1}{n-1}\\sum_{t=2}^{n}|y_t - y_{t-1}|}

Causal Inference
----------------

Interrupted Time Series Analysis (ITSA)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

**Purpose**: Estimate causal effect of an intervention by comparing observed outcomes to counterfactual predictions.

**Model**:

.. math::
   Y_t = \\beta_0 + \\beta_1 T_t + \\beta_2 X_t + \\beta_3 X_t T_t + \\epsilon_t

Where:

* :math:`Y_t`: Outcome at time t
* :math:`T_t`: Time since start
* :math:`X_t`: Intervention indicator (0 before, 1 after)
* :math:`\\beta_2`: Level change (immediate effect)
* :math:`\\beta_3`: Slope change (trend change)

**Assumptions**:

1. **No confounding**: No other interventions at same time
2. **Stable pre-trend**: Pre-intervention trend is stable
3. **No anticipation**: No behavioral changes before intervention

**Average Treatment Effect (ATE)**:

.. math::
   ATE = \\frac{1}{n_{post}}\\sum_{t \\in post}(Y_t^{obs} - Y_t^{cf})

Where:

* :math:`Y_t^{obs}`: Observed outcome
* :math:`Y_t^{cf}`: Counterfactual prediction

**Significance Testing**:

* Permutation test: Randomly assign intervention dates
* P-value: Proportion of permutations with larger effect
* Minimum 1,000 permutations recommended

Difference-in-Differences (DiD)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

**Purpose**: Estimate causal effect by comparing treatment and control groups before and after intervention.

**Model**:

.. math::
   Y_{it} = \\beta_0 + \\beta_1 \\text{Treat}_i + \\beta_2 \\text{Post}_t + \\beta_3 (\\text{Treat}_i \\times \\text{Post}_t) + \\epsilon_{it}

Where:

* :math:`\\beta_3`: DiD estimator (causal effect)

**Assumptions**:

1. **Parallel trends**: Treatment and control would have followed parallel trends without intervention
2. **No spillovers**: Treatment doesn't affect control group
3. **Stable composition**: Groups don't change over time

**Parallel Trends Test**:

* Test if pre-intervention trends are parallel
* Regress outcome on time × treatment interaction in pre-period
* If p > 0.05, parallel trends assumption holds

**Placebo Test**:

* Apply DiD to pre-intervention period
* Should find no effect (p > 0.05)
* Validates parallel trends assumption

Event Study Methodology
~~~~~~~~~~~~~~~~~~~~~~~~

**Purpose**: Measure impact of external events relative to counterfactual baseline.

**Baseline Model**:

* Fit ARIMA or Prophet to pre-event data
* Generate forecast for event period
* Forecast represents "what would have happened"

**Cumulative Abnormal Return (CAR)**:

.. math::
   CAR = \\sum_{t=t_0}^{t_1}(Y_t - \\hat{Y}_t)

Where:

* :math:`Y_t`: Observed value
* :math:`\\hat{Y}_t`: Predicted value from baseline

**Significance Testing**:

* Z-score test:

  .. math::
     Z = \\frac{CAR}{\\sigma_{CAR}}

* If |Z| > 1.96, effect is significant at 95% level

**Half-Life Calculation**:

* Time for effect to decay to 50% of peak
* Fit exponential decay model:

  .. math::
     Y_t = Y_0 e^{-\\lambda t}

* Half-life: :math:`t_{1/2} = \\frac{\\ln(2)}{\\lambda}`

Synthetic Control Method
~~~~~~~~~~~~~~~~~~~~~~~~~

**Purpose**: Construct counterfactual using weighted combination of control units.

**Optimization Problem**:

.. math::
   \\min_W \\sum_{t=1}^{T_0}\\left(Y_{1t} - \\sum_{j=2}^{J}w_j Y_{jt}\\right)^2

Subject to:

* :math:`w_j \\geq 0` for all j
* :math:`\\sum_{j=2}^{J}w_j = 1`

Where:

* :math:`Y_{1t}`: Treated unit outcome
* :math:`Y_{jt}`: Donor unit j outcome
* :math:`w_j`: Weight for donor j
* :math:`T_0`: Pre-intervention period

**Causal Effect**:

.. math::
   \\hat{\\alpha}_{1t} = Y_{1t} - \\sum_{j=2}^{J}w_j^* Y_{jt}

**Inference via Placebo Tests**:

1. Apply method to each control unit (as if treated)
2. Calculate placebo effects
3. P-value: Proportion of placebo effects ≥ actual effect

**Quality Metrics**:

* **Pre-treatment fit**: R² > 0.7 required
* **Weight distribution**: Avoid single donor dominating
* **Donor pool size**: Minimum 5 donors recommended

Multi-Dimensional Analysis
---------------------------

Engagement Metrics
~~~~~~~~~~~~~~~~~~

**Purpose**: Distinguish between passive consumption and active engagement.

**Engagement Ratio**:

.. math::
   ER = \\frac{\\text{Active Editors}}{\\text{Pageviews}} \\times 1000

**Interpretation**:

* Higher ER: More engaged audience
* Lower ER: More passive consumption
* Compare across time periods and platforms

**Statistical Testing**:

* Test if ER differs significantly between periods
* Use t-test or Mann-Whitney U test
* Report effect size (Cohen's d)

Cross-Platform Analysis
~~~~~~~~~~~~~~~~~~~~~~~~

**Purpose**: Compare behavior across desktop, mobile web, and mobile app.

**Platform Concentration**:

* **Herfindahl-Hirschman Index (HHI)**:

  .. math::
     HHI = \\sum_{i=1}^{n}s_i^2 \\times 10000

  Where :math:`s_i` is platform i's market share

**Interpretation**:

* HHI < 1500: Low concentration
* 1500 ≤ HHI ≤ 2500: Moderate concentration
* HHI > 2500: High concentration (risky)

**Platform Risk Assessment**:

* Calculate coefficient of variation for each platform
* Test if mobile dependency exceeds 70% threshold
* Perform scenario analysis (10%, 20%, 30% declines)

Cross-Validation
----------------

Multi-Source Validation
~~~~~~~~~~~~~~~~~~~~~~~

**Purpose**: Confirm findings across multiple data sources.

**Consistency Score**:

.. math::
   CS = \\frac{\\text{Number of sources supporting finding}}{\\text{Total number of sources}}

**Interpretation**:

* CS = 1.0: All sources agree (high confidence)
* CS ≥ 0.67: Majority agree (medium confidence)
* CS < 0.67: Inconsistent (low confidence)

**Data Sources**:

1. Pageviews
2. Active editor counts
3. Edit volumes
4. External benchmarks (when available)

Robustness Checks
~~~~~~~~~~~~~~~~~

**Sensitivity Analysis**:

* Vary key parameters (significance level, window size, etc.)
* Report range of results across parameter space
* Flag findings that change substantially

**Outlier Sensitivity**:

* Run analysis with and without outliers
* Compare results
* Report if conclusions change

**Method Comparison**:

* Apply multiple methods to same question
* Check consistency across methods
* Report consensus findings

Interpreting Results
--------------------

Statistical Significance vs Practical Significance
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

**Statistical Significance** (p-value):

* Indicates if effect is real (not due to chance)
* Depends on sample size
* Small effects can be significant with large samples

**Practical Significance** (effect size):

* Indicates if effect is meaningful
* Independent of sample size
* Large samples can detect tiny, meaningless effects

**Best Practice**:

* Report both p-value and effect size
* Consider practical significance in context
* Don't rely solely on p-value

Confidence Intervals
~~~~~~~~~~~~~~~~~~~~

**Interpretation**:

* "We are 95% confident the true value lies in this range"
* NOT "There is a 95% probability the true value is in this range"
* Wider intervals indicate more uncertainty

**Using CIs for Decisions**:

* If CI excludes zero: Effect is significant
* If CI includes zero: Effect may not exist
* Width of CI: Precision of estimate

Correlation vs Causation
~~~~~~~~~~~~~~~~~~~~~~~~~

**Correlation**: Two variables move together

**Causation**: One variable causes changes in another

**Bradford Hill Criteria** for causation:

1. **Strength**: Large effect size
2. **Consistency**: Replicated across studies
3. **Specificity**: Specific cause-effect relationship
4. **Temporality**: Cause precedes effect
5. **Dose-response**: Larger cause → larger effect
6. **Plausibility**: Biologically/mechanistically plausible
7. **Coherence**: Consistent with existing knowledge
8. **Experiment**: Experimental evidence exists
9. **Analogy**: Similar cause-effect relationships known

**Causal Inference Methods**:

* Use ITSA, DiD, or synthetic controls
* Test assumptions
* Perform robustness checks
* Report limitations

Limitations and Caveats
-----------------------

Data Limitations
~~~~~~~~~~~~~~~~

* **API coverage**: Data availability varies by metric
* **Bot filtering**: May not catch all automated traffic
* **Platform definitions**: May change over time
* **Geographic coverage**: Global aggregates may mask regional patterns

Statistical Limitations
~~~~~~~~~~~~~~~~~~~~~~~

* **Multiple testing**: More tests → higher false positive rate
* **Assumption violations**: Tests assume specific conditions
* **Outliers**: Can distort results
* **Autocorrelation**: Time series violate independence assumption

Causal Inference Limitations
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

* **Confounding**: Unmeasured variables may affect results
* **Selection bias**: Non-random treatment assignment
* **Spillovers**: Treatment may affect control group
* **External validity**: Results may not generalize

Best Practices
~~~~~~~~~~~~~~

1. **Report all tests**: Don't cherry-pick significant results
2. **Adjust for multiple testing**: Use Bonferroni or FDR correction
3. **Test assumptions**: Verify before applying methods
4. **Use robust methods**: When assumptions violated
5. **Cross-validate**: Confirm across data sources
6. **Report limitations**: Be transparent about caveats
7. **Provide context**: Interpret in domain context

References
----------

**Time Series Analysis**:

* Box, G. E. P., Jenkins, G. M., & Reinsel, G. C. (2015). *Time Series Analysis: Forecasting and Control*. Wiley.
* Hyndman, R. J., & Athanasopoulos, G. (2021). *Forecasting: Principles and Practice*. OTexts.

**Causal Inference**:

* Angrist, J. D., & Pischke, J. S. (2009). *Mostly Harmless Econometrics*. Princeton University Press.
* Imbens, G. W., & Rubin, D. B. (2015). *Causal Inference for Statistics, Social, and Biomedical Sciences*. Cambridge University Press.
* Pearl, J., & Mackenzie, D. (2018). *The Book of Why*. Basic Books.

**Statistical Methods**:

* Wasserman, L. (2004). *All of Statistics*. Springer.
* Efron, B., & Hastie, T. (2016). *Computer Age Statistical Inference*. Cambridge University Press.

**Changepoint Detection**:

* Killick, R., Fearnhead, P., & Eckley, I. A. (2012). Optimal detection of changepoints with a linear computational cost. *Journal of the American Statistical Association*, 107(500), 1590-1598.

**Synthetic Control**:

* Abadie, A., Diamond, A., & Hainmueller, J. (2010). Synthetic control methods for comparative case studies. *Journal of the American Statistical Association*, 105(490), 493-505.
