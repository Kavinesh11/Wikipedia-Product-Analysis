User Guides
===========

This section provides step-by-step guides for common analysis workflows.

.. contents:: Table of Contents
   :local:
   :depth: 2

Getting Started
---------------

Installation and Setup
~~~~~~~~~~~~~~~~~~~~~~

1. Install the package::

    pip install -e .

2. Create a configuration file (``config.yaml``)::

    api:
      timeout: 30
      max_retries: 5
    
    statistical:
      significance_level: 0.05
      confidence_level: 0.95

3. Verify installation::

    wikipedia-health --help

Basic Workflow
~~~~~~~~~~~~~~

The typical analysis workflow:

1. **Data Acquisition**: Fetch data from Wikimedia APIs
2. **Data Validation**: Check data quality and completeness
3. **Analysis**: Run statistical and causal analyses
4. **Validation**: Cross-validate findings across data sources
5. **Reporting**: Generate visualizations and reports

Trend Analysis Guide
--------------------

Analyzing Long-Term Trends
~~~~~~~~~~~~~~~~~~~~~~~~~~~

**Objective**: Identify structural shifts in Wikipedia usage patterns.

**Steps**:

1. Fetch historical data::

    from datetime import date
    from wikipedia_health.analysis_system import AnalysisSystem
    
    system = AnalysisSystem()
    results = system.analyze_long_term_trends(
        start_date=date(2015, 1, 1),
        end_date=date(2025, 1, 1),
        platforms=['all']
    )

2. Review detected changepoints::

    for cp in results['changepoints']:
        print(f"{cp.date}: {cp.direction} ({cp.magnitude:+.2%})")
        print(f"  Confidence: {cp.confidence:.2%}")
        print(f"  Significant: {cp.is_significant()}")

3. Examine statistical evidence::

    for finding in results['findings']:
        print(f"{finding.description}")
        print(f"  Confidence: {finding.confidence_level}")
        for evidence in finding.evidence:
            print(f"  - {evidence.test_name}: p={evidence.p_value:.4f}")

**Interpreting Results**:

* **Changepoints**: Dates where usage patterns changed significantly
* **Confidence**: Probability that the change is real (not random)
* **Magnitude**: Size of the change (percentage)
* **P-value**: Statistical significance (< 0.05 is significant)

Attributing Changes to External Events
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

**Objective**: Test whether structural shifts align with external events.

**Steps**:

1. Define external events::

    external_events = [
        (date(2022, 11, 30), "ChatGPT Launch"),
        (date(2024, 5, 14), "Google AI Overviews")
    ]

2. Run analysis with temporal alignment testing::

    results = system.analyze_long_term_trends(
        start_date=date(2020, 1, 1),
        end_date=date(2025, 1, 1),
        platforms=['all'],
        external_events=external_events
    )

3. Review alignment results::

    for event_date, event_name in external_events:
        alignment = results['temporal_alignment'][event_name]
        print(f"{event_name}:")
        print(f"  Nearest changepoint: {alignment['nearest_changepoint']}")
        print(f"  Days difference: {alignment['days_diff']}")
        print(f"  Alignment p-value: {alignment['p_value']:.4f}")

**Interpreting Results**:

* **Days difference < 30**: Strong temporal alignment
* **Alignment p-value < 0.05**: Statistically significant alignment
* **Multiple aligned changepoints**: Stronger evidence for causation

Platform Analysis Guide
-----------------------

Analyzing Platform Mix
~~~~~~~~~~~~~~~~~~~~~~

**Objective**: Understand traffic distribution across platforms.

**Steps**:

1. Run platform analysis::

    results = system.analyze_platform_dependency(
        start_date=date(2020, 1, 1),
        end_date=date(2023, 12, 31),
        platforms=['desktop', 'mobile-web', 'mobile-app']
    )

2. Review platform proportions::

    platform_mix = results['platform_mix']
    for platform, proportion in platform_mix.items():
        ci_lower, ci_upper = results['platform_mix_ci'][platform]
        print(f"{platform}: {proportion:.1%} (95% CI: [{ci_lower:.1%}, {ci_upper:.1%}])")

3. Check platform trends::

    for platform, cagr in results['platform_cagr'].items():
        print(f"{platform} CAGR: {cagr:+.2%}")

**Interpreting Results**:

* **Platform mix**: Current traffic distribution
* **CAGR**: Compound annual growth rate (positive = growing)
* **Confidence intervals**: Uncertainty in estimates

Assessing Platform Dependency Risk
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

**Objective**: Quantify risk from over-dependence on a single platform.

**Steps**:

1. Calculate concentration metrics::

    hhi = results['herfindahl_index']
    print(f"HHI: {hhi:.0f}")
    
    if hhi > 2500:
        print("High concentration risk")
    elif hhi > 1500:
        print("Moderate concentration risk")
    else:
        print("Low concentration risk")

2. Test mobile dependency threshold::

    mobile_proportion = results['mobile_proportion']
    threshold_test = results['mobile_threshold_test']
    
    print(f"Mobile traffic: {mobile_proportion:.1%}")
    print(f"Exceeds 70% threshold: {threshold_test['exceeds_threshold']}")
    print(f"P-value: {threshold_test['p_value']:.4f}")

3. Review scenario analysis::

    for scenario, impact in results['scenario_analysis'].items():
        print(f"{scenario}:")
        print(f"  Total impact: {impact['total_impact']:+.2%}")
        print(f"  Recovery time: {impact['recovery_days']} days")

**Interpreting Results**:

* **HHI > 2500**: High concentration (risky)
* **Mobile > 70%**: High mobile dependency
* **Scenario analysis**: Impact of platform-specific declines

Seasonality Analysis Guide
---------------------------

Detecting Seasonal Patterns
~~~~~~~~~~~~~~~~~~~~~~~~~~~~

**Objective**: Identify and validate seasonal patterns in usage.

**Steps**:

1. Run seasonality analysis::

    results = system.analyze_seasonality(
        start_date=date(2020, 1, 1),
        end_date=date(2023, 12, 31),
        platforms=['all']
    )

2. Review seasonal strength::

    seasonal_strength = results['seasonal_strength']
    print(f"Seasonal strength: {seasonal_strength:.2f}")
    
    if seasonal_strength > 0.6:
        print("Strong seasonality")
    elif seasonal_strength > 0.3:
        print("Moderate seasonality")
    else:
        print("Weak seasonality")

3. Examine seasonal components::

    decomposition = results['decomposition']
    print(f"Trend component: {decomposition.trend.mean():.0f}")
    print(f"Seasonal amplitude: {decomposition.seasonal.std():.0f}")

**Interpreting Results**:

* **Seasonal strength > 0.6**: Strong seasonal pattern
* **Seasonal amplitude**: Size of seasonal fluctuations
* **Spectral analysis**: Dominant frequencies (e.g., weekly, yearly)

Analyzing Day-of-Week Effects
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

**Objective**: Quantify weekday vs weekend differences.

**Steps**:

1. Review day-of-week analysis::

    dow_effects = results['day_of_week_effects']
    for day, effect in dow_effects.items():
        print(f"{day}: {effect['mean']:.0f} pageviews")
        print(f"  Effect size: {effect['effect_size']:.3f}")
        print(f"  P-value: {effect['p_value']:.4f}")

2. Compare weekday vs weekend::

    weekday_mean = results['weekday_mean']
    weekend_mean = results['weekend_mean']
    difference = (weekend_mean - weekday_mean) / weekday_mean
    
    print(f"Weekday average: {weekday_mean:.0f}")
    print(f"Weekend average: {weekend_mean:.0f}")
    print(f"Difference: {difference:+.2%}")

3. Classify usage type::

    usage_type = results['usage_classification']
    print(f"Usage type: {usage_type}")  # 'utility' or 'leisure'

**Interpreting Results**:

* **Weekday > Weekend**: Utility usage (work/school)
* **Weekend > Weekday**: Leisure usage
* **Effect size > 0.5**: Large day-of-week effect

Campaign Analysis Guide
------------------------

Measuring Campaign Effectiveness
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

**Objective**: Estimate the causal impact of a campaign.

**Steps**:

1. Define campaign dates::

    campaign_dates = [
        (date(2021, 6, 15), "Summer Fundraiser"),
        (date(2021, 12, 1), "Winter Fundraiser")
    ]

2. Run campaign analysis::

    results = system.analyze_campaigns(
        start_date=date(2021, 1, 1),
        end_date=date(2022, 12, 31),
        campaign_dates=campaign_dates,
        platforms=['all']
    )

3. Review campaign effects::

    for campaign, effect in results['campaign_effects'].items():
        print(f"{campaign}:")
        print(f"  Effect size: {effect.effect_size:+.2%}")
        print(f"  95% CI: [{effect.confidence_interval[0]:+.2%}, {effect.confidence_interval[1]:+.2%}]")
        print(f"  P-value: {effect.p_value:.4f}")
        print(f"  Significant: {effect.p_value < 0.05}")

**Interpreting Results**:

* **Effect size**: Percentage change in pageviews
* **Positive effect**: Campaign increased traffic
* **P-value < 0.05**: Effect is statistically significant
* **Confidence interval**: Range of plausible effect sizes

Analyzing Campaign Duration
~~~~~~~~~~~~~~~~~~~~~~~~~~~~

**Objective**: Understand how long campaign effects last.

**Steps**:

1. Review duration analysis::

    for campaign, duration in results['campaign_duration'].items():
        print(f"{campaign}:")
        print(f"  Immediate effect (0-7 days): {duration['immediate']:+.2%}")
        print(f"  Short-term effect (8-30 days): {duration['short_term']:+.2%}")
        print(f"  Long-term effect (30+ days): {duration['long_term']:+.2%}")

2. Calculate half-life::

    for campaign, half_life in results['effect_half_life'].items():
        print(f"{campaign} half-life: {half_life} days")

**Interpreting Results**:

* **Immediate effect**: Impact during campaign
* **Short-term effect**: Impact in weeks after campaign
* **Long-term effect**: Sustained impact
* **Half-life**: Days until effect reduces by 50%

Forecasting Guide
-----------------

Generating Traffic Forecasts
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

**Objective**: Predict future traffic with uncertainty quantification.

**Steps**:

1. Generate forecast::

    results = system.generate_forecasts(
        start_date=date(2020, 1, 1),
        end_date=date(2023, 12, 31),
        horizon=90,  # 90-day forecast
        platforms=['all'],
        methods=['arima', 'prophet', 'exponential_smoothing']
    )

2. Review ensemble forecast::

    forecast = results['ensemble_forecast']
    print(f"Mean forecast: {forecast.point_forecast.mean():.0f} daily pageviews")
    print(f"50% PI: [{forecast.lower_bound_50.mean():.0f}, {forecast.upper_bound_50.mean():.0f}]")
    print(f"95% PI: [{forecast.lower_bound_95.mean():.0f}, {forecast.upper_bound_95.mean():.0f}]")

3. Compare individual models::

    for method, accuracy in results['model_accuracy'].items():
        print(f"{method}:")
        print(f"  MAPE: {accuracy['mape']:.2%}")
        print(f"  RMSE: {accuracy['rmse']:.0f}")

**Interpreting Results**:

* **Point forecast**: Most likely outcome
* **50% PI**: Range containing true value with 50% probability
* **95% PI**: Range containing true value with 95% probability
* **MAPE**: Mean absolute percentage error (lower is better)

Scenario Analysis
~~~~~~~~~~~~~~~~~

**Objective**: Explore optimistic, baseline, and pessimistic scenarios.

**Steps**:

1. Review scenarios::

    scenarios = results['scenarios']
    for scenario_name, forecast in scenarios.items():
        print(f"{scenario_name}:")
        print(f"  Mean: {forecast.point_forecast.mean():.0f}")
        print(f"  Probability: {forecast.probability:.1%}")

2. Calculate probability-weighted forecast::

    weighted_forecast = results['probability_weighted_forecast']
    print(f"Expected value: {weighted_forecast:.0f} daily pageviews")

**Interpreting Results**:

* **Optimistic**: Best-case scenario
* **Baseline**: Most likely scenario
* **Pessimistic**: Worst-case scenario
* **Probability-weighted**: Expected value across scenarios

Best Practices
--------------

Data Quality
~~~~~~~~~~~~

1. **Always validate data** before analysis
2. **Check for missing values** and gaps
3. **Inspect outliers** before removing them
4. **Use multiple data sources** for validation

Statistical Testing
~~~~~~~~~~~~~~~~~~~

1. **Report p-values** for all claims
2. **Include confidence intervals** for estimates
3. **Calculate effect sizes** to assess practical significance
4. **Use appropriate tests** (parametric vs non-parametric)

Causal Inference
~~~~~~~~~~~~~~~~

1. **Test assumptions** (parallel trends, etc.)
2. **Use multiple methods** when possible
3. **Perform robustness checks** (placebo tests, sensitivity analysis)
4. **Clearly distinguish** correlation from causation

Reporting
~~~~~~~~~

1. **Include statistical evidence** in all visualizations
2. **Provide plain-language interpretations**
3. **Document methodology** and assumptions
4. **Report limitations** and caveats

Troubleshooting
---------------

Common Issues
~~~~~~~~~~~~~

**Issue**: API timeouts

* **Solution**: Increase timeout in configuration
* **Alternative**: Use smaller date ranges

**Issue**: Insufficient data for analysis

* **Solution**: Extend date range
* **Alternative**: Reduce minimum data requirements

**Issue**: Non-significant results

* **Solution**: Check if effect size is meaningful despite p-value
* **Alternative**: Increase sample size or use more sensitive tests

**Issue**: Assumption violations

* **Solution**: Use non-parametric alternatives
* **Alternative**: Transform data or use robust methods

Getting Help
~~~~~~~~~~~~

1. Check the :doc:`api_reference` for detailed documentation
2. Review example scripts in ``examples/scripts/``
3. Consult the :doc:`methodology` for statistical details
4. Open an issue on GitHub for bugs or feature requests
