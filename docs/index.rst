Wikipedia Product Health Analysis Documentation
===============================================

Welcome to the Wikipedia Product Health Analysis System documentation. This system provides rigorous, evidence-based analytics for evaluating Wikipedia's product health using time-series data from 2015-2025.

.. toctree::
   :maxdepth: 2
   :caption: Contents:

   getting_started
   api_reference
   user_guides
   methodology

Overview
--------

The Wikipedia Product Health Analysis system distinguishes itself from traditional descriptive analytics by implementing:

* **Statistical Validation**: Hypothesis testing, significance analysis, confidence intervals, and effect size quantification
* **Causal Inference**: Interrupted time series, difference-in-differences, event study methodology, and synthetic controls
* **Time Series Analysis**: Seasonal decomposition, changepoint detection, and multi-method forecasting
* **Evidence Framework**: Multi-source validation, robustness checks, and sensitivity analysis
* **Interactive Visualization**: Statistical evidence overlays and publication-quality plots

Quick Start
-----------

Installation::

    pip install -e .

Basic usage::

    from wikipedia_health.config import load_config
    from wikipedia_health.analysis_system import AnalysisSystem
    
    # Load configuration
    config = load_config()
    
    # Initialize analysis system
    system = AnalysisSystem(config=config)
    
    # Run full analysis
    results = system.run_full_analysis(
        start_date="2020-01-01",
        end_date="2023-12-31",
        platforms=["desktop", "mobile-web", "mobile-app"]
    )

Indices and tables
==================

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`
