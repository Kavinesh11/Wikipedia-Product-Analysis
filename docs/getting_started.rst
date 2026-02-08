Getting Started
===============

This guide will help you get started with the Wikipedia Product Health Analysis System.

Installation
------------

Requirements
~~~~~~~~~~~~

* Python >= 3.9
* pip or conda package manager

Install from source::

    git clone <repository-url>
    cd wikipedia-health-analysis
    pip install -e .

For development with testing tools::

    pip install -e ".[dev]"

Configuration
-------------

The system uses a YAML configuration file. Create a ``config.yaml`` file:

.. code-block:: yaml

    api:
      timeout: 30
      max_retries: 5
      backoff_factor: 2.0
    
    statistical:
      significance_level: 0.05
      confidence_level: 0.95
      bootstrap_samples: 10000
    
    time_series:
      seasonal_period: 7
      forecast_methods:
        - arima
        - prophet
        - exponential_smoothing

See :doc:`configuration` for complete configuration options.

Basic Usage
-----------

Command-Line Interface
~~~~~~~~~~~~~~~~~~~~~~

Run a complete analysis::

    wikipedia-health full --start-date 2020-01-01 --end-date 2023-12-31

Run specific analysis types::

    wikipedia-health trends --start-date 2020-01-01 --end-date 2023-12-31
    wikipedia-health platforms --start-date 2020-01-01 --end-date 2023-12-31
    wikipedia-health seasonality --start-date 2020-01-01 --end-date 2023-12-31

Python API
~~~~~~~~~~

.. code-block:: python

    from datetime import date
    from wikipedia_health.config import load_config
    from wikipedia_health.analysis_system import AnalysisSystem
    
    # Load configuration
    config = load_config('config.yaml')
    
    # Initialize system
    system = AnalysisSystem(config=config)
    
    # Run trend analysis
    results = system.analyze_long_term_trends(
        start_date=date(2020, 1, 1),
        end_date=date(2023, 12, 31),
        platforms=['desktop', 'mobile-web', 'mobile-app']
    )
    
    # Access findings
    for finding in results['findings']:
        print(f"{finding.description}: {finding.confidence_level}")

Data Acquisition
----------------

The system fetches data from Wikimedia APIs:

.. code-block:: python

    from wikipedia_health.data_acquisition import WikimediaAPIClient
    from wikipedia_health.config import load_config
    
    config = load_config()
    client = WikimediaAPIClient(config.api)
    
    # Fetch pageview data
    pageviews = client.fetch_pageviews(
        start_date=date(2020, 1, 1),
        end_date=date(2023, 12, 31),
        platforms=['desktop', 'mobile-web', 'mobile-app'],
        agent_type='user'  # Exclude bots
    )
    
    # Fetch editor counts
    editors = client.fetch_editor_counts(
        start_date=date(2020, 1, 1),
        end_date=date(2023, 12, 31)
    )

Next Steps
----------

* Read the :doc:`api_reference` for detailed API documentation
* Explore :doc:`user_guides` for common workflows
* Learn about :doc:`methodology` for statistical methods
