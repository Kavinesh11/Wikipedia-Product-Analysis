# Wikipedia Product Health Analysis System

A rigorous, evidence-based analytics platform for evaluating Wikipedia's product health using time-series data from 2015-2025. The system implements formal statistical validation, causal inference methodologies, and multi-source cross-validation for all findings.

## Features

- **Statistical Validation**: Hypothesis testing, significance analysis, confidence intervals, and effect size quantification
- **Causal Inference**: Interrupted time series, difference-in-differences, event study methodology, and synthetic controls
- **Time Series Analysis**: Seasonal decomposition, changepoint detection, and multi-method forecasting
- **Evidence Framework**: Multi-source validation, robustness checks, and sensitivity analysis
- **Interactive Visualization**: Statistical evidence overlays and publication-quality plots

## Installation

```bash
pip install -e .
```

For development:

```bash
pip install -e ".[dev]"
```

## Project Structure

```
wikipedia_health/
├── data_acquisition/     # Wikimedia API client and data validation
├── time_series/          # Temporal analysis and forecasting
├── statistical_validation/  # Hypothesis testing and significance
├── causal_inference/     # Causal analysis methodologies
├── evidence_framework/   # Cross-validation and robustness
├── visualization/        # Interactive dashboards and plots
├── models/              # Core data structures
├── utils/               # Common utilities
└── config/              # Configuration management

tests/                   # Test suite with property-based tests
```

## Configuration

Configuration is managed through `config.yaml`. Key parameters:

- **API Settings**: Endpoints, timeouts, retry logic
- **Statistical Parameters**: Significance levels, confidence intervals, bootstrap samples
- **Time Series Settings**: Seasonal periods, forecast methods, prediction intervals
- **Causal Analysis**: Pre/post period lengths, baseline windows
- **Validation**: Data quality thresholds, platforms, data sources

## Testing

Run tests:

```bash
pytest
```

Run with coverage:

```bash
pytest --cov=wikipedia_health --cov-report=html
```

Run property-based tests only:

```bash
pytest -m property
```

## Usage

```python
from wikipedia_health.config import load_config
from wikipedia_health.data_acquisition import WikimediaAPIClient

# Load configuration
config = load_config()

# Initialize API client
client = WikimediaAPIClient(config.api)

# Fetch pageview data
data = client.fetch_pageviews(
    start_date="2015-01-01",
    end_date="2025-01-01",
    platforms=["desktop", "mobile-web", "mobile-app"]
)
```

## Requirements

- Python >= 3.9
- pandas >= 2.0.0
- numpy >= 1.24.0
- scipy >= 1.10.0
- statsmodels >= 0.14.0
- scikit-learn >= 1.3.0
- prophet >= 1.1.0
- pmdarima >= 2.0.0
- matplotlib >= 3.7.0
- seaborn >= 0.12.0
- plotly >= 5.14.0

## License

MIT
