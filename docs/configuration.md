# Configuration Guide

This document describes all configuration options for the Wikipedia Product Health Analysis System.

## Configuration File Format

The system supports YAML and JSON configuration files. By default, it looks for `config.yaml` in the current directory.

### Example Configuration (YAML)

```yaml
api:
  pageviews_endpoint: "https://wikimedia.org/api/rest_v1/metrics/pageviews"
  editors_endpoint: "https://wikimedia.org/api/rest_v1/metrics/editors"
  edits_endpoint: "https://wikimedia.org/api/rest_v1/metrics/edits"
  timeout: 30
  max_retries: 5
  backoff_factor: 2.0
  user_agent: "WikipediaHealthAnalysis/0.1.0"

statistical:
  significance_level: 0.05
  confidence_level: 0.95
  bootstrap_samples: 10000
  permutation_iterations: 10000
  numerical_precision: 1.0e-10
  outlier_threshold: 3.0
  min_data_points_trend: 90
  min_data_points_causal: 30

time_series:
  seasonal_period: 7
  changepoint_min_size: 30
  forecast_methods:
    - arima
    - prophet
    - exponential_smoothing
  prediction_intervals:
    - 0.50
    - 0.80
    - 0.95
  holdout_percentage: 0.10

causal:
  pre_period_length: 90
  post_period_length: 90
  baseline_window: 90
  event_post_window: 30
  event_max_window: 180
  placebo_iterations: 100

validation:
  max_missing_percentage: 0.10
  max_gap_days: 3
  staleness_threshold_hours: 24
  data_sources:
    - pageviews
    - editors
    - edits
  platforms:
    - desktop
    - mobile-web
    - mobile-app
```

### Example Configuration (JSON)

```json
{
  "api": {
    "pageviews_endpoint": "https://wikimedia.org/api/rest_v1/metrics/pageviews",
    "editors_endpoint": "https://wikimedia.org/api/rest_v1/metrics/editors",
    "edits_endpoint": "https://wikimedia.org/api/rest_v1/metrics/edits",
    "timeout": 30,
    "max_retries": 5,
    "backoff_factor": 2.0,
    "user_agent": "WikipediaHealthAnalysis/0.1.0"
  },
  "statistical": {
    "significance_level": 0.05,
    "confidence_level": 0.95,
    "bootstrap_samples": 10000,
    "permutation_iterations": 10000,
    "numerical_precision": 1e-10,
    "outlier_threshold": 3.0,
    "min_data_points_trend": 90,
    "min_data_points_causal": 30
  }
}
```

## Configuration Sections

### API Configuration (`api`)

Controls how the system interacts with Wikimedia APIs.

| Option | Type | Default | Description |
|--------|------|---------|-------------|
| `pageviews_endpoint` | string | `https://wikimedia.org/api/rest_v1/metrics/pageviews` | Wikimedia Pageviews API endpoint |
| `editors_endpoint` | string | `https://wikimedia.org/api/rest_v1/metrics/editors` | Wikimedia Editors API endpoint |
| `edits_endpoint` | string | `https://wikimedia.org/api/rest_v1/metrics/edits` | Wikimedia Edits API endpoint |
| `timeout` | integer | `30` | Request timeout in seconds |
| `max_retries` | integer | `5` | Maximum number of retry attempts for failed requests |
| `backoff_factor` | float | `2.0` | Exponential backoff factor for retries |
| `user_agent` | string | `WikipediaHealthAnalysis/0.1.0` | User agent string for API requests |

**Validation Rules:**
- All endpoints must be valid URLs starting with `http://` or `https://`
- `timeout` must be positive (recommended: 10-300 seconds)
- `max_retries` must be non-negative (recommended: 3-10)
- `backoff_factor` must be positive (recommended: >= 1.0)
- `user_agent` cannot be empty

### Statistical Configuration (`statistical`)

Controls statistical analysis parameters.

| Option | Type | Default | Description |
|--------|------|---------|-------------|
| `significance_level` | float | `0.05` | Alpha level for hypothesis testing (α) |
| `confidence_level` | float | `0.95` | Confidence level for intervals (1 - α) |
| `bootstrap_samples` | integer | `10000` | Number of bootstrap samples for CI estimation |
| `permutation_iterations` | integer | `10000` | Number of permutations for permutation tests |
| `numerical_precision` | float | `1e-10` | Numerical precision tolerance |
| `outlier_threshold` | float | `3.0` | Z-score threshold for outlier detection |
| `min_data_points_trend` | integer | `90` | Minimum data points required for trend analysis |
| `min_data_points_causal` | integer | `30` | Minimum data points required for causal inference |

**Validation Rules:**
- `significance_level` must be between 0 and 1 (recommended: 0.01-0.10)
- `confidence_level` must be between 0 and 1 (recommended: 0.90-0.99)
- `bootstrap_samples` must be positive (recommended: >= 1000)
- `permutation_iterations` must be positive (recommended: >= 1000)
- `numerical_precision` must be positive
- `outlier_threshold` must be positive (recommended: 2.0-4.0)
- `min_data_points_trend` must be positive (recommended: >= 30)
- `min_data_points_causal` must be positive (recommended: >= 30)

### Time Series Configuration (`time_series`)

Controls time series analysis and forecasting.

| Option | Type | Default | Description |
|--------|------|---------|-------------|
| `seasonal_period` | integer | `7` | Seasonal period in days (7 = weekly) |
| `changepoint_min_size` | integer | `30` | Minimum segment size for changepoint detection |
| `forecast_methods` | list[string] | `['arima', 'prophet', 'exponential_smoothing']` | Forecasting methods to use |
| `prediction_intervals` | list[float] | `[0.50, 0.80, 0.95]` | Prediction interval levels |
| `holdout_percentage` | float | `0.10` | Percentage of data to hold out for validation |

**Validation Rules:**
- `seasonal_period` must be positive (recommended: >= 2)
- `changepoint_min_size` must be positive (recommended: >= 10)
- `forecast_methods` must contain at least one valid method: `arima`, `prophet`, or `exponential_smoothing`
- `prediction_intervals` must be between 0 and 1
- `holdout_percentage` must be between 0 and 1 (recommended: 0.05-0.30)

### Causal Inference Configuration (`causal`)

Controls causal inference analysis parameters.

| Option | Type | Default | Description |
|--------|------|---------|-------------|
| `pre_period_length` | integer | `90` | Pre-intervention period length in days |
| `post_period_length` | integer | `90` | Post-intervention period length in days |
| `baseline_window` | integer | `90` | Baseline window for event studies in days |
| `event_post_window` | integer | `30` | Post-event window in days |
| `event_max_window` | integer | `180` | Maximum event analysis window in days |
| `placebo_iterations` | integer | `100` | Number of placebo tests for inference |

**Validation Rules:**
- All period lengths must be positive
- `pre_period_length` recommended: >= 30 days
- `baseline_window` recommended: >= 30 days
- `event_max_window` must be >= `event_post_window`
- `placebo_iterations` must be positive (recommended: >= 50)

### Validation Configuration (`validation`)

Controls data validation and quality checks.

| Option | Type | Default | Description |
|--------|------|---------|-------------|
| `max_missing_percentage` | float | `0.10` | Maximum allowed percentage of missing data |
| `max_gap_days` | integer | `3` | Maximum allowed gap in days for interpolation |
| `staleness_threshold_hours` | integer | `24` | Data staleness threshold in hours |
| `data_sources` | list[string] | `['pageviews', 'editors', 'edits']` | Data sources to validate |
| `platforms` | list[string] | `['desktop', 'mobile-web', 'mobile-app']` | Platforms to analyze |

**Validation Rules:**
- `max_missing_percentage` must be between 0 and 1 (recommended: <= 0.20)
- `max_gap_days` must be non-negative (recommended: <= 7)
- `staleness_threshold_hours` must be positive
- `data_sources` must contain at least one of: `pageviews`, `editors`, `edits`
- `platforms` must contain at least one of: `desktop`, `mobile-web`, `mobile-app`, `all`

## Using Configuration Files

### Command-Line Usage

Specify a configuration file using the `--config` option:

```bash
wikipedia-health full --config my_config.yaml --start-date 2020-01-01 --end-date 2023-12-31
```

### Python API Usage

```python
from wikipedia_health.config import Config, load_config, validate_config

# Load from file
config = load_config('my_config.yaml')

# Validate configuration
validation_result = validate_config(config)
if not validation_result.is_valid:
    print(validation_result.get_summary())
    exit(1)

# Use configuration
from wikipedia_health.analysis_system import AnalysisSystem
system = AnalysisSystem(config=config)
```

### Creating Configuration Programmatically

```python
from wikipedia_health.config import Config, APIConfig, StatisticalConfig

# Create custom configuration
config = Config(
    api=APIConfig(
        timeout=60,
        max_retries=3
    ),
    statistical=StatisticalConfig(
        significance_level=0.01,
        bootstrap_samples=5000
    )
)

# Save to file
config.to_yaml('custom_config.yaml')
```

## Configuration Validation

The system automatically validates configuration when loaded. Validation checks:

1. **Type checking**: Ensures all values are of the correct type
2. **Range checking**: Ensures numeric values are within valid ranges
3. **Dependency checking**: Ensures related values are consistent
4. **Format checking**: Ensures strings (URLs, etc.) are properly formatted

Validation produces:
- **Errors**: Configuration issues that prevent execution
- **Warnings**: Configuration issues that may cause problems but don't prevent execution

### Example Validation Output

```
Configuration validation failed

Errors (2):
  - api.timeout: Timeout must be positive
  - statistical.significance_level: Significance level must be between 0 and 1

Warnings (1):
  - time_series.holdout_percentage: Holdout percentage < 0.05 may be too small
```

## Environment-Specific Configurations

You can maintain different configurations for different environments:

```bash
# Development
wikipedia-health full --config config.dev.yaml ...

# Production
wikipedia-health full --config config.prod.yaml ...

# Testing
wikipedia-health full --config config.test.yaml ...
```

## Configuration Best Practices

1. **Start with defaults**: The default configuration is suitable for most use cases
2. **Validate early**: Always validate configuration before running analyses
3. **Document changes**: Comment your configuration files to explain non-standard values
4. **Version control**: Keep configuration files in version control
5. **Environment separation**: Use different configs for dev/test/prod
6. **Security**: Never commit sensitive credentials to configuration files
7. **Backup**: Keep backups of working configurations

## Troubleshooting

### Common Configuration Issues

**Issue**: API requests timing out
- **Solution**: Increase `api.timeout` value
- **Recommended**: 60-120 seconds for large date ranges

**Issue**: Statistical tests taking too long
- **Solution**: Reduce `statistical.bootstrap_samples` or `statistical.permutation_iterations`
- **Recommended**: 5000-10000 for faster execution

**Issue**: Changepoint detection too sensitive
- **Solution**: Increase `time_series.changepoint_min_size`
- **Recommended**: 30-60 days for stable detection

**Issue**: Insufficient data for causal analysis
- **Solution**: Reduce `causal.pre_period_length` or `causal.post_period_length`
- **Minimum**: 30 days for each period

## Configuration Schema

For programmatic validation, the configuration schema is available:

```python
from wikipedia_health.config import Config

# Get default configuration
default_config = Config()

# Convert to dictionary
config_dict = default_config.to_dict()

# Inspect schema
print(config_dict.keys())  # ['api', 'statistical', 'time_series', 'causal', 'validation']
```

## Support

For configuration questions or issues:
1. Check this documentation
2. Review the default `config.yaml` file
3. Run configuration validation to identify issues
4. Consult the API documentation for advanced options
