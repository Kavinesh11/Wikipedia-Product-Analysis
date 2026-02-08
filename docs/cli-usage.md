# CLI Usage Guide

This guide provides examples and instructions for using the Wikipedia Product Health Analysis CLI.

## Installation

After installing the package, the CLI is available as `wikipedia-health`:

```bash
pip install -e .
```

Or run directly with Python:

```bash
python -m wikipedia_health [command] [options]
```

## Quick Start

Run a full analysis:

```bash
wikipedia-health full --start-date 2020-01-01 --end-date 2023-12-31
```

## Available Commands

### `full` - Complete Analysis Pipeline

Runs all analysis types in sequence.

```bash
wikipedia-health full \
  --start-date 2020-01-01 \
  --end-date 2023-12-31 \
  --platforms desktop mobile-web mobile-app \
  --analysis-types trends platforms seasonality
```

**Options:**
- `--analysis-types`: Specific analyses to run (default: all)
  - Choices: `trends`, `platforms`, `seasonality`, `campaigns`, `events`, `forecasts`

### `trends` - Long-Term Trend Analysis

Analyzes structural shifts and long-term patterns.

```bash
wikipedia-health trends \
  --start-date 2020-01-01 \
  --end-date 2023-12-31 \
  --external-events "2022-11-30:ChatGPT Launch,2024-05-14:Google AI Overviews"
```

**Options:**
- `--external-events`: External events for temporal alignment testing
  - Format: `YYYY-MM-DD:event_name,YYYY-MM-DD:event_name,...`

### `platforms` - Platform Dependency Analysis

Analyzes platform mix and dependency risks.

```bash
wikipedia-health platforms \
  --start-date 2020-01-01 \
  --end-date 2023-12-31 \
  --platforms desktop mobile-web mobile-app \
  --mobile-threshold 0.70
```

**Options:**
- `--mobile-threshold`: Mobile dependency threshold (default: 0.70)

### `seasonality` - Seasonal Pattern Analysis

Analyzes seasonal patterns and temporal effects.

```bash
wikipedia-health seasonality \
  --start-date 2020-01-01 \
  --end-date 2023-12-31 \
  --seasonal-period 7
```

**Options:**
- `--seasonal-period`: Seasonal period in days (default: 7 for weekly)

### `campaigns` - Campaign Effectiveness Analysis

Evaluates campaign impacts using causal inference.

```bash
wikipedia-health campaigns \
  --start-date 2020-01-01 \
  --end-date 2023-12-31 \
  --campaign-dates "2021-06-15:Summer Campaign,2021-12-01:Winter Campaign"
```

**Options:**
- `--campaign-dates`: Campaign dates (required)
  - Format: `YYYY-MM-DD:campaign_name,YYYY-MM-DD:campaign_name,...`

### `events` - External Event Impact Analysis

Measures Wikipedia's response to external shocks.

```bash
wikipedia-health events \
  --start-date 2020-01-01 \
  --end-date 2023-12-31 \
  --event-dates "2021-01-06:Capitol Riot:political,2021-03-23:Suez Canal:natural_disaster"
```

**Options:**
- `--event-dates`: Event dates with categories (required)
  - Format: `YYYY-MM-DD:event_name:category,YYYY-MM-DD:event_name:category,...`
  - Categories: `political`, `natural_disaster`, `celebrity`, `scientific`

### `forecasts` - Traffic Forecasting

Generates traffic forecasts with uncertainty quantification.

```bash
wikipedia-health forecasts \
  --start-date 2020-01-01 \
  --end-date 2023-12-31 \
  --horizon 90 \
  --methods arima prophet
```

**Options:**
- `--horizon`: Forecast horizon in days (default: 90)
- `--methods`: Forecasting methods to use
  - Choices: `arima`, `prophet`, `exponential_smoothing`

## Global Options

These options apply to all commands:

### Configuration

```bash
--config PATH
```

Specify a custom configuration file (YAML or JSON).

Example:
```bash
wikipedia-health full --config my_config.yaml --start-date 2020-01-01 --end-date 2023-12-31
```

### Verbosity

```bash
--verbose, -v
```

Enable verbose logging for debugging.

Example:
```bash
wikipedia-health --verbose trends --start-date 2020-01-01 --end-date 2023-12-31
```

### Output Directory

```bash
--output-dir PATH
```

Specify output directory for results (default: `output/TIMESTAMP`).

Example:
```bash
wikipedia-health --output-dir results/analysis1 full --start-date 2020-01-01 --end-date 2023-12-31
```

### Output Format

```bash
--output-format {json,html,pdf}
```

Specify output format for reports (default: html).

Example:
```bash
wikipedia-health --output-format json full --start-date 2020-01-01 --end-date 2023-12-31
```

## Common Options

These options are available for most analysis commands:

### Date Range

```bash
--start-date YYYY-MM-DD  # Required
--end-date YYYY-MM-DD    # Required
```

Specify the analysis date range.

### Platforms

```bash
--platforms {desktop,mobile-web,mobile-app,all}
```

Specify platforms to analyze (default: all platforms).

Example:
```bash
wikipedia-health trends \
  --start-date 2020-01-01 \
  --end-date 2023-12-31 \
  --platforms desktop mobile-web
```

### Significance Level

```bash
--significance-level FLOAT
```

Override statistical significance level (default: 0.05).

Example:
```bash
wikipedia-health trends \
  --start-date 2020-01-01 \
  --end-date 2023-12-31 \
  --significance-level 0.01
```

## Complete Examples

### Example 1: Full Analysis with Custom Config

```bash
wikipedia-health \
  --config production.yaml \
  --output-dir results/2024-analysis \
  --verbose \
  full \
  --start-date 2020-01-01 \
  --end-date 2023-12-31 \
  --platforms desktop mobile-web mobile-app
```

### Example 2: Trend Analysis with AI Impact

```bash
wikipedia-health trends \
  --start-date 2020-01-01 \
  --end-date 2024-12-31 \
  --external-events "2022-11-30:ChatGPT Launch,2024-05-14:Google AI Overviews" \
  --significance-level 0.01
```

### Example 3: Campaign Analysis

```bash
wikipedia-health campaigns \
  --start-date 2021-01-01 \
  --end-date 2023-12-31 \
  --campaign-dates "2021-06-15:Summer Fundraiser,2021-12-01:Winter Fundraiser,2022-06-15:Summer Fundraiser,2022-12-01:Winter Fundraiser"
```

### Example 4: Event Impact Analysis

```bash
wikipedia-health events \
  --start-date 2020-01-01 \
  --end-date 2023-12-31 \
  --event-dates "2020-03-11:COVID-19 Pandemic:natural_disaster,2021-01-06:Capitol Riot:political,2022-02-24:Ukraine Invasion:political"
```

### Example 5: Forecasting with Custom Horizon

```bash
wikipedia-health forecasts \
  --start-date 2020-01-01 \
  --end-date 2023-12-31 \
  --horizon 180 \
  --methods arima prophet exponential_smoothing \
  --output-format json
```

## Output Structure

Results are saved to the output directory with the following structure:

```
output/
└── YYYYMMDD_HHMMSS/
    ├── results.json              # Complete results in JSON format
    ├── findings_report.html      # Human-readable findings report
    ├── evidence_report.html      # Statistical evidence report
    ├── trends_results.json       # Trends analysis results (if run)
    ├── platforms_results.json    # Platform analysis results (if run)
    ├── seasonality_results.json  # Seasonality analysis results (if run)
    ├── campaigns_results.json    # Campaign analysis results (if run)
    ├── events_results.json       # Events analysis results (if run)
    └── forecasts_results.json    # Forecast results (if run)
```

## Error Handling

The CLI provides informative error messages:

```bash
# Invalid date format
wikipedia-health trends --start-date 01-01-2020 --end-date 2023-12-31
# Error: Invalid date format: 01-01-2020. Expected YYYY-MM-DD

# Missing required argument
wikipedia-health campaigns --start-date 2020-01-01 --end-date 2023-12-31
# Error: the following arguments are required: --campaign-dates

# Invalid configuration
wikipedia-health --config invalid.yaml full --start-date 2020-01-01 --end-date 2023-12-31
# Error: Configuration validation failed
```

## Exit Codes

- `0`: Success
- `1`: Error (invalid arguments, analysis failure, etc.)
- `2`: Invalid command-line arguments

## Tips and Best Practices

1. **Use verbose mode for debugging**: Add `--verbose` to see detailed logs
2. **Specify output directory**: Use `--output-dir` to organize results
3. **Start with smaller date ranges**: Test with shorter periods before running full analyses
4. **Use custom configs for production**: Create environment-specific configuration files
5. **Check output directory**: Results are timestamped to avoid overwriting
6. **Validate dates**: Ensure start date is before end date
7. **Use appropriate significance levels**: Lower values (0.01) for stricter testing

## Getting Help

View help for any command:

```bash
# General help
wikipedia-health --help

# Command-specific help
wikipedia-health trends --help
wikipedia-health campaigns --help
```

## Troubleshooting

### Command not found

If `wikipedia-health` command is not found, try:

```bash
python -m wikipedia_health [command] [options]
```

Or reinstall the package:

```bash
pip install -e .
```

### API timeouts

If experiencing API timeouts, increase the timeout in your config file:

```yaml
api:
  timeout: 120  # Increase from default 30 seconds
```

### Memory issues with large date ranges

For very large date ranges (>5 years), consider:
- Running analyses separately by year
- Increasing system memory
- Using a more powerful machine

## Support

For issues or questions:
1. Check the configuration guide: `docs/configuration.md`
2. Review error messages carefully
3. Use `--verbose` for detailed logs
4. Check the output directory for partial results
