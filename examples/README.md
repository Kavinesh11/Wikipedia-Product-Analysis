# Usage Examples

This directory contains examples demonstrating how to use the Wikipedia Product Health Analysis System.

## Directory Structure

```
examples/
├── notebooks/              # Jupyter notebooks with interactive examples
│   ├── 01_data_acquisition.ipynb
│   ├── 02_trend_analysis.ipynb
│   ├── 03_platform_analysis.ipynb
│   ├── 04_seasonality_analysis.ipynb
│   ├── 05_campaign_analysis.ipynb
│   ├── 06_event_analysis.ipynb
│   └── 07_forecasting.ipynb
├── scripts/               # Python scripts for common workflows
│   ├── basic_analysis.py
│   ├── full_pipeline.py
│   ├── custom_analysis.py
│   └── batch_processing.py
└── data/                  # Sample data for examples
    └── sample_config.yaml
```

## Quick Start

### Running Notebooks

Install Jupyter:

```bash
pip install jupyter
```

Start Jupyter:

```bash
jupyter notebook examples/notebooks/
```

### Running Scripts

Basic analysis:

```bash
python examples/scripts/basic_analysis.py
```

Full pipeline:

```bash
python examples/scripts/full_pipeline.py --start-date 2020-01-01 --end-date 2023-12-31
```

## Example Notebooks

### 1. Data Acquisition (`01_data_acquisition.ipynb`)

Learn how to:
- Fetch pageview data from Wikimedia APIs
- Fetch editor and edit volume data
- Validate data quality
- Handle missing data and anomalies

### 2. Trend Analysis (`02_trend_analysis.ipynb`)

Learn how to:
- Decompose time series into trend, seasonal, and residual components
- Detect structural breaks and changepoints
- Test for statistical significance
- Attribute changes to external events

### 3. Platform Analysis (`03_platform_analysis.ipynb`)

Learn how to:
- Analyze platform mix (desktop, mobile web, mobile app)
- Calculate platform concentration metrics
- Assess platform dependency risks
- Perform scenario analysis

### 4. Seasonality Analysis (`04_seasonality_analysis.ipynb`)

Learn how to:
- Detect and validate seasonal patterns
- Analyze day-of-week effects
- Model holiday impacts
- Distinguish utility vs leisure usage

### 5. Campaign Analysis (`05_campaign_analysis.ipynb`)

Learn how to:
- Measure campaign effectiveness with causal inference
- Construct counterfactual baselines
- Calculate average treatment effects
- Perform robustness checks

### 6. Event Analysis (`06_event_analysis.ipynb`)

Learn how to:
- Measure Wikipedia's response to external events
- Calculate cumulative abnormal returns
- Test event significance
- Compare responses across event categories

### 7. Forecasting (`07_forecasting.ipynb`)

Learn how to:
- Generate multi-method forecasts
- Quantify forecast uncertainty
- Evaluate forecast accuracy
- Perform scenario analysis

## Example Scripts

### Basic Analysis (`basic_analysis.py`)

Simple script demonstrating:
- Loading configuration
- Fetching data
- Running a single analysis
- Saving results

### Full Pipeline (`full_pipeline.py`)

Complete analysis pipeline:
- Data acquisition from APIs
- All analysis types
- Cross-validation
- Report generation

### Custom Analysis (`custom_analysis.py`)

Advanced customization:
- Custom statistical tests
- Custom causal inference methods
- Custom visualization
- Integration with external data

### Batch Processing (`batch_processing.py`)

Processing multiple analyses:
- Batch data acquisition
- Parallel analysis execution
- Result aggregation
- Automated reporting

## Sample Data

The `data/` directory contains:
- `sample_config.yaml` - Example configuration file
- Sample datasets for testing (if available)

## Tips

1. **Start with notebooks**: Interactive exploration is easier in notebooks
2. **Use scripts for automation**: Convert notebook workflows to scripts for production
3. **Customize examples**: Modify examples to fit your specific use case
4. **Check documentation**: Refer to API documentation for detailed parameter descriptions

## Common Workflows

### Workflow 1: Quick Trend Check

```python
from datetime import date
from wikipedia_health.config import load_config
from wikipedia_health.analysis_system import AnalysisSystem

config = load_config()
system = AnalysisSystem(config=config)

results = system.analyze_long_term_trends(
    start_date=date(2020, 1, 1),
    end_date=date(2023, 12, 31),
    platforms=['all']
)

print(f"Found {len(results['findings'])} findings")
for finding in results['findings']:
    print(f"- {finding.description}")
```

### Workflow 2: Campaign Evaluation

```python
from datetime import date
from wikipedia_health.analysis_system import AnalysisSystem

system = AnalysisSystem()

results = system.analyze_campaigns(
    start_date=date(2021, 1, 1),
    end_date=date(2023, 12, 31),
    campaign_dates=[
        (date(2021, 6, 15), "Summer Campaign"),
        (date(2021, 12, 1), "Winter Campaign")
    ],
    platforms=['all']
)

for campaign, effect in results['campaign_effects'].items():
    print(f"{campaign}: {effect.effect_size:.2%} impact (p={effect.p_value:.4f})")
```

### Workflow 3: Forecasting

```python
from datetime import date
from wikipedia_health.analysis_system import AnalysisSystem

system = AnalysisSystem()

results = system.generate_forecasts(
    start_date=date(2020, 1, 1),
    end_date=date(2023, 12, 31),
    horizon=90,
    platforms=['all'],
    methods=['arima', 'prophet', 'exponential_smoothing']
)

forecast = results['ensemble_forecast']
print(f"90-day forecast: {forecast.point_forecast.mean():.0f} daily pageviews")
print(f"95% CI: [{forecast.lower_bound.mean():.0f}, {forecast.upper_bound.mean():.0f}]")
```

## Interpreting Results

### Statistical Significance

- **p-value < 0.05**: Statistically significant at 95% confidence level
- **p-value < 0.01**: Highly significant at 99% confidence level
- **p-value ≥ 0.05**: Not statistically significant

### Effect Sizes

- **Cohen's d**:
  - Small: 0.2
  - Medium: 0.5
  - Large: 0.8

- **Percentage change**:
  - Interpret in context of baseline
  - Consider confidence intervals

### Confidence Intervals

- **95% CI**: Range containing true value with 95% probability
- **Narrow CI**: Precise estimate
- **Wide CI**: Uncertain estimate

## Troubleshooting

### API Rate Limiting

If you encounter rate limiting:
- Reduce date range
- Add delays between requests
- Use cached data when available

### Memory Issues

For large datasets:
- Process data in chunks
- Use smaller date ranges
- Increase system memory

### Slow Execution

To speed up analysis:
- Reduce bootstrap/permutation iterations
- Use fewer forecasting methods
- Parallelize when possible

## Contributing Examples

To contribute new examples:

1. Create a new notebook or script
2. Add clear comments and documentation
3. Include sample output
4. Update this README
5. Submit a pull request

## Support

For questions about examples:
1. Check the API documentation
2. Review the configuration guide
3. Examine existing examples
4. Open an issue on GitHub
