# Wikipedia Intelligence System

A production-ready business intelligence platform that transforms Wikipedia data into actionable business insights. The system leverages the Wikimedia Pageviews API, Wikipedia Edit History, and web crawling to provide real-time demand forecasting, brand reputation monitoring, competitive intelligence, and emerging trend detection.

## Features

### Data Collection
- **Pageviews Analytics**: Collect and analyze Wikipedia article traffic with bot filtering and device segmentation
- **Edit History Monitoring**: Track edit patterns, detect vandalism, and assess reputation risks
- **Content Crawling**: Extract structured data from Wikipedia articles including infoboxes, tables, and relationships

### Analytics & Intelligence
- **Demand Forecasting**: Time series predictions with confidence intervals using Prophet and ARIMA models
- **Hype Detection**: Identify trending topics and viral content with composite scoring algorithms
- **Reputation Monitoring**: Real-time alerts for brand reputation risks based on edit velocity and vandalism patterns
- **Topic Clustering**: Group related articles by industry and calculate comparative growth metrics
- **Knowledge Graphs**: Visualize entity relationships and identify influential nodes in domain networks

### Visualization & Reporting
- **Interactive Dashboards**: Real-time Streamlit dashboards with auto-refresh and filtering
- **Competitive Analysis**: Compare metrics across competitors with sortable tables and charts
- **Alert System**: High-priority notifications for reputation risks and emerging trends
- **Data Export**: Generate reports in CSV and PDF formats

## Architecture

The system follows a layered architecture with five primary components:

```
Data Sources → Data Ingestion → Processing → Storage → Analytics → Visualization
```

- **Data Ingestion Layer**: Collectors for Pageviews API, Edit History, and Crawl4AI pipeline
- **Processing Layer**: ETL pipelines with validation, deduplication, and data lineage tracking
- **Storage Layer**: PostgreSQL data warehouse with Redis caching for real-time queries
- **Analytics Layer**: ML models for forecasting, clustering, hype detection, and reputation scoring
- **Visualization Layer**: Streamlit dashboard with interactive charts and export capabilities

See [Architecture Documentation](docs/architecture.md) for detailed component diagrams.

## Quick Start

### Prerequisites

- Python 3.11+
- PostgreSQL 14+
- Redis 7+
- 4GB+ RAM recommended

### Installation

1. **Clone the repository**
   ```bash
   git clone https://github.com/your-org/wikipedia-intelligence-system.git
   cd wikipedia-intelligence-system
   ```

2. **Create virtual environment**
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

4. **Configure environment**
   ```bash
   cp .env.example .env
   # Edit .env with your database credentials and API settings
   ```

5. **Initialize database**
   ```bash
   # Start PostgreSQL and Redis
   # Then run migrations
   alembic upgrade head
   ```

6. **Verify setup**
   ```bash
   python verify_setup.py
   ```

### Running the System

**Start the dashboard:**
```bash
streamlit run src/visualization/dashboard_app.py
```

**Run data collection:**
```bash
python -m src.scheduling.job_scheduler
```

**Run a single analysis:**
```bash
python examples/scripts/basic_analysis.py --article "Python_(programming_language)" --days 90
```

## Configuration

### Configuration Files

The system uses a hierarchical configuration approach:

1. **Base configuration**: `config/config.yaml`
2. **Environment-specific**: `config/development.yaml`, `config/staging.yaml`, `config/production.yaml`
3. **Environment variables**: Override any config value (highest priority)
4. **`.env` file**: Loaded automatically for local development

### Key Configuration Parameters

```yaml
# Database
database:
  host: localhost
  port: 5432
  name: wikipedia_intelligence
  user: your_user
  password: your_password

# Redis Cache
redis:
  host: localhost
  port: 6379
  ttl: 300  # 5 minutes

# API Rate Limiting
api:
  wikimedia:
    rate_limit: 200  # requests per second
    timeout: 30
  
# Analytics
analytics:
  forecasting:
    min_training_days: 90
    confidence_level: 0.95
  hype_detection:
    threshold: 0.75
  reputation:
    alert_threshold: 0.7

# Dashboard
dashboard:
  auto_refresh_seconds: 300
  max_articles_display: 100
```

See [Configuration Guide](docs/configuration.md) for complete reference.

### Environment Variables

Critical settings can be overridden via environment variables:

```bash
export DB_HOST=localhost
export DB_PORT=5432
export DB_NAME=wikipedia_intelligence
export DB_USER=your_user
export DB_PASSWORD=your_password
export REDIS_HOST=localhost
export REDIS_PORT=6379
export LOG_LEVEL=INFO
export ENVIRONMENT=development
```

## Usage Examples

### Example 1: Collect Pageviews for an Article

```python
from src.data_ingestion.pageviews_collector import PageviewsCollector
from src.utils.api_client import WikimediaAPIClient
from src.utils.rate_limiter import RateLimiter
from datetime import datetime, timedelta

# Initialize components
rate_limiter = RateLimiter(requests_per_second=200)
api_client = WikimediaAPIClient(rate_limiter=rate_limiter)
collector = PageviewsCollector(api_client=api_client, rate_limiter=rate_limiter)

# Collect pageviews
end_date = datetime.now()
start_date = end_date - timedelta(days=30)

pageviews = await collector.fetch_per_article(
    article="Python_(programming_language)",
    start_date=start_date,
    end_date=end_date,
    granularity="daily"
)

print(f"Collected {len(pageviews)} daily pageview records")
```

### Example 2: Generate Demand Forecast

```python
from src.analytics.time_series_forecaster import TimeSeriesForecaster
import pandas as pd

# Prepare historical data
historical_data = pd.DataFrame({
    'ds': pd.date_range('2023-01-01', periods=180),
    'y': [1000, 1050, 1100, ...]  # pageview counts
})

# Train and predict
forecaster = TimeSeriesForecaster(model_type="prophet")
model = forecaster.train(historical_data, article="Python_(programming_language)")
forecast = forecaster.predict(model, periods=30)

print(f"30-day forecast: {forecast.predictions['yhat'].mean():.0f} avg daily views")
print(f"Growth rate: {forecaster.calculate_growth_rate(historical_data, period_days=30):.2f}%")
```

### Example 3: Monitor Reputation Risk

```python
from src.analytics.reputation_monitor import ReputationMonitor
from src.data_ingestion.edit_history_scraper import EditHistoryScraper

# Fetch edit history
scraper = EditHistoryScraper(api_client=api_client)
revisions = await scraper.fetch_revisions(
    article="Company_Name",
    start_date=start_date,
    end_date=end_date
)

# Calculate metrics
edit_velocity = scraper.calculate_edit_velocity(revisions, window_hours=24)
vandalism_metrics = scraper.detect_vandalism_signals(revisions)

# Assess reputation risk
monitor = ReputationMonitor(alert_threshold=0.7)
risk_score = monitor.calculate_reputation_risk({
    'edit_velocity': edit_velocity,
    'vandalism_rate': vandalism_metrics.vandalism_rate,
    'anonymous_edit_pct': vandalism_metrics.anonymous_pct
})

if risk_score.risk_score > 0.7:
    alert = monitor.generate_alert("Company_Name", risk_score.risk_score)
    print(f"HIGH RISK ALERT: {alert.message}")
```

### Example 4: Detect Trending Topics

```python
from src.analytics.hype_detection_engine import HypeDetectionEngine

# Calculate hype metrics
hype_engine = HypeDetectionEngine(hype_threshold=0.75)
hype_score = hype_engine.calculate_hype_score(
    view_velocity=0.85,  # 85% growth rate
    edit_growth=0.60,    # 60% increase in edits
    content_expansion=0.40  # 40% content growth
)

attention_density = hype_engine.calculate_attention_density(
    pageviews_df, 
    window_days=7
)

if hype_score > 0.75:
    print(f"TRENDING: Hype score {hype_score:.2f}, Attention density {attention_density:.0f}")
```

See [examples/](examples/) directory for complete working examples.

## Testing

The system uses a dual testing strategy combining unit tests and property-based tests.

### Run All Tests

```bash
pytest
```

### Run Unit Tests Only

```bash
pytest tests/unit -m unit
```

### Run Property-Based Tests

```bash
pytest tests/property -m property
```

### Run Specific Test Module

```bash
pytest tests/unit/test_pageviews_collector.py -v
```

### Run with Coverage

```bash
pytest --cov=src --cov-report=html
```

### Test Organization

- **Unit Tests** (`tests/unit/`): Specific examples and edge cases
- **Property Tests** (`tests/property/`): Universal correctness properties using Hypothesis
- **Integration Tests** (`tests/`): End-to-end workflows

See [Testing Documentation](docs/testing.md) for detailed testing guidelines.

## Development

### Code Quality Tools

**Format code:**
```bash
black src tests
```

**Lint code:**
```bash
flake8 src tests
```

**Type checking:**
```bash
mypy src
```

**Run all quality checks:**
```bash
make quality
```

### Project Structure

```
wikipedia-intelligence-system/
├── src/
│   ├── data_ingestion/       # Data collectors
│   │   ├── pageviews_collector.py
│   │   ├── edit_history_scraper.py
│   │   └── crawl4ai_pipeline.py
│   ├── processing/           # ETL pipelines
│   │   └── etl_pipeline_manager.py
│   ├── storage/             # Database and cache
│   │   ├── models.py
│   │   ├── database.py
│   │   └── redis_cache.py
│   ├── analytics/           # ML and analytics
│   │   ├── time_series_forecaster.py
│   │   ├── reputation_monitor.py
│   │   ├── topic_clustering_engine.py
│   │   ├── hype_detection_engine.py
│   │   └── knowledge_graph_builder.py
│   ├── visualization/       # Dashboard
│   │   └── dashboard_app.py
│   ├── scheduling/          # Job orchestration
│   │   └── job_scheduler.py
│   └── utils/              # Shared utilities
│       ├── config.py
│       ├── logging.py
│       ├── api_client.py
│       └── rate_limiter.py
├── tests/
│   ├── unit/               # Unit tests
│   ├── property/           # Property-based tests
│   └── conftest.py         # Pytest fixtures
├── config/                 # Configuration files
├── alembic/               # Database migrations
├── docs/                  # Documentation
├── examples/              # Usage examples
├── scripts/               # Utility scripts
└── data/                  # Data storage
```

## Deployment

### Docker Deployment

**Build and run with Docker Compose:**
```bash
docker-compose up -d
```

This starts:
- Application container
- PostgreSQL database
- Redis cache

**Access the dashboard:**
```
http://localhost:8501
```

### Manual Deployment

1. **Set up infrastructure**
   - PostgreSQL 14+ database
   - Redis 7+ cache
   - Python 3.11+ runtime

2. **Configure production settings**
   ```bash
   export ENVIRONMENT=production
   export DB_HOST=your-db-host
   export REDIS_HOST=your-redis-host
   ```

3. **Run database migrations**
   ```bash
   alembic upgrade head
   ```

4. **Start services**
   ```bash
   # Start job scheduler
   python -m src.scheduling.job_scheduler &
   
   # Start dashboard
   streamlit run src/visualization/dashboard_app.py --server.port 8501
   ```

See [Deployment Guide](DEPLOYMENT.md) for detailed instructions.

## Monitoring & Operations

### Health Checks

```bash
curl http://localhost:8501/health
```

### Logs

Logs are written to `logs/` directory in structured JSON format:

```bash
tail -f logs/wikipedia_intelligence.log
```

### Metrics

System metrics are exposed for monitoring:
- Data ingestion rates
- Processing latency
- Storage utilization
- API usage
- Pipeline health

### Scheduled Jobs

The system runs scheduled jobs for:
- **Hourly**: Pageview collection
- **Daily**: Edit history scraping, deep crawls
- **Weekly**: Model retraining

Monitor job status in the dashboard or logs.

## Documentation

- [Getting Started Guide](docs/getting_started.rst)
- [API Reference](docs/api_reference.rst)
- [Configuration Guide](docs/configuration.md)
- [User Guides](docs/user_guides.rst)
- [Architecture Documentation](docs/architecture.md)
- [Testing Guide](docs/testing.md)
- [Deployment Guide](DEPLOYMENT.md)

## Contributing

Contributions are welcome! Please:

1. Fork the repository
2. Create a feature branch
3. Make your changes with tests
4. Run quality checks (`make quality`)
5. Submit a pull request

See [CONTRIBUTING.md](CONTRIBUTING.md) for detailed guidelines.

## License

MIT License - see [LICENSE](LICENSE) file for details.

## Support

- **Documentation**: [docs/](docs/)
- **Examples**: [examples/](examples/)
- **Issues**: [GitHub Issues](https://github.com/your-org/wikipedia-intelligence-system/issues)
- **Discussions**: [GitHub Discussions](https://github.com/your-org/wikipedia-intelligence-system/discussions)

## Acknowledgments

- Wikimedia Foundation for providing the Pageviews API
- Wikipedia community for maintaining comprehensive article data
- Open source libraries: Prophet, scikit-learn, NetworkX, Streamlit, Hypothesis
