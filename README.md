# Wikipedia Intelligence System

A real-time business intelligence platform that transforms Wikipedia data into actionable business insights.

## Project Structure

```
wikipedia-intelligence-system/
├── src/
│   ├── data_ingestion/    # Data collectors (Pageviews, Edit History, Crawler)
│   ├── processing/        # ETL pipelines and data transformation
│   ├── storage/          # Database models and cache management
│   ├── analytics/        # ML models and statistical analysis
│   ├── visualization/    # Dashboard and reporting
│   └── utils/           # Shared utilities (config, logging)
├── tests/
│   ├── unit/            # Unit tests
│   └── property/        # Property-based tests (Hypothesis)
├── config/              # Configuration files
├── data/               # Data storage
└── docs/               # Documentation
```

## Setup

### Prerequisites

- Python 3.11+
- PostgreSQL 14+
- Redis 7+

### Installation

1. Clone the repository
2. Create a virtual environment:
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

4. Configure environment:
   ```bash
   cp .env.example .env
   # Edit .env with your configuration
   ```

5. Initialize database:
   ```bash
   alembic upgrade head
   ```

## Configuration

Configuration is loaded from:
1. `config/config.yaml` - Base configuration with profiles
2. Environment variables (override config file)
3. `.env` file (loaded automatically)

Supported profiles: `development`, `staging`, `production`

## Testing

Run all tests:
```bash
pytest
```

Run unit tests only:
```bash
pytest tests/unit -m unit
```

Run property-based tests:
```bash
pytest tests/property -m property
```

## Code Quality

Format code:
```bash
black src tests
```

Lint code:
```bash
flake8 src tests
```

Type checking:
```bash
mypy src
```

## License

MIT
