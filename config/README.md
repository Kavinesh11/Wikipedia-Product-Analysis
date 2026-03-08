# Configuration Documentation

## Overview

The Wikipedia Intelligence System uses a hierarchical configuration system that supports multiple environments (development, staging, production) and allows configuration through both YAML files and environment variables.

## Configuration Precedence

Configuration values are loaded in the following order (later sources override earlier ones):

1. Default values in `config.yaml`
2. Environment-specific section in `config.yaml` (development/staging/production)
3. Environment variables (highest priority)

## Configuration Files

### config.yaml

Main configuration file with environment-specific sections. Located at `config/config.yaml`.

### Environment Variables

Environment variables override YAML configuration. Use the `.env` file for local development or set them in your deployment environment.

## Configuration Parameters

### API Configuration

**wikimedia_base_url** (string)
- Description: Base URL for Wikimedia REST API
- Default: `https://wikimedia.org/api/rest_v1`
- Environment Variable: `WIKIMEDIA_API_BASE_URL`
- Required: Yes

**rate_limit** (integer)
- Description: Maximum API requests per second
- Default: `200`
- Environment Variable: `WIKIMEDIA_RATE_LIMIT`
- Required: Yes
- Valid Range: 1-200

**timeout** (integer)
- Description: API request timeout in seconds
- Default: `30`
- Environment Variable: `API_TIMEOUT`
- Required: No

**max_retries** (integer)
- Description: Maximum number of retry attempts for failed API requests
- Default: Development: 3, Production: 5
- Environment Variable: `API_MAX_RETRIES`
- Required: No
- Valid Range: 0-10

### Database Configuration

**postgres_host** (string)
- Description: PostgreSQL server hostname
- Default: `localhost` (development)
- Environment Variable: `POSTGRES_HOST`
- Required: Yes (production)

**postgres_port** (integer)
- Description: PostgreSQL server port
- Default: `5432`
- Environment Variable: `POSTGRES_PORT`
- Required: No

**postgres_db** (string)
- Description: PostgreSQL database name
- Default: `wikipedia_intelligence_dev`
- Environment Variable: `POSTGRES_DB`
- Required: Yes

**postgres_user** (string)
- Description: PostgreSQL username
- Default: `dev_user`
- Environment Variable: `POSTGRES_USER`
- Required: Yes

**postgres_password** (string)
- Description: PostgreSQL password (should be encrypted in production)
- Default: None
- Environment Variable: `POSTGRES_PASSWORD`
- Required: Yes
- Security: Sensitive - must be encrypted

**pool_size** (integer)
- Description: Database connection pool size
- Default: Development: 10, Staging: 20, Production: 50
- Environment Variable: `DB_POOL_SIZE`
- Required: No
- Valid Range: 5-100

**use_sqlite** (boolean)
- Description: Use SQLite instead of PostgreSQL (development only)
- Default: `true` (development), `false` (staging/production)
- Environment Variable: `USE_SQLITE`
- Required: No

**sqlite_path** (string)
- Description: Path to SQLite database file
- Default: `data/wikipedia_intelligence_dev.db`
- Environment Variable: `SQLITE_PATH`
- Required: No (only if use_sqlite is true)

### Cache Configuration

**redis_host** (string)
- Description: Redis server hostname
- Default: `localhost`
- Environment Variable: `REDIS_HOST`
- Required: Yes

**redis_port** (integer)
- Description: Redis server port
- Default: `6379`
- Environment Variable: `REDIS_PORT`
- Required: No

**redis_db** (integer)
- Description: Redis database number
- Default: `0`
- Environment Variable: `REDIS_DB`
- Required: No
- Valid Range: 0-15

**max_connections** (integer)
- Description: Maximum Redis connection pool size
- Default: Development: 50, Staging: 100, Production: 200
- Environment Variable: `REDIS_MAX_CONNECTIONS`
- Required: No
- Valid Range: 10-500

### Logging Configuration

**level** (string)
- Description: Logging level
- Default: Development: `DEBUG`, Production: `INFO`
- Environment Variable: `LOG_LEVEL`
- Required: No
- Valid Values: DEBUG, INFO, WARNING, ERROR, CRITICAL

**json_format** (boolean)
- Description: Use structured JSON logging format
- Default: `true`
- Environment Variable: `LOG_JSON_FORMAT`
- Required: No

**log_file** (string)
- Description: Path to log file
- Default: `logs/app.log`
- Environment Variable: `LOG_FILE`
- Required: No

### Crawler Configuration

**max_concurrent** (integer)
- Description: Maximum concurrent crawl operations
- Default: Development: 10, Staging: 20, Production: 50
- Environment Variable: `CRAWL_MAX_CONCURRENT`
- Required: No
- Valid Range: 1-100

**crawl_delay** (float)
- Description: Delay between crawl requests in seconds
- Default: Development: 1.0, Staging: 0.5, Production: 0.2
- Environment Variable: `CRAWL_DELAY`
- Required: No
- Valid Range: 0.1-5.0

**max_depth** (integer)
- Description: Maximum depth for deep crawl operations
- Default: Development: 2, Production: 3
- Environment Variable: `CRAWL_MAX_DEPTH`
- Required: No
- Valid Range: 1-5

**respect_robots_txt** (boolean)
- Description: Respect robots.txt directives
- Default: `true`
- Environment Variable: `CRAWL_RESPECT_ROBOTS`
- Required: No

### Analytics Configuration

**min_training_days** (integer)
- Description: Minimum days of historical data required for model training
- Default: `90`
- Environment Variable: `ANALYTICS_MIN_TRAINING_DAYS`
- Required: No
- Valid Range: 30-365

**forecast_periods** (integer)
- Description: Number of periods to forecast into the future
- Default: `30`
- Environment Variable: `ANALYTICS_FORECAST_PERIODS`
- Required: No
- Valid Range: 1-365

**hype_threshold** (float)
- Description: Threshold for flagging articles as trending
- Default: `0.75`
- Environment Variable: `ANALYTICS_HYPE_THRESHOLD`
- Required: No
- Valid Range: 0.0-1.0

**reputation_alert_threshold** (float)
- Description: Threshold for generating reputation risk alerts
- Default: `0.7`
- Environment Variable: `ANALYTICS_REPUTATION_THRESHOLD`
- Required: No
- Valid Range: 0.0-1.0

### Dashboard Configuration

**refresh_interval** (integer)
- Description: Dashboard auto-refresh interval in seconds
- Default: `300` (5 minutes)
- Environment Variable: `DASHBOARD_AUTO_REFRESH_INTERVAL`
- Required: No
- Valid Range: 60-3600

**cache_ttl** (integer)
- Description: Dashboard data cache TTL in seconds
- Default: `300`
- Environment Variable: `DASHBOARD_CACHE_TTL`
- Required: No
- Valid Range: 60-3600

### Scheduling Configuration

**enable_scheduled_jobs** (boolean)
- Description: Enable automatic scheduled data collection
- Default: `true`
- Environment Variable: `ENABLE_SCHEDULED_JOBS`
- Required: No

**pageviews_collection_interval** (integer)
- Description: Interval for pageviews collection in seconds
- Default: `3600` (1 hour)
- Environment Variable: `PAGEVIEWS_COLLECTION_INTERVAL`
- Required: No

**edits_collection_interval** (integer)
- Description: Interval for edit history collection in seconds
- Default: `86400` (24 hours)
- Environment Variable: `EDITS_COLLECTION_INTERVAL`
- Required: No

**model_retraining_interval** (integer)
- Description: Interval for model retraining in seconds
- Default: `604800` (7 days)
- Environment Variable: `MODEL_RETRAINING_INTERVAL`
- Required: No

### Alert Configuration

**alert_email_enabled** (boolean)
- Description: Enable email alerts
- Default: `false`
- Environment Variable: `ALERT_EMAIL_ENABLED`
- Required: No

**alert_webhook_url** (string)
- Description: Webhook URL for alert notifications
- Default: None
- Environment Variable: `ALERT_WEBHOOK_URL`
- Required: No

## Environment-Specific Configuration

### Development

Optimized for local development with:
- SQLite support for easy setup
- Debug logging enabled
- Lower concurrency limits
- Longer crawl delays

### Staging

Mirrors production with:
- PostgreSQL database
- INFO level logging
- Moderate concurrency
- Production-like settings for testing

### Production

Optimized for performance with:
- High connection pool sizes
- Maximum concurrency
- Minimal crawl delays
- All sensitive values from environment variables

## Configuration Validation

The system validates configuration on startup and will fail fast with clear error messages if:

- Required parameters are missing
- Values are outside valid ranges
- Type mismatches occur
- Database connection fails
- Redis connection fails

## Security Best Practices

1. **Never commit sensitive values** to version control
2. **Use environment variables** for all passwords, API keys, and tokens
3. **Encrypt sensitive configuration** in production using the built-in encryption
4. **Rotate credentials** regularly
5. **Use different credentials** for each environment
6. **Restrict database user permissions** to minimum required

## Example: Setting Up Development Environment

1. Copy `.env.example` to `.env`:
   ```bash
   cp .env.example .env
   ```

2. Edit `.env` with your local settings:
   ```bash
   ENVIRONMENT=development
   POSTGRES_PASSWORD=your_dev_password
   ```

3. The system will automatically load development configuration from `config.yaml`

## Example: Setting Up Production Environment

1. Set environment variables in your deployment platform:
   ```bash
   export ENVIRONMENT=production
   export POSTGRES_HOST=prod-db.example.com
   export POSTGRES_DB=wikipedia_intelligence
   export POSTGRES_USER=wiki_prod_user
   export POSTGRES_PASSWORD=encrypted_password
   export REDIS_HOST=prod-redis.example.com
   ```

2. The system will use production configuration from `config.yaml` with environment variable overrides

## Troubleshooting

### Configuration Not Loading

- Check that `ENVIRONMENT` variable is set correctly
- Verify `config.yaml` exists in the `config/` directory
- Check file permissions

### Database Connection Fails

- Verify `POSTGRES_HOST`, `POSTGRES_PORT`, `POSTGRES_DB` are correct
- Check that `POSTGRES_USER` has proper permissions
- Ensure `POSTGRES_PASSWORD` is set correctly
- Test connection manually: `psql -h $POSTGRES_HOST -U $POSTGRES_USER -d $POSTGRES_DB`

### Redis Connection Fails

- Verify `REDIS_HOST` and `REDIS_PORT` are correct
- Check that Redis server is running
- Test connection manually: `redis-cli -h $REDIS_HOST -p $REDIS_PORT ping`

## Configuration Updates

### Runtime Updates

Non-critical parameters can be updated at runtime:
- Dashboard refresh interval
- Cache TTL
- Logging level

### Restart Required

Critical parameters require application restart:
- Database connection settings
- Redis connection settings
- API endpoints
- Pool sizes
