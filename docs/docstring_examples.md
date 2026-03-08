# Docstring Examples for Wikipedia Intelligence System

This document provides comprehensive examples of properly documented classes and methods for the Wikipedia Intelligence System.

## Table of Contents

1. [Data Ingestion Layer Examples](#data-ingestion-layer-examples)
2. [Processing Layer Examples](#processing-layer-examples)
3. [Analytics Layer Examples](#analytics-layer-examples)
4. [Storage Layer Examples](#storage-layer-examples)
5. [Visualization Layer Examples](#visualization-layer-examples)

## Data Ingestion Layer Examples

### PageviewsCollector Class

```python
class PageviewsCollector:
    """Collector for Wikipedia pageview statistics from Wikimedia API.
    
    This collector fetches article traffic data with bot filtering, device
    segmentation, and automatic rate limiting. It implements exponential
    backoff for rate limit errors and validates all API responses against
    expected schemas.
    
    The collector supports three types of queries:
    - Per-article pageviews: Traffic for specific articles over time
    - Top articles: Most viewed articles for a given period
    - Aggregate statistics: Total Wikipedia traffic metrics
    
    Args:
        api_client: WikimediaAPIClient instance for making HTTP requests.
        rate_limiter: RateLimiter instance for throttling requests.
        
    Attributes:
        api_client: The API client for HTTP requests.
        rate_limiter: The rate limiter for request throttling.
        base_url: Base URL for Wikimedia Pageviews API.
        
    Example:
        >>> from src.utils.api_client import WikimediaAPIClient
        >>> from src.utils.rate_limiter import RateLimiter
        >>> from datetime import datetime, timedelta
        >>> 
        >>> rate_limiter = RateLimiter(requests_per_second=200)
        >>> api_client = WikimediaAPIClient(rate_limiter=rate_limiter)
        >>> collector = PageviewsCollector(api_client, rate_limiter)
        >>> 
        >>> end_date = datetime.now()
        >>> start_date = end_date - timedelta(days=30)
        >>> pageviews = await collector.fetch_per_article(
        ...     article="Python_(programming_language)",
        ...     start_date=start_date,
        ...     end_date=end_date,
        ...     granularity="daily"
        ... )
        >>> print(f"Collected {len(pageviews)} records")
        Collected 30 records
    
    Note:
        All methods are async and must be awaited. The collector automatically
        filters bot traffic unless explicitly requested.
        
    See Also:
        :class:`WikimediaAPIClient`: For HTTP client configuration.
        :class:`RateLimiter`: For rate limiting configuration.
    """
    
    async def fetch_per_article(
        self,
        article: str,
        start_date: datetime,
        end_date: datetime,
        granularity: str = "daily"
    ) -> List[PageviewRecord]:
        """Fetch pageviews for a specific article over a date range.
        
        Retrieves pageview statistics with bot filtering and device segmentation.
        The method automatically handles pagination for large date ranges and
        validates response schemas before returning data.
        
        Args:
            article: Article title (URL-encoded). Use underscores for spaces.
                Example: "Python_(programming_language)".
            start_date: Start date for data retrieval (inclusive).
            end_date: End date for data retrieval (inclusive).
            granularity: Time granularity for data. Valid values: "hourly",
                "daily", "monthly". Defaults to "daily".
                
        Returns:
            List of PageviewRecord objects containing:
                - article: Article title
                - timestamp: Date/time of measurement
                - device_type: Device category (desktop, mobile-web, mobile-app)
                - views_human: Human pageviews (bot traffic filtered)
                - views_bot: Bot pageviews
                - views_total: Total pageviews (human + bot)
                
        Raises:
            ValueError: If date range is invalid (start_date > end_date) or
                granularity is not one of the valid values.
            APIError: If API request fails after all retry attempts.
            RateLimitError: If rate limit is exceeded and backoff fails.
            
        Example:
            >>> collector = PageviewsCollector(api_client, rate_limiter)
            >>> pageviews = await collector.fetch_per_article(
            ...     article="Artificial_intelligence",
            ...     start_date=datetime(2024, 1, 1),
            ...     end_date=datetime(2024, 1, 31),
            ...     granularity="daily"
            ... )
            >>> for record in pageviews[:3]:
            ...     print(f"{record.timestamp}: {record.views_human} views")
            2024-01-01: 125000 views
            2024-01-02: 130000 views
            2024-01-03: 128000 views
            
        Note:
            - Bot traffic is automatically filtered using agent_type='user'
            - Device types are aggregated from all available platforms
            - Large date ranges may take several minutes to fetch
            - Results are cached in Redis for 5 minutes
        """
        pass
```

### EditHistoryScraper Class

```python
class EditHistoryScraper:
    """Scraper for Wikipedia edit history and revision data.
    
    This scraper extracts edit patterns, editor metadata, and vandalism signals
    from Wikipedia revision history. It classifies editors, detects reverted
    edits, and calculates edit velocity metrics over rolling time windows.
    
    Args:
        api_client: WikipediaAPIClient instance for API requests.
        
    Attributes:
        api_client: The API client for making requests.
        
    Example:
        >>> scraper = EditHistoryScraper(api_client)
        >>> revisions = await scraper.fetch_revisions(
        ...     article="Company_Name",
        ...     start_date=datetime(2024, 1, 1),
        ...     end_date=datetime(2024, 1, 31)
        ... )
        >>> velocity = scraper.calculate_edit_velocity(revisions, window_hours=24)
        >>> print(f"Edit velocity: {velocity:.2f} edits/hour")
        Edit velocity: 2.50 edits/hour
    """
    
    def calculate_edit_velocity(
        self,
        revisions: List[RevisionRecord],
        window_hours: int = 24
    ) -> float:
        """Calculate edit velocity over a rolling time window.
        
        Computes the rate of edits per hour by counting edits within the
        specified time window. This metric helps identify edit spikes and
        abnormal editing patterns.
        
        Args:
            revisions: List of RevisionRecord objects sorted by timestamp.
            window_hours: Size of rolling window in hours. Common values:
                - 24: Daily edit rate
                - 168: Weekly edit rate (24 * 7)
                - 720: Monthly edit rate (24 * 30)
                Defaults to 24 hours.
                
        Returns:
            Edit velocity as edits per hour (float). Returns 0.0 if no edits
            in the window or if revisions list is empty.
            
        Raises:
            ValueError: If window_hours is not positive.
            
        Example:
            >>> revisions = [
            ...     RevisionRecord(timestamp=datetime(2024, 1, 1, 0, 0), ...),
            ...     RevisionRecord(timestamp=datetime(2024, 1, 1, 1, 0), ...),
            ...     RevisionRecord(timestamp=datetime(2024, 1, 1, 2, 0), ...),
            ... ]
            >>> velocity = scraper.calculate_edit_velocity(revisions, window_hours=24)
            >>> print(f"{velocity:.2f} edits/hour")
            0.13 edits/hour
            
        Note:
            - Velocity is calculated as: (edit_count / window_hours)
            - Revisions must be sorted by timestamp (ascending)
            - Only considers edits within the most recent window
        """
        pass
```

## Analytics Layer Examples

### TimeSeriesForecaster Class

```python
class TimeSeriesForecaster:
    """Time series forecasting engine for demand prediction.
    
    Generates demand predictions using Prophet or ARIMA models with confidence
    intervals. Detects seasonal patterns, calculates growth rates, and flags
    launch hype events when growth exceeds statistical thresholds.
    
    Args:
        model_type: Forecasting model to use. Valid values:
            - "prophet": Facebook Prophet (recommended for seasonal data)
            - "arima": ARIMA model (better for short-term forecasts)
            Defaults to "prophet".
            
    Attributes:
        model_type: The selected forecasting model.
        min_training_days: Minimum days of historical data required (90).
        
    Example:
        >>> import pandas as pd
        >>> forecaster = TimeSeriesForecaster(model_type="prophet")
        >>> 
        >>> # Prepare historical data
        >>> historical_data = pd.DataFrame({
        ...     'ds': pd.date_range('2023-01-01', periods=180),
        ...     'y': [1000, 1050, 1100, ...]  # pageview counts
        ... })
        >>> 
        >>> # Train and predict
        >>> model = forecaster.train(historical_data, article="Python")
        >>> forecast = forecaster.predict(model, periods=30)
        >>> 
        >>> print(f"30-day forecast: {forecast.predictions['yhat'].mean():.0f}")
        30-day forecast: 1250
        
    Note:
        - Requires minimum 90 days of historical data for training
        - Prophet handles missing data and outliers automatically
        - ARIMA requires stationary data (may need differencing)
        
    See Also:
        :meth:`train`: Train forecasting model on historical data.
        :meth:`predict`: Generate predictions with confidence intervals.
        :meth:`detect_seasonality`: Identify seasonal patterns.
    """
    
    def train(
        self,
        historical_data: pd.DataFrame,
        article: str
    ) -> ForecastModel:
        """Train forecasting model on historical pageview data.
        
        Fits the selected model (Prophet or ARIMA) to historical data and
        validates that sufficient training data is available. The model learns
        trend, seasonality, and holiday effects from the data.
        
        Args:
            historical_data: DataFrame with columns:
                - 'ds': Date column (datetime64)
                - 'y': Pageview counts (int or float)
                Must have at least 90 days of data.
            article: Article name for logging and model identification.
                
        Returns:
            ForecastModel object containing:
                - model: Trained Prophet or ARIMA model
                - article: Article name
                - training_end_date: Last date in training data
                - metrics: Training metrics (MAE, RMSE, MAPE)
                
        Raises:
            ValueError: If historical_data has fewer than 90 days or
                missing required columns ('ds', 'y').
            ModelTrainingError: If model fails to converge or training fails.
            
        Example:
            >>> historical_data = pd.DataFrame({
            ...     'ds': pd.date_range('2023-01-01', periods=180),
            ...     'y': np.random.randint(1000, 2000, 180)
            ... })
            >>> model = forecaster.train(historical_data, article="Python")
            >>> print(f"Trained on {len(historical_data)} days")
            Trained on 180 days
            
        Note:
            - Prophet automatically detects yearly, weekly, and daily seasonality
            - ARIMA order is selected automatically using AIC criterion
            - Training may take 30-60 seconds for large datasets
            - Model is cached in Redis for 7 days
        """
        pass
    
    def calculate_growth_rate(
        self,
        data: pd.DataFrame,
        period_days: int = 30
    ) -> float:
        """Calculate percentage growth rate over a time period.
        
        Computes the percentage change in pageviews between the start and end
        of the specified period. Used to identify rapid growth or decline.
        
        Args:
            data: DataFrame with 'ds' (date) and 'y' (pageviews) columns.
            period_days: Number of days for growth calculation. Common values:
                - 7: Weekly growth
                - 30: Monthly growth
                - 90: Quarterly growth
                Defaults to 30 days.
                
        Returns:
            Growth rate as percentage (float). Positive values indicate growth,
            negative values indicate decline. Formula:
            ((views_end - views_start) / views_start) * 100
            
        Raises:
            ValueError: If data has fewer rows than period_days or
                if views_start is zero (division by zero).
                
        Example:
            >>> data = pd.DataFrame({
            ...     'ds': pd.date_range('2024-01-01', periods=60),
            ...     'y': [1000] * 30 + [1500] * 30  # 50% growth after 30 days
            ... })
            >>> growth = forecaster.calculate_growth_rate(data, period_days=30)
            >>> print(f"Growth rate: {growth:.2f}%")
            Growth rate: 50.00%
            
        Note:
            - Growth is calculated using first and last values in period
            - Does not account for volatility within the period
            - Returns None if insufficient data
        """
        pass
```

### ReputationMonitor Class

```python
class ReputationMonitor:
    """Monitor for brand reputation risk assessment.
    
    Analyzes edit patterns to calculate reputation risk scores and generate
    alerts for potential brand damage. Combines edit velocity, vandalism rate,
    and anonymous edit percentage into a composite risk score.
    
    Args:
        alert_threshold: Risk score threshold (0-1) for generating high-priority
            alerts. Defaults to 0.7 (70% risk).
            
    Attributes:
        alert_threshold: The configured alert threshold.
        
    Example:
        >>> monitor = ReputationMonitor(alert_threshold=0.7)
        >>> 
        >>> # Calculate risk from edit metrics
        >>> metrics = EditMetrics(
        ...     edit_velocity=50.0,  # 50 edits/hour
        ...     vandalism_rate=0.4,  # 40% vandalism
        ...     anonymous_edit_pct=0.8  # 80% anonymous
        ... )
        >>> 
        >>> score = monitor.calculate_reputation_risk(metrics)
        >>> if score.risk_score > 0.7:
        ...     alert = monitor.generate_alert("Company_Name", score.risk_score)
        ...     print(f"ALERT: {alert.message}")
        ALERT: High reputation risk detected for Company_Name (risk: 0.85)
    """
    
    def calculate_reputation_risk(
        self,
        edit_metrics: EditMetrics
    ) -> ReputationScore:
        """Calculate composite reputation risk score.
        
        Combines multiple signals into a single risk score (0-1) using weighted
        averaging. Higher scores indicate greater reputation risk.
        
        The risk score formula:
        risk = 0.4 * normalized_velocity + 0.4 * vandalism_rate + 0.2 * anon_pct
        
        Args:
            edit_metrics: EditMetrics object containing:
                - edit_velocity: Edits per hour (float)
                - vandalism_rate: Percentage of reverted edits (0-1)
                - anonymous_edit_pct: Percentage of anonymous edits (0-1)
                
        Returns:
            ReputationScore object containing:
                - article: Article name
                - risk_score: Composite risk score (0-1)
                - edit_velocity: Input edit velocity
                - vandalism_rate: Input vandalism rate
                - anonymous_edit_pct: Input anonymous percentage
                - alert_level: "low" (<0.3), "medium" (0.3-0.7), "high" (>0.7)
                - timestamp: Calculation timestamp
                
        Raises:
            ValueError: If any metric is outside valid range (0-1 for rates,
                non-negative for velocity).
                
        Example:
            >>> metrics = EditMetrics(
            ...     edit_velocity=10.0,
            ...     vandalism_rate=0.2,
            ...     anonymous_edit_pct=0.5
            ... )
            >>> score = monitor.calculate_reputation_risk(metrics)
            >>> print(f"Risk: {score.risk_score:.2f} ({score.alert_level})")
            Risk: 0.45 (medium)
            
        Note:
            - Velocity is normalized using baseline (5 edits/hour)
            - Weights: velocity (40%), vandalism (40%), anonymous (20%)
            - Score is clamped to [0, 1] range
        """
        pass
```

## Storage Layer Examples

### RedisCache Class

```python
class RedisCache:
    """Redis-based caching layer for real-time data.
    
    Provides get/set operations with TTL (time-to-live) for caching metrics,
    dashboard data, and API responses. Implements automatic serialization for
    complex objects and fallback to database on cache misses.
    
    Args:
        redis_client: Redis client instance.
        default_ttl: Default TTL in seconds. Defaults to 300 (5 minutes).
        
    Attributes:
        redis_client: The Redis client.
        default_ttl: Default cache expiration time.
        
    Example:
        >>> import redis
        >>> redis_client = redis.Redis(host='localhost', port=6379, db=0)
        >>> cache = RedisCache(redis_client, default_ttl=300)
        >>> 
        >>> # Cache metrics
        >>> metrics = {'views': 1000, 'growth': 0.15}
        >>> cache.set('metrics:article:123', metrics, ttl=600)
        >>> 
        >>> # Retrieve metrics
        >>> cached = cache.get('metrics:article:123')
        >>> print(cached)
        {'views': 1000, 'growth': 0.15}
    """
    
    def get(
        self,
        key: str,
        fallback_fn: Optional[Callable] = None
    ) -> Optional[Any]:
        """Retrieve value from cache with optional fallback.
        
        Attempts to retrieve value from Redis cache. If key doesn't exist
        and fallback function is provided, calls fallback to generate value,
        caches it, and returns it.
        
        Args:
            key: Cache key (string). Use namespaced keys like
                "metrics:article:123" or "dashboard:trends:abc".
            fallback_fn: Optional function to call on cache miss. Should
                return the value to cache. If None, returns None on miss.
                
        Returns:
            Cached value (deserialized from JSON) or None if key doesn't exist
            and no fallback provided.
            
        Raises:
            RedisError: If Redis connection fails.
            SerializationError: If cached value cannot be deserialized.
            
        Example:
            >>> def fetch_from_db():
            ...     return {'views': 1000, 'growth': 0.15}
            >>> 
            >>> # First call: cache miss, calls fallback
            >>> metrics = cache.get('metrics:article:123', fallback_fn=fetch_from_db)
            >>> print(metrics)
            {'views': 1000, 'growth': 0.15}
            >>> 
            >>> # Second call: cache hit, no fallback needed
            >>> metrics = cache.get('metrics:article:123')
            >>> print(metrics)
            {'views': 1000, 'growth': 0.15}
            
        Note:
            - Values are serialized as JSON
            - Complex objects (dataclasses, models) are converted to dicts
            - Cache misses with fallback are logged for monitoring
        """
        pass
```

## Visualization Layer Examples

### DashboardApp Class

```python
class DashboardApp:
    """Streamlit-based interactive dashboard application.
    
    Provides real-time visualizations for demand trends, competitor comparisons,
    reputation alerts, and emerging topics. Supports filtering, auto-refresh,
    and data export to CSV/PDF formats.
    
    Args:
        db: Database connection instance.
        cache: RedisCache instance for caching dashboard data.
        
    Attributes:
        db: Database connection.
        cache: Redis cache.
        refresh_interval: Auto-refresh interval in seconds (default: 300).
        
    Example:
        >>> from src.storage.database import Database
        >>> from src.storage.redis_cache import RedisCache
        >>> 
        >>> db = Database(config.database)
        >>> cache = RedisCache(redis_client, default_ttl=300)
        >>> app = DashboardApp(db, cache)
        >>> 
        >>> # Run dashboard (blocking)
        >>> app.run()
    """
    
    def render_demand_trends(
        self,
        articles: List[str],
        date_range: DateRange
    ) -> Chart:
        """Render time series chart for product demand trends.
        
        Creates interactive Plotly chart showing pageview trends for multiple
        articles with tooltips, zoom, and pan capabilities.
        
        Args:
            articles: List of article titles to display.
            date_range: DateRange object with start_date and end_date.
                
        Returns:
            Chart object containing Plotly figure and metadata.
            
        Raises:
            ValueError: If articles list is empty or date range is invalid.
            DataNotFoundError: If no data exists for specified articles/dates.
            
        Example:
            >>> articles = ["Python_(programming_language)", "Java_(programming_language)"]
            >>> date_range = DateRange(
            ...     start_date=datetime(2024, 1, 1),
            ...     end_date=datetime(2024, 3, 31)
            ... )
            >>> chart = app.render_demand_trends(articles, date_range)
            >>> # Chart is automatically displayed in Streamlit
            
        Note:
            - Chart includes 7-day moving average
            - Tooltips show exact values and percentage changes
            - Data is cached for 5 minutes
            - Maximum 10 articles can be displayed simultaneously
        """
        pass
```

## Best Practices Summary

1. **Always include**:
   - One-line summary
   - Extended description (2-3 sentences)
   - All parameters with types and descriptions
   - Return value description
   - At least one example
   - Common exceptions

2. **Use clear examples**:
   - Show typical usage
   - Include expected output
   - Use realistic data
   - Keep examples concise

3. **Document edge cases**:
   - Empty inputs
   - Boundary values
   - Error conditions
   - Performance considerations

4. **Cross-reference related items**:
   - Related classes
   - Related methods
   - Configuration options
   - External documentation

5. **Keep it maintainable**:
   - Update docstrings when code changes
   - Test examples with doctest
   - Use consistent terminology
   - Follow Google style guide
