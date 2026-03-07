# Job Scheduling and Orchestration Guide

## Overview

The Wikipedia Intelligence System includes a comprehensive job scheduling and orchestration framework built on APScheduler. This guide explains how to use the scheduling system to automate data collection, processing, and analytics tasks.

## Components

### 1. JobScheduler

The `JobScheduler` class provides a high-level interface for scheduling jobs using cron or interval triggers.

**Key Features:**
- Cron-based scheduling (e.g., "every hour at minute 0")
- Interval-based scheduling (e.g., "every 30 minutes")
- Job monitoring with event listeners
- Graceful error handling
- Health status reporting

**Example Usage:**

```python
from src.scheduling import JobScheduler, JobConfig

# Initialize scheduler
scheduler = JobScheduler()

# Define a job function
async def collect_data():
    print("Collecting data...")
    # Your data collection logic here

# Add a job with cron trigger (runs every hour)
job_config = JobConfig(
    job_id="hourly_collection",
    name="Hourly Data Collection",
    func=collect_data,
    trigger_type="cron",
    trigger_args={"minute": "0"},
    description="Collect data every hour"
)
scheduler.add_job(job_config)

# Start the scheduler
scheduler.start()

# Get job status
status = scheduler.get_job_status("hourly_collection")
print(f"Next run: {status['next_run_time']}")

# List all jobs
jobs = scheduler.list_jobs()
for job in jobs:
    print(f"{job['name']}: {job['trigger']}")
```

### 2. Standard Jobs Configuration

The scheduler provides a convenient method to configure all standard Wikipedia Intelligence System jobs:

```python
from src.scheduling import JobScheduler

scheduler = JobScheduler()

# Configure standard jobs
scheduler.configure_standard_jobs(
    pageview_collector_func=collect_pageviews,
    edit_scraper_func=scrape_edits,
    model_retraining_func=retrain_models,
    deep_crawl_func=perform_crawl
)

# This creates:
# - Hourly pageview collection (at minute 0)
# - Daily edit history scraping (at 2:00 AM)
# - Weekly model retraining (Sunday at 3:00 AM)
# - Daily deep crawls (at 4:00 AM)
```

### 3. DataCollectionOrchestrator

The `DataCollectionOrchestrator` coordinates data collection jobs and provides health monitoring.

**Key Features:**
- Pageview collection with ETL pipeline integration
- Edit history scraping with error handling
- Deep crawling with checkpoint support
- Health metrics tracking
- Automatic metric storage in Redis

**Example Usage:**

```python
from src.scheduling import DataCollectionOrchestrator
from src.storage import Database, RedisCache

# Initialize components
db = Database()
cache = RedisCache()

# Create orchestrator
orchestrator = DataCollectionOrchestrator(
    db=db,
    cache=cache,
    pageviews_collector=pageviews_collector,
    edit_scraper=edit_scraper,
    crawler=crawler
)

# Collect pageviews for specific articles
metrics = await orchestrator.collect_pageviews(
    articles=["Python_(programming_language)", "Machine_learning"]
)

print(f"Status: {metrics.status}")
print(f"Records processed: {metrics.records_processed}")
print(f"Execution time: {metrics.execution_time_seconds}s")

# Get health status
health = orchestrator.get_health_status()
print(f"Success rate: {health['success_rate']:.2%}")
```

### 4. AnalyticsPipelineOrchestrator

The `AnalyticsPipelineOrchestrator` coordinates analytics jobs including forecasting, clustering, and hype detection.

**Example Usage:**

```python
from src.scheduling import AnalyticsPipelineOrchestrator

# Create orchestrator
orchestrator = AnalyticsPipelineOrchestrator(
    db=db,
    cache=cache
)

# Retrain forecasting models
metrics = await orchestrator.retrain_models(
    articles=["Python_(programming_language)", "Machine_learning"]
)

# Run complete analytics pipeline
metrics = await orchestrator.run_analytics_pipeline(
    articles=["Python_(programming_language)"]
)
```

## Job Configuration

### Cron Triggers

Cron triggers use standard cron syntax:

```python
# Every hour at minute 0
trigger_args={"minute": "0"}

# Every day at 2:00 AM
trigger_args={"hour": "2", "minute": "0"}

# Every Sunday at 3:00 AM
trigger_args={"day_of_week": "sun", "hour": "3", "minute": "0"}

# Every 15 minutes
trigger_args={"minute": "*/15"}
```

### Interval Triggers

Interval triggers run at fixed intervals:

```python
# Every 30 minutes
trigger_args={"minutes": 30}

# Every 2 hours
trigger_args={"hours": 2}

# Every day
trigger_args={"days": 1}
```

## Health Monitoring

### Job Health Metrics

Each job execution generates health metrics:

```python
class JobHealthMetrics:
    job_name: str
    start_time: datetime
    end_time: Optional[datetime]
    status: str  # "running", "success", "failed"
    records_processed: int
    errors: List[str]
    execution_time_seconds: float
```

### Accessing Health Status

```python
# Get orchestrator health status
health = orchestrator.get_health_status()

print(f"Total jobs: {health['total_jobs']}")
print(f"Successful: {health['successful']}")
print(f"Failed: {health['failed']}")
print(f"Success rate: {health['success_rate']:.2%}")

# View recent job details
for job in health['recent_jobs']:
    print(f"{job['job_name']}: {job['status']} ({job['execution_time_seconds']:.2f}s)")
```

## Error Handling

The scheduling system includes comprehensive error handling:

1. **Job Failures**: Jobs that fail don't crash the scheduler; errors are logged and tracked
2. **Partial Failures**: If some articles fail during collection, successful ones are still processed
3. **Retry Logic**: Failed operations can be retried based on configuration
4. **Health Metrics**: All failures are recorded in health metrics for monitoring

## Best Practices

1. **Use Async Functions**: All job functions should be async for non-blocking execution
2. **Monitor Health**: Regularly check health status to detect issues early
3. **Set Appropriate Intervals**: Balance data freshness with API rate limits
4. **Handle Errors Gracefully**: Log errors but allow jobs to continue
5. **Use Checkpoints**: For long-running operations, implement checkpointing
6. **Test Schedules**: Test job schedules in development before production deployment

## Production Deployment

### Running the Scheduler

```python
import asyncio
from src.scheduling import JobScheduler, DataCollectionOrchestrator

async def main():
    # Initialize components
    scheduler = JobScheduler()
    orchestrator = DataCollectionOrchestrator(db, cache)
    
    # Configure jobs
    scheduler.configure_standard_jobs(
        pageview_collector_func=orchestrator.collect_pageviews,
        edit_scraper_func=orchestrator.scrape_edit_history,
        model_retraining_func=analytics_orchestrator.retrain_models,
        deep_crawl_func=orchestrator.perform_deep_crawl
    )
    
    # Start scheduler
    scheduler.start()
    
    # Keep running
    try:
        while True:
            await asyncio.sleep(60)
    except KeyboardInterrupt:
        scheduler.shutdown(wait=True)

if __name__ == "__main__":
    asyncio.run(main())
```

### Monitoring

Monitor job execution through:
- Health status endpoints
- Redis metrics (stored with 24-hour TTL)
- Application logs (structured JSON format)
- APScheduler event listeners

## Troubleshooting

### Common Issues

1. **"No running event loop" error**
   - Ensure scheduler is started within an async context
   - Use `asyncio.run()` or create an event loop

2. **Jobs not executing**
   - Check scheduler is running: `scheduler.scheduler.running`
   - Verify job was added: `scheduler.list_jobs()`
   - Check logs for errors

3. **High failure rate**
   - Review health metrics for error patterns
   - Check API rate limits
   - Verify database connectivity

4. **Memory issues**
   - Limit concurrent job instances
   - Implement proper cleanup in job functions
   - Monitor health metrics list size

## References

- [APScheduler Documentation](https://apscheduler.readthedocs.io/)
- [Wikipedia Intelligence System Design](../design.md)
- [ETL Pipeline Documentation](./etl_guide.md)
