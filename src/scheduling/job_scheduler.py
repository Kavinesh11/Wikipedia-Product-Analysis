"""
Job scheduler using APScheduler for Wikipedia Intelligence System.

This module provides scheduled job execution for:
- Hourly pageview collection
- Daily edit history scraping
- Weekly model retraining
- Daily deep crawls for new articles
"""

import logging
from dataclasses import dataclass
from datetime import datetime
from typing import Callable, Dict, Optional, Any
from apscheduler.schedulers.asyncio import AsyncIOScheduler
from apscheduler.triggers.cron import CronTrigger
from apscheduler.triggers.interval import IntervalTrigger
from apscheduler.events import EVENT_JOB_EXECUTED, EVENT_JOB_ERROR
import asyncio

logger = logging.getLogger(__name__)


@dataclass
class JobConfig:
    """Configuration for a scheduled job."""
    
    job_id: str
    name: str
    func: Callable
    trigger_type: str  # "cron" or "interval"
    trigger_args: Dict[str, Any]
    description: str = ""
    max_instances: int = 1
    coalesce: bool = True
    misfire_grace_time: int = 300  # 5 minutes


class JobScheduler:
    """
    Job scheduler for Wikipedia Intelligence System.
    
    Manages scheduled execution of data collection, processing, and analytics jobs.
    Uses APScheduler with AsyncIO support for non-blocking job execution.
    """
    
    def __init__(self):
        """Initialize the job scheduler."""
        self.scheduler = AsyncIOScheduler()
        self.jobs: Dict[str, JobConfig] = {}
        self._setup_event_listeners()
        logger.info("JobScheduler initialized")
    
    def _setup_event_listeners(self):
        """Set up event listeners for job execution monitoring."""
        self.scheduler.add_listener(
            self._job_executed_listener,
            EVENT_JOB_EXECUTED
        )
        self.scheduler.add_listener(
            self._job_error_listener,
            EVENT_JOB_ERROR
        )
    
    def _job_executed_listener(self, event):
        """Handle successful job execution events."""
        job_id = event.job_id
        logger.info(
            f"Job executed successfully",
            extra={
                "job_id": job_id,
                "scheduled_run_time": event.scheduled_run_time,
                "retval": str(event.retval)[:100] if event.retval else None
            }
        )
    
    def _job_error_listener(self, event):
        """Handle job execution errors."""
        job_id = event.job_id
        logger.error(
            f"Job execution failed",
            extra={
                "job_id": job_id,
                "exception": str(event.exception),
                "traceback": event.traceback
            },
            exc_info=event.exception
        )
    
    def add_job(self, job_config: JobConfig) -> None:
        """
        Add a job to the scheduler.
        
        Args:
            job_config: Configuration for the job to add
        """
        # Create trigger based on type
        if job_config.trigger_type == "cron":
            trigger = CronTrigger(**job_config.trigger_args)
        elif job_config.trigger_type == "interval":
            trigger = IntervalTrigger(**job_config.trigger_args)
        else:
            raise ValueError(f"Unknown trigger type: {job_config.trigger_type}")
        
        # Add job to scheduler
        self.scheduler.add_job(
            job_config.func,
            trigger=trigger,
            id=job_config.job_id,
            name=job_config.name,
            max_instances=job_config.max_instances,
            coalesce=job_config.coalesce,
            misfire_grace_time=job_config.misfire_grace_time
        )
        
        # Store job config
        self.jobs[job_config.job_id] = job_config
        
        logger.info(
            f"Job added to scheduler",
            extra={
                "job_id": job_config.job_id,
                "name": job_config.name,
                "trigger_type": job_config.trigger_type,
                "description": job_config.description
            }
        )
    
    def remove_job(self, job_id: str) -> None:
        """
        Remove a job from the scheduler.
        
        Args:
            job_id: ID of the job to remove
        """
        if job_id in self.jobs:
            self.scheduler.remove_job(job_id)
            del self.jobs[job_id]
            logger.info(f"Job removed from scheduler", extra={"job_id": job_id})
        else:
            logger.warning(f"Job not found", extra={"job_id": job_id})
    
    def start(self) -> None:
        """Start the scheduler."""
        if not self.scheduler.running:
            self.scheduler.start()
            logger.info("JobScheduler started")
        else:
            logger.warning("JobScheduler already running")
    
    def shutdown(self, wait: bool = True) -> None:
        """
        Shutdown the scheduler.
        
        Args:
            wait: Whether to wait for running jobs to complete
        """
        if self.scheduler.running:
            self.scheduler.shutdown(wait=wait)
            logger.info("JobScheduler shutdown", extra={"wait": wait})
        else:
            logger.warning("JobScheduler not running")
    
    def get_job_status(self, job_id: str) -> Optional[Dict[str, Any]]:
        """
        Get status information for a job.
        
        Args:
            job_id: ID of the job
            
        Returns:
            Dictionary with job status information, or None if job not found
        """
        job = self.scheduler.get_job(job_id)
        if job:
            return {
                "job_id": job.id,
                "name": job.name,
                "next_run_time": str(job.next_run_time) if hasattr(job, 'next_run_time') else None,
                "trigger": str(job.trigger),
                "pending": job.pending if hasattr(job, 'pending') else None
            }
        return None
    
    def list_jobs(self) -> list[Dict[str, Any]]:
        """
        List all scheduled jobs.
        
        Returns:
            List of job status dictionaries
        """
        jobs = []
        for job in self.scheduler.get_jobs():
            jobs.append({
                "job_id": job.id,
                "name": job.name,
                "next_run_time": str(job.next_run_time) if hasattr(job, 'next_run_time') else None,
                "trigger": str(job.trigger)
            })
        return jobs
    
    def configure_standard_jobs(
        self,
        pageview_collector_func: Callable,
        edit_scraper_func: Callable,
        model_retraining_func: Callable,
        deep_crawl_func: Callable
    ) -> None:
        """
        Configure standard jobs for Wikipedia Intelligence System.
        
        Args:
            pageview_collector_func: Function for hourly pageview collection
            edit_scraper_func: Function for daily edit history scraping
            model_retraining_func: Function for weekly model retraining
            deep_crawl_func: Function for daily deep crawls
        """
        # Hourly pageview collection (at minute 0 of every hour)
        self.add_job(JobConfig(
            job_id="pageview_collection",
            name="Hourly Pageview Collection",
            func=pageview_collector_func,
            trigger_type="cron",
            trigger_args={"minute": "0"},
            description="Collect pageview statistics from Wikimedia API every hour"
        ))
        
        # Daily edit history scraping (at 2:00 AM every day)
        self.add_job(JobConfig(
            job_id="edit_history_scraping",
            name="Daily Edit History Scraping",
            func=edit_scraper_func,
            trigger_type="cron",
            trigger_args={"hour": "2", "minute": "0"},
            description="Scrape edit history for monitored articles daily"
        ))
        
        # Weekly model retraining (Sunday at 3:00 AM)
        self.add_job(JobConfig(
            job_id="model_retraining",
            name="Weekly Model Retraining",
            func=model_retraining_func,
            trigger_type="cron",
            trigger_args={"day_of_week": "sun", "hour": "3", "minute": "0"},
            description="Retrain forecasting models with latest data weekly"
        ))
        
        # Daily deep crawls (at 4:00 AM every day)
        self.add_job(JobConfig(
            job_id="deep_crawl",
            name="Daily Deep Crawl",
            func=deep_crawl_func,
            trigger_type="cron",
            trigger_args={"hour": "4", "minute": "0"},
            description="Perform deep crawls for new articles daily"
        ))
        
        logger.info("Standard jobs configured")
