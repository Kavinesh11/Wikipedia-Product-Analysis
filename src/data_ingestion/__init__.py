"""Data Ingestion Layer - Collectors for Wikimedia APIs and web crawling"""

from .edit_history_scraper import EditHistoryScraper
from .api_client import WikimediaAPIClient
from .rate_limiter import RateLimiter
from .crawl4ai_pipeline import Crawl4AIPipeline, ExtractionConfig, CrawlCheckpoint

__all__ = [
    "EditHistoryScraper",
    "WikimediaAPIClient",
    "RateLimiter",
    "Crawl4AIPipeline",
    "ExtractionConfig",
    "CrawlCheckpoint",
]
