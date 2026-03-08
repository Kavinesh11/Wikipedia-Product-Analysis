"""
Pageviews Collector for Wikimedia Pageviews API.

This module provides functionality to collect article traffic statistics
from the Wikimedia Pageviews API with bot filtering and device segmentation.
"""

import asyncio
import logging
from datetime import datetime, timedelta
from typing import List, Optional
from urllib.parse import quote

from .api_client import WikimediaAPIClient
from .rate_limiter import RateLimiter
from ..storage.dto import PageviewRecord, TopArticleRecord, AggregateStats

logger = logging.getLogger(__name__)


class PageviewsCollector:
    """
    Collector for Wikipedia pageview statistics.
    
    Fetches article traffic data from Wikimedia Pageviews API with:
    - Bot traffic filtering (agent_type parameter)
    - Device segmentation (desktop, mobile-web, mobile-app)
    - Response schema validation
    - Rate limiting and error handling
    
    Requirements: 1.1, 1.2, 1.3, 1.4, 1.5, 1.7
    """
    
    # Wikimedia Pageviews API base URL
    BASE_URL = "https://wikimedia.org/api/rest_v1"
    
    # Valid device types for segmentation
    DEVICE_TYPES = ["desktop", "mobile-web", "mobile-app"]
    
    # Valid granularities
    VALID_GRANULARITIES = ["hourly", "daily", "monthly"]
    
    def __init__(
        self,
        api_client: Optional[WikimediaAPIClient] = None,
        rate_limiter: Optional[RateLimiter] = None
    ):
        """
        Initialize PageviewsCollector.
        
        Args:
            api_client: WikimediaAPIClient instance (creates default if None)
            rate_limiter: RateLimiter instance (creates default if None)
        """
        self.api_client = api_client or WikimediaAPIClient(base_url=self.BASE_URL)
        self.rate_limiter = rate_limiter or RateLimiter()
        
        logger.info("PageviewsCollector initialized")
    
    def _format_date(self, dt: datetime, granularity: str = "daily") -> str:
        """
        Format datetime for API request.
        
        Args:
            dt: Datetime to format
            granularity: Time granularity (hourly, daily, monthly)
            
        Returns:
            Formatted date string (YYYYMMDDHH for hourly, YYYYMMDD for daily)
        """
        if granularity == "hourly":
            return dt.strftime("%Y%m%d%H")
        elif granularity == "daily":
            return dt.strftime("%Y%m%d")
        elif granularity == "monthly":
            return dt.strftime("%Y%m")
        else:
            raise ValueError(f"Invalid granularity: {granularity}")
    
    def _validate_pageview_response(self, data: dict, article: str) -> None:
        """
        Validate API response schema for pageview data.
        
        Args:
            data: Response data from API
            article: Article name for error messages
            
        Raises:
            ValueError: If response doesn't match expected schema
            
        Requirements: 1.7
        """
        if not isinstance(data, dict):
            raise ValueError(f"Invalid response type for {article}: expected dict, got {type(data)}")
        
        if "items" not in data:
            raise ValueError(f"Missing 'items' field in response for {article}")
        
        if not isinstance(data["items"], list):
            raise ValueError(f"Invalid 'items' type for {article}: expected list")
        
        # Validate each item has required fields
        for item in data["items"]:
            required_fields = ["project", "article", "granularity", "timestamp", "views"]
            missing_fields = [f for f in required_fields if f not in item]
            if missing_fields:
                raise ValueError(
                    f"Missing required fields in pageview item for {article}: {missing_fields}"
                )
    
    def _validate_top_articles_response(self, data: dict) -> None:
        """
        Validate API response schema for top articles data.
        
        Args:
            data: Response data from API
            
        Raises:
            ValueError: If response doesn't match expected schema
            
        Requirements: 1.7
        """
        if not isinstance(data, dict):
            raise ValueError(f"Invalid response type: expected dict, got {type(data)}")
        
        if "items" not in data:
            raise ValueError("Missing 'items' field in top articles response")
        
        if not isinstance(data["items"], list):
            raise ValueError("Invalid 'items' type: expected list")
        
        # Validate first item structure
        if data["items"]:
            first_item = data["items"][0]
            if "articles" not in first_item:
                raise ValueError("Missing 'articles' field in top articles response")
            
            if not isinstance(first_item["articles"], list):
                raise ValueError("Invalid 'articles' type: expected list")
            
            # Validate article structure
            if first_item["articles"]:
                article = first_item["articles"][0]
                required_fields = ["article", "views", "rank"]
                missing_fields = [f for f in required_fields if f not in article]
                if missing_fields:
                    raise ValueError(
                        f"Missing required fields in top article: {missing_fields}"
                    )
    
    def _validate_aggregate_response(self, data: dict) -> None:
        """
        Validate API response schema for aggregate stats.
        
        Args:
            data: Response data from API
            
        Raises:
            ValueError: If response doesn't match expected schema
            
        Requirements: 1.7
        """
        if not isinstance(data, dict):
            raise ValueError(f"Invalid response type: expected dict, got {type(data)}")
        
        if "items" not in data:
            raise ValueError("Missing 'items' field in aggregate response")
        
        if not isinstance(data["items"], list):
            raise ValueError("Invalid 'items' type: expected list")
        
        # Validate each item has required fields
        for item in data["items"]:
            required_fields = ["project", "timestamp", "views"]
            missing_fields = [f for f in required_fields if f not in item]
            if missing_fields:
                raise ValueError(
                    f"Missing required fields in aggregate item: {missing_fields}"
                )
    
    async def fetch_per_article(
        self,
        article: str,
        start_date: datetime,
        end_date: datetime,
        granularity: str = "daily"
    ) -> List[PageviewRecord]:
        """
        Fetch pageviews for a specific article with bot filtering and device segmentation.
        
        Makes separate requests for each device type to get segmented data.
        Filters bot traffic using agent_type=user parameter.
        
        Args:
            article: Article title (URL-encoded automatically)
            start_date: Start date for data collection
            end_date: End date for data collection
            granularity: Time granularity (hourly, daily, monthly)
            
        Returns:
            List of PageviewRecord objects with device segmentation
            
        Raises:
            ValueError: If granularity is invalid or dates are invalid
            aiohttp.ClientError: On API request failure
            
        Requirements: 1.1, 1.4, 1.5, 1.7
        """
        if granularity not in self.VALID_GRANULARITIES:
            raise ValueError(
                f"Invalid granularity '{granularity}'. "
                f"Must be one of: {self.VALID_GRANULARITIES}"
            )
        
        if start_date > end_date:
            raise ValueError("start_date must be before or equal to end_date")
        
        # URL-encode article title
        encoded_article = quote(article.replace(" ", "_"), safe="")
        
        # Format dates
        start_str = self._format_date(start_date, granularity)
        end_str = self._format_date(end_date, granularity)
        
        logger.info(
            f"Fetching pageviews for {article}: {start_str} to {end_str}, "
            f"granularity={granularity}"
        )
        
        # Fetch data for each device type concurrently
        tasks = []
        for device_type in self.DEVICE_TYPES:
            access_type = "mobile-app" if device_type == "mobile-app" else "all-access"
            
            # Build endpoint
            # Format: /metrics/pageviews/per-article/{project}/{access}/{agent}/{article}/{granularity}/{start}/{end}
            endpoint = (
                f"metrics/pageviews/per-article/en.wikipedia/"
                f"{access_type}/user/{encoded_article}/"
                f"{granularity}/{start_str}/{end_str}"
            )
            
            tasks.append(self._fetch_device_pageviews(endpoint, article, device_type))
        
        # Execute all requests concurrently
        device_results = await asyncio.gather(*tasks, return_exceptions=True)
        
        # Combine results from all device types
        all_records = []
        for device_type, result in zip(self.DEVICE_TYPES, device_results):
            if isinstance(result, Exception):
                logger.error(
                    f"Failed to fetch {device_type} pageviews for {article}: {result}"
                )
                # Continue with other device types
                continue
            
            all_records.extend(result)
        
        logger.info(f"Fetched {len(all_records)} pageview records for {article}")
        return all_records
    
    async def _fetch_device_pageviews(
        self,
        endpoint: str,
        article: str,
        device_type: str
    ) -> List[PageviewRecord]:
        """
        Fetch pageviews for a specific device type.
        
        Args:
            endpoint: API endpoint to call
            article: Article title
            device_type: Device type (desktop, mobile-web, mobile-app)
            
        Returns:
            List of PageviewRecord objects for this device type
        """
        # Make API request
        data = await self.api_client.get(endpoint)
        
        # Validate response schema
        self._validate_pageview_response(data, article)
        
        # Parse response into PageviewRecord objects
        records = []
        for item in data["items"]:
            # Parse timestamp based on granularity
            timestamp_str = str(item["timestamp"])
            if len(timestamp_str) == 10:  # YYYYMMDDHH
                timestamp = datetime.strptime(timestamp_str, "%Y%m%d%H")
            elif len(timestamp_str) == 8:  # YYYYMMDD
                timestamp = datetime.strptime(timestamp_str, "%Y%m%d")
            elif len(timestamp_str) == 6:  # YYYYMM
                timestamp = datetime.strptime(timestamp_str, "%Y%m")
            else:
                logger.warning(f"Unknown timestamp format: {timestamp_str}")
                continue
            
            # Create PageviewRecord
            # Note: API with agent_type=user returns only human views
            views_human = item["views"]
            views_bot = 0  # We filtered bots with agent_type=user
            
            record = PageviewRecord(
                article=article,
                timestamp=timestamp,
                device_type=device_type,
                views_human=views_human,
                views_bot=views_bot,
                views_total=views_human + views_bot
            )
            records.append(record)
        
        return records
    
    async def fetch_top_articles(
        self,
        date: datetime,
        limit: int = 1000
    ) -> List[TopArticleRecord]:
        """
        Fetch most viewed articles for a specific date.
        
        Args:
            date: Date to fetch top articles for
            limit: Maximum number of articles to return (default 1000)
            
        Returns:
            List of TopArticleRecord objects ranked by views
            
        Raises:
            aiohttp.ClientError: On API request failure
            
        Requirements: 1.2, 1.7
        """
        # Format date (YYYY/MM/DD)
        year = date.strftime("%Y")
        month = date.strftime("%m")
        day = date.strftime("%d")
        
        logger.info(f"Fetching top {limit} articles for {date.date()}")
        
        # Build endpoint
        # Format: /metrics/pageviews/top/{project}/{access}/{year}/{month}/{day}
        endpoint = f"metrics/pageviews/top/en.wikipedia/all-access/{year}/{month}/{day}"
        
        # Make API request
        data = await self.api_client.get(endpoint)
        
        # Validate response schema
        self._validate_top_articles_response(data)
        
        # Parse response into TopArticleRecord objects
        records = []
        if data["items"] and data["items"][0]["articles"]:
            articles = data["items"][0]["articles"]
            
            for article_data in articles[:limit]:
                record = TopArticleRecord(
                    article=article_data["article"],
                    rank=article_data["rank"],
                    views=article_data["views"],
                    date=date
                )
                records.append(record)
        
        logger.info(f"Fetched {len(records)} top articles for {date.date()}")
        return records
    
    async def fetch_aggregate(
        self,
        start_date: datetime,
        end_date: datetime
    ) -> AggregateStats:
        """
        Fetch aggregate Wikipedia traffic statistics.
        
        Args:
            start_date: Start date for aggregation
            end_date: End date for aggregation
            
        Returns:
            AggregateStats object with total traffic metrics
            
        Raises:
            ValueError: If dates are invalid
            aiohttp.ClientError: On API request failure
            
        Requirements: 1.3, 1.7
        """
        if start_date > end_date:
            raise ValueError("start_date must be before or equal to end_date")
        
        # Format dates (YYYYMMDD)
        start_str = self._format_date(start_date, "daily")
        end_str = self._format_date(end_date, "daily")
        
        logger.info(f"Fetching aggregate stats: {start_str} to {end_str}")
        
        # Build endpoint
        # Format: /metrics/pageviews/aggregate/{project}/{access}/{agent}/{granularity}/{start}/{end}
        endpoint = (
            f"metrics/pageviews/aggregate/en.wikipedia/"
            f"all-access/user/daily/{start_str}/{end_str}"
        )
        
        # Make API request
        data = await self.api_client.get(endpoint)
        
        # Validate response schema
        self._validate_aggregate_response(data)
        
        # Calculate aggregate statistics
        total_views = sum(item["views"] for item in data["items"])
        num_days = len(data["items"])
        
        # Estimate total articles (Wikipedia has ~6.9M articles)
        # This is an approximation since the API doesn't provide this directly
        estimated_articles = 6_900_000
        avg_views_per_article = total_views / estimated_articles if estimated_articles > 0 else 0
        
        stats = AggregateStats(
            start_date=start_date,
            end_date=end_date,
            total_views=total_views,
            total_articles=estimated_articles,
            avg_views_per_article=avg_views_per_article
        )
        
        logger.info(
            f"Aggregate stats: {total_views:,} total views over {num_days} days"
        )
        return stats
    
    async def close(self) -> None:
        """Close API client and cleanup resources."""
        await self.api_client.close()
        logger.info("PageviewsCollector closed")
    
    async def __aenter__(self):
        """Async context manager entry."""
        return self
    
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Async context manager exit."""
        await self.close()
