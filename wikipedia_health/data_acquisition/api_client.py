"""Wikimedia API client for fetching pageviews, editor counts, and edit volumes."""

import time
from datetime import date, timedelta
from typing import Dict, List, Tuple, Optional
import logging

import pandas as pd
from pandas import DataFrame
import requests

from wikipedia_health.config.config import load_config


logger = logging.getLogger(__name__)


class WikimediaAPIClient:
    """Client for fetching data from Wikimedia REST APIs.
    
    Implements exponential backoff retry logic and response validation
    for robust data acquisition from Wikimedia metrics APIs.
    """
    
    def __init__(self, config = None):
        """Initialize API client with configuration.
        
        Args:
            config: Optional Config object. If None, loads default config
        """
        from wikipedia_health.config.config import Config
        
        if config is None:
            config = load_config()
        
        self.config = config
        self.api_config = self.config.api
        
        self.pageviews_endpoint = self.api_config.pageviews_endpoint
        self.editors_endpoint = self.api_config.editors_endpoint
        self.edits_endpoint = self.api_config.edits_endpoint
        
        self.timeout = self.api_config.timeout
        self.max_retries = self.api_config.max_retries
        self.backoff_factor = self.api_config.backoff_factor
        self.user_agent = self.api_config.user_agent
        
        self.session = requests.Session()
        self.session.headers.update({'User-Agent': self.user_agent})
    
    def _make_request_with_retry(
        self,
        url: str,
        params: Optional[Dict] = None
    ) -> requests.Response:
        """Make HTTP request with exponential backoff retry logic.
        
        Args:
            url: API endpoint URL
            params: Query parameters
            
        Returns:
            Response object
            
        Raises:
            requests.RequestException: If all retries fail
        """
        last_exception = None
        
        for attempt in range(self.max_retries):
            try:
                response = self.session.get(
                    url,
                    params=params,
                    timeout=self.timeout
                )
                response.raise_for_status()
                return response
                
            except requests.RequestException as e:
                last_exception = e
                wait_time = self.backoff_factor ** attempt
                
                logger.warning(
                    f"Request failed (attempt {attempt + 1}/{self.max_retries}): {e}. "
                    f"Retrying in {wait_time:.1f}s..."
                )
                
                if attempt < self.max_retries - 1:
                    time.sleep(wait_time)
                else:
                    logger.error(f"All {self.max_retries} retry attempts failed")
        
        raise last_exception
    
    def fetch_pageviews(
        self,
        start_date: date,
        end_date: date,
        platforms: List[str],
        agent_type: str = 'user'
    ) -> DataFrame:
        """Fetch pageview data from Wikimedia API.
        
        Args:
            start_date: Start date (inclusive)
            end_date: End date (inclusive)
            platforms: List of platforms ('desktop', 'mobile-web', 'mobile-app')
            agent_type: Agent type filter ('user' to exclude bots, 'all' for all traffic)
            
        Returns:
            DataFrame with columns: date, platform, views, agent_type
            
        Raises:
            requests.RequestException: If API request fails after retries
            ValueError: If date range or platforms are invalid
        """
        if end_date < start_date:
            raise ValueError(f"end_date ({end_date}) must be >= start_date ({start_date})")
        
        valid_platforms = {'desktop', 'mobile-web', 'mobile-app', 'all-access'}
        invalid_platforms = set(platforms) - valid_platforms
        if invalid_platforms:
            raise ValueError(f"Invalid platforms: {invalid_platforms}. Valid: {valid_platforms}")
        
        all_data = []
        
        for platform in platforms:
            # Wikimedia API uses 'all-access' instead of 'all'
            api_platform = 'all-access' if platform == 'all' else platform
            
            # Format dates as YYYYMMDD
            start_str = start_date.strftime('%Y%m%d')
            end_str = end_date.strftime('%Y%m%d')
            
            # Construct URL for aggregate endpoint
            url = f"{self.pageviews_endpoint}/aggregate/all-projects/{api_platform}/{agent_type}/daily/{start_str}/{end_str}"
            
            logger.info(f"Fetching pageviews for platform={platform}, dates={start_date} to {end_date}")
            
            try:
                response = self._make_request_with_retry(url)
                data = response.json()
                
                # Extract items from response
                items = data.get('items', [])
                
                for item in items:
                    # Parse timestamp (format: YYYYMMDD00)
                    timestamp_str = item.get('timestamp', '')
                    if len(timestamp_str) >= 8:
                        item_date = pd.to_datetime(timestamp_str[:8], format='%Y%m%d').date()
                    else:
                        logger.warning(f"Invalid timestamp format: {timestamp_str}")
                        continue
                    
                    all_data.append({
                        'date': item_date,
                        'platform': platform,
                        'views': item.get('views', 0),
                        'agent_type': agent_type
                    })
                
                logger.info(f"Successfully fetched {len(items)} records for platform={platform}")
                
            except requests.RequestException as e:
                logger.error(f"Failed to fetch pageviews for platform={platform}: {e}")
                raise
        
        if not all_data:
            logger.warning("No pageview data retrieved")
            return pd.DataFrame(columns=['date', 'platform', 'views', 'agent_type'])
        
        df = pd.DataFrame(all_data)
        df['date'] = pd.to_datetime(df['date'])
        
        return df
    
    def fetch_editor_counts(
        self,
        start_date: date,
        end_date: date,
        granularity: str = 'daily'
    ) -> DataFrame:
        """Fetch active editor counts from Wikimedia API.
        
        Args:
            start_date: Start date (inclusive)
            end_date: End date (inclusive)
            granularity: Time granularity ('daily', 'monthly')
            
        Returns:
            DataFrame with columns: date, editors, granularity
            
        Raises:
            requests.RequestException: If API request fails after retries
            ValueError: If date range or granularity is invalid
        """
        if end_date < start_date:
            raise ValueError(f"end_date ({end_date}) must be >= start_date ({start_date})")
        
        if granularity not in ['daily', 'monthly']:
            raise ValueError(f"Invalid granularity: {granularity}. Must be 'daily' or 'monthly'")
        
        # Format dates as YYYYMMDD
        start_str = start_date.strftime('%Y%m%d')
        end_str = end_date.strftime('%Y%m%d')
        
        # Construct URL for editors endpoint
        url = f"{self.editors_endpoint}/aggregate/all-projects/all-editor-types/all-page-types/{granularity}/{start_str}/{end_str}"
        
        logger.info(f"Fetching editor counts for dates={start_date} to {end_date}, granularity={granularity}")
        
        try:
            response = self._make_request_with_retry(url)
            data = response.json()
            
            items = data.get('items', [])
            
            all_data = []
            for item in items:
                # Parse timestamp
                timestamp_str = item.get('timestamp', '')
                if len(timestamp_str) >= 8:
                    item_date = pd.to_datetime(timestamp_str[:8], format='%Y%m%d').date()
                else:
                    logger.warning(f"Invalid timestamp format: {timestamp_str}")
                    continue
                
                all_data.append({
                    'date': item_date,
                    'editors': item.get('editors', 0),
                    'granularity': granularity
                })
            
            logger.info(f"Successfully fetched {len(items)} editor count records")
            
            if not all_data:
                logger.warning("No editor data retrieved")
                return pd.DataFrame(columns=['date', 'editors', 'granularity'])
            
            df = pd.DataFrame(all_data)
            df['date'] = pd.to_datetime(df['date'])
            
            return df
            
        except requests.RequestException as e:
            logger.error(f"Failed to fetch editor counts: {e}")
            raise
    
    def fetch_edit_volumes(
        self,
        start_date: date,
        end_date: date,
        granularity: str = 'daily'
    ) -> DataFrame:
        """Fetch edit volume data from Wikimedia API.
        
        Args:
            start_date: Start date (inclusive)
            end_date: End date (inclusive)
            granularity: Time granularity ('daily', 'monthly')
            
        Returns:
            DataFrame with columns: date, edits, granularity
            
        Raises:
            requests.RequestException: If API request fails after retries
            ValueError: If date range or granularity is invalid
        """
        if end_date < start_date:
            raise ValueError(f"end_date ({end_date}) must be >= start_date ({start_date})")
        
        if granularity not in ['daily', 'monthly']:
            raise ValueError(f"Invalid granularity: {granularity}. Must be 'daily' or 'monthly'")
        
        # Format dates as YYYYMMDD
        start_str = start_date.strftime('%Y%m%d')
        end_str = end_date.strftime('%Y%m%d')
        
        # Construct URL for edits endpoint
        url = f"{self.edits_endpoint}/aggregate/all-projects/all-editor-types/all-page-types/{granularity}/{start_str}/{end_str}"
        
        logger.info(f"Fetching edit volumes for dates={start_date} to {end_date}, granularity={granularity}")
        
        try:
            response = self._make_request_with_retry(url)
            data = response.json()
            
            items = data.get('items', [])
            
            all_data = []
            for item in items:
                # Parse timestamp
                timestamp_str = item.get('timestamp', '')
                if len(timestamp_str) >= 8:
                    item_date = pd.to_datetime(timestamp_str[:8], format='%Y%m%d').date()
                else:
                    logger.warning(f"Invalid timestamp format: {timestamp_str}")
                    continue
                
                all_data.append({
                    'date': item_date,
                    'edits': item.get('edits', 0),
                    'granularity': granularity
                })
            
            logger.info(f"Successfully fetched {len(items)} edit volume records")
            
            if not all_data:
                logger.warning("No edit volume data retrieved")
                return pd.DataFrame(columns=['date', 'edits', 'granularity'])
            
            df = pd.DataFrame(all_data)
            df['date'] = pd.to_datetime(df['date'])
            
            return df
            
        except requests.RequestException as e:
            logger.error(f"Failed to fetch edit volumes: {e}")
            raise
    
    def validate_response(self, response_data: Dict) -> Tuple[bool, List[str]]:
        """Validate API response structure and content.
        
        Args:
            response_data: Parsed JSON response from API
            
        Returns:
            Tuple of (is_valid, error_messages)
        """
        errors = []
        
        if not isinstance(response_data, dict):
            errors.append("Response is not a dictionary")
            return False, errors
        
        if 'items' not in response_data:
            errors.append("Response missing 'items' field")
            return False, errors
        
        items = response_data.get('items', [])
        if not isinstance(items, list):
            errors.append("'items' field is not a list")
            return False, errors
        
        if len(items) == 0:
            errors.append("Response contains no data items")
            # This is a warning, not necessarily invalid
        
        # Validate structure of first item if present
        if items:
            first_item = items[0]
            if not isinstance(first_item, dict):
                errors.append("Items are not dictionaries")
                return False, errors
            
            if 'timestamp' not in first_item:
                errors.append("Items missing 'timestamp' field")
                return False, errors
        
        is_valid = len(errors) == 0 or (len(errors) == 1 and "no data items" in errors[0])
        
        return is_valid, errors
