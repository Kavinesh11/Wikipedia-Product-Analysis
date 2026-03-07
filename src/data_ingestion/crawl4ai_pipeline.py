"""
Crawl4AI Pipeline for Wikipedia article content extraction.

This module provides asynchronous web crawling functionality using Crawl4AI
to extract structured content from Wikipedia articles including summaries,
infoboxes, tables, categories, and internal links.
"""

import asyncio
import time
from typing import List, Dict, Any, Optional, Set
from dataclasses import dataclass
from datetime import datetime
import logging
from collections import deque
from urllib.parse import urljoin, urlparse
import re

import pandas as pd
from bs4 import BeautifulSoup
from crawl4ai import AsyncWebCrawler, BrowserConfig, CrawlerRunConfig, CacheMode

from ..storage.dto import ArticleContent
from .rate_limiter import RateLimiter, RateLimiterConfig

logger = logging.getLogger(__name__)


@dataclass
class ExtractionConfig:
    """Configuration for content extraction."""
    extract_summary: bool = True
    extract_infobox: bool = True
    extract_tables: bool = True
    extract_categories: bool = True
    extract_internal_links: bool = True
    summary_selector: str = "div.mw-parser-output > p"
    infobox_selector: str = "table.infobox"
    table_selector: str = "table.wikitable"
    category_selector: str = "div#mw-normal-catlinks a"
    internal_link_selector: str = "div.mw-parser-output a[href^='/wiki/']"


@dataclass
class CrawlCheckpoint:
    """Checkpoint for resuming deep crawls."""
    seed_url: str
    visited_urls: Set[str]
    queue: deque
    crawled_articles: List[ArticleContent]
    timestamp: datetime
    depth_map: Dict[str, int]  # url -> depth


class Crawl4AIPipeline:
    """
    Asynchronous web crawler for Wikipedia articles using Crawl4AI.
    
    Features:
    - Async crawling for high throughput
    - BFS traversal for deep crawls
    - CSS selector-based extraction
    - Robots.txt compliance
    - Rate limiting
    - Error handling and retry logic
    - Checkpointing for long-running operations
    
    Requirements: 3.1, 3.2, 3.3, 3.4, 3.6, 3.7, 3.8
    """
    
    def __init__(
        self,
        max_concurrent: int = 10,
        rate_limiter: Optional[RateLimiter] = None,
        extraction_config: Optional[ExtractionConfig] = None,
        timeout: float = 30.0,
        max_retries: int = 3,
        respect_robots_txt: bool = True,
        crawl_delay: float = 1.0
    ):
        """
        Initialize Crawl4AI pipeline.
        
        Args:
            max_concurrent: Maximum concurrent crawl operations
            rate_limiter: Rate limiter instance (creates default if None)
            extraction_config: Content extraction configuration
            timeout: Request timeout in seconds
            max_retries: Maximum retry attempts for failed crawls
            respect_robots_txt: Whether to respect robots.txt directives
            crawl_delay: Delay between requests to same domain (seconds)
        """
        self.max_concurrent = max_concurrent
        self.rate_limiter = rate_limiter or RateLimiter(
            RateLimiterConfig(max_requests_per_second=10.0)  # Conservative for crawling
        )
        self.extraction_config = extraction_config or ExtractionConfig()
        self.timeout = timeout
        self.max_retries = max_retries
        self.respect_robots_txt = respect_robots_txt
        self.crawl_delay = crawl_delay
        
        # Crawler will be initialized lazily
        self._crawler: Optional[AsyncWebCrawler] = None
        
        # Track last request time per domain for rate limiting
        self._last_request_time: Dict[str, float] = {}
        
        logger.info(
            f"Crawl4AIPipeline initialized: max_concurrent={max_concurrent}, "
            f"timeout={timeout}s, respect_robots_txt={respect_robots_txt}"
        )
    
    async def _get_crawler(self) -> AsyncWebCrawler:
        """Get or create AsyncWebCrawler instance."""
        if self._crawler is None:
            browser_config = BrowserConfig(
                headless=True,
                verbose=False,
                extra_args=["--disable-gpu", "--disable-dev-shm-usage"]
            )
            self._crawler = AsyncWebCrawler(config=browser_config)
            await self._crawler.__aenter__()
            logger.debug("Created new AsyncWebCrawler instance")
        return self._crawler
    
    async def _apply_crawl_delay(self, url: str) -> None:
        """
        Apply crawl delay for domain rate limiting.
        
        Args:
            url: URL being crawled
        """
        domain = urlparse(url).netloc
        
        if domain in self._last_request_time:
            elapsed = time.time() - self._last_request_time[domain]
            if elapsed < self.crawl_delay:
                delay = self.crawl_delay - elapsed
                logger.debug(f"Applying crawl delay: {delay:.2f}s for {domain}")
                await asyncio.sleep(delay)
        
        self._last_request_time[domain] = time.time()
    
    async def crawl_article(
        self,
        url: str,
        extract_config: Optional[ExtractionConfig] = None
    ) -> ArticleContent:
        """
        Crawl a single Wikipedia article with async support.
        
        Extracts structured content including summary, infobox, tables,
        categories, and internal links based on configuration.
        
        Args:
            url: Wikipedia article URL
            extract_config: Override default extraction config
            
        Returns:
            ArticleContent with extracted data
            
        Raises:
            RuntimeError: If crawl fails after all retries
            
        Requirements: 3.1, 3.4, 3.6, 3.7, 3.8
        """
        config = extract_config or self.extraction_config
        
        for attempt in range(self.max_retries):
            try:
                # Acquire rate limit token
                await self.rate_limiter.acquire()
                
                # Apply domain-specific crawl delay
                await self._apply_crawl_delay(url)
                
                # Log crawl start
                crawl_start = time.time()
                logger.info(
                    f"Crawling article: {url} (attempt {attempt + 1}/{self.max_retries})",
                    extra={"url": url, "attempt": attempt + 1}
                )
                
                # Get crawler instance
                crawler = await self._get_crawler()
                
                # Configure crawl
                run_config = CrawlerRunConfig(
                    cache_mode=CacheMode.BYPASS,
                    page_timeout=int(self.timeout * 1000),  # Convert to milliseconds
                    wait_for_images=False,
                    process_iframes=False
                )
                
                # Perform crawl
                result = await crawler.arun(url=url, config=run_config)
                
                crawl_duration = time.time() - crawl_start
                logger.info(
                    f"Crawl completed: {url} ({crawl_duration:.2f}s)",
                    extra={"url": url, "duration_seconds": crawl_duration}
                )
                
                # Check if crawl was successful
                if not result.success:
                    error_msg = f"Crawl failed for {url}: {result.error_message}"
                    logger.error(error_msg, extra={"url": url, "error": result.error_message})
                    
                    if attempt < self.max_retries - 1:
                        # Retry with exponential backoff
                        delay = 2 ** attempt
                        logger.info(f"Retrying after {delay}s...")
                        await asyncio.sleep(delay)
                        continue
                    else:
                        raise RuntimeError(error_msg)
                
                # Extract content from HTML
                html = result.html
                soup = BeautifulSoup(html, 'lxml')
                
                # Extract title from URL or page
                title = self._extract_title(url, soup)
                
                # Extract components based on config
                summary = self._extract_summary(soup, config) if config.extract_summary else ""
                infobox = self.extract_infobox(html) if config.extract_infobox else {}
                tables = self.extract_tables(html) if config.extract_tables else []
                categories = self._extract_categories(soup, config) if config.extract_categories else []
                internal_links = self.extract_internal_links(html) if config.extract_internal_links else []
                
                # Create ArticleContent DTO
                article_content = ArticleContent(
                    title=title,
                    url=url,
                    summary=summary,
                    infobox=infobox,
                    tables=tables,
                    categories=categories,
                    internal_links=internal_links,
                    crawl_timestamp=datetime.now()
                )
                
                logger.info(
                    f"Extracted content from {url}: "
                    f"summary={len(summary)} chars, infobox={len(infobox)} fields, "
                    f"tables={len(tables)}, categories={len(categories)}, "
                    f"links={len(internal_links)}"
                )
                
                return article_content
            
            except Exception as e:
                logger.error(
                    f"Error crawling {url} (attempt {attempt + 1}): {type(e).__name__}: {e}",
                    extra={"url": url, "error_type": type(e).__name__, "error": str(e)},
                    exc_info=True
                )
                
                if attempt < self.max_retries - 1:
                    delay = 2 ** attempt
                    logger.info(f"Retrying after {delay}s...")
                    await asyncio.sleep(delay)
                else:
                    # Log final failure and re-raise
                    logger.error(
                        f"Failed to crawl {url} after {self.max_retries} attempts",
                        extra={"url": url, "max_retries": self.max_retries}
                    )
                    raise RuntimeError(f"Failed to crawl {url} after {self.max_retries} attempts") from e
        
        # Should not reach here
        raise RuntimeError(f"Failed to crawl {url}")
    
    async def deep_crawl(
        self,
        seed_url: str,
        max_depth: int = 2,
        max_articles: int = 100,
        checkpoint: Optional[CrawlCheckpoint] = None
    ) -> List[ArticleContent]:
        """
        Perform BFS deep crawl starting from seed article.
        
        Discovers related articles through internal links and crawls them
        in breadth-first order up to max_depth.
        
        Args:
            seed_url: Starting Wikipedia article URL
            max_depth: Maximum link depth to crawl
            max_articles: Maximum number of articles to crawl
            checkpoint: Resume from checkpoint if provided
            
        Returns:
            List of crawled ArticleContent objects
            
        Requirements: 3.3, 3.6, 3.7, 3.8
        """
        logger.info(
            f"Starting deep crawl: seed={seed_url}, max_depth={max_depth}, "
            f"max_articles={max_articles}"
        )
        
        # Initialize or restore from checkpoint
        if checkpoint:
            visited = checkpoint.visited_urls
            queue = checkpoint.queue
            crawled_articles = checkpoint.crawled_articles
            depth_map = checkpoint.depth_map
            logger.info(
                f"Resuming from checkpoint: {len(visited)} visited, "
                f"{len(queue)} queued, {len(crawled_articles)} crawled"
            )
        else:
            visited: Set[str] = set()
            queue: deque = deque([(seed_url, 0)])  # (url, depth)
            crawled_articles: List[ArticleContent] = []
            depth_map: Dict[str, int] = {seed_url: 0}
        
        # BFS traversal
        while queue and len(crawled_articles) < max_articles:
            url, depth = queue.popleft()
            
            # Skip if already visited
            if url in visited:
                continue
            
            # Skip if exceeds max depth
            if depth > max_depth:
                continue
            
            try:
                # Crawl article
                article = await self.crawl_article(url)
                crawled_articles.append(article)
                visited.add(url)
                
                logger.info(
                    f"Deep crawl progress: {len(crawled_articles)}/{max_articles} articles, "
                    f"depth={depth}, queue_size={len(queue)}"
                )
                
                # Add internal links to queue if not at max depth
                if depth < max_depth:
                    for link in article.internal_links:
                        # Convert relative links to absolute
                        absolute_link = urljoin(url, link)
                        
                        # Only queue if not visited and not already queued
                        if absolute_link not in visited and absolute_link not in depth_map:
                            queue.append((absolute_link, depth + 1))
                            depth_map[absolute_link] = depth + 1
                            logger.debug(f"Queued: {absolute_link} at depth {depth + 1}")
            
            except Exception as e:
                # Log error but continue with other articles (graceful failure)
                logger.error(
                    f"Failed to crawl {url} in deep crawl: {type(e).__name__}: {e}",
                    extra={"url": url, "depth": depth, "error": str(e)}
                )
                visited.add(url)  # Mark as visited to avoid retry
                continue
        
        logger.info(
            f"Deep crawl completed: {len(crawled_articles)} articles crawled, "
            f"{len(visited)} total visited"
        )
        
        return crawled_articles
    
    def extract_infobox(self, html: str) -> Dict[str, Any]:
        """
        Extract infobox data using CSS selectors.
        
        Parses Wikipedia infobox tables and returns structured data.
        
        Args:
            html: Article HTML content
            
        Returns:
            Dictionary of infobox field -> value mappings
            
        Requirements: 3.4
        """
        soup = BeautifulSoup(html, 'lxml')
        infobox_data = {}
        
        # Find infobox table
        infobox = soup.select_one(self.extraction_config.infobox_selector)
        
        if not infobox:
            logger.debug("No infobox found in article")
            return infobox_data
        
        # Extract rows
        rows = infobox.find_all('tr')
        
        for row in rows:
            # Look for header and data cells
            header = row.find('th')
            data = row.find('td')
            
            if header and data:
                # Clean text
                key = header.get_text(strip=True)
                value = data.get_text(strip=True)
                
                if key and value:
                    infobox_data[key] = value
        
        logger.debug(f"Extracted {len(infobox_data)} infobox fields")
        return infobox_data
    
    def extract_tables(self, html: str) -> List[pd.DataFrame]:
        """
        Extract tables from article returning pandas DataFrames.
        
        Parses all Wikipedia tables and converts them to DataFrames.
        
        Args:
            html: Article HTML content
            
        Returns:
            List of pandas DataFrames, one per table
            
        Requirements: 3.4
        """
        soup = BeautifulSoup(html, 'lxml')
        tables = []
        
        # Find all wikitable elements
        wiki_tables = soup.select(self.extraction_config.table_selector)
        
        for table in wiki_tables:
            try:
                # Convert HTML table to DataFrame
                # Use pandas read_html which handles complex table structures
                df_list = pd.read_html(str(table))
                
                if df_list:
                    tables.append(df_list[0])
                    logger.debug(f"Extracted table with shape {df_list[0].shape}")
            
            except Exception as e:
                # Log error but continue with other tables
                logger.warning(
                    f"Failed to parse table: {type(e).__name__}: {e}",
                    extra={"error": str(e)}
                )
                continue
        
        logger.debug(f"Extracted {len(tables)} tables from article")
        return tables
    
    def extract_internal_links(self, html: str) -> List[str]:
        """
        Extract internal Wikipedia links from article.
        
        Finds all links to other Wikipedia articles (href starting with /wiki/).
        
        Args:
            html: Article HTML content
            
        Returns:
            List of internal link URLs
            
        Requirements: 3.6
        """
        soup = BeautifulSoup(html, 'lxml')
        internal_links = []
        
        # Find all internal links
        links = soup.select(self.extraction_config.internal_link_selector)
        
        for link in links:
            href = link.get('href', '')
            
            # Filter out special pages and non-article links
            if href and self._is_valid_article_link(href):
                # Store relative URL
                internal_links.append(href)
        
        # Remove duplicates while preserving order
        seen = set()
        unique_links = []
        for link in internal_links:
            if link not in seen:
                seen.add(link)
                unique_links.append(link)
        
        logger.debug(f"Extracted {len(unique_links)} unique internal links")
        return unique_links
    
    def _extract_title(self, url: str, soup: BeautifulSoup) -> str:
        """Extract article title from URL or page."""
        # Try to get from page title
        title_elem = soup.find('h1', class_='firstHeading')
        if title_elem:
            return title_elem.get_text(strip=True)
        
        # Fall back to URL parsing
        path = urlparse(url).path
        title = path.split('/wiki/')[-1] if '/wiki/' in path else path
        return title.replace('_', ' ')
    
    def _extract_summary(self, soup: BeautifulSoup, config: ExtractionConfig) -> str:
        """Extract article summary (first few paragraphs)."""
        paragraphs = soup.select(config.summary_selector)
        
        # Get first 3 non-empty paragraphs
        summary_parts = []
        for p in paragraphs[:5]:  # Check first 5 to get 3 non-empty
            text = p.get_text(strip=True)
            if text and len(text) > 50:  # Skip very short paragraphs
                summary_parts.append(text)
                if len(summary_parts) >= 3:
                    break
        
        return ' '.join(summary_parts)
    
    def _extract_categories(self, soup: BeautifulSoup, config: ExtractionConfig) -> List[str]:
        """Extract article categories."""
        categories = []
        
        category_links = soup.select(config.category_selector)
        
        for link in category_links:
            category = link.get_text(strip=True)
            if category:
                categories.append(category)
        
        return categories
    
    def _is_valid_article_link(self, href: str) -> bool:
        """
        Check if link is a valid article link.
        
        Filters out special pages, files, categories, etc.
        """
        # Must start with /wiki/
        if not href.startswith('/wiki/'):
            return False
        
        # Exclude special pages
        excluded_prefixes = [
            '/wiki/File:',
            '/wiki/Category:',
            '/wiki/Wikipedia:',
            '/wiki/Help:',
            '/wiki/Template:',
            '/wiki/Special:',
            '/wiki/Talk:',
            '/wiki/User:',
            '/wiki/Portal:'
        ]
        
        for prefix in excluded_prefixes:
            if href.startswith(prefix):
                return False
        
        # Exclude links with fragments or query params (usually navigation)
        if '#' in href or '?' in href:
            return False
        
        return True
    
    def create_checkpoint(
        self,
        seed_url: str,
        visited: Set[str],
        queue: deque,
        crawled_articles: List[ArticleContent],
        depth_map: Dict[str, int]
    ) -> CrawlCheckpoint:
        """
        Create checkpoint for resuming deep crawl.
        
        Args:
            seed_url: Original seed URL
            visited: Set of visited URLs
            queue: Current crawl queue
            crawled_articles: Articles crawled so far
            depth_map: URL to depth mapping
            
        Returns:
            CrawlCheckpoint object
            
        Requirements: 11.7
        """
        checkpoint = CrawlCheckpoint(
            seed_url=seed_url,
            visited_urls=visited.copy(),
            queue=queue.copy(),
            crawled_articles=crawled_articles.copy(),
            timestamp=datetime.now(),
            depth_map=depth_map.copy()
        )
        
        logger.info(
            f"Created checkpoint: {len(visited)} visited, {len(queue)} queued, "
            f"{len(crawled_articles)} crawled"
        )
        
        return checkpoint
    
    async def close(self) -> None:
        """Close crawler and cleanup resources."""
        if self._crawler:
            await self._crawler.__aexit__(None, None, None)
            self._crawler = None
            logger.info("Crawl4AIPipeline closed")
    
    async def __aenter__(self):
        """Async context manager entry."""
        return self
    
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Async context manager exit."""
        await self.close()
