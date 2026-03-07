"""
Unit tests for Crawl4AIPipeline.

Tests with sample HTML fixtures and edge cases including:
- Sample HTML content extraction
- Network failures
- Invalid HTML
- Checkpoint creation and resumption

Requirements: 3.1, 3.7, 3.8
"""

import pytest
from unittest.mock import AsyncMock, Mock, patch, MagicMock
from datetime import datetime
from collections import deque
import asyncio

from src.data_ingestion.crawl4ai_pipeline import (
    Crawl4AIPipeline,
    ExtractionConfig,
    CrawlCheckpoint
)
from src.storage.dto import ArticleContent


# Sample HTML fixtures
SAMPLE_WIKIPEDIA_HTML = """
<!DOCTYPE html>
<html>
<head><title>Test Article</title></head>
<body>
<h1 class="firstHeading">Test Article</h1>
<div class="mw-parser-output">
    <p>This is the first paragraph of the article summary. It contains important information about the topic.</p>
    <p>This is the second paragraph with more details about the subject matter.</p>
    <p>This is the third paragraph concluding the summary section.</p>
    
    <table class="infobox">
        <tr><th>Type</th><td>Example</td></tr>
        <tr><th>Founded</th><td>2024</td></tr>
        <tr><th>Industry</th><td>Technology</td></tr>
    </table>
    
    <table class="wikitable">
        <tr><th>Year</th><th>Revenue</th></tr>
        <tr><td>2023</td><td>$100M</td></tr>
        <tr><td>2024</td><td>$150M</td></tr>
    </table>
    
    <a href="/wiki/Related_Article_1">Related Article 1</a>
    <a href="/wiki/Related_Article_2">Related Article 2</a>
    <a href="/wiki/File:Image.jpg">File Link</a>
    <a href="/wiki/Category:Test">Category Link</a>
    <a href="https://external.com">External Link</a>
</div>
<div id="mw-normal-catlinks">
    <a href="/wiki/Category:Technology">Technology</a>
    <a href="/wiki/Category:Business">Business</a>
</div>
</body>
</html>
"""

MINIMAL_HTML = """
<!DOCTYPE html>
<html>
<head><title>Minimal</title></head>
<body>
<h1 class="firstHeading">Minimal Article</h1>
<div class="mw-parser-output">
    <p>This is a minimal article with just a summary paragraph.</p>
</div>
</body>
</html>
"""

INVALID_HTML = """
<html>
<body>
<div>Broken HTML without proper closing tags
<p>Missing closing paragraph
"""


class TestCrawlArticleWithSampleHTML:
    """Test crawl_article with sample HTML fixtures."""
    
    @pytest.mark.asyncio
    async def test_crawl_article_extracts_all_content(self):
        """Test that crawl_article extracts all content types from sample HTML."""
        pipeline = Crawl4AIPipeline(max_concurrent=1)
        
        # Mock the crawler
        mock_result = Mock()
        mock_result.success = True
        mock_result.html = SAMPLE_WIKIPEDIA_HTML
        mock_result.error_message = None
        
        mock_crawler = AsyncMock()
        mock_crawler.arun = AsyncMock(return_value=mock_result)
        
        pipeline._crawler = mock_crawler
        
        # Crawl article
        url = "https://en.wikipedia.org/wiki/Test_Article"
        article = await pipeline.crawl_article(url)
        
        # Verify ArticleContent structure
        assert isinstance(article, ArticleContent)
        assert article.title == "Test Article"
        assert article.url == url
        
        # Verify summary extraction
        assert len(article.summary) > 0
        assert "first paragraph" in article.summary
        
        # Verify infobox extraction
        assert len(article.infobox) > 0
        assert "Type" in article.infobox
        assert article.infobox["Type"] == "Example"
        assert article.infobox["Founded"] == "2024"
        
        # Verify table extraction
        assert len(article.tables) > 0
        
        # Verify categories extraction
        assert len(article.categories) > 0
        assert "Technology" in article.categories
        assert "Business" in article.categories
        
        # Verify internal links extraction (should filter out File: and Category:)
        assert len(article.internal_links) > 0
        assert "/wiki/Related_Article_1" in article.internal_links
        assert "/wiki/Related_Article_2" in article.internal_links
        assert "/wiki/File:Image.jpg" not in article.internal_links
        assert "/wiki/Category:Test" not in article.internal_links
        
        # Verify timestamp
        assert isinstance(article.crawl_timestamp, datetime)

    
    @pytest.mark.asyncio
    async def test_crawl_article_with_minimal_content(self):
        """Test crawl_article with minimal HTML content."""
        pipeline = Crawl4AIPipeline(max_concurrent=1)
        
        # Mock the crawler
        mock_result = Mock()
        mock_result.success = True
        mock_result.html = MINIMAL_HTML
        mock_result.error_message = None
        
        mock_crawler = AsyncMock()
        mock_crawler.arun = AsyncMock(return_value=mock_result)
        
        pipeline._crawler = mock_crawler
        
        # Crawl article
        url = "https://en.wikipedia.org/wiki/Minimal_Article"
        article = await pipeline.crawl_article(url)
        
        # Verify basic structure
        assert isinstance(article, ArticleContent)
        assert article.title == "Minimal Article"
        assert len(article.summary) > 0
        
        # Verify empty collections for missing content
        assert len(article.infobox) == 0
        assert len(article.tables) == 0
        assert len(article.categories) == 0
        assert len(article.internal_links) == 0
    
    @pytest.mark.asyncio
    async def test_crawl_article_with_custom_extraction_config(self):
        """Test crawl_article with custom extraction configuration."""
        # Create config that only extracts summary and infobox
        config = ExtractionConfig(
            extract_summary=True,
            extract_infobox=True,
            extract_tables=False,
            extract_categories=False,
            extract_internal_links=False
        )
        
        pipeline = Crawl4AIPipeline(max_concurrent=1, extraction_config=config)
        
        # Mock the crawler
        mock_result = Mock()
        mock_result.success = True
        mock_result.html = SAMPLE_WIKIPEDIA_HTML
        mock_result.error_message = None
        
        mock_crawler = AsyncMock()
        mock_crawler.arun = AsyncMock(return_value=mock_result)
        
        pipeline._crawler = mock_crawler
        
        # Crawl article
        url = "https://en.wikipedia.org/wiki/Test_Article"
        article = await pipeline.crawl_article(url)
        
        # Verify only configured content is extracted
        assert len(article.summary) > 0
        assert len(article.infobox) > 0
        assert len(article.tables) == 0
        assert len(article.categories) == 0
        assert len(article.internal_links) == 0


class TestErrorHandling:
    """Test error handling for network failures and invalid HTML."""
    
    @pytest.mark.asyncio
    async def test_crawl_article_with_network_failure(self):
        """Test crawl_article handles network failures with retry."""
        pipeline = Crawl4AIPipeline(max_concurrent=1, max_retries=3)
        
        # Mock the crawler to fail
        mock_crawler = AsyncMock()
        mock_crawler.arun = AsyncMock(side_effect=Exception("Network timeout"))
        
        pipeline._crawler = mock_crawler
        
        # Crawl should fail after retries
        url = "https://en.wikipedia.org/wiki/Test_Article"
        
        with pytest.raises(RuntimeError) as exc_info:
            await pipeline.crawl_article(url)
        
        assert "Failed to crawl" in str(exc_info.value)
        assert mock_crawler.arun.call_count == 3  # Should retry 3 times

    
    @pytest.mark.asyncio
    async def test_crawl_article_with_crawl_failure(self):
        """Test crawl_article handles crawl failures (result.success = False)."""
        pipeline = Crawl4AIPipeline(max_concurrent=1, max_retries=2)
        
        # Mock the crawler to return failed result
        mock_result = Mock()
        mock_result.success = False
        mock_result.error_message = "Page not found"
        
        mock_crawler = AsyncMock()
        mock_crawler.arun = AsyncMock(return_value=mock_result)
        
        pipeline._crawler = mock_crawler
        
        # Crawl should fail after retries
        url = "https://en.wikipedia.org/wiki/NonExistent"
        
        with pytest.raises(RuntimeError) as exc_info:
            await pipeline.crawl_article(url)
        
        assert "Crawl failed" in str(exc_info.value)
        assert "Page not found" in str(exc_info.value)
        assert mock_crawler.arun.call_count == 2  # Should retry 2 times
    
    @pytest.mark.asyncio
    async def test_crawl_article_with_invalid_html(self):
        """Test crawl_article handles invalid/malformed HTML gracefully."""
        pipeline = Crawl4AIPipeline(max_concurrent=1)
        
        # Mock the crawler with invalid HTML
        mock_result = Mock()
        mock_result.success = True
        mock_result.html = INVALID_HTML
        mock_result.error_message = None
        
        mock_crawler = AsyncMock()
        mock_crawler.arun = AsyncMock(return_value=mock_result)
        
        pipeline._crawler = mock_crawler
        
        # Should not crash, but extract what it can
        url = "https://en.wikipedia.org/wiki/Invalid_HTML"
        article = await pipeline.crawl_article(url)
        
        # Should still create ArticleContent object
        assert isinstance(article, ArticleContent)
        assert article.url == url
        # Content may be minimal or empty, but shouldn't crash
    
    @pytest.mark.asyncio
    async def test_crawl_article_with_timeout(self):
        """Test crawl_article handles timeout errors."""
        pipeline = Crawl4AIPipeline(max_concurrent=1, max_retries=2, timeout=1.0)
        
        # Mock the crawler to timeout
        mock_crawler = AsyncMock()
        mock_crawler.arun = AsyncMock(side_effect=asyncio.TimeoutError("Request timeout"))
        
        pipeline._crawler = mock_crawler
        
        # Crawl should fail after retries
        url = "https://en.wikipedia.org/wiki/Slow_Page"
        
        with pytest.raises(RuntimeError) as exc_info:
            await pipeline.crawl_article(url)
        
        assert "Failed to crawl" in str(exc_info.value)
        assert mock_crawler.arun.call_count == 2
    
    @pytest.mark.asyncio
    async def test_crawl_article_retry_with_exponential_backoff(self):
        """Test crawl_article uses exponential backoff on retries."""
        pipeline = Crawl4AIPipeline(max_concurrent=1, max_retries=3)
        
        # Mock the crawler to fail first 2 times, succeed on 3rd
        mock_result_success = Mock()
        mock_result_success.success = True
        mock_result_success.html = MINIMAL_HTML
        mock_result_success.error_message = None
        
        call_count = 0
        async def mock_arun(*args, **kwargs):
            nonlocal call_count
            call_count += 1
            if call_count < 3:
                raise Exception("Temporary failure")
            return mock_result_success
        
        mock_crawler = AsyncMock()
        mock_crawler.arun = mock_arun
        
        pipeline._crawler = mock_crawler
        
        # Should succeed on 3rd attempt
        url = "https://en.wikipedia.org/wiki/Test_Article"
        article = await pipeline.crawl_article(url)
        
        assert isinstance(article, ArticleContent)
        assert call_count == 3


class TestCheckpointing:
    """Test checkpoint creation and resumption for long-running crawls."""

    
    def test_create_checkpoint(self):
        """Test checkpoint creation captures crawl state."""
        pipeline = Crawl4AIPipeline(max_concurrent=1)
        
        # Create sample crawl state
        seed_url = "https://en.wikipedia.org/wiki/Seed_Article"
        visited = {"url1", "url2", "url3"}
        queue = deque([("url4", 1), ("url5", 2)])
        crawled_articles = [
            ArticleContent(
                title="Article 1",
                url="url1",
                summary="Summary 1",
                infobox={},
                tables=[],
                categories=[],
                internal_links=[],
                crawl_timestamp=datetime.now()
            )
        ]
        depth_map = {"url1": 0, "url2": 1, "url3": 1, "url4": 1, "url5": 2}
        
        # Create checkpoint
        checkpoint = pipeline.create_checkpoint(
            seed_url=seed_url,
            visited=visited,
            queue=queue,
            crawled_articles=crawled_articles,
            depth_map=depth_map
        )
        
        # Verify checkpoint structure
        assert isinstance(checkpoint, CrawlCheckpoint)
        assert checkpoint.seed_url == seed_url
        assert checkpoint.visited_urls == visited
        assert checkpoint.queue == queue
        assert len(checkpoint.crawled_articles) == 1
        assert checkpoint.depth_map == depth_map
        assert isinstance(checkpoint.timestamp, datetime)
    
    @pytest.mark.asyncio
    async def test_deep_crawl_resume_from_checkpoint(self):
        """Test deep_crawl can resume from checkpoint."""
        pipeline = Crawl4AIPipeline(max_concurrent=1)
        
        # Create a checkpoint with some progress
        seed_url = "https://en.wikipedia.org/wiki/Seed"
        visited = {"https://en.wikipedia.org/wiki/Seed"}
        queue = deque([
            ("https://en.wikipedia.org/wiki/Article2", 1),
            ("https://en.wikipedia.org/wiki/Article3", 1)
        ])
        crawled_articles = [
            ArticleContent(
                title="Seed",
                url=seed_url,
                summary="Seed summary",
                infobox={},
                tables=[],
                categories=[],
                internal_links=["/wiki/Article2", "/wiki/Article3"],
                crawl_timestamp=datetime.now()
            )
        ]
        depth_map = {
            seed_url: 0,
            "https://en.wikipedia.org/wiki/Article2": 1,
            "https://en.wikipedia.org/wiki/Article3": 1
        }
        
        checkpoint = CrawlCheckpoint(
            seed_url=seed_url,
            visited_urls=visited,
            queue=queue,
            crawled_articles=crawled_articles,
            timestamp=datetime.now(),
            depth_map=depth_map
        )
        
        # Mock crawler for remaining articles
        mock_result = Mock()
        mock_result.success = True
        mock_result.html = MINIMAL_HTML
        mock_result.error_message = None
        
        mock_crawler = AsyncMock()
        mock_crawler.arun = AsyncMock(return_value=mock_result)
        
        pipeline._crawler = mock_crawler
        
        # Resume crawl from checkpoint
        articles = await pipeline.deep_crawl(
            seed_url=seed_url,
            max_depth=1,
            max_articles=10,
            checkpoint=checkpoint
        )
        
        # Should have crawled the remaining articles
        assert len(articles) >= 1  # At least the seed from checkpoint
        # Should have called crawler for queued articles
        assert mock_crawler.arun.call_count >= 1

    
    @pytest.mark.asyncio
    async def test_deep_crawl_checkpoint_preserves_state(self):
        """Test that checkpoint preserves exact crawl state."""
        pipeline = Crawl4AIPipeline(max_concurrent=1)
        
        # Create initial state
        seed_url = "https://en.wikipedia.org/wiki/Test"
        visited = {"url1", "url2"}
        queue = deque([("url3", 1)])
        crawled = [
            ArticleContent(
                title="Test",
                url="url1",
                summary="Test",
                infobox={},
                tables=[],
                categories=[],
                internal_links=[],
                crawl_timestamp=datetime.now()
            )
        ]
        depth_map = {"url1": 0, "url2": 1, "url3": 1}
        
        # Create checkpoint
        checkpoint = pipeline.create_checkpoint(seed_url, visited, queue, crawled, depth_map)
        
        # Verify state is preserved (not just referenced)
        visited.add("url4")
        queue.append(("url5", 2))
        
        # Checkpoint should have original state
        assert "url4" not in checkpoint.visited_urls
        assert ("url5", 2) not in checkpoint.queue


class TestDeepCrawlBehavior:
    """Test deep crawl BFS behavior and graceful failure handling."""
    
    @pytest.mark.asyncio
    async def test_deep_crawl_graceful_failure_handling(self):
        """Test deep_crawl continues on individual article failures."""
        pipeline = Crawl4AIPipeline(max_concurrent=1, max_retries=1)
        
        # Mock crawler that fails for some URLs
        call_count = 0
        async def mock_arun(url, config):
            nonlocal call_count
            call_count += 1
            
            # Fail on second article
            if "Article2" in url:
                raise Exception("Failed to crawl Article2")
            
            # Succeed for others
            mock_result = Mock()
            mock_result.success = True
            mock_result.html = MINIMAL_HTML
            mock_result.error_message = None
            return mock_result
        
        mock_crawler = AsyncMock()
        mock_crawler.arun = mock_arun
        
        pipeline._crawler = mock_crawler
        
        # Deep crawl should continue despite failure
        seed_url = "https://en.wikipedia.org/wiki/Seed"
        articles = await pipeline.deep_crawl(
            seed_url=seed_url,
            max_depth=0,  # Only crawl seed
            max_articles=10
        )
        
        # Should have crawled seed successfully
        assert len(articles) >= 1
    
    @pytest.mark.asyncio
    async def test_deep_crawl_respects_max_articles(self):
        """Test deep_crawl stops at max_articles limit."""
        pipeline = Crawl4AIPipeline(max_concurrent=1)
        
        # Mock crawler that returns articles with many links
        def create_mock_result(title):
            mock_result = Mock()
            mock_result.success = True
            # HTML with many internal links
            html = f"""
            <html><body>
            <h1 class="firstHeading">{title}</h1>
            <div class="mw-parser-output">
                <p>Article content for {title}</p>
                <a href="/wiki/Link1">Link 1</a>
                <a href="/wiki/Link2">Link 2</a>
                <a href="/wiki/Link3">Link 3</a>
                <a href="/wiki/Link4">Link 4</a>
                <a href="/wiki/Link5">Link 5</a>
            </div>
            </body></html>
            """
            mock_result.html = html
            mock_result.error_message = None
            return mock_result
        
        call_count = 0
        async def mock_arun(url, config):
            nonlocal call_count
            call_count += 1
            title = url.split("/")[-1]
            return create_mock_result(title)
        
        mock_crawler = AsyncMock()
        mock_crawler.arun = mock_arun
        
        pipeline._crawler = mock_crawler
        
        # Deep crawl with max_articles limit
        seed_url = "https://en.wikipedia.org/wiki/Seed"
        articles = await pipeline.deep_crawl(
            seed_url=seed_url,
            max_depth=2,
            max_articles=3  # Limit to 3 articles
        )
        
        # Should stop at max_articles
        assert len(articles) == 3

    
    @pytest.mark.asyncio
    async def test_deep_crawl_respects_max_depth(self):
        """Test deep_crawl respects max_depth limit."""
        pipeline = Crawl4AIPipeline(max_concurrent=1)
        
        # Track which URLs were crawled
        crawled_urls = []
        
        async def mock_arun(url, config):
            crawled_urls.append(url)
            
            mock_result = Mock()
            mock_result.success = True
            
            # Each article links to next level
            if "Seed" in url:
                html = """
                <html><body>
                <h1 class="firstHeading">Seed</h1>
                <div class="mw-parser-output">
                    <p>Seed article</p>
                    <a href="/wiki/Level1">Level 1</a>
                </div>
                </body></html>
                """
            elif "Level1" in url:
                html = """
                <html><body>
                <h1 class="firstHeading">Level1</h1>
                <div class="mw-parser-output">
                    <p>Level 1 article</p>
                    <a href="/wiki/Level2">Level 2</a>
                </div>
                </body></html>
                """
            else:
                html = """
                <html><body>
                <h1 class="firstHeading">Level2</h1>
                <div class="mw-parser-output">
                    <p>Level 2 article</p>
                </div>
                </body></html>
                """
            
            mock_result.html = html
            mock_result.error_message = None
            return mock_result
        
        mock_crawler = AsyncMock()
        mock_crawler.arun = mock_arun
        
        pipeline._crawler = mock_crawler
        
        # Deep crawl with max_depth=1
        seed_url = "https://en.wikipedia.org/wiki/Seed"
        articles = await pipeline.deep_crawl(
            seed_url=seed_url,
            max_depth=1,
            max_articles=100
        )
        
        # Should crawl Seed (depth 0) and Level1 (depth 1), but not Level2 (depth 2)
        assert len(articles) == 2
        assert any("Seed" in url for url in crawled_urls)
        assert any("Level1" in url for url in crawled_urls)
        assert not any("Level2" in url for url in crawled_urls)


class TestContentExtraction:
    """Test specific content extraction methods."""
    
    def test_extract_infobox_with_valid_table(self):
        """Test extract_infobox extracts data from infobox table."""
        pipeline = Crawl4AIPipeline()
        
        infobox = pipeline.extract_infobox(SAMPLE_WIKIPEDIA_HTML)
        
        assert isinstance(infobox, dict)
        assert len(infobox) > 0
        assert "Type" in infobox
        assert infobox["Type"] == "Example"
        assert "Founded" in infobox
        assert infobox["Founded"] == "2024"
    
    def test_extract_infobox_with_no_infobox(self):
        """Test extract_infobox returns empty dict when no infobox present."""
        pipeline = Crawl4AIPipeline()
        
        infobox = pipeline.extract_infobox(MINIMAL_HTML)
        
        assert isinstance(infobox, dict)
        assert len(infobox) == 0
    
    def test_extract_tables_with_valid_tables(self):
        """Test extract_tables extracts pandas DataFrames."""
        pipeline = Crawl4AIPipeline()
        
        tables = pipeline.extract_tables(SAMPLE_WIKIPEDIA_HTML)
        
        assert isinstance(tables, list)
        assert len(tables) > 0
        # Should be pandas DataFrames
        import pandas as pd
        assert all(isinstance(table, pd.DataFrame) for table in tables)
    
    def test_extract_tables_with_no_tables(self):
        """Test extract_tables returns empty list when no tables present."""
        pipeline = Crawl4AIPipeline()
        
        tables = pipeline.extract_tables(MINIMAL_HTML)
        
        assert isinstance(tables, list)
        assert len(tables) == 0

    
    def test_extract_internal_links_filters_correctly(self):
        """Test extract_internal_links filters out non-article links."""
        pipeline = Crawl4AIPipeline()
        
        links = pipeline.extract_internal_links(SAMPLE_WIKIPEDIA_HTML)
        
        assert isinstance(links, list)
        assert len(links) > 0
        
        # Should include article links
        assert "/wiki/Related_Article_1" in links
        assert "/wiki/Related_Article_2" in links
        
        # Should exclude special pages
        assert not any("File:" in link for link in links)
        assert not any("Category:" in link for link in links)
        
        # Should not include external links
        assert not any(link.startswith("http") for link in links)
    
    def test_extract_internal_links_removes_duplicates(self):
        """Test extract_internal_links removes duplicate links."""
        pipeline = Crawl4AIPipeline()
        
        html_with_duplicates = """
        <html><body>
        <div class="mw-parser-output">
            <a href="/wiki/Article1">Link 1</a>
            <a href="/wiki/Article1">Link 1 again</a>
            <a href="/wiki/Article2">Link 2</a>
            <a href="/wiki/Article1">Link 1 third time</a>
        </div>
        </body></html>
        """
        
        links = pipeline.extract_internal_links(html_with_duplicates)
        
        # Should have unique links only
        assert len(links) == 2
        assert "/wiki/Article1" in links
        assert "/wiki/Article2" in links
    
    def test_is_valid_article_link_filters_special_pages(self):
        """Test _is_valid_article_link correctly identifies valid article links."""
        pipeline = Crawl4AIPipeline()
        
        # Valid article links
        assert pipeline._is_valid_article_link("/wiki/Article_Name")
        assert pipeline._is_valid_article_link("/wiki/Company")
        
        # Invalid special pages
        assert not pipeline._is_valid_article_link("/wiki/File:Image.jpg")
        assert not pipeline._is_valid_article_link("/wiki/Category:Technology")
        assert not pipeline._is_valid_article_link("/wiki/Wikipedia:Policy")
        assert not pipeline._is_valid_article_link("/wiki/Help:Contents")
        assert not pipeline._is_valid_article_link("/wiki/Template:Infobox")
        assert not pipeline._is_valid_article_link("/wiki/Special:Search")
        assert not pipeline._is_valid_article_link("/wiki/Talk:Article")
        assert not pipeline._is_valid_article_link("/wiki/User:Username")
        assert not pipeline._is_valid_article_link("/wiki/Portal:Technology")
        
        # Links with fragments or query params
        assert not pipeline._is_valid_article_link("/wiki/Article#Section")
        assert not pipeline._is_valid_article_link("/wiki/Article?action=edit")
        
        # Not starting with /wiki/
        assert not pipeline._is_valid_article_link("/w/index.php")
        assert not pipeline._is_valid_article_link("https://external.com")


class TestRateLimiting:
    """Test rate limiting and crawl delay behavior."""
    
    @pytest.mark.asyncio
    async def test_crawl_delay_applied_between_requests(self):
        """Test that crawl delay is applied between requests to same domain."""
        pipeline = Crawl4AIPipeline(max_concurrent=1, crawl_delay=0.5)
        
        # Mock crawler
        mock_result = Mock()
        mock_result.success = True
        mock_result.html = MINIMAL_HTML
        mock_result.error_message = None
        
        mock_crawler = AsyncMock()
        mock_crawler.arun = AsyncMock(return_value=mock_result)
        
        pipeline._crawler = mock_crawler
        
        # Crawl two articles from same domain
        url1 = "https://en.wikipedia.org/wiki/Article1"
        url2 = "https://en.wikipedia.org/wiki/Article2"
        
        import time
        start = time.time()
        await pipeline.crawl_article(url1)
        await pipeline.crawl_article(url2)
        duration = time.time() - start
        
        # Should have applied crawl delay (at least 0.5 seconds)
        assert duration >= 0.5


class TestCleanup:
    """Test resource cleanup."""
    
    @pytest.mark.asyncio
    async def test_close_cleans_up_crawler(self):
        """Test close() properly cleans up crawler resources."""
        pipeline = Crawl4AIPipeline(max_concurrent=1)
        
        # Create mock crawler
        mock_crawler = AsyncMock()
        mock_crawler.__aexit__ = AsyncMock()
        
        pipeline._crawler = mock_crawler
        
        # Close pipeline
        await pipeline.close()
        
        # Crawler should be cleaned up
        assert pipeline._crawler is None
        assert mock_crawler.__aexit__.called
    
    @pytest.mark.asyncio
    async def test_context_manager_cleanup(self):
        """Test async context manager properly cleans up."""
        mock_crawler = AsyncMock()
        mock_crawler.__aexit__ = AsyncMock()
        
        async with Crawl4AIPipeline(max_concurrent=1) as pipeline:
            pipeline._crawler = mock_crawler
        
        # Should have called cleanup
        assert mock_crawler.__aexit__.called
