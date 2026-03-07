"""Property-based tests for Crawl4AI pipeline

Feature: wikipedia-intelligence-system
"""
import pytest
import asyncio
from datetime import datetime, timedelta
from hypothesis import given, strategies as st, assume, settings, HealthCheck
from typing import List, Dict, Any
from collections import deque
from unittest.mock import Mock, AsyncMock, patch, MagicMock
import pandas as pd
from bs4 import BeautifulSoup

from src.data_ingestion.crawl4ai_pipeline import (
    Crawl4AIPipeline,
    ExtractionConfig,
    CrawlCheckpoint
)
from src.storage.dto import ArticleContent


# ============================================================================
# HYPOTHESIS STRATEGIES
# ============================================================================

def html_with_elements_strategy(
    include_summary: bool = True,
    include_infobox: bool = True,
    include_tables: bool = True,
    include_categories: bool = True,
    include_links: bool = True
):
    """Strategy for generating HTML with various Wikipedia elements."""
    html_parts = ['<html><head><title>Test Article</title></head><body>']
    html_parts.append('<h1 class="firstHeading">Test Article</h1>')
    html_parts.append('<div class="mw-parser-output">')
    
    if include_summary:
        html_parts.append('<p>This is the first paragraph of the article summary with enough content.</p>')
        html_parts.append('<p>This is the second paragraph providing more details about the topic.</p>')
        html_parts.append('<p>This is the third paragraph concluding the summary section.</p>')
    
    if include_infobox:
        html_parts.append('''
        <table class="infobox">
            <tr><th>Founded</th><td>2024</td></tr>
            <tr><th>Headquarters</th><td>San Francisco</td></tr>
            <tr><th>Industry</th><td>Technology</td></tr>
        </table>
        ''')
    
    if include_tables:
        html_parts.append('''
        <table class="wikitable">
            <tr><th>Year</th><th>Revenue</th></tr>
            <tr><td>2023</td><td>$100M</td></tr>
            <tr><td>2024</td><td>$150M</td></tr>
        </table>
        ''')
    
    if include_categories:
        html_parts.append('''
        <div id="mw-normal-catlinks">
            <a href="/wiki/Category:Technology">Technology</a>
            <a href="/wiki/Category:Companies">Companies</a>
        </div>
        ''')
    
    if include_links:
        html_parts.append('''
        <a href="/wiki/Related_Article_1">Related Article 1</a>
        <a href="/wiki/Related_Article_2">Related Article 2</a>
        <a href="/wiki/Related_Article_3">Related Article 3</a>
        ''')
    
    html_parts.append('</div></body></html>')
    return ''.join(html_parts)


def css_selector_strategy():
    """Strategy for generating CSS selectors."""
    return st.sampled_from([
        "div.mw-parser-output > p",
        "table.infobox",
        "table.wikitable",
        "div#mw-normal-catlinks a",
        "div.mw-parser-output a[href^='/wiki/']",
        "h1.firstHeading",
        "span.mw-headline"
    ])


def internal_link_strategy():
    """Strategy for generating internal Wikipedia links."""
    return st.builds(
        lambda title: f"/wiki/{title.replace(' ', '_')}",
        st.text(
            alphabet=st.characters(whitelist_categories=('Lu', 'Ll', 'Nd'), min_codepoint=65, max_codepoint=122),
            min_size=3,
            max_size=30
        ).filter(lambda x: ':' not in x and '/' not in x)
    )


# ============================================================================
# PROPERTY 11: Article Content Extraction Completeness
# ============================================================================

# Feature: wikipedia-intelligence-system, Property 11: Article Content Extraction Completeness
@given(
    include_summary=st.booleans(),
    include_infobox=st.booleans(),
    include_tables=st.booleans(),
    include_categories=st.booleans(),
    include_links=st.booleans()
)
@settings(
    suppress_health_check=[HealthCheck.function_scoped_fixture],
    max_examples=100,
    deadline=None
)
@pytest.mark.asyncio
async def test_property_11_article_content_extraction_completeness(
    include_summary, include_infobox, include_tables, include_categories, include_links
):
    """Property 11: Article Content Extraction Completeness
    
    For any Wikipedia article, the System should extract all available elements
    (summary, infobox, tables, categories, internal links) when present in the HTML.
    
    Validates: Requirements 3.1
    """
    # Generate HTML with specified elements
    html = html_with_elements_strategy(
        include_summary=include_summary,
        include_infobox=include_infobox,
        include_tables=include_tables,
        include_categories=include_categories,
        include_links=include_links
    )
    
    # Create pipeline with extraction config
    config = ExtractionConfig(
        extract_summary=True,
        extract_infobox=True,
        extract_tables=True,
        extract_categories=True,
        extract_internal_links=True
    )
    pipeline = Crawl4AIPipeline(extraction_config=config)
    
    # Mock the crawler to return our HTML
    mock_result = Mock()
    mock_result.success = True
    mock_result.html = html
    mock_result.error_message = None
    
    mock_crawler = AsyncMock()
    mock_crawler.arun = AsyncMock(return_value=mock_result)
    mock_crawler.__aenter__ = AsyncMock(return_value=mock_crawler)
    mock_crawler.__aexit__ = AsyncMock()
    
    pipeline._crawler = mock_crawler
    
    # Crawl the article
    url = "https://en.wikipedia.org/wiki/Test_Article"
    article = await pipeline.crawl_article(url)
    
    # Verify all available elements were extracted
    if include_summary:
        assert len(article.summary) > 0, "Summary should be extracted when present"
        assert "first paragraph" in article.summary.lower(), "Summary should contain content"
    
    if include_infobox:
        assert len(article.infobox) > 0, "Infobox should be extracted when present"
        # Check for expected infobox fields
        assert any(key in article.infobox for key in ["Founded", "Headquarters", "Industry"]), \
            "Infobox should contain expected fields"
    
    if include_tables:
        assert len(article.tables) > 0, "Tables should be extracted when present"
        assert isinstance(article.tables[0], pd.DataFrame), "Tables should be DataFrames"
    
    if include_categories:
        assert len(article.categories) > 0, "Categories should be extracted when present"
    
    if include_links:
        assert len(article.internal_links) > 0, "Internal links should be extracted when present"
    
    # Verify ArticleContent structure
    assert isinstance(article, ArticleContent), "Should return ArticleContent object"
    assert article.url == url, "URL should match"
    assert isinstance(article.crawl_timestamp, datetime), "Should have crawl timestamp"
    
    await pipeline.close()




# ============================================================================
# PROPERTY 12: BFS Crawl Order
# ============================================================================

# Feature: wikipedia-intelligence-system, Property 12: BFS Crawl Order
@given(
    max_depth=st.integers(min_value=1, max_value=3),
    links_per_article=st.integers(min_value=1, max_value=5)
)
@settings(
    suppress_health_check=[HealthCheck.function_scoped_fixture],
    max_examples=50,
    deadline=None
)
@pytest.mark.asyncio
async def test_property_12_bfs_crawl_order(max_depth, links_per_article):
    """Property 12: BFS Crawl Order
    
    For any deep crawl starting from a seed article, all articles at depth N
    should be discovered and queued before any article at depth N+1 is crawled.
    
    Validates: Requirements 3.3
    """
    # Track crawl order
    crawl_order = []
    depth_when_crawled = {}
    
    # Create mock HTML generator that returns links
    def generate_html_with_links(article_name, depth):
        links_html = []
        for i in range(links_per_article):
            link_name = f"{article_name}_Child_{i}"
            links_html.append(f'<a href="/wiki/{link_name}">{link_name}</a>')
        
        html = f'''
        <html><body>
        <h1 class="firstHeading">{article_name}</h1>
        <div class="mw-parser-output">
            <p>This is article {article_name} at depth {depth} with enough content for summary.</p>
            {''.join(links_html)}
        </div>
        </body></html>
        '''
        return html
    
    # Create pipeline
    pipeline = Crawl4AIPipeline(max_concurrent=1)  # Sequential for deterministic order
    
    # Mock crawler
    async def mock_arun(url, config):
        # Extract article name from URL
        article_name = url.split('/wiki/')[-1]
        
        # Determine depth based on article name structure
        depth = article_name.count('_Child_')
        
        # Record crawl order
        crawl_order.append(article_name)
        depth_when_crawled[article_name] = depth
        
        # Generate HTML with links
        html = generate_html_with_links(article_name, depth)
        
        result = Mock()
        result.success = True
        result.html = html
        result.error_message = None
        return result
    
    mock_crawler = AsyncMock()
    mock_crawler.arun = mock_arun
    mock_crawler.__aenter__ = AsyncMock(return_value=mock_crawler)
    mock_crawler.__aexit__ = AsyncMock()
    
    pipeline._crawler = mock_crawler
    
    # Perform deep crawl
    seed_url = "https://en.wikipedia.org/wiki/Seed_Article"
    max_articles = min(50, (links_per_article ** (max_depth + 1) - 1) // (links_per_article - 1) if links_per_article > 1 else max_depth + 1)
    
    articles = await pipeline.deep_crawl(
        seed_url=seed_url,
        max_depth=max_depth,
        max_articles=max_articles
    )
    
    # Verify BFS order: all articles at depth N should be crawled before depth N+1
    for i in range(len(crawl_order) - 1):
        current_article = crawl_order[i]
        current_depth = depth_when_crawled[current_article]
        
        # Check all subsequent articles
        for j in range(i + 1, len(crawl_order)):
            next_article = crawl_order[j]
            next_depth = depth_when_crawled[next_article]
            
            # If next article has greater depth, all articles between should not have even greater depth
            if next_depth > current_depth:
                for k in range(i + 1, j):
                    intermediate_depth = depth_when_crawled[crawl_order[k]]
                    assert intermediate_depth <= next_depth, \
                        f"BFS violation: article at depth {intermediate_depth} crawled between " \
                        f"depth {current_depth} and {next_depth}"
    
    # Verify no article at depth N+1 was crawled before all articles at depth N
    for depth in range(max_depth):
        articles_at_depth = [a for a, d in depth_when_crawled.items() if d == depth]
        articles_at_next_depth = [a for a, d in depth_when_crawled.items() if d == depth + 1]
        
        if articles_at_depth and articles_at_next_depth:
            # Find last article at current depth
            last_at_depth_idx = max(crawl_order.index(a) for a in articles_at_depth)
            # Find first article at next depth
            first_at_next_depth_idx = min(crawl_order.index(a) for a in articles_at_next_depth)
            
            assert last_at_depth_idx < first_at_next_depth_idx, \
                f"BFS violation: article at depth {depth + 1} crawled before " \
                f"all articles at depth {depth} were complete"
    
    await pipeline.close()


# ============================================================================
# PROPERTY 13: CSS Selector Extraction
# ============================================================================

# Feature: wikipedia-intelligence-system, Property 13: CSS Selector Extraction
@given(
    num_elements=st.integers(min_value=1, max_value=20)
)
@settings(
    suppress_health_check=[HealthCheck.function_scoped_fixture],
    max_examples=100,
    deadline=None
)
def test_property_13_css_selector_extraction(num_elements):
    """Property 13: CSS Selector Extraction
    
    For any article and CSS selector configuration, the System should extract
    all elements matching the selector.
    
    Validates: Requirements 3.4
    """
    # Generate HTML with known number of elements
    html_parts = ['<html><body><div class="mw-parser-output">']
    
    # Add paragraphs
    for i in range(num_elements):
        html_parts.append(f'<p>Paragraph {i} with sufficient content for extraction.</p>')
    
    html_parts.append('</div></body></html>')
    html = ''.join(html_parts)
    
    # Create pipeline
    config = ExtractionConfig(
        extract_summary=True,
        summary_selector="div.mw-parser-output > p"
    )
    pipeline = Crawl4AIPipeline(extraction_config=config)
    
    # Extract summary using CSS selector
    soup = BeautifulSoup(html, 'lxml')
    summary = pipeline._extract_summary(soup, config)
    
    # Verify all matching elements were extracted
    # Summary extracts first 3 non-empty paragraphs
    expected_paragraphs = min(3, num_elements)
    
    if num_elements > 0:
        assert len(summary) > 0, "Summary should be extracted when paragraphs exist"
        
        # Count how many paragraph contents appear in summary
        paragraphs_found = sum(1 for i in range(num_elements) if f"Paragraph {i}" in summary)
        assert paragraphs_found == expected_paragraphs, \
            f"Expected {expected_paragraphs} paragraphs in summary, found {paragraphs_found}"


# Feature: wikipedia-intelligence-system, Property 13: CSS Selector Infobox Extraction
@given(
    num_fields=st.integers(min_value=0, max_value=15)
)
@settings(
    suppress_health_check=[HealthCheck.function_scoped_fixture],
    max_examples=100
)
def test_property_13_css_selector_infobox_extraction(num_fields):
    """Property 13: CSS Selector Infobox Extraction
    
    For any article with an infobox, the System should extract all fields
    using CSS selectors.
    
    Validates: Requirements 3.4
    """
    # Generate HTML with infobox
    html_parts = ['<html><body>']
    
    if num_fields > 0:
        html_parts.append('<table class="infobox">')
        for i in range(num_fields):
            html_parts.append(f'<tr><th>Field{i}</th><td>Value{i}</td></tr>')
        html_parts.append('</table>')
    
    html_parts.append('</body></html>')
    html = ''.join(html_parts)
    
    # Create pipeline
    pipeline = Crawl4AIPipeline()
    
    # Extract infobox
    infobox = pipeline.extract_infobox(html)
    
    # Verify all fields were extracted
    if num_fields > 0:
        assert len(infobox) == num_fields, \
            f"Expected {num_fields} infobox fields, got {len(infobox)}"
        
        # Verify each field
        for i in range(num_fields):
            assert f"Field{i}" in infobox, f"Field{i} should be in infobox"
            assert infobox[f"Field{i}"] == f"Value{i}", \
                f"Field{i} should have value Value{i}"
    else:
        assert len(infobox) == 0, "Empty infobox should return empty dict"


# ============================================================================
# PROPERTY 15: Internal Link Extraction
# ============================================================================

# Feature: wikipedia-intelligence-system, Property 15: Internal Link Extraction
@given(
    num_internal_links=st.integers(min_value=0, max_value=30),
    num_external_links=st.integers(min_value=0, max_value=10),
    num_special_links=st.integers(min_value=0, max_value=10)
)
@settings(
    suppress_health_check=[HealthCheck.function_scoped_fixture],
    max_examples=100
)
def test_property_15_internal_link_extraction(num_internal_links, num_external_links, num_special_links):
    """Property 15: Internal Link Extraction
    
    For any article, the System should extract all internal Wikipedia links
    (href starting with /wiki/) and filter out special pages.
    
    Validates: Requirements 3.6
    """
    # Generate HTML with various link types
    html_parts = ['<html><body><div class="mw-parser-output">']
    
    # Add internal article links
    expected_links = []
    for i in range(num_internal_links):
        link = f"/wiki/Article_{i}"
        html_parts.append(f'<a href="{link}">Article {i}</a>')
        expected_links.append(link)
    
    # Add external links (should be filtered out)
    for i in range(num_external_links):
        html_parts.append(f'<a href="https://example.com/page{i}">External {i}</a>')
    
    # Add special page links (should be filtered out)
    special_prefixes = ["/wiki/File:", "/wiki/Category:", "/wiki/Wikipedia:", 
                       "/wiki/Help:", "/wiki/Template:", "/wiki/Special:"]
    for i in range(num_special_links):
        prefix = special_prefixes[i % len(special_prefixes)]
        html_parts.append(f'<a href="{prefix}Page{i}">Special {i}</a>')
    
    html_parts.append('</div></body></html>')
    html = ''.join(html_parts)
    
    # Create pipeline
    pipeline = Crawl4AIPipeline()
    
    # Extract internal links
    internal_links = pipeline.extract_internal_links(html)
    
    # Verify only internal article links were extracted
    assert len(internal_links) == num_internal_links, \
        f"Expected {num_internal_links} internal links, got {len(internal_links)}"
    
    # Verify all expected links are present
    for expected_link in expected_links:
        assert expected_link in internal_links, \
            f"Expected link {expected_link} not found in extracted links"
    
    # Verify no external or special links were included
    for link in internal_links:
        assert link.startswith("/wiki/"), "All links should start with /wiki/"
        assert not any(link.startswith(prefix) for prefix in special_prefixes), \
            f"Special page link {link} should be filtered out"


# Feature: wikipedia-intelligence-system, Property 15: Internal Link Deduplication
@given(
    num_unique_links=st.integers(min_value=1, max_value=20),
    duplicates_per_link=st.integers(min_value=1, max_value=5)
)
@settings(
    suppress_health_check=[HealthCheck.function_scoped_fixture],
    max_examples=100
)
def test_property_15_internal_link_deduplication(num_unique_links, duplicates_per_link):
    """Property 15: Internal Link Deduplication
    
    For any article with duplicate internal links, the System should return
    only unique links while preserving order.
    
    Validates: Requirements 3.6
    """
    # Generate HTML with duplicate links
    html_parts = ['<html><body><div class="mw-parser-output">']
    
    expected_unique_links = []
    for i in range(num_unique_links):
        link = f"/wiki/Article_{i}"
        expected_unique_links.append(link)
        
        # Add link multiple times
        for _ in range(duplicates_per_link):
            html_parts.append(f'<a href="{link}">Article {i}</a>')
    
    html_parts.append('</div></body></html>')
    html = ''.join(html_parts)
    
    # Create pipeline
    pipeline = Crawl4AIPipeline()
    
    # Extract internal links
    internal_links = pipeline.extract_internal_links(html)
    
    # Verify deduplication
    assert len(internal_links) == num_unique_links, \
        f"Expected {num_unique_links} unique links, got {len(internal_links)}"
    
    # Verify all unique links are present
    for expected_link in expected_unique_links:
        assert expected_link in internal_links, \
            f"Expected unique link {expected_link} not found"
    
    # Verify no duplicates
    assert len(internal_links) == len(set(internal_links)), \
        "Internal links should not contain duplicates"


# ============================================================================
# PROPERTY 16: Crawl Rate Limiting
# ============================================================================

# Feature: wikipedia-intelligence-system, Property 16: Crawl Rate Limiting
@given(
    num_requests=st.integers(min_value=2, max_value=10),
    crawl_delay=st.floats(min_value=0.1, max_value=2.0)
)
@settings(
    suppress_health_check=[HealthCheck.function_scoped_fixture],
    max_examples=50,
    deadline=None
)
@pytest.mark.asyncio
async def test_property_16_crawl_rate_limiting(num_requests, crawl_delay):
    """Property 16: Crawl Rate Limiting
    
    For any crawl session, the System should respect robots.txt directives
    and implement delays when rate limits are encountered.
    
    Validates: Requirements 3.7
    """
    # Create pipeline with specific crawl delay
    pipeline = Crawl4AIPipeline(crawl_delay=crawl_delay, respect_robots_txt=True)
    
    # Track request timestamps
    request_times = []
    
    # Mock crawler
    async def mock_arun(url, config):
        request_times.append(asyncio.get_event_loop().time())
        
        result = Mock()
        result.success = True
        result.html = '<html><body><h1>Test</h1><p>Content with enough text for summary.</p></body></html>'
        result.error_message = None
        return result
    
    mock_crawler = AsyncMock()
    mock_crawler.arun = mock_arun
    mock_crawler.__aenter__ = AsyncMock(return_value=mock_crawler)
    mock_crawler.__aexit__ = AsyncMock()
    
    pipeline._crawler = mock_crawler
    
    # Make multiple requests to same domain
    base_url = "https://en.wikipedia.org/wiki/"
    for i in range(num_requests):
        url = f"{base_url}Article_{i}"
        await pipeline.crawl_article(url)
    
    # Verify rate limiting delays were applied
    for i in range(1, len(request_times)):
        time_diff = request_times[i] - request_times[i-1]
        
        # Allow small tolerance for timing variations
        assert time_diff >= (crawl_delay - 0.1), \
            f"Request {i} violated crawl delay: {time_diff:.3f}s < {crawl_delay}s"
    
    await pipeline.close()


# ============================================================================
# PROPERTY 17: Graceful Crawl Failure Handling
# ============================================================================

# Feature: wikipedia-intelligence-system, Property 17: Graceful Crawl Failure Handling
@given(
    failure_type=st.sampled_from(["network_error", "timeout", "invalid_html", "crawl_failure"])
)
@settings(
    suppress_health_check=[HealthCheck.function_scoped_fixture],
    max_examples=100,
    deadline=None
)
@pytest.mark.asyncio
async def test_property_17_graceful_crawl_failure_handling(failure_type):
    """Property 17: Graceful Crawl Failure Handling
    
    For any crawl failure (network error, timeout, invalid HTML), the System
    should log the error with context and continue processing other articles
    without crashing.
    
    Validates: Requirements 3.8
    """
    # Create pipeline with retries
    pipeline = Crawl4AIPipeline(max_retries=2, timeout=5.0)
    
    # Mock crawler to simulate failure
    if failure_type == "network_error":
        error = ConnectionError("Network connection failed")
    elif failure_type == "timeout":
        error = asyncio.TimeoutError("Request timed out")
    elif failure_type == "invalid_html":
        error = ValueError("Invalid HTML structure")
    else:  # crawl_failure
        error = None  # Will use result.success = False
    
    async def mock_arun(url, config):
        if error:
            raise error
        else:
            result = Mock()
            result.success = False
            result.html = ""
            result.error_message = "Crawl failed"
            return result
    
    mock_crawler = AsyncMock()
    mock_crawler.arun = mock_arun
    mock_crawler.__aenter__ = AsyncMock(return_value=mock_crawler)
    mock_crawler.__aexit__ = AsyncMock()
    
    pipeline._crawler = mock_crawler
    
    # Attempt to crawl - should raise RuntimeError after retries
    url = "https://en.wikipedia.org/wiki/Test_Article"
    
    with pytest.raises(RuntimeError) as exc_info:
        await pipeline.crawl_article(url)
    
    # Verify error message contains context
    error_message = str(exc_info.value)
    assert "Test_Article" in error_message or url in error_message, \
        "Error message should contain article context"
    
    await pipeline.close()


# Feature: wikipedia-intelligence-system, Property 17: Deep Crawl Continues After Failure
@given(
    num_articles=st.integers(min_value=3, max_value=10),
    failure_index=st.integers(min_value=0, max_value=2)
)
@settings(
    suppress_health_check=[HealthCheck.function_scoped_fixture],
    max_examples=50,
    deadline=None
)
@pytest.mark.asyncio
async def test_property_17_deep_crawl_continues_after_failure(num_articles, failure_index):
    """Property 17: Deep Crawl Continues After Individual Failures
    
    For any deep crawl where individual articles fail, the System should
    continue crawling other articles gracefully.
    
    Validates: Requirements 3.8
    """
    # Ensure failure_index is within range
    failure_index = failure_index % num_articles
    
    # Create pipeline
    pipeline = Crawl4AIPipeline(max_retries=1)
    
    # Track crawl attempts
    crawl_attempts = []
    
    # Mock crawler
    async def mock_arun(url, config):
        article_name = url.split('/wiki/')[-1]
        crawl_attempts.append(article_name)
        
        # Simulate failure for specific article
        if len(crawl_attempts) - 1 == failure_index:
            raise ConnectionError(f"Failed to crawl {article_name}")
        
        # Generate HTML with links
        html = f'''
        <html><body>
        <h1 class="firstHeading">{article_name}</h1>
        <div class="mw-parser-output">
            <p>This is article {article_name} with sufficient content for summary extraction.</p>
        </div>
        </body></html>
        '''
        
        result = Mock()
        result.success = True
        result.html = html
        result.error_message = None
        return result
    
    mock_crawler = AsyncMock()
    mock_crawler.arun = mock_arun
    mock_crawler.__aenter__ = AsyncMock(return_value=mock_crawler)
    mock_crawler.__aexit__ = AsyncMock()
    
    pipeline._crawler = mock_crawler
    
    # Perform deep crawl
    seed_url = "https://en.wikipedia.org/wiki/Seed_Article"
    articles = await pipeline.deep_crawl(
        seed_url=seed_url,
        max_depth=0,  # Only crawl seed
        max_articles=num_articles
    )
    
    # Verify crawl continued despite failure
    # Should have attempted all articles but only succeeded for non-failing ones
    assert len(crawl_attempts) >= 1, "Should have attempted to crawl articles"
    
    # Verify some articles were successfully crawled (all except the failed one)
    # Note: deep_crawl only crawls seed at depth 0, so we expect 1 article unless seed fails
    if failure_index == 0:
        # Seed failed, no articles crawled
        assert len(articles) == 0, "No articles should be crawled if seed fails"
    else:
        # Seed succeeded
        assert len(articles) >= 1, "Should have crawled at least the seed article"
    
    await pipeline.close()


