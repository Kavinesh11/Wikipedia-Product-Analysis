"""
Unit tests for Pageviews Collector.

Tests specific examples, edge cases, and error handling for pageview collection.
"""

import pytest
from datetime import datetime, timedelta
from unittest.mock import AsyncMock, MagicMock, patch
from aiohttp import ClientError

from src.data_ingestion.pageviews_collector import PageviewsCollector
from src.data_ingestion.api_client import WikimediaAPIClient
from src.storage.dto import PageviewRecord, TopArticleRecord, AggregateStats


# ============================================================================
# Fixtures
# ============================================================================

@pytest.fixture
def mock_api_client():
    """Create mock API client."""
    client = AsyncMock(spec=WikimediaAPIClient)
    return client


@pytest.fixture
def collector(mock_api_client):
    """Create PageviewsCollector with mock API client."""
    return PageviewsCollector(api_client=mock_api_client)


# ============================================================================
# Test fetch_per_article with Mock API Responses
# ============================================================================

@pytest.mark.asyncio
async def test_fetch_per_article_success(collector, mock_api_client):
    """Test successful pageview collection for a specific article."""
    article = "Python_(programming_language)"
    start_date = datetime(2024, 1, 1)
    end_date = datetime(2024, 1, 3)
    
    # Mock API response for each device type
    mock_response = {
        "items": [
            {
                "project": "en.wikipedia",
                "article": article,
                "granularity": "daily",
                "timestamp": 20240101,
                "views": 10000
            },
            {
                "project": "en.wikipedia",
                "article": article,
                "granularity": "daily",
                "timestamp": 20240102,
                "views": 12000
            },
            {
                "project": "en.wikipedia",
                "article": article,
                "granularity": "daily",
                "timestamp": 20240103,
                "views": 11000
            }
        ]
    }
    
    mock_api_client.get = AsyncMock(return_value=mock_response)
    
    # Fetch pageviews
    records = await collector.fetch_per_article(
        article=article,
        start_date=start_date,
        end_date=end_date,
        granularity="daily"
    )
    
    # Verify results
    assert len(records) == 9  # 3 days * 3 device types
    assert all(isinstance(r, PageviewRecord) for r in records)
    assert all(r.article == article for r in records)
    
    # Verify device types
    device_types = set(r.device_type for r in records)
    assert device_types == {"desktop", "mobile-web", "mobile-app"}
    
    # Verify API was called 3 times (once per device type)
    assert mock_api_client.get.call_count == 3


@pytest.mark.asyncio
async def test_fetch_per_article_hourly_granularity(collector, mock_api_client):
    """Test pageview collection with hourly granularity."""
    article = "Machine_Learning"
    start_date = datetime(2024, 1, 1, 0)
    end_date = datetime(2024, 1, 1, 2)
    
    # Mock API response with hourly data
    mock_response = {
        "items": [
            {
                "project": "en.wikipedia",
                "article": article,
                "granularity": "hourly",
                "timestamp": 2024010100,
                "views": 1000
            },
            {
                "project": "en.wikipedia",
                "article": article,
                "granularity": "hourly",
                "timestamp": 2024010101,
                "views": 1100
            },
            {
                "project": "en.wikipedia",
                "article": article,
                "granularity": "hourly",
                "timestamp": 2024010102,
                "views": 1200
            }
        ]
    }
    
    mock_api_client.get = AsyncMock(return_value=mock_response)
    
    # Fetch pageviews
    records = await collector.fetch_per_article(
        article=article,
        start_date=start_date,
        end_date=end_date,
        granularity="hourly"
    )
    
    # Verify hourly timestamps
    assert len(records) == 9  # 3 hours * 3 device types
    
    # Check that timestamps are hourly
    timestamps = sorted(set(r.timestamp for r in records))
    assert len(timestamps) == 3


@pytest.mark.asyncio
async def test_fetch_per_article_special_characters_in_title(collector, mock_api_client):
    """Test article title with special characters is properly URL-encoded."""
    article = "C++ (programming language)"
    start_date = datetime(2024, 1, 1)
    end_date = datetime(2024, 1, 1)
    
    mock_response = {
        "items": [{
            "project": "en.wikipedia",
            "article": article,
            "granularity": "daily",
            "timestamp": 20240101,
            "views": 5000
        }]
    }
    
    mock_api_client.get = AsyncMock(return_value=mock_response)
    
    # Fetch pageviews
    records = await collector.fetch_per_article(
        article=article,
        start_date=start_date,
        end_date=end_date
    )
    
    # Verify API was called with URL-encoded article name
    for call in mock_api_client.get.call_args_list:
        endpoint = call[0][0]
        # Should contain URL-encoded version (spaces become underscores, special chars encoded)
        assert "C%2B%2B" in endpoint or "C++" in endpoint.replace("%20", "_")


# ============================================================================
# Test fetch_top_articles
# ============================================================================

@pytest.mark.asyncio
async def test_fetch_top_articles_success(collector, mock_api_client):
    """Test successful top articles collection."""
    date = datetime(2024, 1, 15)
    
    # Mock API response
    mock_response = {
        "items": [{
            "project": "en.wikipedia",
            "access": "all-access",
            "year": "2024",
            "month": "01",
            "day": "15",
            "articles": [
                {"article": "Main_Page", "rank": 1, "views": 5000000},
                {"article": "Python_(programming_language)", "rank": 2, "views": 1000000},
                {"article": "Machine_Learning", "rank": 3, "views": 500000}
            ]
        }]
    }
    
    mock_api_client.get = AsyncMock(return_value=mock_response)
    
    # Fetch top articles
    records = await collector.fetch_top_articles(date=date, limit=10)
    
    # Verify results
    assert len(records) == 3
    assert all(isinstance(r, TopArticleRecord) for r in records)
    assert records[0].article == "Main_Page"
    assert records[0].rank == 1
    assert records[0].views == 5000000
    assert all(r.date == date for r in records)


@pytest.mark.asyncio
async def test_fetch_top_articles_with_limit(collector, mock_api_client):
    """Test top articles collection respects limit parameter."""
    date = datetime(2024, 1, 15)
    limit = 2
    
    # Mock API response with more articles than limit
    articles = [
        {"article": f"Article_{i}", "rank": i, "views": 1000000 - i * 1000}
        for i in range(1, 11)
    ]
    
    mock_response = {
        "items": [{
            "project": "en.wikipedia",
            "access": "all-access",
            "year": "2024",
            "month": "01",
            "day": "15",
            "articles": articles
        }]
    }
    
    mock_api_client.get = AsyncMock(return_value=mock_response)
    
    # Fetch top articles with limit
    records = await collector.fetch_top_articles(date=date, limit=limit)
    
    # Verify limit is respected
    assert len(records) == limit
    assert records[0].rank == 1
    assert records[1].rank == 2


@pytest.mark.asyncio
async def test_fetch_top_articles_empty_results(collector, mock_api_client):
    """Test top articles with empty results."""
    date = datetime(2024, 1, 15)
    
    # Mock API response with no articles
    mock_response = {
        "items": [{
            "project": "en.wikipedia",
            "access": "all-access",
            "year": "2024",
            "month": "01",
            "day": "15",
            "articles": []
        }]
    }
    
    mock_api_client.get = AsyncMock(return_value=mock_response)
    
    # Fetch top articles
    records = await collector.fetch_top_articles(date=date)
    
    # Verify empty results
    assert len(records) == 0


# ============================================================================
# Test fetch_aggregate
# ============================================================================

@pytest.mark.asyncio
async def test_fetch_aggregate_success(collector, mock_api_client):
    """Test successful aggregate statistics collection."""
    start_date = datetime(2024, 1, 1)
    end_date = datetime(2024, 1, 3)
    
    # Mock API response
    mock_response = {
        "items": [
            {
                "project": "en.wikipedia",
                "access": "all-access",
                "agent": "user",
                "granularity": "daily",
                "timestamp": 20240101,
                "views": 100000000
            },
            {
                "project": "en.wikipedia",
                "access": "all-access",
                "agent": "user",
                "granularity": "daily",
                "timestamp": 20240102,
                "views": 110000000
            },
            {
                "project": "en.wikipedia",
                "access": "all-access",
                "agent": "user",
                "granularity": "daily",
                "timestamp": 20240103,
                "views": 105000000
            }
        ]
    }
    
    mock_api_client.get = AsyncMock(return_value=mock_response)
    
    # Fetch aggregate stats
    stats = await collector.fetch_aggregate(
        start_date=start_date,
        end_date=end_date
    )
    
    # Verify results
    assert isinstance(stats, AggregateStats)
    assert stats.start_date == start_date
    assert stats.end_date == end_date
    assert stats.total_views == 315000000  # Sum of all views
    assert stats.total_articles > 0
    assert stats.avg_views_per_article > 0


# ============================================================================
# Test Error Handling
# ============================================================================

@pytest.mark.asyncio
async def test_fetch_per_article_404_error(collector, mock_api_client):
    """Test handling of 404 error for non-existent article."""
    article = "NonExistent_Article_12345"
    start_date = datetime(2024, 1, 1)
    end_date = datetime(2024, 1, 1)
    
    # Mock API to raise ClientError (404)
    mock_api_client.get = AsyncMock(
        side_effect=ClientError("Client error 404: Not Found")
    )
    
    # Should return empty list (all device types failed gracefully)
    records = await collector.fetch_per_article(
        article=article,
        start_date=start_date,
        end_date=end_date
    )
    
    # Verify empty results due to all failures
    assert len(records) == 0


@pytest.mark.asyncio
async def test_fetch_per_article_5xx_error(collector, mock_api_client):
    """Test handling of 5xx server error."""
    article = "Test_Article"
    start_date = datetime(2024, 1, 1)
    end_date = datetime(2024, 1, 1)
    
    # Mock API to raise ClientError (500)
    mock_api_client.get = AsyncMock(
        side_effect=ClientError("Server error 500 after retries")
    )
    
    # Should return empty list (all device types failed gracefully)
    records = await collector.fetch_per_article(
        article=article,
        start_date=start_date,
        end_date=end_date
    )
    
    # Verify empty results due to all failures
    assert len(records) == 0


@pytest.mark.asyncio
async def test_fetch_per_article_timeout(collector, mock_api_client):
    """Test handling of timeout error."""
    article = "Test_Article"
    start_date = datetime(2024, 1, 1)
    end_date = datetime(2024, 1, 1)
    
    # Mock API to raise timeout
    import asyncio
    mock_api_client.get = AsyncMock(side_effect=asyncio.TimeoutError())
    
    # Should return empty list (all device types failed gracefully)
    records = await collector.fetch_per_article(
        article=article,
        start_date=start_date,
        end_date=end_date
    )
    
    # Verify empty results due to all failures
    assert len(records) == 0


@pytest.mark.asyncio
async def test_fetch_per_article_invalid_granularity(collector):
    """Test error handling for invalid granularity."""
    article = "Test_Article"
    start_date = datetime(2024, 1, 1)
    end_date = datetime(2024, 1, 1)
    
    # Should raise ValueError for invalid granularity
    with pytest.raises(ValueError, match="Invalid granularity"):
        await collector.fetch_per_article(
            article=article,
            start_date=start_date,
            end_date=end_date,
            granularity="weekly"  # Invalid
        )


@pytest.mark.asyncio
async def test_fetch_per_article_invalid_date_range(collector):
    """Test error handling for invalid date range (start > end)."""
    article = "Test_Article"
    start_date = datetime(2024, 1, 10)
    end_date = datetime(2024, 1, 1)  # Before start
    
    # Should raise ValueError
    with pytest.raises(ValueError, match="start_date must be before"):
        await collector.fetch_per_article(
            article=article,
            start_date=start_date,
            end_date=end_date
        )


@pytest.mark.asyncio
async def test_fetch_aggregate_invalid_date_range(collector):
    """Test error handling for invalid date range in aggregate."""
    start_date = datetime(2024, 1, 10)
    end_date = datetime(2024, 1, 1)  # Before start
    
    # Should raise ValueError
    with pytest.raises(ValueError, match="start_date must be before"):
        await collector.fetch_aggregate(
            start_date=start_date,
            end_date=end_date
        )


# ============================================================================
# Test Edge Cases
# ============================================================================

@pytest.mark.asyncio
async def test_fetch_per_article_empty_results(collector, mock_api_client):
    """Test handling of empty results (no pageviews)."""
    article = "Obscure_Article"
    start_date = datetime(2024, 1, 1)
    end_date = datetime(2024, 1, 1)
    
    # Mock API response with empty items
    mock_response = {"items": []}
    mock_api_client.get = AsyncMock(return_value=mock_response)
    
    # Fetch pageviews
    records = await collector.fetch_per_article(
        article=article,
        start_date=start_date,
        end_date=end_date
    )
    
    # Should return empty list
    assert len(records) == 0


@pytest.mark.asyncio
async def test_fetch_per_article_single_day(collector, mock_api_client):
    """Test pageview collection for a single day."""
    article = "Test_Article"
    date = datetime(2024, 1, 15)
    
    # Mock API response
    mock_response = {
        "items": [{
            "project": "en.wikipedia",
            "article": article,
            "granularity": "daily",
            "timestamp": 20240115,
            "views": 5000
        }]
    }
    
    mock_api_client.get = AsyncMock(return_value=mock_response)
    
    # Fetch pageviews for single day
    records = await collector.fetch_per_article(
        article=article,
        start_date=date,
        end_date=date
    )
    
    # Should have 3 records (one per device type)
    assert len(records) == 3
    assert all(r.timestamp.date() == date.date() for r in records)


@pytest.mark.asyncio
async def test_fetch_per_article_partial_device_failure(collector, mock_api_client):
    """Test handling when one device type fails but others succeed."""
    article = "Test_Article"
    start_date = datetime(2024, 1, 1)
    end_date = datetime(2024, 1, 1)
    
    # Mock API to succeed for first two calls, fail for third
    mock_response = {
        "items": [{
            "project": "en.wikipedia",
            "article": article,
            "granularity": "daily",
            "timestamp": 20240101,
            "views": 1000
        }]
    }
    
    mock_api_client.get = AsyncMock(
        side_effect=[
            mock_response,  # desktop - success
            mock_response,  # mobile-web - success
            ClientError("Failed")  # mobile-app - failure
        ]
    )
    
    # Fetch pageviews
    records = await collector.fetch_per_article(
        article=article,
        start_date=start_date,
        end_date=end_date
    )
    
    # Should have 2 records (from successful device types)
    assert len(records) == 2
    device_types = set(r.device_type for r in records)
    assert len(device_types) == 2  # Two device types succeeded


# ============================================================================
# Test Schema Validation
# ============================================================================

def test_validate_pageview_response_missing_items(collector):
    """Test validation rejects response without 'items' field."""
    invalid_response = {"data": []}
    
    with pytest.raises(ValueError, match="Missing 'items' field"):
        collector._validate_pageview_response(invalid_response, "Test_Article")


def test_validate_pageview_response_invalid_items_type(collector):
    """Test validation rejects response with non-list 'items'."""
    invalid_response = {"items": "not a list"}
    
    with pytest.raises(ValueError, match="Invalid 'items' type"):
        collector._validate_pageview_response(invalid_response, "Test_Article")


def test_validate_pageview_response_missing_required_fields(collector):
    """Test validation rejects items missing required fields."""
    invalid_response = {
        "items": [{
            "project": "en.wikipedia",
            "article": "Test",
            # Missing: granularity, timestamp, views
        }]
    }
    
    with pytest.raises(ValueError, match="Missing required fields"):
        collector._validate_pageview_response(invalid_response, "Test_Article")


def test_validate_top_articles_response_missing_articles(collector):
    """Test validation rejects top articles response without 'articles' field."""
    invalid_response = {
        "items": [{
            "project": "en.wikipedia",
            # Missing: articles
        }]
    }
    
    with pytest.raises(ValueError, match="Missing 'articles' field"):
        collector._validate_top_articles_response(invalid_response)


def test_validate_aggregate_response_valid(collector):
    """Test validation accepts valid aggregate response."""
    valid_response = {
        "items": [{
            "project": "en.wikipedia",
            "timestamp": 20240101,
            "views": 100000000
        }]
    }
    
    # Should not raise exception
    collector._validate_aggregate_response(valid_response)


# ============================================================================
# Test Date Formatting
# ============================================================================

def test_format_date_daily(collector):
    """Test date formatting for daily granularity."""
    date = datetime(2024, 1, 15, 14, 30)
    formatted = collector._format_date(date, "daily")
    assert formatted == "20240115"


def test_format_date_hourly(collector):
    """Test date formatting for hourly granularity."""
    date = datetime(2024, 1, 15, 14, 30)
    formatted = collector._format_date(date, "hourly")
    assert formatted == "2024011514"


def test_format_date_monthly(collector):
    """Test date formatting for monthly granularity."""
    date = datetime(2024, 1, 15, 14, 30)
    formatted = collector._format_date(date, "monthly")
    assert formatted == "202401"


def test_format_date_invalid_granularity(collector):
    """Test date formatting with invalid granularity."""
    date = datetime(2024, 1, 15)
    
    with pytest.raises(ValueError, match="Invalid granularity"):
        collector._format_date(date, "weekly")


# ============================================================================
# Test Context Manager
# ============================================================================

@pytest.mark.asyncio
async def test_context_manager(mock_api_client):
    """Test PageviewsCollector as async context manager."""
    async with PageviewsCollector(api_client=mock_api_client) as collector:
        assert collector is not None
        assert isinstance(collector, PageviewsCollector)
    
    # Verify close was called
    mock_api_client.close.assert_called_once()
