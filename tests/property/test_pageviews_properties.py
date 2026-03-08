"""
Property-based tests for Pageviews Collector.

Tests universal correctness properties for pageview data collection
using Hypothesis for randomized testing.

Feature: wikipedia-intelligence-system
"""

import pytest
from hypothesis import given, strategies as st, settings, assume
from datetime import datetime, timedelta
from unittest.mock import AsyncMock, MagicMock, patch
import asyncio

from src.data_ingestion.pageviews_collector import PageviewsCollector
from src.data_ingestion.api_client import WikimediaAPIClient
from src.storage.dto import PageviewRecord, TopArticleRecord, AggregateStats


# ============================================================================
# Test Strategies
# ============================================================================

@st.composite
def article_name_strategy(draw):
    """Generate valid Wikipedia article names."""
    # Wikipedia article names can contain letters, numbers, underscores, parentheses
    name = draw(st.text(
        min_size=1,
        max_size=50,
        alphabet=st.characters(
            whitelist_categories=('Lu', 'Ll', 'Nd'),
            whitelist_characters='_()- '
        )
    ))
    # Remove leading/trailing spaces
    return name.strip() or "Test_Article"


@st.composite
def date_range_strategy(draw):
    """Generate valid date ranges."""
    start = draw(st.datetimes(
        min_value=datetime(2020, 1, 1),
        max_value=datetime(2024, 12, 31)
    ))
    # End date is 1-30 days after start
    days_diff = draw(st.integers(min_value=1, max_value=30))
    end = start + timedelta(days=days_diff)
    return start, end


@st.composite
def pageview_api_response_strategy(draw, article: str, num_items: int = None):
    """Generate valid pageview API responses."""
    if num_items is None:
        num_items = draw(st.integers(min_value=1, max_value=10))
    
    items = []
    for i in range(num_items):
        timestamp = draw(st.integers(min_value=2020010100, max_value=2024123123))
        views = draw(st.integers(min_value=0, max_value=1000000))
        
        items.append({
            "project": "en.wikipedia",
            "article": article,
            "granularity": "daily",
            "timestamp": timestamp,
            "views": views
        })
    
    return {"items": items}


@st.composite
def top_articles_api_response_strategy(draw, num_articles: int = None):
    """Generate valid top articles API responses."""
    if num_articles is None:
        num_articles = draw(st.integers(min_value=1, max_value=100))
    
    articles = []
    for rank in range(1, num_articles + 1):
        article_name = draw(article_name_strategy())
        views = draw(st.integers(min_value=1000, max_value=10000000))
        
        articles.append({
            "article": article_name,
            "rank": rank,
            "views": views
        })
    
    # Sort by views descending to maintain rank order
    articles.sort(key=lambda x: x["views"], reverse=True)
    # Update ranks
    for i, article in enumerate(articles, 1):
        article["rank"] = i
    
    return {
        "items": [{
            "project": "en.wikipedia",
            "access": "all-access",
            "year": "2024",
            "month": "01",
            "day": "01",
            "articles": articles
        }]
    }


@st.composite
def aggregate_api_response_strategy(draw, num_days: int = None):
    """Generate valid aggregate API responses."""
    if num_days is None:
        num_days = draw(st.integers(min_value=1, max_value=30))
    
    items = []
    for i in range(num_days):
        timestamp = 20240101 + i
        views = draw(st.integers(min_value=1000000, max_value=1000000000))
        
        items.append({
            "project": "en.wikipedia",
            "access": "all-access",
            "agent": "user",
            "granularity": "daily",
            "timestamp": timestamp,
            "views": views
        })
    
    return {"items": items}


# ============================================================================
# Property 1: API Response Schema Validation
# ============================================================================

# Feature: wikipedia-intelligence-system, Property 1: API Response Schema Validation
@given(
    article=article_name_strategy(),
    response_data=pageview_api_response_strategy("Test_Article")
)
@settings(max_examples=100, deadline=None)
def test_property_1_api_response_schema_validation_valid(article, response_data):
    """
    Property 1: For any API response from Wikimedia Pageviews API,
    the System should validate the response against the expected schema
    and accept responses that match.
    
    **Validates: Requirements 1.1, 1.2, 1.3, 1.7**
    """
    # Create mock API client to avoid event loop issues
    mock_client = MagicMock(spec=WikimediaAPIClient)
    collector = PageviewsCollector(api_client=mock_client)
    
    # Update article name in response
    for item in response_data["items"]:
        item["article"] = article
    
    # Should not raise exception for valid response
    try:
        collector._validate_pageview_response(response_data, article)
    except ValueError as e:
        pytest.fail(f"Valid response rejected: {e}")


# Feature: wikipedia-intelligence-system, Property 1: API Response Schema Validation
@given(
    article=article_name_strategy(),
    missing_field=st.sampled_from(["project", "article", "granularity", "timestamp", "views"])
)
@settings(max_examples=100, deadline=None)
def test_property_1_api_response_schema_validation_invalid(article, missing_field):
    """
    Property 1: For any API response missing required fields,
    the System should reject the response with a validation error.
    
    **Validates: Requirements 1.1, 1.2, 1.3, 1.7**
    """
    # Create mock API client to avoid event loop issues
    mock_client = MagicMock(spec=WikimediaAPIClient)
    collector = PageviewsCollector(api_client=mock_client)
    
    # Create response with missing field
    invalid_response = {
        "items": [{
            "project": "en.wikipedia",
            "article": article,
            "granularity": "daily",
            "timestamp": 2024010100,
            "views": 1000
        }]
    }
    
    # Remove the specified field
    del invalid_response["items"][0][missing_field]
    
    # Should raise ValueError for invalid response
    with pytest.raises(ValueError, match="Missing required fields"):
        collector._validate_pageview_response(invalid_response, article)


# ============================================================================
# Property 2: Bot Traffic Filtering
# ============================================================================

# Feature: wikipedia-intelligence-system, Property 2: Bot Traffic Filtering
@given(
    article=article_name_strategy(),
    start_date=st.datetimes(min_value=datetime(2020, 1, 1), max_value=datetime(2024, 1, 1)),
    num_days=st.integers(min_value=1, max_value=7)
)
@settings(max_examples=100, deadline=None)
@pytest.mark.asyncio
async def test_property_2_bot_traffic_filtering(article, start_date, num_days):
    """
    Property 2: For any pageview request, the System should include bot filtering
    parameters and return separate counts for human and bot traffic.
    
    **Validates: Requirements 1.4**
    """
    end_date = start_date + timedelta(days=num_days)
    
    # Create mock API client
    mock_client = AsyncMock(spec=WikimediaAPIClient)
    
    # Create mock response with human views only (bot filtered)
    mock_response = {
        "items": [
            {
                "project": "en.wikipedia",
                "article": article,
                "granularity": "daily",
                "timestamp": int((start_date + timedelta(days=i)).strftime("%Y%m%d")),
                "views": 1000 + i * 100
            }
            for i in range(num_days)
        ]
    }
    
    mock_client.get = AsyncMock(return_value=mock_response)
    
    collector = PageviewsCollector(api_client=mock_client)
    
    # Fetch pageviews
    records = await collector.fetch_per_article(
        article=article,
        start_date=start_date,
        end_date=end_date,
        granularity="daily"
    )
    
    # Verify bot filtering: all records should have views_bot = 0
    # because we use agent_type=user which filters bots
    for record in records:
        assert record.views_bot == 0, \
            f"Bot views should be 0 (filtered), got {record.views_bot}"
        assert record.views_human >= 0, \
            f"Human views should be non-negative, got {record.views_human}"
        assert record.views_total == record.views_human + record.views_bot, \
            "Total views should equal human + bot views"
    
    # Verify API was called with agent_type=user (bot filtering)
    for call in mock_client.get.call_args_list:
        endpoint = call[0][0]
        assert "/user/" in endpoint, \
            f"Endpoint should include '/user/' for bot filtering: {endpoint}"


# ============================================================================
# Property 3: Device Segmentation
# ============================================================================

# Feature: wikipedia-intelligence-system, Property 3: Device Segmentation
@given(
    article=article_name_strategy(),
    start_date=st.datetimes(min_value=datetime(2020, 1, 1), max_value=datetime(2024, 1, 1)),
    num_days=st.integers(min_value=1, max_value=5)
)
@settings(max_examples=100, deadline=None)
@pytest.mark.asyncio
async def test_property_3_device_segmentation(article, start_date, num_days):
    """
    Property 3: For any pageview request, the System should return data
    segmented by all device types (desktop, mobile-web, mobile-app).
    
    **Validates: Requirements 1.5**
    """
    end_date = start_date + timedelta(days=num_days)
    
    # Create mock API client
    mock_client = AsyncMock(spec=WikimediaAPIClient)
    
    # Create mock responses for each device type
    def create_device_response(device_type: str):
        return {
            "items": [
                {
                    "project": "en.wikipedia",
                    "article": article,
                    "granularity": "daily",
                    "timestamp": int((start_date + timedelta(days=i)).strftime("%Y%m%d")),
                    "views": 1000 + i * 100
                }
                for i in range(num_days)
            ]
        }
    
    # Mock API to return different responses for each device type
    responses = [
        create_device_response("desktop"),
        create_device_response("mobile-web"),
        create_device_response("mobile-app")
    ]
    mock_client.get = AsyncMock(side_effect=responses)
    
    collector = PageviewsCollector(api_client=mock_client)
    
    # Fetch pageviews
    records = await collector.fetch_per_article(
        article=article,
        start_date=start_date,
        end_date=end_date,
        granularity="daily"
    )
    
    # Verify device segmentation: should have records for all device types
    device_types_found = set(record.device_type for record in records)
    expected_device_types = {"desktop", "mobile-web", "mobile-app"}
    
    assert device_types_found == expected_device_types, \
        f"Should have all device types. Expected {expected_device_types}, got {device_types_found}"
    
    # Verify each device type has the expected number of records
    for device_type in expected_device_types:
        device_records = [r for r in records if r.device_type == device_type]
        assert len(device_records) == num_days, \
            f"Device type {device_type} should have {num_days} records, got {len(device_records)}"
    
    # Verify API was called 3 times (once per device type)
    assert mock_client.get.call_count == 3, \
        f"API should be called 3 times (once per device), got {mock_client.get.call_count}"


# ============================================================================
# Additional Property Tests
# ============================================================================

# Feature: wikipedia-intelligence-system, Property 1: API Response Schema Validation
@given(response_data=top_articles_api_response_strategy())
@settings(max_examples=100, deadline=None)
def test_property_1_top_articles_schema_validation(response_data):
    """
    Property 1: For any top articles API response,
    the System should validate the response schema.
    
    **Validates: Requirements 1.2, 1.7**
    """
    # Create mock API client to avoid event loop issues
    mock_client = MagicMock(spec=WikimediaAPIClient)
    collector = PageviewsCollector(api_client=mock_client)
    
    # Should not raise exception for valid response
    try:
        collector._validate_top_articles_response(response_data)
    except ValueError as e:
        pytest.fail(f"Valid top articles response rejected: {e}")


# Feature: wikipedia-intelligence-system, Property 1: API Response Schema Validation
@given(response_data=aggregate_api_response_strategy())
@settings(max_examples=100, deadline=None)
def test_property_1_aggregate_schema_validation(response_data):
    """
    Property 1: For any aggregate API response,
    the System should validate the response schema.
    
    **Validates: Requirements 1.3, 1.7**
    """
    # Create mock API client to avoid event loop issues
    mock_client = MagicMock(spec=WikimediaAPIClient)
    collector = PageviewsCollector(api_client=mock_client)
    
    # Should not raise exception for valid response
    try:
        collector._validate_aggregate_response(response_data)
    except ValueError as e:
        pytest.fail(f"Valid aggregate response rejected: {e}")


# Feature: wikipedia-intelligence-system, Property 2: Bot Traffic Filtering
@given(
    date=st.datetimes(min_value=datetime(2020, 1, 1), max_value=datetime(2024, 12, 31)),
    num_articles=st.integers(min_value=10, max_value=100)
)
@settings(max_examples=100, deadline=None)
@pytest.mark.asyncio
async def test_property_2_top_articles_returns_ranked_list(date, num_articles):
    """
    Property 2: For any top articles request, the System should return
    articles ranked by views in descending order.
    
    **Validates: Requirements 1.2**
    """
    # Create mock API client
    mock_client = AsyncMock(spec=WikimediaAPIClient)
    
    # Create mock response with ranked articles
    articles = []
    for rank in range(1, num_articles + 1):
        articles.append({
            "article": f"Article_{rank}",
            "rank": rank,
            "views": 1000000 - (rank * 1000)  # Descending views
        })
    
    mock_response = {
        "items": [{
            "project": "en.wikipedia",
            "access": "all-access",
            "year": date.strftime("%Y"),
            "month": date.strftime("%m"),
            "day": date.strftime("%d"),
            "articles": articles
        }]
    }
    
    mock_client.get = AsyncMock(return_value=mock_response)
    
    collector = PageviewsCollector(api_client=mock_client)
    
    # Fetch top articles
    records = await collector.fetch_top_articles(date=date, limit=num_articles)
    
    # Verify ranking: ranks should be sequential from 1 to num_articles
    ranks = [record.rank for record in records]
    assert ranks == list(range(1, len(records) + 1)), \
        f"Ranks should be sequential from 1, got {ranks}"
    
    # Verify views are in descending order
    views = [record.views for record in records]
    assert views == sorted(views, reverse=True), \
        "Views should be in descending order"


# Feature: wikipedia-intelligence-system, Property 3: Device Segmentation
@given(
    start_date=st.datetimes(min_value=datetime(2020, 1, 1), max_value=datetime(2024, 1, 1)),
    num_days=st.integers(min_value=1, max_value=10)
)
@settings(max_examples=100, deadline=None)
@pytest.mark.asyncio
async def test_property_3_aggregate_calculates_totals(start_date, num_days):
    """
    Property 3: For any aggregate request, the System should calculate
    total views across all days in the range.
    
    **Validates: Requirements 1.3**
    """
    end_date = start_date + timedelta(days=num_days - 1)
    
    # Create mock API client
    mock_client = AsyncMock(spec=WikimediaAPIClient)
    
    # Create mock response with daily views
    daily_views = [1000000 + i * 100000 for i in range(num_days)]
    items = []
    for i, views in enumerate(daily_views):
        timestamp = int((start_date + timedelta(days=i)).strftime("%Y%m%d"))
        items.append({
            "project": "en.wikipedia",
            "access": "all-access",
            "agent": "user",
            "granularity": "daily",
            "timestamp": timestamp,
            "views": views
        })
    
    mock_response = {"items": items}
    mock_client.get = AsyncMock(return_value=mock_response)
    
    collector = PageviewsCollector(api_client=mock_client)
    
    # Fetch aggregate stats
    stats = await collector.fetch_aggregate(
        start_date=start_date,
        end_date=end_date
    )
    
    # Verify total views equals sum of daily views
    expected_total = sum(daily_views)
    assert stats.total_views == expected_total, \
        f"Total views should be {expected_total}, got {stats.total_views}"
    
    # Verify date range
    assert stats.start_date == start_date, "Start date should match"
    assert stats.end_date == end_date, "End date should match"
    
    # Verify average calculation
    expected_avg = expected_total / stats.total_articles
    assert abs(stats.avg_views_per_article - expected_avg) < 0.01, \
        f"Average views per article should be {expected_avg}, got {stats.avg_views_per_article}"
