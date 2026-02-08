"""Property-based tests for data acquisition module.

Tests Properties 1, 2, and 3 from the design document:
- Property 1: Data Acquisition Completeness
- Property 2: Bot Traffic Exclusion
- Property 3: Multi-Source Data Alignment
"""

from datetime import date, timedelta
from unittest.mock import Mock, patch
import pandas as pd
import pytest
from hypothesis import given, settings, strategies as st, assume, HealthCheck

from wikipedia_health.data_acquisition import WikimediaAPIClient, DataValidator
from wikipedia_health.models.data_models import TimeSeriesData


# Strategy for generating valid dates within 2015-2025 range
valid_dates = st.dates(
    min_value=date(2015, 1, 1),
    max_value=date(2025, 12, 31)
)

# Strategy for platforms
platforms_strategy = st.lists(
    st.sampled_from(['desktop', 'mobile-web', 'mobile-app']),
    min_size=1,
    max_size=3,
    unique=True
)


def create_mock_pageviews_response(start_date: date, end_date: date, platform: str, agent_type: str):
    """Create a mock API response for pageviews."""
    items = []
    current_date = start_date
    while current_date <= end_date:
        timestamp = current_date.strftime('%Y%m%d') + '00'
        items.append({
            'timestamp': timestamp,
            'views': 1000000 + (hash(current_date) % 100000),  # Deterministic but varied
            'project': 'all-projects',
            'access': platform,
            'agent': agent_type
        })
        current_date += timedelta(days=1)
    
    return {'items': items}


def create_mock_editors_response(start_date: date, end_date: date):
    """Create a mock API response for editor counts."""
    items = []
    current_date = start_date
    while current_date <= end_date:
        timestamp = current_date.strftime('%Y%m%d') + '00'
        items.append({
            'timestamp': timestamp,
            'editors': 5000 + (hash(current_date) % 500),
            'project': 'all-projects'
        })
        current_date += timedelta(days=1)
    
    return {'items': items}


def create_mock_edits_response(start_date: date, end_date: date):
    """Create a mock API response for edit volumes."""
    items = []
    current_date = start_date
    while current_date <= end_date:
        timestamp = current_date.strftime('%Y%m%d') + '00'
        items.append({
            'timestamp': timestamp,
            'edits': 50000 + (hash(current_date) % 5000),
            'project': 'all-projects'
        })
        current_date += timedelta(days=1)
    
    return {'items': items}


@pytest.mark.property
@settings(max_examples=20, deadline=None, suppress_health_check=[HealthCheck.filter_too_much])
@given(
    start_date=valid_dates,
    end_date=valid_dates,
    platforms=platforms_strategy
)
def test_property_1_data_acquisition_completeness(start_date, end_date, platforms):
    """
    Feature: wikipedia-product-health-analysis
    Property 1: For any requested date range within 2015-2025 and any platform combination,
    when data is fetched from the Wikimedia APIs, the returned data should cover the complete
    requested range with all specified platforms, and any gaps or anomalies should be flagged.
    
    Validates: Requirements 1.1, 1.3, 1.6
    """
    # Ensure start_date <= end_date
    if end_date < start_date:
        start_date, end_date = end_date, start_date
    
    # Limit date range to avoid excessive test time (increased to 180 days)
    assume((end_date - start_date).days <= 180)
    
    client = WikimediaAPIClient()
    validator = DataValidator()
    
    # Mock the API requests
    with patch.object(client, '_make_request_with_retry') as mock_request:
        # Setup mock to return complete data for each platform
        def mock_response(url, params=None):
            mock_resp = Mock()
            
            # Determine which endpoint based on URL
            if 'pageviews' in url:
                # Extract platform from URL
                for platform in platforms:
                    api_platform = 'all-access' if platform == 'all' else platform
                    if api_platform in url:
                        mock_resp.json.return_value = create_mock_pageviews_response(
                            start_date, end_date, platform, 'user'
                        )
                        break
            
            return mock_resp
        
        mock_request.side_effect = mock_response
        
        # Act: Fetch pageviews
        data = client.fetch_pageviews(start_date, end_date, platforms, agent_type='user')
        
        # Assert: Data should not be None or empty
        assert data is not None
        assert not data.empty
        
        # Assert: All requested platforms should be present
        actual_platforms = set(data['platform'].unique())
        expected_platforms = set(platforms)
        assert actual_platforms == expected_platforms, \
            f"Expected platforms {expected_platforms}, got {actual_platforms}"
        
        # Assert: Date range should be covered
        data['date'] = pd.to_datetime(data['date'])
        assert data['date'].min().date() >= start_date
        assert data['date'].max().date() <= end_date
        
        # Check completeness with validator
        validation_report = validator.check_completeness(
            data,
            (start_date, end_date)
        )
        
        # Assert: If there are missing dates, they should be flagged
        expected_dates = pd.date_range(start=start_date, end=end_date, freq='D')
        actual_dates = set(data['date'].dt.date)
        missing_dates = set(d.date() for d in expected_dates) - actual_dates
        
        if missing_dates:
            assert len(validation_report.missing_dates) > 0, \
                "Missing dates should be flagged in validation report"
            assert set(validation_report.missing_dates) == missing_dates, \
                "Flagged missing dates should match actual missing dates"
        else:
            # Complete data should have high completeness score
            assert validation_report.completeness_score >= 0.9


@pytest.mark.property
@settings(max_examples=20, deadline=None, suppress_health_check=[HealthCheck.filter_too_much])
@given(
    start_date=valid_dates,
    end_date=valid_dates,
    platforms=platforms_strategy
)
def test_property_2_bot_traffic_exclusion(start_date, end_date, platforms):
    """
    Feature: wikipedia-product-health-analysis
    Property 2: For any pageview data fetched by the system, all records should have
    agent_type='user' and no bot traffic should be present in the dataset.
    
    Validates: Requirements 1.2
    """
    # Ensure start_date <= end_date
    if end_date < start_date:
        start_date, end_date = end_date, start_date
    
    # Limit date range
    assume((end_date - start_date).days <= 180)
    
    client = WikimediaAPIClient()
    
    # Mock the API requests
    with patch.object(client, '_make_request_with_retry') as mock_request:
        def mock_response(url, params=None):
            mock_resp = Mock()
            
            # Verify that 'user' agent type is in the URL
            assert '/user/' in url, "API should be called with 'user' agent type"
            
            # Return mock data
            for platform in platforms:
                api_platform = 'all-access' if platform == 'all' else platform
                if api_platform in url:
                    mock_resp.json.return_value = create_mock_pageviews_response(
                        start_date, end_date, platform, 'user'
                    )
                    break
            
            return mock_resp
        
        mock_request.side_effect = mock_response
        
        # Act: Fetch pageviews with agent_type='user'
        data = client.fetch_pageviews(start_date, end_date, platforms, agent_type='user')
        
        # Assert: All records should have agent_type='user'
        assert 'agent_type' in data.columns
        assert (data['agent_type'] == 'user').all(), \
            "All records should have agent_type='user' (no bot traffic)"


@pytest.mark.property
@settings(max_examples=20, deadline=None, suppress_health_check=[HealthCheck.filter_too_much])
@given(
    start_date=valid_dates,
    end_date=valid_dates
)
def test_property_3_multi_source_data_alignment(start_date, end_date):
    """
    Feature: wikipedia-product-health-analysis
    Property 3: For any analysis period, pageview data, editor count data, and edit volume
    data should all cover the same date range, enabling cross-validation across data sources.
    
    Validates: Requirements 1.4, 1.5
    """
    # Ensure start_date <= end_date
    if end_date < start_date:
        start_date, end_date = end_date, start_date
    
    # Limit date range
    assume((end_date - start_date).days <= 180)
    
    client = WikimediaAPIClient()
    
    # Mock the API requests
    with patch.object(client, '_make_request_with_retry') as mock_request:
        def mock_response(url, params=None):
            mock_resp = Mock()
            
            if 'pageviews' in url:
                mock_resp.json.return_value = create_mock_pageviews_response(
                    start_date, end_date, 'desktop', 'user'
                )
            elif 'editors' in url:
                mock_resp.json.return_value = create_mock_editors_response(
                    start_date, end_date
                )
            elif 'edits' in url:
                mock_resp.json.return_value = create_mock_edits_response(
                    start_date, end_date
                )
            
            return mock_resp
        
        mock_request.side_effect = mock_response
        
        # Act: Fetch data from all three sources
        pageviews_data = client.fetch_pageviews(start_date, end_date, ['desktop'])
        editors_data = client.fetch_editor_counts(start_date, end_date)
        edits_data = client.fetch_edit_volumes(start_date, end_date)
        
        # Assert: All data sources should be non-empty
        assert not pageviews_data.empty
        assert not editors_data.empty
        assert not edits_data.empty
        
        # Assert: All data sources should cover the same date range
        pageviews_data['date'] = pd.to_datetime(pageviews_data['date'])
        editors_data['date'] = pd.to_datetime(editors_data['date'])
        edits_data['date'] = pd.to_datetime(edits_data['date'])
        
        pageviews_min = pageviews_data['date'].min().date()
        pageviews_max = pageviews_data['date'].max().date()
        
        editors_min = editors_data['date'].min().date()
        editors_max = editors_data['date'].max().date()
        
        edits_min = edits_data['date'].min().date()
        edits_max = edits_data['date'].max().date()
        
        # All sources should start at or after the requested start date
        assert pageviews_min >= start_date
        assert editors_min >= start_date
        assert edits_min >= start_date
        
        # All sources should end at or before the requested end date
        assert pageviews_max <= end_date
        assert editors_max <= end_date
        assert edits_max <= end_date
        
        # Date ranges should be aligned (within tolerance)
        # They should all cover approximately the same period
        date_range_days = (end_date - start_date).days
        
        pageviews_days = (pageviews_max - pageviews_min).days
        editors_days = (editors_max - editors_min).days
        edits_days = (edits_max - edits_min).days
        
        # All should cover at least 90% of the requested range
        min_coverage = 0.9 * date_range_days
        assert pageviews_days >= min_coverage
        assert editors_days >= min_coverage
        assert edits_days >= min_coverage
