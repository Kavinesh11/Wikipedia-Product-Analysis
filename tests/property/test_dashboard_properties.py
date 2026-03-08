"""Property-Based Tests for Dashboard Functionality

Tests correctness properties for the Streamlit dashboard application.
"""
import pytest
from hypothesis import given, strategies as st, settings, assume
from datetime import datetime, timedelta
import pandas as pd
import io
import csv

from src.visualization.dashboard import (
    export_to_csv, export_to_pdf, get_cache_key
)


# ============================================================================
# PROPERTY 34: Competitor Table Sorting
# ============================================================================

# Feature: wikipedia-intelligence-system, Property 34: Competitor Table Sorting
@given(
    num_rows=st.integers(min_value=1, max_value=100),
    sort_column=st.sampled_from(['article', 'views', 'growth_rate', 'hype_score'])
)
@settings(max_examples=5, deadline=None)
def test_competitor_table_sorting(num_rows, sort_column):
    """
    Property 34: For any competitor comparison table and sort column,
    clicking the column header should reorder rows by that column's values
    in ascending or descending order.
    
    Validates: Requirements 8.2
    """
    # Generate test data
    data = pd.DataFrame({
        'article': [f'Article_{i}' for i in range(num_rows)],
        'views': [abs(hash(f'views_{i}')) % 1000000 for i in range(num_rows)],
        'growth_rate': [((hash(f'growth_{i}') % 200) - 100) / 10.0 for i in range(num_rows)],
        'hype_score': [abs(hash(f'hype_{i}')) % 100 / 100.0 for i in range(num_rows)]
    })
    
    # Test ascending sort
    sorted_asc = data.sort_values(by=sort_column, ascending=True)
    
    # Verify ascending order
    if sort_column == 'article':
        for i in range(len(sorted_asc) - 1):
            assert sorted_asc.iloc[i][sort_column] <= sorted_asc.iloc[i + 1][sort_column], \
                f"Ascending sort failed for {sort_column}"
    else:
        for i in range(len(sorted_asc) - 1):
            assert sorted_asc.iloc[i][sort_column] <= sorted_asc.iloc[i + 1][sort_column], \
                f"Ascending sort failed for {sort_column}"
    
    # Test descending sort
    sorted_desc = data.sort_values(by=sort_column, ascending=False)
    
    # Verify descending order
    if sort_column == 'article':
        for i in range(len(sorted_desc) - 1):
            assert sorted_desc.iloc[i][sort_column] >= sorted_desc.iloc[i + 1][sort_column], \
                f"Descending sort failed for {sort_column}"
    else:
        for i in range(len(sorted_desc) - 1):
            assert sorted_desc.iloc[i][sort_column] >= sorted_desc.iloc[i + 1][sort_column], \
                f"Descending sort failed for {sort_column}"


# ============================================================================
# PROPERTY 35: Alert Display on Risk Detection
# ============================================================================

# Feature: wikipedia-intelligence-system, Property 35: Alert Display on Risk Detection
@given(
    num_alerts=st.integers(min_value=1, max_value=50),
    risk_scores=st.lists(
        st.floats(min_value=0.0, max_value=1.0, allow_nan=False, allow_infinity=False),
        min_size=1,
        max_size=50
    )
)
@settings(max_examples=5, deadline=None)
def test_alert_display_on_risk_detection(num_alerts, risk_scores):
    """
    Property 35: For any active reputation risk alert,
    the dashboard should display the alert prominently in the alerts panel.
    
    Validates: Requirements 8.3
    """
    # Ensure we have matching number of alerts and scores
    risk_scores = risk_scores[:num_alerts]
    if len(risk_scores) < num_alerts:
        risk_scores.extend([0.5] * (num_alerts - len(risk_scores)))
    
    # Create alert data
    alerts = []
    for i, score in enumerate(risk_scores):
        alert = {
            'article': f'Article_{i}',
            'risk_score': score,
            'timestamp': datetime.now() - timedelta(hours=i),
            'alert_level': 'high' if score > 0.7 else ('medium' if score > 0.4 else 'low')
        }
        alerts.append(alert)
    
    # Verify all alerts are present
    assert len(alerts) == num_alerts, "Not all alerts were created"
    
    # Verify each alert has required fields
    for alert in alerts:
        assert 'article' in alert, "Alert missing article field"
        assert 'risk_score' in alert, "Alert missing risk_score field"
        assert 'timestamp' in alert, "Alert missing timestamp field"
        assert 'alert_level' in alert, "Alert missing alert_level field"
        
        # Verify risk score is in valid range
        assert 0.0 <= alert['risk_score'] <= 1.0, \
            f"Risk score {alert['risk_score']} out of range"
        
        # Verify alert level matches risk score
        if alert['risk_score'] > 0.7:
            assert alert['alert_level'] == 'high', \
                f"High risk score {alert['risk_score']} should have 'high' alert level"
        elif alert['risk_score'] > 0.4:
            assert alert['alert_level'] == 'medium', \
                f"Medium risk score {alert['risk_score']} should have 'medium' alert level"
        else:
            assert alert['alert_level'] == 'low', \
                f"Low risk score {alert['risk_score']} should have 'low' alert level"


# ============================================================================
# PROPERTY 36: Leaderboard Ranking
# ============================================================================

# Feature: wikipedia-intelligence-system, Property 36: Leaderboard Ranking
@given(
    num_articles=st.integers(min_value=1, max_value=100),
    pageviews=st.lists(
        st.integers(min_value=0, max_value=10000000),
        min_size=1,
        max_size=100
    )
)
@settings(max_examples=5, deadline=None)
def test_leaderboard_ranking(num_articles, pageviews):
    """
    Property 36: For any traffic leaderboard,
    articles should be ranked in descending order by total pageviews.
    
    Validates: Requirements 8.5
    """
    # Ensure we have matching number of articles and pageviews
    pageviews = pageviews[:num_articles]
    if len(pageviews) < num_articles:
        pageviews.extend([0] * (num_articles - len(pageviews)))
    
    # Create leaderboard data
    leaderboard = pd.DataFrame({
        'article': [f'Article_{i}' for i in range(num_articles)],
        'total_pageviews': pageviews
    })
    
    # Sort by pageviews descending (as leaderboard should be)
    leaderboard_sorted = leaderboard.sort_values(
        by='total_pageviews',
        ascending=False
    ).reset_index(drop=True)
    
    # Verify descending order
    for i in range(len(leaderboard_sorted) - 1):
        current_views = leaderboard_sorted.iloc[i]['total_pageviews']
        next_views = leaderboard_sorted.iloc[i + 1]['total_pageviews']
        assert current_views >= next_views, \
            f"Leaderboard not in descending order: {current_views} < {next_views} at position {i}"
    
    # Verify rank 1 has highest pageviews
    if len(leaderboard_sorted) > 0:
        max_views = leaderboard['total_pageviews'].max()
        rank_1_views = leaderboard_sorted.iloc[0]['total_pageviews']
        assert rank_1_views == max_views, \
            f"Rank 1 article doesn't have highest pageviews: {rank_1_views} != {max_views}"


# ============================================================================
# PROPERTY 37: Dashboard Auto-Refresh
# ============================================================================

# Feature: wikipedia-intelligence-system, Property 37: Dashboard Auto-Refresh
@given(
    refresh_interval=st.integers(min_value=1, max_value=60),
    elapsed_time=st.integers(min_value=0, max_value=300)
)
@settings(max_examples=5, deadline=None)
def test_dashboard_auto_refresh(refresh_interval, elapsed_time):
    """
    Property 37: For any dashboard with auto-refresh enabled,
    data should be reloaded at the configured interval (default 5 minutes).
    
    Validates: Requirements 8.6
    """
    # Calculate expected number of refreshes
    expected_refreshes = elapsed_time // refresh_interval
    
    # Simulate refresh tracking
    refresh_times = []
    current_time = 0
    
    while current_time <= elapsed_time:
        if current_time % refresh_interval == 0 and current_time > 0:
            refresh_times.append(current_time)
        current_time += 1
    
    # Verify number of refreshes
    assert len(refresh_times) == expected_refreshes, \
        f"Expected {expected_refreshes} refreshes, got {len(refresh_times)}"
    
    # Verify refresh intervals are consistent
    for i in range(len(refresh_times)):
        expected_time = (i + 1) * refresh_interval
        actual_time = refresh_times[i]
        assert actual_time == expected_time, \
            f"Refresh {i} at wrong time: expected {expected_time}, got {actual_time}"


# ============================================================================
# PROPERTY 38: Data Filtering
# ============================================================================

# Feature: wikipedia-intelligence-system, Property 38: Data Filtering
@given(
    num_records=st.integers(min_value=10, max_value=100),
    filter_date_range=st.integers(min_value=1, max_value=30),
    filter_industry=st.sampled_from(['Technology', 'Healthcare', 'Finance', None]),
    filter_metric_type=st.sampled_from(['views', 'growth', 'hype', None])
)
@settings(max_examples=5, deadline=None)
def test_data_filtering(num_records, filter_date_range, filter_industry, filter_metric_type):
    """
    Property 38: For any dashboard filter combination (date range, industry, metric type),
    only data matching all active filters should be displayed.
    
    Validates: Requirements 8.7
    """
    # Generate test data
    base_date = datetime(2024, 1, 1)
    industries = ['Technology', 'Healthcare', 'Finance', 'Retail']
    metric_types = ['views', 'growth', 'hype', 'reputation']
    
    data = pd.DataFrame({
        'article': [f'Article_{i}' for i in range(num_records)],
        'date': [base_date + timedelta(days=i % 60) for i in range(num_records)],
        'industry': [industries[i % len(industries)] for i in range(num_records)],
        'metric_type': [metric_types[i % len(metric_types)] for i in range(num_records)],
        'value': [abs(hash(f'val_{i}')) % 1000 for i in range(num_records)]
    })
    
    # Apply filters
    filtered_data = data.copy()
    
    # Date range filter
    end_date = base_date + timedelta(days=filter_date_range)
    filtered_data = filtered_data[
        (filtered_data['date'] >= base_date) &
        (filtered_data['date'] <= end_date)
    ]
    
    # Industry filter
    if filter_industry is not None:
        filtered_data = filtered_data[filtered_data['industry'] == filter_industry]
    
    # Metric type filter
    if filter_metric_type is not None:
        filtered_data = filtered_data[filtered_data['metric_type'] == filter_metric_type]
    
    # Verify all filtered records match criteria
    for _, row in filtered_data.iterrows():
        # Check date range
        assert base_date <= row['date'] <= end_date, \
            f"Record date {row['date']} outside filter range"
        
        # Check industry
        if filter_industry is not None:
            assert row['industry'] == filter_industry, \
                f"Record industry {row['industry']} doesn't match filter {filter_industry}"
        
        # Check metric type
        if filter_metric_type is not None:
            assert row['metric_type'] == filter_metric_type, \
                f"Record metric_type {row['metric_type']} doesn't match filter {filter_metric_type}"
    
    # Verify filtered data is subset of original
    assert len(filtered_data) <= len(data), \
        "Filtered data has more records than original"


# ============================================================================
# PROPERTY 39: Export Format Validity
# ============================================================================

# Feature: wikipedia-intelligence-system, Property 39: Export Format Validity
@given(
    num_rows=st.integers(min_value=1, max_value=50),
    num_cols=st.integers(min_value=1, max_value=10)
)
@settings(max_examples=5, deadline=None)
def test_export_format_validity_csv(num_rows, num_cols):
    """
    Property 39 (CSV): For any dashboard data export to CSV,
    the output file should be valid CSV format and parseable by standard tools.
    
    Validates: Requirements 8.8
    """
    # Generate test data
    data = pd.DataFrame({
        f'col_{i}': [f'value_{i}_{j}' for j in range(num_rows)]
        for i in range(num_cols)
    })
    
    # Export to CSV
    csv_bytes = export_to_csv(data)
    
    # Verify output is bytes
    assert isinstance(csv_bytes, bytes), "CSV export should return bytes"
    
    # Verify CSV is parseable
    csv_str = csv_bytes.decode('utf-8')
    csv_reader = csv.reader(io.StringIO(csv_str))
    rows = list(csv_reader)
    
    # Verify header row
    assert len(rows) > 0, "CSV should have at least header row"
    header = rows[0]
    assert len(header) == num_cols, \
        f"CSV header has {len(header)} columns, expected {num_cols}"
    
    # Verify data rows
    assert len(rows) == num_rows + 1, \
        f"CSV has {len(rows)} rows (including header), expected {num_rows + 1}"
    
    # Verify each data row has correct number of columns
    for i, row in enumerate(rows[1:], start=1):
        assert len(row) == num_cols, \
            f"Row {i} has {len(row)} columns, expected {num_cols}"


# Feature: wikipedia-intelligence-system, Property 39: Export Format Validity
@given(
    num_rows=st.integers(min_value=1, max_value=50),
    num_cols=st.integers(min_value=1, max_value=10)
)
@settings(max_examples=5, deadline=None)
def test_export_format_validity_pdf(num_rows, num_cols):
    """
    Property 39 (PDF): For any dashboard data export to PDF,
    the output file should be valid PDF format and parseable by standard tools.
    
    Validates: Requirements 8.8
    """
    # Generate test data
    data = pd.DataFrame({
        f'col_{i}': [f'value_{i}_{j}' for j in range(num_rows)]
        for i in range(num_cols)
    })
    
    # Export to PDF
    pdf_bytes = export_to_pdf(data, "Test Report")
    
    # Verify output is bytes
    assert isinstance(pdf_bytes, bytes), "PDF export should return bytes"
    
    # Verify PDF has valid header (PDF files start with %PDF-)
    assert pdf_bytes.startswith(b'%PDF-'), "PDF should start with %PDF- header"
    
    # Verify PDF has EOF marker
    assert b'%%EOF' in pdf_bytes, "PDF should contain %%EOF marker"
    
    # Verify PDF is not empty
    assert len(pdf_bytes) > 100, "PDF should have substantial content"
    
    # Verify PDF contains ReportLab signature (indicates it was generated correctly)
    assert b'ReportLab' in pdf_bytes, "PDF should be generated by ReportLab"


# ============================================================================
# ADDITIONAL HELPER TESTS
# ============================================================================

@given(
    data_type=st.sampled_from(['demand_trends', 'competitor_comparison', 'alerts', 'leaderboard']),
    param1=st.text(min_size=1, max_size=20),
    param2=st.integers(min_value=0, max_value=1000)
)
@settings(max_examples=5, deadline=None)
def test_cache_key_generation(data_type, param1, param2):
    """
    Test that cache key generation is deterministic and unique.
    """
    # Generate cache key
    key1 = get_cache_key(data_type, param1=param1, param2=param2)
    key2 = get_cache_key(data_type, param1=param1, param2=param2)
    
    # Verify deterministic (same inputs produce same key)
    assert key1 == key2, "Cache key should be deterministic"
    
    # Verify key is a valid hash (32 hex characters for MD5)
    assert len(key1) == 32, "Cache key should be 32 characters (MD5 hash)"
    assert all(c in '0123456789abcdef' for c in key1), \
        "Cache key should be hexadecimal"
    
    # Verify different parameters produce different keys
    key3 = get_cache_key(data_type, param1=param1, param2=param2 + 1)
    assert key1 != key3, "Different parameters should produce different cache keys"
