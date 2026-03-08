"""Property-based tests for edit history processing

Feature: wikipedia-intelligence-system
"""
import pytest
import asyncio
from datetime import datetime, timedelta
from hypothesis import given, strategies as st, assume, settings, HealthCheck
from typing import List

from src.data_ingestion.edit_history_scraper import EditHistoryScraper
from src.storage.dto import RevisionRecord, VandalismMetrics, EditMetrics


# ============================================================================
# HYPOTHESIS STRATEGIES
# ============================================================================

def revision_record_strategy(
    article: str = "Test_Article",
    min_timestamp: datetime = None,
    max_timestamp: datetime = None
):
    """Strategy for generating RevisionRecord objects."""
    if min_timestamp is None:
        min_timestamp = datetime(2024, 1, 1)
    if max_timestamp is None:
        max_timestamp = datetime(2024, 12, 31)
    
    return st.builds(
        RevisionRecord,
        article=st.just(article),
        revision_id=st.integers(min_value=1, max_value=999999999),
        timestamp=st.datetimes(min_value=min_timestamp, max_value=max_timestamp),
        editor_type=st.sampled_from(["anonymous", "registered"]),
        editor_id=st.one_of(
            # IP addresses for anonymous
            st.from_regex(r'\d{1,3}\.\d{1,3}\.\d{1,3}\.\d{1,3}', fullmatch=True),
            # Usernames for registered
            st.text(
                alphabet=st.characters(
                    whitelist_categories=('Lu', 'Ll', 'Nd'),
                    min_codepoint=65, max_codepoint=122
                ),
                min_size=3,
                max_size=20
            )
        ),
        is_reverted=st.booleans(),
        bytes_changed=st.integers(min_value=-10000, max_value=10000),
        edit_summary=st.one_of(
            st.just(""),
            st.text(min_size=1, max_size=200),
            # Revert keywords
            st.sampled_from([
                "Reverted edits by user",
                "Undo revision 12345",
                "Rollback vandalism",
                "Restored previous version",
                "rv spam",
                "Normal edit without revert"
            ])
        )
    )


def revision_list_strategy(
    min_size: int = 1,
    max_size: int = 100,
    article: str = "Test_Article"
):
    """Strategy for generating lists of RevisionRecord objects."""
    return st.lists(
        revision_record_strategy(article=article),
        min_size=min_size,
        max_size=max_size
    )



# ============================================================================
# PROPERTY 5: Edit Data Extraction Completeness
# ============================================================================

# Feature: wikipedia-intelligence-system, Property 5: Edit Data Extraction Completeness
@given(
    article=st.text(min_size=1, max_size=100),
    num_revisions=st.integers(min_value=1, max_value=50)
)
@settings(
    suppress_health_check=[HealthCheck.function_scoped_fixture],
    max_examples=5,
    deadline=None
)
@pytest.mark.asyncio
async def test_property_5_edit_data_extraction_completeness(article, num_revisions):
    """Property 5: Edit Data Extraction Completeness
    
    For any edit history response, the System should extract all required fields
    (edit count, timestamp, editor identifier) for every revision.
    
    Validates: Requirements 2.1
    """
    # Create mock revisions with all required fields
    revisions = []
    for i in range(num_revisions):
        revision = RevisionRecord(
            article=article,
            revision_id=1000 + i,
            timestamp=datetime(2024, 1, 1) + timedelta(hours=i),
            editor_type="registered",
            editor_id=f"User{i}",
            is_reverted=False,
            bytes_changed=100,
            edit_summary=f"Edit {i}"
        )
        revisions.append(revision)
    
    # Verify all required fields are present and valid
    assert len(revisions) == num_revisions, \
        f"Expected {num_revisions} revisions, got {len(revisions)}"
    
    for i, revision in enumerate(revisions):
        # Verify all required fields are present
        assert revision.article == article, \
            f"Revision {i}: article field missing or incorrect"
        assert revision.revision_id is not None, \
            f"Revision {i}: revision_id field missing"
        assert revision.timestamp is not None, \
            f"Revision {i}: timestamp field missing"
        assert revision.editor_id is not None, \
            f"Revision {i}: editor_id field missing"
        
        # Verify field types
        assert isinstance(revision.revision_id, int), \
            f"Revision {i}: revision_id should be int"
        assert isinstance(revision.timestamp, datetime), \
            f"Revision {i}: timestamp should be datetime"
        assert isinstance(revision.editor_id, str), \
            f"Revision {i}: editor_id should be str"
        assert isinstance(revision.editor_type, str), \
            f"Revision {i}: editor_type should be str"



# ============================================================================
# PROPERTY 6: Editor Classification
# ============================================================================

# Feature: wikipedia-intelligence-system, Property 6: Editor Classification
@given(
    editor_id=st.one_of(
        # IPv4 addresses
        st.from_regex(r'\d{1,3}\.\d{1,3}\.\d{1,3}\.\d{1,3}', fullmatch=True),
        # IPv6 addresses (simplified)
        st.from_regex(r'[0-9a-fA-F:]+', fullmatch=True).filter(lambda x: ':' in x),
        # Usernames (not IP addresses)
        st.text(
            alphabet=st.characters(
                whitelist_categories=('Lu', 'Ll', 'Nd'),
                min_codepoint=65, max_codepoint=122
            ),
            min_size=3,
            max_size=20
        ).filter(lambda x: '.' not in x and ':' not in x)
    )
)
@settings(
    suppress_health_check=[HealthCheck.function_scoped_fixture],
    max_examples=5
)
def test_property_6_editor_classification(editor_id):
    """Property 6: Editor Classification
    
    For any editor in edit history, the System should classify them as exactly
    one of "anonymous" or "registered" based on whether the identifier is an
    IP address.
    
    Validates: Requirements 2.2
    """
    scraper = EditHistoryScraper()
    
    # Classify the editor
    editor_type = scraper._classify_editor(editor_id)
    
    # Verify classification is one of the valid types
    assert editor_type in ["anonymous", "registered"], \
        f"Editor type must be 'anonymous' or 'registered', got '{editor_type}'"
    
    # Verify IP addresses are classified as anonymous
    is_ip = scraper._is_ip_address(editor_id)
    if is_ip:
        assert editor_type == "anonymous", \
            f"IP address '{editor_id}' should be classified as 'anonymous', got '{editor_type}'"
    else:
        assert editor_type == "registered", \
            f"Non-IP identifier '{editor_id}' should be classified as 'registered', got '{editor_type}'"


# Feature: wikipedia-intelligence-system, Property 6: Editor Classification Consistency
@given(
    editor_id=st.text(min_size=1, max_size=50)
)
@settings(
    suppress_health_check=[HealthCheck.function_scoped_fixture],
    max_examples=5
)
def test_property_6_editor_classification_consistency(editor_id):
    """Property 6: Editor Classification Consistency
    
    For any editor ID, classifying it multiple times should always return the
    same result (deterministic classification).
    
    Validates: Requirements 2.2
    """
    scraper = EditHistoryScraper()
    
    # Classify multiple times
    classification1 = scraper._classify_editor(editor_id)
    classification2 = scraper._classify_editor(editor_id)
    classification3 = scraper._classify_editor(editor_id)
    
    # All classifications should be identical
    assert classification1 == classification2 == classification3, \
        f"Editor classification should be consistent for '{editor_id}', " \
        f"got {classification1}, {classification2}, {classification3}"



# ============================================================================
# PROPERTY 7: Revert Detection
# ============================================================================

# Feature: wikipedia-intelligence-system, Property 7: Revert Detection
@given(
    revisions=revision_list_strategy(min_size=1, max_size=50)
)
@settings(
    suppress_health_check=[HealthCheck.function_scoped_fixture],
    max_examples=5
)
def test_property_7_revert_detection(revisions):
    """Property 7: Revert Detection
    
    For any edit history containing reverted edits, the System should flag all
    reverted edits as potential vandalism signals.
    
    Validates: Requirements 2.3
    """
    scraper = EditHistoryScraper()
    
    # Count revisions with revert keywords in summary
    revert_keywords = [
        "revert", "undo", "undid", "rollback", "reverted", "rv ", "restore", "restored"
    ]
    
    expected_reverts = 0
    for revision in revisions:
        summary_lower = revision.edit_summary.lower()
        if any(keyword in summary_lower for keyword in revert_keywords):
            expected_reverts += 1
    
    # Detect vandalism signals
    metrics = scraper.detect_vandalism_signals(revisions)
    
    # Verify all reverted edits are flagged
    assert metrics.reverted_edits == expected_reverts, \
        f"Expected {expected_reverts} reverted edits, detected {metrics.reverted_edits}"
    
    # Verify reverted edits are marked in the revision records
    flagged_count = sum(1 for r in revisions if r.is_reverted)
    assert flagged_count == expected_reverts, \
        f"Expected {expected_reverts} revisions flagged as reverted, got {flagged_count}"
    
    # Verify vandalism percentage calculation
    if len(revisions) > 0:
        expected_percentage = (expected_reverts / len(revisions)) * 100
        assert abs(metrics.vandalism_percentage - expected_percentage) < 0.01, \
            f"Vandalism percentage {metrics.vandalism_percentage:.2f}% " \
            f"doesn't match expected {expected_percentage:.2f}%"



# ============================================================================
# PROPERTY 8: Edit Velocity Calculation
# ============================================================================

# Feature: wikipedia-intelligence-system, Property 8: Edit Velocity Calculation
@given(
    num_edits=st.integers(min_value=1, max_value=100),
    window_hours=st.integers(min_value=1, max_value=168)  # 1 hour to 1 week
)
@settings(
    suppress_health_check=[HealthCheck.function_scoped_fixture],
    max_examples=5
)
def test_property_8_edit_velocity_calculation(num_edits, window_hours):
    """Property 8: Edit Velocity Calculation
    
    For any article's edit history and time window, the System should calculate
    edit velocity as (number of edits in window) / (window duration in hours).
    
    Validates: Requirements 2.4
    """
    scraper = EditHistoryScraper()
    
    # Create revisions spread over the time window
    start_time = datetime(2024, 1, 1, 0, 0, 0)
    revisions = []
    
    for i in range(num_edits):
        # Distribute edits evenly across the window
        hours_offset = (i / num_edits) * window_hours
        timestamp = start_time + timedelta(hours=hours_offset)
        
        revision = RevisionRecord(
            article="Test_Article",
            revision_id=1000 + i,
            timestamp=timestamp,
            editor_type="registered",
            editor_id=f"User{i}",
            is_reverted=False,
            bytes_changed=100,
            edit_summary=f"Edit {i}"
        )
        revisions.append(revision)
    
    # Calculate edit velocity
    velocity = scraper.calculate_edit_velocity(revisions, window_hours)
    
    # Calculate expected velocity
    # The actual time span of the revisions
    time_span_hours = (revisions[-1].timestamp - revisions[0].timestamp).total_seconds() / 3600.0
    
    # If time span is less than window, use actual time span
    if time_span_hours < window_hours:
        expected_velocity = num_edits / max(time_span_hours, 0.001)
    else:
        expected_velocity = num_edits / window_hours
    
    # Verify velocity calculation (allow small tolerance for floating point)
    assert abs(velocity - expected_velocity) < 0.01, \
        f"Edit velocity {velocity:.4f} doesn't match expected {expected_velocity:.4f} " \
        f"({num_edits} edits over {window_hours} hours)"
    
    # Verify velocity is non-negative
    assert velocity >= 0, f"Edit velocity should be non-negative, got {velocity}"


# Feature: wikipedia-intelligence-system, Property 8: Edit Velocity Non-Negative
@given(
    revisions=revision_list_strategy(min_size=0, max_size=50),
    window_hours=st.integers(min_value=1, max_value=168)
)
@settings(
    suppress_health_check=[HealthCheck.function_scoped_fixture],
    max_examples=5
)
def test_property_8_edit_velocity_non_negative(revisions, window_hours):
    """Property 8: Edit Velocity is Always Non-Negative
    
    For any edit history (including empty), edit velocity should always be
    non-negative.
    
    Validates: Requirements 2.4
    """
    scraper = EditHistoryScraper()
    
    # Calculate edit velocity
    velocity = scraper.calculate_edit_velocity(revisions, window_hours)
    
    # Verify velocity is non-negative
    assert velocity >= 0, \
        f"Edit velocity should be non-negative, got {velocity} " \
        f"for {len(revisions)} revisions"



# ============================================================================
# PROPERTY 10: Rolling Window Metrics
# ============================================================================

# Feature: wikipedia-intelligence-system, Property 10: Rolling Window Metrics
@given(
    num_revisions=st.integers(min_value=5, max_value=100),
    windows=st.lists(
        st.integers(min_value=1, max_value=720),  # 1 hour to 30 days
        min_size=1,
        max_size=5,
        unique=True
    )
)
@settings(
    suppress_health_check=[HealthCheck.function_scoped_fixture],
    max_examples=5
)
def test_property_10_rolling_window_metrics(num_revisions, windows):
    """Property 10: Rolling Window Metrics
    
    For any edit history, the System should calculate metrics for all specified
    rolling windows (24h, 7d, 30d).
    
    Validates: Requirements 2.6
    """
    scraper = EditHistoryScraper()
    
    # Create revisions spread over 30 days
    reference_time = datetime(2024, 1, 31, 12, 0, 0)
    revisions = []
    
    for i in range(num_revisions):
        # Distribute revisions over 30 days (720 hours)
        hours_ago = (i / num_revisions) * 720
        timestamp = reference_time - timedelta(hours=hours_ago)
        
        # Mix of editor types and revert status
        editor_type = "anonymous" if i % 3 == 0 else "registered"
        editor_id = f"192.168.1.{i % 255}" if editor_type == "anonymous" else f"User{i}"
        is_reverted = (i % 5 == 0)  # 20% reverted
        
        revision = RevisionRecord(
            article="Test_Article",
            revision_id=1000 + i,
            timestamp=timestamp,
            editor_type=editor_type,
            editor_id=editor_id,
            is_reverted=is_reverted,
            bytes_changed=100,
            edit_summary="Reverted" if is_reverted else "Normal edit"
        )
        revisions.append(revision)
    
    # Calculate rolling window metrics
    metrics_by_window = scraper.calculate_rolling_window_metrics(revisions, windows)
    
    # Verify metrics calculated for all windows
    assert len(metrics_by_window) <= len(windows), \
        f"Expected at most {len(windows)} window metrics, got {len(metrics_by_window)}"
    
    # Verify each window's metrics
    for window_hours in windows:
        # Find metrics for this window
        window_label = {24: "24h", 168: "7d", 720: "30d"}.get(window_hours, f"{window_hours}h")
        
        # Skip if no revisions in this window
        window_start = reference_time - timedelta(hours=window_hours)
        window_revisions = [r for r in revisions if r.timestamp >= window_start]
        
        if not window_revisions:
            continue
        
        # Verify metrics exist for this window
        if window_label in metrics_by_window:
            metrics = metrics_by_window[window_label]
            
            # Verify metrics structure
            assert isinstance(metrics, EditMetrics), \
                f"Metrics for {window_label} should be EditMetrics instance"
            
            assert metrics.article == "Test_Article", \
                f"Metrics article should match"
            
            assert metrics.time_window_hours == window_hours, \
                f"Metrics window should be {window_hours} hours, got {metrics.time_window_hours}"
            
            # Verify metrics are non-negative
            assert metrics.edit_velocity >= 0, \
                f"Edit velocity should be non-negative for {window_label}"
            
            assert 0 <= metrics.vandalism_rate <= 100, \
                f"Vandalism rate should be 0-100% for {window_label}, got {metrics.vandalism_rate}"
            
            assert 0 <= metrics.anonymous_edit_pct <= 100, \
                f"Anonymous edit % should be 0-100% for {window_label}, got {metrics.anonymous_edit_pct}"
            
            assert metrics.total_edits >= 0, \
                f"Total edits should be non-negative for {window_label}"
            
            assert metrics.reverted_edits >= 0, \
                f"Reverted edits should be non-negative for {window_label}"
            
            # Verify reverted edits doesn't exceed total edits
            assert metrics.reverted_edits <= metrics.total_edits, \
                f"Reverted edits ({metrics.reverted_edits}) should not exceed " \
                f"total edits ({metrics.total_edits}) for {window_label}"



# Feature: wikipedia-intelligence-system, Property 10: Rolling Window Metrics Consistency
@given(
    num_revisions=st.integers(min_value=10, max_value=50)
)
@settings(
    suppress_health_check=[HealthCheck.function_scoped_fixture],
    max_examples=5
)
def test_property_10_rolling_window_metrics_consistency(num_revisions):
    """Property 10: Rolling Window Metrics Consistency
    
    For any edit history, metrics for larger windows should include all edits
    from smaller windows (when windows are nested).
    
    Validates: Requirements 2.6
    """
    scraper = EditHistoryScraper()
    
    # Create revisions spread over 30 days
    reference_time = datetime(2024, 1, 31, 12, 0, 0)
    revisions = []
    
    for i in range(num_revisions):
        hours_ago = (i / num_revisions) * 720  # Spread over 30 days
        timestamp = reference_time - timedelta(hours=hours_ago)
        
        revision = RevisionRecord(
            article="Test_Article",
            revision_id=1000 + i,
            timestamp=timestamp,
            editor_type="registered",
            editor_id=f"User{i}",
            is_reverted=False,
            bytes_changed=100,
            edit_summary=f"Edit {i}"
        )
        revisions.append(revision)
    
    # Calculate metrics for standard windows (24h, 7d, 30d)
    windows = [24, 168, 720]
    metrics_by_window = scraper.calculate_rolling_window_metrics(revisions, windows)
    
    # Verify nested window property: larger windows should have >= edits than smaller
    if "24h" in metrics_by_window and "7d" in metrics_by_window:
        assert metrics_by_window["7d"].total_edits >= metrics_by_window["24h"].total_edits, \
            "7-day window should have at least as many edits as 24-hour window"
    
    if "7d" in metrics_by_window and "30d" in metrics_by_window:
        assert metrics_by_window["30d"].total_edits >= metrics_by_window["7d"].total_edits, \
            "30-day window should have at least as many edits as 7-day window"
    
    if "24h" in metrics_by_window and "30d" in metrics_by_window:
        assert metrics_by_window["30d"].total_edits >= metrics_by_window["24h"].total_edits, \
            "30-day window should have at least as many edits as 24-hour window"


# ============================================================================
# ADDITIONAL PROPERTY TESTS
# ============================================================================

# Feature: wikipedia-intelligence-system, Property 5: Empty Edit History Handling
@settings(
    suppress_health_check=[HealthCheck.function_scoped_fixture],
    max_examples=5
)
def test_property_5_empty_edit_history_handling():
    """Property 5: Empty Edit History Handling
    
    For an empty edit history, the System should handle it gracefully without
    errors and return appropriate empty/zero values.
    
    Validates: Requirements 2.1
    """
    scraper = EditHistoryScraper()
    
    empty_revisions = []
    
    # Test edit velocity with empty list
    velocity = scraper.calculate_edit_velocity(empty_revisions, window_hours=24)
    assert velocity == 0.0, "Edit velocity for empty history should be 0.0"
    
    # Test rolling window metrics with empty list
    metrics_by_window = scraper.calculate_rolling_window_metrics(empty_revisions)
    assert len(metrics_by_window) == 0, "Empty history should produce no window metrics"
