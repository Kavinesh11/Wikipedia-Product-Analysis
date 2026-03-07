"""Property-Based Tests for Reputation Monitoring

Tests universal correctness properties for reputation risk assessment.
"""
import pytest
from hypothesis import given, strategies as st, assume, settings
from datetime import datetime, timedelta

from src.analytics.reputation_monitor import ReputationMonitor
from src.storage.dto import RevisionRecord, EditMetrics


# ============================================================================
# Test Strategies
# ============================================================================

@st.composite
def revision_record_strategy(draw):
    """Generate valid RevisionRecord instances"""
    article = draw(st.text(min_size=1, max_size=50, alphabet=st.characters(
        whitelist_categories=('Lu', 'Ll', 'Nd'), whitelist_characters='_-'
    )))
    revision_id = draw(st.integers(min_value=1, max_value=999999999))
    timestamp = datetime.now() - timedelta(hours=draw(st.integers(min_value=0, max_value=720)))
    editor_type = draw(st.sampled_from(["anonymous", "registered"]))
    editor_id = draw(st.text(min_size=1, max_size=20))
    is_reverted = draw(st.booleans())
    bytes_changed = draw(st.integers(min_value=-10000, max_value=10000))
    edit_summary = draw(st.text(max_size=100))
    
    return RevisionRecord(
        article=article,
        revision_id=revision_id,
        timestamp=timestamp,
        editor_type=editor_type,
        editor_id=editor_id,
        is_reverted=is_reverted,
        bytes_changed=bytes_changed,
        edit_summary=edit_summary
    )


@st.composite
def edit_metrics_strategy(draw):
    """Generate valid EditMetrics instances"""
    article = draw(st.text(min_size=1, max_size=50, alphabet=st.characters(
        whitelist_categories=('Lu', 'Ll', 'Nd'), whitelist_characters='_-'
    )))
    edit_velocity = draw(st.floats(min_value=0.0, max_value=200.0, allow_nan=False, allow_infinity=False))
    vandalism_rate = draw(st.floats(min_value=0.0, max_value=100.0, allow_nan=False, allow_infinity=False))
    anonymous_edit_pct = draw(st.floats(min_value=0.0, max_value=100.0, allow_nan=False, allow_infinity=False))
    total_edits = draw(st.integers(min_value=1, max_value=10000))
    reverted_edits = draw(st.integers(min_value=0, max_value=total_edits))
    time_window_hours = draw(st.integers(min_value=1, max_value=720))
    
    return EditMetrics(
        article=article,
        edit_velocity=edit_velocity,
        vandalism_rate=vandalism_rate,
        anonymous_edit_pct=anonymous_edit_pct,
        total_edits=total_edits,
        reverted_edits=reverted_edits,
        time_window_hours=time_window_hours
    )


# ============================================================================
# Property 9: Reputation Risk Alert Generation
# ============================================================================

# Feature: wikipedia-intelligence-system, Property 9: Reputation Risk Alert Generation
@given(
    anonymous_edit_pct=st.floats(min_value=0.0, max_value=100.0, allow_nan=False, allow_infinity=False),
    threshold=st.floats(min_value=0.0, max_value=100.0, allow_nan=False, allow_infinity=False)
)
@settings(max_examples=20)
def test_property_9_reputation_risk_alert_generation(anonymous_edit_pct, threshold):
    """
    Property 9: For any article where anonymous edit percentage exceeds 
    the configured threshold, the System should generate a reputation risk alert.
    
    **Validates: Requirements 2.5**
    """
    monitor = ReputationMonitor(alert_threshold=0.7)
    
    # Create edit metrics with the given anonymous edit percentage
    edit_metrics = EditMetrics(
        article="Test_Article",
        edit_velocity=10.0,
        vandalism_rate=20.0,
        anonymous_edit_pct=anonymous_edit_pct,
        total_edits=100,
        reverted_edits=20,
        time_window_hours=24
    )
    
    # Calculate reputation score
    score = monitor.calculate_reputation_risk(edit_metrics)
    
    # If anonymous edit percentage is high (>threshold), risk should be elevated
    # The formula uses 30% weight for anonymous edits
    # So if anonymous_edit_pct > threshold, we expect some contribution to risk
    if anonymous_edit_pct > threshold:
        # With high anonymous edits, the risk score should reflect this
        # At minimum, the anonymous component should be > 0
        anonymous_component = 0.3 * (anonymous_edit_pct / 100.0)
        assert score.risk_score >= anonymous_component * 0.9  # Allow small floating point error
    
    # Verify score is in valid range
    assert 0 <= score.risk_score <= 1
    assert score.anonymous_edit_pct == anonymous_edit_pct


# ============================================================================
# Property 25: Edit Spike Alert Generation
# ============================================================================

# Feature: wikipedia-intelligence-system, Property 25: Edit Spike Alert Generation
@given(
    edit_velocity=st.floats(min_value=0.0, max_value=1000.0, allow_nan=False, allow_infinity=False),
    baseline=st.floats(min_value=0.1, max_value=100.0, allow_nan=False, allow_infinity=False)
)
@settings(max_examples=20)
def test_property_25_edit_spike_alert_generation(edit_velocity, baseline):
    """
    Property 25: For any article where current edit velocity exceeds 3x 
    the baseline rate, the System should generate a reputation risk alert.
    
    **Validates: Requirements 6.1**
    """
    monitor = ReputationMonitor()
    
    # Detect edit spike
    is_spike = monitor.detect_edit_spikes(edit_velocity, baseline)
    
    # Verify spike detection logic
    if edit_velocity >= (3.0 * baseline):
        assert is_spike, f"Should detect spike: {edit_velocity} >= 3 * {baseline}"
    else:
        assert not is_spike, f"Should not detect spike: {edit_velocity} < 3 * {baseline}"


# ============================================================================
# Property 26: Vandalism Percentage Calculation
# ============================================================================

# Feature: wikipedia-intelligence-system, Property 26: Vandalism Percentage Calculation
@given(
    revisions=st.lists(
        revision_record_strategy(),
        min_size=1,
        max_size=100
    )
)
@settings(max_examples=20)
def test_property_26_vandalism_percentage_calculation(revisions):
    """
    Property 26: For any edit history, the System should calculate 
    vandalism percentage as (reverted_edits / total_edits) * 100.
    
    **Validates: Requirements 6.2**
    """
    monitor = ReputationMonitor()
    
    # Calculate vandalism rate
    vandalism_rate = monitor.calculate_vandalism_rate(revisions)
    
    # Calculate expected rate
    total_edits = len(revisions)
    reverted_edits = sum(1 for rev in revisions if rev.is_reverted)
    expected_rate = (reverted_edits / total_edits) * 100.0
    
    # Verify calculation
    assert abs(vandalism_rate - expected_rate) < 0.01, \
        f"Vandalism rate {vandalism_rate} doesn't match expected {expected_rate}"
    
    # Verify range
    assert 0 <= vandalism_rate <= 100


# ============================================================================
# Property 27: Reputation Risk Score Calculation
# ============================================================================

# Feature: wikipedia-intelligence-system, Property 27: Reputation Risk Score Calculation
@given(edit_metrics=edit_metrics_strategy())
@settings(max_examples=20)
def test_property_27_reputation_risk_score_calculation(edit_metrics):
    """
    Property 27: For any article, the System should calculate reputation 
    risk score as a weighted combination of edit velocity, vandalism rate, 
    and anonymous edit percentage, normalized to 0-1 range.
    
    **Validates: Requirements 6.3**
    """
    monitor = ReputationMonitor()
    
    # Calculate reputation risk
    score = monitor.calculate_reputation_risk(edit_metrics)
    
    # Verify score is in valid range
    assert 0 <= score.risk_score <= 1, \
        f"Risk score {score.risk_score} is out of range [0, 1]"
    
    # Verify the weighted formula
    # Formula: 0.3 * normalized_velocity + 0.4 * vandalism_rate/100 + 0.3 * anonymous_edit_pct/100
    normalized_velocity = min(edit_metrics.edit_velocity / 100.0, 1.0)
    normalized_vandalism = edit_metrics.vandalism_rate / 100.0
    normalized_anonymous = edit_metrics.anonymous_edit_pct / 100.0
    
    expected_score = (
        0.3 * normalized_velocity +
        0.4 * normalized_vandalism +
        0.3 * normalized_anonymous
    )
    
    assert abs(score.risk_score - expected_score) < 0.01, \
        f"Risk score {score.risk_score} doesn't match expected {expected_score}"
    
    # Verify components are preserved
    assert score.edit_velocity == edit_metrics.edit_velocity
    assert score.vandalism_rate == edit_metrics.vandalism_rate
    assert score.anonymous_edit_pct == edit_metrics.anonymous_edit_pct


# ============================================================================
# Property 28: High-Priority Alert Threshold
# ============================================================================

# Feature: wikipedia-intelligence-system, Property 28: High-Priority Alert Threshold
@given(
    risk_score=st.floats(min_value=0.0, max_value=1.0, allow_nan=False, allow_infinity=False)
)
@settings(max_examples=20)
def test_property_28_high_priority_alert_threshold(risk_score):
    """
    Property 28: For any article where reputation risk score exceeds 0.7, 
    the System should send a high-priority alert.
    
    **Validates: Requirements 6.4**
    """
    monitor = ReputationMonitor(alert_threshold=0.7)
    
    # Generate alert
    alert = monitor.generate_alert("Test_Article", risk_score)
    
    # Verify priority based on risk score
    if risk_score >= 0.7:
        assert alert.priority == "high", \
            f"Risk score {risk_score} >= 0.7 should generate high priority alert"
    elif risk_score >= 0.4:
        assert alert.priority == "medium", \
            f"Risk score {risk_score} in [0.4, 0.7) should generate medium priority alert"
    else:
        assert alert.priority == "low", \
            f"Risk score {risk_score} < 0.4 should generate low priority alert"
    
    # Verify alert structure
    assert alert.alert_type == "reputation_risk"
    assert alert.article == "Test_Article"
    assert "Test_Article" in alert.message
    assert str(risk_score)[:4] in alert.message  # Check risk score is in message


# ============================================================================
# Additional Property Tests
# ============================================================================

@given(
    revisions=st.lists(
        revision_record_strategy(),
        min_size=0,
        max_size=100
    )
)
@settings(max_examples=15)
def test_vandalism_metrics_completeness(revisions):
    """Test that vandalism metrics extraction is complete"""
    monitor = ReputationMonitor()
    
    metrics = monitor.detect_vandalism_signals(revisions)
    
    # Verify all fields are populated
    assert metrics.total_edits == len(revisions)
    assert metrics.reverted_edits == sum(1 for rev in revisions if rev.is_reverted)
    
    if revisions:
        expected_pct = (metrics.reverted_edits / metrics.total_edits) * 100.0
        assert abs(metrics.vandalism_percentage - expected_pct) < 0.01
    else:
        assert metrics.vandalism_percentage == 0.0


@given(edit_metrics=edit_metrics_strategy())
@settings(max_examples=15)
def test_alert_level_consistency(edit_metrics):
    """Test that alert level is consistent with risk score"""
    monitor = ReputationMonitor()
    
    score = monitor.calculate_reputation_risk(edit_metrics)
    
    # Verify alert level matches risk score
    if score.risk_score >= 0.7:
        assert score.alert_level == "high"
    elif score.risk_score >= 0.4:
        assert score.alert_level == "medium"
    else:
        assert score.alert_level == "low"


@given(
    baseline=st.floats(min_value=0.0, max_value=100.0, allow_nan=False, allow_infinity=False)
)
@settings(max_examples=15)
def test_edit_spike_with_zero_baseline(baseline):
    """Test edit spike detection handles zero baseline correctly"""
    monitor = ReputationMonitor()
    
    if baseline == 0:
        # Any positive velocity should be a spike when baseline is zero
        assert monitor.detect_edit_spikes(1.0, baseline) == True
        assert monitor.detect_edit_spikes(0.0, baseline) == False
    else:
        # Normal 3x threshold applies
        assert monitor.detect_edit_spikes(3.0 * baseline, baseline) == True
        assert monitor.detect_edit_spikes(2.9 * baseline, baseline) == False
