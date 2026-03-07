"""Unit Tests for Reputation Monitor

Tests specific examples, edge cases, and error conditions for reputation monitoring.
"""
import pytest
from datetime import datetime, timedelta

from src.analytics.reputation_monitor import ReputationMonitor
from src.storage.dto import RevisionRecord, EditMetrics


class TestReputationMonitor:
    """Test suite for ReputationMonitor class"""
    
    def test_initialization_default_threshold(self):
        """Test monitor initializes with default threshold"""
        monitor = ReputationMonitor()
        assert monitor.alert_threshold == 0.7
    
    def test_initialization_custom_threshold(self):
        """Test monitor initializes with custom threshold"""
        monitor = ReputationMonitor(alert_threshold=0.5)
        assert monitor.alert_threshold == 0.5
    
    def test_initialization_invalid_threshold(self):
        """Test monitor rejects invalid threshold values"""
        with pytest.raises(ValueError):
            ReputationMonitor(alert_threshold=1.5)
        
        with pytest.raises(ValueError):
            ReputationMonitor(alert_threshold=-0.1)
    
    def test_calculate_reputation_risk_high_risk(self):
        """Test reputation risk calculation for high-risk scenario"""
        monitor = ReputationMonitor()
        
        # High risk: high velocity, high vandalism, high anonymous edits
        metrics = EditMetrics(
            article="Test_Article",
            edit_velocity=50.0,  # 50 edits/hour
            vandalism_rate=40.0,  # 40% vandalism
            anonymous_edit_pct=80.0,  # 80% anonymous
            total_edits=100,
            reverted_edits=40,
            time_window_hours=24
        )
        
        score = monitor.calculate_reputation_risk(metrics)
        
        assert score.risk_score > 0.7
        assert score.alert_level == "high"
        assert score.article == "Test_Article"
        assert score.edit_velocity == 50.0
        assert score.vandalism_rate == 40.0
        assert score.anonymous_edit_pct == 80.0
    
    def test_calculate_reputation_risk_medium_risk(self):
        """Test reputation risk calculation for medium-risk scenario"""
        monitor = ReputationMonitor()
        
        # Medium risk: moderate values
        metrics = EditMetrics(
            article="Test_Article",
            edit_velocity=20.0,
            vandalism_rate=20.0,
            anonymous_edit_pct=40.0,
            total_edits=100,
            reverted_edits=20,
            time_window_hours=24
        )
        
        score = monitor.calculate_reputation_risk(metrics)
        
        assert 0.4 <= score.risk_score < 0.7
        assert score.alert_level == "medium"
    
    def test_calculate_reputation_risk_low_risk(self):
        """Test reputation risk calculation for low-risk scenario"""
        monitor = ReputationMonitor()
        
        # Low risk: low values across the board
        metrics = EditMetrics(
            article="Test_Article",
            edit_velocity=2.0,
            vandalism_rate=5.0,
            anonymous_edit_pct=10.0,
            total_edits=100,
            reverted_edits=5,
            time_window_hours=24
        )
        
        score = monitor.calculate_reputation_risk(metrics)
        
        assert score.risk_score < 0.4
        assert score.alert_level == "low"
    
    def test_detect_edit_spikes_true(self):
        """Test edit spike detection when velocity exceeds 3x baseline"""
        monitor = ReputationMonitor()
        
        baseline = 10.0
        velocity = 31.0  # Just over 3x baseline
        
        assert monitor.detect_edit_spikes(velocity, baseline) is True
    
    def test_detect_edit_spikes_false(self):
        """Test edit spike detection when velocity is below 3x baseline"""
        monitor = ReputationMonitor()
        
        baseline = 10.0
        velocity = 29.0  # Just under 3x baseline
        
        assert monitor.detect_edit_spikes(velocity, baseline) is False
    
    def test_detect_edit_spikes_exact_threshold(self):
        """Test edit spike detection at exact 3x threshold"""
        monitor = ReputationMonitor()
        
        baseline = 10.0
        velocity = 30.0  # Exactly 3x baseline
        
        assert monitor.detect_edit_spikes(velocity, baseline) is True
    
    def test_detect_edit_spikes_zero_baseline(self):
        """Test edit spike detection with zero baseline"""
        monitor = ReputationMonitor()
        
        # Any positive velocity should be a spike when baseline is zero
        assert monitor.detect_edit_spikes(1.0, 0.0) is True
        assert monitor.detect_edit_spikes(0.0, 0.0) is False
    
    def test_calculate_vandalism_rate_normal(self):
        """Test vandalism rate calculation with normal edit history"""
        monitor = ReputationMonitor()
        
        revisions = [
            RevisionRecord("Test", 1, datetime.now(), "registered", "user1", False, 100, "edit 1"),
            RevisionRecord("Test", 2, datetime.now(), "registered", "user2", True, -50, "revert"),
            RevisionRecord("Test", 3, datetime.now(), "anonymous", "192.168.1.1", False, 200, "edit 3"),
            RevisionRecord("Test", 4, datetime.now(), "registered", "user3", True, -100, "revert"),
        ]
        
        rate = monitor.calculate_vandalism_rate(revisions)
        
        # 2 reverted out of 4 = 50%
        assert rate == 50.0
    
    def test_calculate_vandalism_rate_no_vandalism(self):
        """Test vandalism rate calculation with no reverted edits"""
        monitor = ReputationMonitor()
        
        revisions = [
            RevisionRecord("Test", 1, datetime.now(), "registered", "user1", False, 100, "edit 1"),
            RevisionRecord("Test", 2, datetime.now(), "registered", "user2", False, 50, "edit 2"),
            RevisionRecord("Test", 3, datetime.now(), "registered", "user3", False, 200, "edit 3"),
        ]
        
        rate = monitor.calculate_vandalism_rate(revisions)
        
        assert rate == 0.0
    
    def test_calculate_vandalism_rate_all_vandalism(self):
        """Test vandalism rate calculation when all edits are reverted"""
        monitor = ReputationMonitor()
        
        revisions = [
            RevisionRecord("Test", 1, datetime.now(), "anonymous", "192.168.1.1", True, 100, "spam"),
            RevisionRecord("Test", 2, datetime.now(), "anonymous", "192.168.1.2", True, -50, "vandalism"),
            RevisionRecord("Test", 3, datetime.now(), "anonymous", "192.168.1.3", True, 200, "spam"),
        ]
        
        rate = monitor.calculate_vandalism_rate(revisions)
        
        assert rate == 100.0
    
    def test_calculate_vandalism_rate_empty_list(self):
        """Test vandalism rate calculation with empty revision list"""
        monitor = ReputationMonitor()
        
        rate = monitor.calculate_vandalism_rate([])
        
        assert rate == 0.0
    
    def test_generate_alert_high_priority(self):
        """Test alert generation for high-risk article"""
        monitor = ReputationMonitor(alert_threshold=0.7)
        
        alert = monitor.generate_alert("Test_Article", 0.85)
        
        assert alert.priority == "high"
        assert alert.alert_type == "reputation_risk"
        assert alert.article == "Test_Article"
        assert "Test_Article" in alert.message
        assert "0.85" in alert.message
    
    def test_generate_alert_medium_priority(self):
        """Test alert generation for medium-risk article"""
        monitor = ReputationMonitor()
        
        alert = monitor.generate_alert("Test_Article", 0.55)
        
        assert alert.priority == "medium"
        assert alert.alert_type == "reputation_risk"
    
    def test_generate_alert_low_priority(self):
        """Test alert generation for low-risk article"""
        monitor = ReputationMonitor()
        
        alert = monitor.generate_alert("Test_Article", 0.25)
        
        assert alert.priority == "low"
        assert alert.alert_type == "reputation_risk"
    
    def test_generate_alert_with_metadata(self):
        """Test alert generation with additional metadata"""
        monitor = ReputationMonitor()
        
        metadata = {"source": "automated_scan", "timestamp": datetime.now().isoformat()}
        alert = monitor.generate_alert("Test_Article", 0.75, metadata=metadata)
        
        assert alert.metadata == metadata
        assert alert.metadata["source"] == "automated_scan"
    
    def test_calculate_edit_metrics_normal(self):
        """Test edit metrics calculation with normal revision history"""
        monitor = ReputationMonitor()
        
        revisions = [
            RevisionRecord("Test", 1, datetime.now(), "registered", "user1", False, 100, "edit 1"),
            RevisionRecord("Test", 2, datetime.now(), "anonymous", "192.168.1.1", True, -50, "revert"),
            RevisionRecord("Test", 3, datetime.now(), "anonymous", "192.168.1.2", False, 200, "edit 3"),
            RevisionRecord("Test", 4, datetime.now(), "registered", "user2", False, 50, "edit 4"),
        ]
        
        metrics = monitor.calculate_edit_metrics("Test", revisions, time_window_hours=24)
        
        assert metrics.article == "Test"
        assert metrics.total_edits == 4
        assert metrics.reverted_edits == 1
        assert metrics.vandalism_rate == 25.0  # 1/4
        assert metrics.anonymous_edit_pct == 50.0  # 2/4
        assert metrics.edit_velocity == 4.0 / 24.0  # 4 edits / 24 hours
        assert metrics.time_window_hours == 24
    
    def test_calculate_edit_metrics_empty(self):
        """Test edit metrics calculation with empty revision list"""
        monitor = ReputationMonitor()
        
        metrics = monitor.calculate_edit_metrics("Test", [], time_window_hours=24)
        
        assert metrics.article == "Test"
        assert metrics.total_edits == 0
        assert metrics.reverted_edits == 0
        assert metrics.vandalism_rate == 0.0
        assert metrics.anonymous_edit_pct == 0.0
        assert metrics.edit_velocity == 0.0
    
    def test_detect_vandalism_signals_normal(self):
        """Test vandalism signal detection with normal edit history"""
        monitor = ReputationMonitor()
        
        revisions = [
            RevisionRecord("Test", 1, datetime.now(), "registered", "user1", False, 100, "edit 1"),
            RevisionRecord("Test", 2, datetime.now(), "anonymous", "192.168.1.1", True, -50, "revert"),
            RevisionRecord("Test", 3, datetime.now(), "registered", "user2", False, 200, "edit 3"),
        ]
        
        metrics = monitor.detect_vandalism_signals(revisions)
        
        assert metrics.article == "Test"
        assert metrics.total_edits == 3
        assert metrics.reverted_edits == 1
        assert metrics.vandalism_percentage == pytest.approx(33.33, rel=0.01)
        assert len(metrics.revert_patterns) == 1
        assert metrics.revert_patterns[0]["revision_id"] == 2
        assert metrics.revert_patterns[0]["editor_type"] == "anonymous"
    
    def test_detect_vandalism_signals_no_vandalism(self):
        """Test vandalism signal detection with no reverted edits"""
        monitor = ReputationMonitor()
        
        revisions = [
            RevisionRecord("Test", 1, datetime.now(), "registered", "user1", False, 100, "edit 1"),
            RevisionRecord("Test", 2, datetime.now(), "registered", "user2", False, 50, "edit 2"),
        ]
        
        metrics = monitor.detect_vandalism_signals(revisions)
        
        assert metrics.total_edits == 2
        assert metrics.reverted_edits == 0
        assert metrics.vandalism_percentage == 0.0
        assert len(metrics.revert_patterns) == 0
    
    def test_detect_vandalism_signals_empty(self):
        """Test vandalism signal detection with empty revision list"""
        monitor = ReputationMonitor()
        
        metrics = monitor.detect_vandalism_signals([])
        
        assert metrics.total_edits == 0
        assert metrics.reverted_edits == 0
        assert metrics.vandalism_percentage == 0.0
        assert len(metrics.revert_patterns) == 0
    
    def test_reputation_score_formula_weights(self):
        """Test that reputation score uses correct formula weights"""
        monitor = ReputationMonitor()
        
        # Test with known values to verify formula
        # Formula: 0.3 * normalized_velocity + 0.4 * vandalism_rate/100 + 0.3 * anonymous_edit_pct/100
        metrics = EditMetrics(
            article="Test",
            edit_velocity=100.0,  # Will be capped at 1.0 after normalization
            vandalism_rate=50.0,  # 0.5 after normalization
            anonymous_edit_pct=100.0,  # 1.0 after normalization
            total_edits=100,
            reverted_edits=50,
            time_window_hours=24
        )
        
        score = monitor.calculate_reputation_risk(metrics)
        
        # Expected: 0.3 * 1.0 + 0.4 * 0.5 + 0.3 * 1.0 = 0.3 + 0.2 + 0.3 = 0.8
        assert score.risk_score == pytest.approx(0.8, rel=0.01)
    
    def test_alert_threshold_boundary(self):
        """Test alert generation at exact threshold boundaries"""
        monitor = ReputationMonitor(alert_threshold=0.7)
        
        # Test at 0.7 (should be high)
        alert_high = monitor.generate_alert("Test", 0.7)
        assert alert_high.priority == "high"
        
        # Test at 0.69 (should be medium)
        alert_medium = monitor.generate_alert("Test", 0.69)
        assert alert_medium.priority == "medium"
        
        # Test at 0.4 (should be medium)
        alert_medium2 = monitor.generate_alert("Test", 0.4)
        assert alert_medium2.priority == "medium"
        
        # Test at 0.39 (should be low)
        alert_low = monitor.generate_alert("Test", 0.39)
        assert alert_low.priority == "low"
    
    def test_single_edit_metrics(self):
        """Test metrics calculation with single edit"""
        monitor = ReputationMonitor()
        
        revisions = [
            RevisionRecord("Test", 1, datetime.now(), "anonymous", "192.168.1.1", True, 100, "spam")
        ]
        
        metrics = monitor.calculate_edit_metrics("Test", revisions, time_window_hours=1)
        
        assert metrics.total_edits == 1
        assert metrics.reverted_edits == 1
        assert metrics.vandalism_rate == 100.0
        assert metrics.anonymous_edit_pct == 100.0
        assert metrics.edit_velocity == 1.0  # 1 edit / 1 hour
