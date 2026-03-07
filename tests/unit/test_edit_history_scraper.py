"""
Unit tests for EditHistoryScraper.

Tests with sample edit histories and edge cases including:
- No edits
- Single edit
- All reverted edits
- Various editor types

Requirements: 2.1, 2.2, 2.3, 2.4
"""

import pytest
from unittest.mock import AsyncMock, Mock, patch
from datetime import datetime, timedelta
import logging

from src.data_ingestion.edit_history_scraper import EditHistoryScraper
from src.data_ingestion.api_client import WikimediaAPIClient
from src.storage.dto import RevisionRecord, VandalismMetrics, EditMetrics


class TestEditorClassification:
    """Test editor classification functionality."""
    
    def test_classify_ipv4_as_anonymous(self):
        """Test IPv4 addresses are classified as anonymous."""
        assert EditHistoryScraper._is_ip_address("192.168.1.1")
        assert EditHistoryScraper._is_ip_address("10.0.0.1")
        assert EditHistoryScraper._is_ip_address("172.16.0.1")
        assert EditHistoryScraper._classify_editor("192.168.1.1") == "anonymous"
    
    def test_classify_ipv6_as_anonymous(self):
        """Test IPv6 addresses are classified as anonymous."""
        assert EditHistoryScraper._is_ip_address("2001:0db8:85a3:0000:0000:8a2e:0370:7334")
        assert EditHistoryScraper._is_ip_address("::1")
        assert EditHistoryScraper._is_ip_address("fe80::1")
        assert EditHistoryScraper._classify_editor("2001:0db8::1") == "anonymous"
    
    def test_classify_username_as_registered(self):
        """Test usernames are classified as registered."""
        assert not EditHistoryScraper._is_ip_address("JohnDoe")
        assert not EditHistoryScraper._is_ip_address("User123")
        assert not EditHistoryScraper._is_ip_address("WikiEditor_2024")
        assert EditHistoryScraper._classify_editor("JohnDoe") == "registered"
        assert EditHistoryScraper._classify_editor("User123") == "registered"
    
    def test_classify_empty_string(self):
        """Test empty string is classified as registered (not an IP)."""
        assert not EditHistoryScraper._is_ip_address("")
        assert EditHistoryScraper._classify_editor("") == "registered"
    
    def test_classify_special_characters(self):
        """Test usernames with special characters are classified as registered."""
        assert not EditHistoryScraper._is_ip_address("User@Example")
        assert not EditHistoryScraper._is_ip_address("Test-User_123")
        assert EditHistoryScraper._classify_editor("User@Example") == "registered"


class TestEditVelocityCalculation:
    """Test edit velocity calculation."""
    
    def test_velocity_with_no_edits(self):
        """Test velocity calculation with empty revision list."""
        scraper = EditHistoryScraper()
        velocity = scraper.calculate_edit_velocity([])
        assert velocity == 0.0
    
    def test_velocity_with_single_edit(self):
        """Test velocity calculation with single edit."""
        scraper = EditHistoryScraper()
        
        revision = RevisionRecord(
            article="Test Article",
            revision_id=12345,
            timestamp=datetime(2024, 1, 1, 12, 0, 0),
            editor_type="registered",
            editor_id="TestUser",
            is_reverted=False,
            bytes_changed=100,
            edit_summary="Test edit"
        )
        
        velocity = scraper.calculate_edit_velocity([revision])
        # Single edit should have very high velocity (1 edit / minimal duration)
        assert velocity > 0
    
    def test_velocity_with_multiple_edits_24h_window(self):
        """Test velocity calculation with multiple edits over 24 hours."""
        scraper = EditHistoryScraper()
        
        base_time = datetime(2024, 1, 1, 0, 0, 0)
        revisions = [
            RevisionRecord(
                article="Test Article",
                revision_id=i,
                timestamp=base_time + timedelta(hours=i),
                editor_type="registered",
                editor_id=f"User{i}",
                is_reverted=False,
                bytes_changed=100,
                edit_summary=f"Edit {i}"
            )
            for i in range(24)  # 24 edits over 24 hours
        ]
        
        velocity = scraper.calculate_edit_velocity(revisions, window_hours=24)
        # 24 edits over 24 hours = 1 edit per hour
        assert abs(velocity - 1.0) < 0.1
    
    def test_velocity_with_rapid_edits(self):
        """Test velocity calculation with rapid edits in short time."""
        scraper = EditHistoryScraper()
        
        base_time = datetime(2024, 1, 1, 12, 0, 0)
        revisions = [
            RevisionRecord(
                article="Test Article",
                revision_id=i,
                timestamp=base_time + timedelta(minutes=i * 5),
                editor_type="registered",
                editor_id=f"User{i}",
                is_reverted=False,
                bytes_changed=100,
                edit_summary=f"Edit {i}"
            )
            for i in range(12)  # 12 edits over 1 hour
        ]
        
        velocity = scraper.calculate_edit_velocity(revisions, window_hours=24)
        # 12 edits over ~1 hour = ~12 edits per hour
        assert velocity > 10.0


class TestVandalismDetection:
    """Test vandalism detection functionality."""
    
    def test_vandalism_detection_with_no_edits(self):
        """Test vandalism detection with empty revision list."""
        scraper = EditHistoryScraper()
        metrics = scraper.detect_vandalism_signals([])
        
        assert metrics.total_edits == 0
        assert metrics.reverted_edits == 0
        assert metrics.vandalism_percentage == 0.0
        assert len(metrics.revert_patterns) == 0
    
    def test_vandalism_detection_with_no_reverts(self):
        """Test vandalism detection with no reverted edits."""
        scraper = EditHistoryScraper()
        
        revisions = [
            RevisionRecord(
                article="Test Article",
                revision_id=i,
                timestamp=datetime(2024, 1, 1, 12, i, 0),
                editor_type="registered",
                editor_id=f"User{i}",
                is_reverted=False,
                bytes_changed=100,
                edit_summary=f"Normal edit {i}"
            )
            for i in range(10)
        ]
        
        metrics = scraper.detect_vandalism_signals(revisions)
        
        assert metrics.total_edits == 10
        assert metrics.reverted_edits == 0
        assert metrics.vandalism_percentage == 0.0
        assert len(metrics.revert_patterns) == 0
    
    def test_vandalism_detection_with_all_reverts(self):
        """Test vandalism detection when all edits are reverted."""
        scraper = EditHistoryScraper()
        
        revisions = [
            RevisionRecord(
                article="Test Article",
                revision_id=i,
                timestamp=datetime(2024, 1, 1, 12, i, 0),
                editor_type="anonymous",
                editor_id=f"192.168.1.{i}",
                is_reverted=False,
                bytes_changed=100,
                edit_summary="Reverted vandalism"
            )
            for i in range(5)
        ]
        
        metrics = scraper.detect_vandalism_signals(revisions)
        
        assert metrics.total_edits == 5
        assert metrics.reverted_edits == 5
        assert metrics.vandalism_percentage == 100.0
        assert len(metrics.revert_patterns) == 5
        # Check that is_reverted flag was set
        assert all(r.is_reverted for r in revisions)
    
    def test_vandalism_detection_with_revert_keywords(self):
        """Test vandalism detection identifies various revert keywords."""
        scraper = EditHistoryScraper()
        
        revert_summaries = [
            "Reverted edits by User123",
            "Undo revision 12345",
            "Undid vandalism",
            "Rollback to previous version",
            "Reverted to last good version",
            "rv spam",
            "Restored previous content",
            "restored article"
        ]
        
        revisions = [
            RevisionRecord(
                article="Test Article",
                revision_id=i,
                timestamp=datetime(2024, 1, 1, 12, i, 0),
                editor_type="registered",
                editor_id=f"User{i}",
                is_reverted=False,
                bytes_changed=100,
                edit_summary=summary
            )
            for i, summary in enumerate(revert_summaries)
        ]
        
        metrics = scraper.detect_vandalism_signals(revisions)
        
        assert metrics.total_edits == len(revert_summaries)
        assert metrics.reverted_edits == len(revert_summaries)
        assert metrics.vandalism_percentage == 100.0
    
    def test_vandalism_detection_with_partial_reverts(self):
        """Test vandalism detection with some reverted edits."""
        scraper = EditHistoryScraper()
        
        revisions = [
            RevisionRecord(
                article="Test Article",
                revision_id=1,
                timestamp=datetime(2024, 1, 1, 12, 0, 0),
                editor_type="registered",
                editor_id="User1",
                is_reverted=False,
                bytes_changed=100,
                edit_summary="Normal edit"
            ),
            RevisionRecord(
                article="Test Article",
                revision_id=2,
                timestamp=datetime(2024, 1, 1, 12, 1, 0),
                editor_type="anonymous",
                editor_id="192.168.1.1",
                is_reverted=False,
                bytes_changed=-50,
                edit_summary="Vandalism"
            ),
            RevisionRecord(
                article="Test Article",
                revision_id=3,
                timestamp=datetime(2024, 1, 1, 12, 2, 0),
                editor_type="registered",
                editor_id="User2",
                is_reverted=False,
                bytes_changed=50,
                edit_summary="Reverted vandalism by 192.168.1.1"
            ),
            RevisionRecord(
                article="Test Article",
                revision_id=4,
                timestamp=datetime(2024, 1, 1, 12, 3, 0),
                editor_type="registered",
                editor_id="User3",
                is_reverted=False,
                bytes_changed=200,
                edit_summary="Added new section"
            )
        ]
        
        metrics = scraper.detect_vandalism_signals(revisions)
        
        assert metrics.total_edits == 4
        assert metrics.reverted_edits == 1  # Only the revert edit itself
        assert metrics.vandalism_percentage == 25.0
        assert len(metrics.revert_patterns) == 1
    
    def test_vandalism_detection_case_insensitive(self):
        """Test vandalism detection is case-insensitive."""
        scraper = EditHistoryScraper()
        
        revisions = [
            RevisionRecord(
                article="Test Article",
                revision_id=1,
                timestamp=datetime(2024, 1, 1, 12, 0, 0),
                editor_type="registered",
                editor_id="User1",
                is_reverted=False,
                bytes_changed=100,
                edit_summary="REVERTED EDITS"
            ),
            RevisionRecord(
                article="Test Article",
                revision_id=2,
                timestamp=datetime(2024, 1, 1, 12, 1, 0),
                editor_type="registered",
                editor_id="User2",
                is_reverted=False,
                bytes_changed=100,
                edit_summary="Undo Previous Change"
            )
        ]
        
        metrics = scraper.detect_vandalism_signals(revisions)
        
        assert metrics.reverted_edits == 2
        assert all(r.is_reverted for r in revisions)


class TestRollingWindowMetrics:
    """Test rolling window metrics calculation."""
    
    def test_rolling_window_with_no_edits(self):
        """Test rolling window metrics with empty revision list."""
        scraper = EditHistoryScraper()
        metrics = scraper.calculate_rolling_window_metrics([])
        
        assert len(metrics) == 0
    
    def test_rolling_window_with_single_edit(self):
        """Test rolling window metrics with single edit."""
        scraper = EditHistoryScraper()
        
        revision = RevisionRecord(
            article="Test Article",
            revision_id=12345,
            timestamp=datetime(2024, 1, 1, 12, 0, 0),
            editor_type="registered",
            editor_id="TestUser",
            is_reverted=False,
            bytes_changed=100,
            edit_summary="Test edit"
        )
        
        metrics = scraper.calculate_rolling_window_metrics([revision])
        
        # Should have metrics for all windows
        assert "24h" in metrics
        assert "7d" in metrics
        assert "30d" in metrics
        
        # All windows should show 1 edit
        for window_label, window_metrics in metrics.items():
            assert window_metrics.total_edits == 1
            assert window_metrics.reverted_edits == 0
            assert window_metrics.vandalism_rate == 0.0
            assert window_metrics.anonymous_edit_pct == 0.0
    
    def test_rolling_window_with_multiple_windows(self):
        """Test rolling window metrics with edits across different time periods."""
        scraper = EditHistoryScraper()
        
        # Create edits spanning 30 days
        base_time = datetime(2024, 1, 30, 12, 0, 0)  # Reference time
        revisions = []
        
        # Add edits at different time offsets
        # 5 edits in last 24 hours
        for i in range(5):
            revisions.append(RevisionRecord(
                article="Test Article",
                revision_id=i,
                timestamp=base_time - timedelta(hours=i),
                editor_type="registered",
                editor_id=f"User{i}",
                is_reverted=False,
                bytes_changed=100,
                edit_summary=f"Recent edit {i}"
            ))
        
        # 10 edits in last 7 days (but not in last 24h)
        for i in range(5, 15):
            revisions.append(RevisionRecord(
                article="Test Article",
                revision_id=i,
                timestamp=base_time - timedelta(days=i - 4),
                editor_type="registered",
                editor_id=f"User{i}",
                is_reverted=False,
                bytes_changed=100,
                edit_summary=f"Week edit {i}"
            ))
        
        # 15 edits in last 30 days (but not in last 7d)
        for i in range(15, 30):
            revisions.append(RevisionRecord(
                article="Test Article",
                revision_id=i,
                timestamp=base_time - timedelta(days=i - 7),
                editor_type="anonymous",
                editor_id=f"192.168.1.{i}",
                is_reverted=False,
                bytes_changed=100,
                edit_summary=f"Month edit {i}"
            ))
        
        metrics = scraper.calculate_rolling_window_metrics(revisions)
        
        # 24h window should have 5 edits
        assert metrics["24h"].total_edits == 5
        
        # 7d window should have 15 edits (5 + 10)
        assert metrics["7d"].total_edits == 15
        
        # 30d window should have all 30 edits
        assert metrics["30d"].total_edits == 30
    
    def test_rolling_window_with_vandalism(self):
        """Test rolling window metrics correctly calculate vandalism rates."""
        scraper = EditHistoryScraper()
        
        base_time = datetime(2024, 1, 1, 12, 0, 0)
        revisions = [
            RevisionRecord(
                article="Test Article",
                revision_id=1,
                timestamp=base_time,
                editor_type="registered",
                editor_id="User1",
                is_reverted=False,
                bytes_changed=100,
                edit_summary="Normal edit"
            ),
            RevisionRecord(
                article="Test Article",
                revision_id=2,
                timestamp=base_time - timedelta(hours=1),
                editor_type="anonymous",
                editor_id="192.168.1.1",
                is_reverted=False,
                bytes_changed=-50,
                edit_summary="Vandalism"
            ),
            RevisionRecord(
                article="Test Article",
                revision_id=3,
                timestamp=base_time - timedelta(hours=2),
                editor_type="registered",
                editor_id="User2",
                is_reverted=False,
                bytes_changed=50,
                edit_summary="Reverted vandalism"
            ),
            RevisionRecord(
                article="Test Article",
                revision_id=4,
                timestamp=base_time - timedelta(hours=3),
                editor_type="anonymous",
                editor_id="10.0.0.1",
                is_reverted=False,
                bytes_changed=100,
                edit_summary="Another edit"
            )
        ]
        
        # First detect vandalism to set is_reverted flags
        scraper.detect_vandalism_signals(revisions)
        
        metrics = scraper.calculate_rolling_window_metrics(revisions)
        
        # Check 24h window
        assert metrics["24h"].total_edits == 4
        assert metrics["24h"].reverted_edits == 1  # One revert
        assert metrics["24h"].vandalism_rate == 25.0
        assert metrics["24h"].anonymous_edit_pct == 50.0  # 2 out of 4
    
    def test_rolling_window_custom_windows(self):
        """Test rolling window metrics with custom window sizes."""
        scraper = EditHistoryScraper()
        
        base_time = datetime(2024, 1, 1, 12, 0, 0)
        revisions = [
            RevisionRecord(
                article="Test Article",
                revision_id=i,
                timestamp=base_time - timedelta(hours=i),
                editor_type="registered",
                editor_id=f"User{i}",
                is_reverted=False,
                bytes_changed=100,
                edit_summary=f"Edit {i}"
            )
            for i in range(50)
        ]
        
        # Use custom windows: 12h, 48h
        metrics = scraper.calculate_rolling_window_metrics(revisions, windows=[12, 48])
        
        assert "12h" in metrics
        assert "48h" in metrics
        assert metrics["12h"].total_edits == 12
        assert metrics["48h"].total_edits == 48


class TestFetchRevisions:
    """Test fetch_revisions method with mocked API responses."""
    
    @pytest.mark.asyncio
    async def test_fetch_revisions_with_sample_data(self):
        """Test fetching revisions with sample API response."""
        # Mock API client
        mock_client = AsyncMock(spec=WikimediaAPIClient)
        
        # Sample API response
        mock_response = {
            "query": {
                "pages": {
                    "12345": {
                        "pageid": 12345,
                        "title": "Test Article",
                        "revisions": [
                            {
                                "revid": 1001,
                                "timestamp": "2024-01-01T12:00:00Z",
                                "user": "TestUser1",
                                "size": 1000,
                                "comment": "Added content"
                            },
                            {
                                "revid": 1002,
                                "timestamp": "2024-01-01T13:00:00Z",
                                "user": "192.168.1.1",
                                "size": 1100,
                                "comment": "Minor edit"
                            },
                            {
                                "revid": 1003,
                                "timestamp": "2024-01-01T14:00:00Z",
                                "user": "TestUser2",
                                "size": 1050,
                                "comment": "Reverted vandalism"
                            }
                        ]
                    }
                }
            }
        }
        
        mock_client.get = AsyncMock(return_value=mock_response)
        
        scraper = EditHistoryScraper(api_client=mock_client)
        
        start_date = datetime(2024, 1, 1, 0, 0, 0)
        end_date = datetime(2024, 1, 2, 0, 0, 0)
        
        revisions = await scraper.fetch_revisions("Test Article", start_date, end_date)
        
        assert len(revisions) == 3
        assert revisions[0].revision_id == 1001
        assert revisions[0].editor_type == "registered"
        assert revisions[1].revision_id == 1002
        assert revisions[1].editor_type == "anonymous"
        assert revisions[2].revision_id == 1003
        assert revisions[2].editor_type == "registered"
    
    @pytest.mark.asyncio
    async def test_fetch_revisions_with_no_edits(self):
        """Test fetching revisions when article has no edits."""
        mock_client = AsyncMock(spec=WikimediaAPIClient)
        
        # API response with no revisions
        mock_response = {
            "query": {
                "pages": {
                    "12345": {
                        "pageid": 12345,
                        "title": "Test Article",
                        "revisions": []
                    }
                }
            }
        }
        
        mock_client.get = AsyncMock(return_value=mock_response)
        
        scraper = EditHistoryScraper(api_client=mock_client)
        
        start_date = datetime(2024, 1, 1, 0, 0, 0)
        end_date = datetime(2024, 1, 2, 0, 0, 0)
        
        revisions = await scraper.fetch_revisions("Test Article", start_date, end_date)
        
        assert len(revisions) == 0
    
    @pytest.mark.asyncio
    async def test_fetch_revisions_with_missing_article(self):
        """Test fetching revisions for non-existent article."""
        mock_client = AsyncMock(spec=WikimediaAPIClient)
        
        # API response for missing article
        mock_response = {
            "query": {
                "pages": {
                    "-1": {
                        "missing": ""
                    }
                }
            }
        }
        
        mock_client.get = AsyncMock(return_value=mock_response)
        
        scraper = EditHistoryScraper(api_client=mock_client)
        
        start_date = datetime(2024, 1, 1, 0, 0, 0)
        end_date = datetime(2024, 1, 2, 0, 0, 0)
        
        revisions = await scraper.fetch_revisions("NonExistent Article", start_date, end_date)
        
        assert len(revisions) == 0
    
    @pytest.mark.asyncio
    async def test_fetch_revisions_with_single_edit(self):
        """Test fetching revisions when article has only one edit."""
        mock_client = AsyncMock(spec=WikimediaAPIClient)
        
        # API response with single revision
        mock_response = {
            "query": {
                "pages": {
                    "12345": {
                        "pageid": 12345,
                        "title": "Test Article",
                        "revisions": [
                            {
                                "revid": 1001,
                                "timestamp": "2024-01-01T12:00:00Z",
                                "user": "TestUser",
                                "size": 1000,
                                "comment": "Initial creation"
                            }
                        ]
                    }
                }
            }
        }
        
        mock_client.get = AsyncMock(return_value=mock_response)
        
        scraper = EditHistoryScraper(api_client=mock_client)
        
        start_date = datetime(2024, 1, 1, 0, 0, 0)
        end_date = datetime(2024, 1, 2, 0, 0, 0)
        
        revisions = await scraper.fetch_revisions("Test Article", start_date, end_date)
        
        assert len(revisions) == 1
        assert revisions[0].revision_id == 1001
        assert revisions[0].editor_id == "TestUser"
        assert revisions[0].editor_type == "registered"
        assert revisions[0].edit_summary == "Initial creation"
        assert not revisions[0].is_reverted
    
    @pytest.mark.asyncio
    async def test_fetch_revisions_with_all_reverted(self):
        """Test fetching revisions where all edits are reverted."""
        mock_client = AsyncMock(spec=WikimediaAPIClient)
        
        # API response with all reverted edits
        mock_response = {
            "query": {
                "pages": {
                    "12345": {
                        "pageid": 12345,
                        "title": "Test Article",
                        "revisions": [
                            {
                                "revid": 1001,
                                "timestamp": "2024-01-01T12:00:00Z",
                                "user": "Admin1",
                                "size": 1000,
                                "comment": "Reverted vandalism by 192.168.1.1"
                            },
                            {
                                "revid": 1002,
                                "timestamp": "2024-01-01T13:00:00Z",
                                "user": "Admin2",
                                "size": 1100,
                                "comment": "Undo revision 1001"
                            },
                            {
                                "revid": 1003,
                                "timestamp": "2024-01-01T14:00:00Z",
                                "user": "Admin3",
                                "size": 1050,
                                "comment": "Rollback to previous version"
                            }
                        ]
                    }
                }
            }
        }
        
        mock_client.get = AsyncMock(return_value=mock_response)
        
        scraper = EditHistoryScraper(api_client=mock_client)
        
        start_date = datetime(2024, 1, 1, 0, 0, 0)
        end_date = datetime(2024, 1, 2, 0, 0, 0)
        
        revisions = await scraper.fetch_revisions("Test Article", start_date, end_date)
        
        assert len(revisions) == 3
        
        # Detect vandalism to set is_reverted flags
        metrics = scraper.detect_vandalism_signals(revisions)
        
        # All should be marked as reverted
        assert metrics.reverted_edits == 3
        assert metrics.vandalism_percentage == 100.0
        assert all(r.is_reverted for r in revisions)
