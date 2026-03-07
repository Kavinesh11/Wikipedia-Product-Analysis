"""Unit Tests for ETL Pipeline Manager

Tests specific examples, edge cases, and error conditions for ETL pipelines.
"""
import pytest
from datetime import datetime
from unittest.mock import Mock, MagicMock, patch
import asyncio

from src.processing.etl_pipeline import ETLPipelineManager, DataLineage
from src.storage.dto import (
    PageviewRecord, RevisionRecord, ArticleContent,
    ValidationResult, PipelineResult
)


class TestETLPipelineManager:
    """Test suite for ETL Pipeline Manager"""
    
    def setup_method(self):
        """Set up test fixtures"""
        self.db = Mock()
        self.cache = Mock()
        self.etl = ETLPipelineManager(self.db, self.cache)
    
    # ========================================================================
    # Validation Tests
    # ========================================================================
    
    def test_validate_valid_pageview_records(self):
        """Test validation of valid pageview records"""
        records = [
            PageviewRecord(
                article="Python_(programming_language)",
                timestamp=datetime(2024, 1, 1, 12, 0, 0),
                device_type="desktop",
                views_human=1000,
                views_bot=50,
                views_total=1050
            ),
            PageviewRecord(
                article="Machine_Learning",
                timestamp=datetime(2024, 1, 1, 13, 0, 0),
                device_type="mobile-web",
                views_human=500,
                views_bot=25,
                views_total=525
            )
        ]
        
        result = self.etl.validate_data(records)
        
        assert result.is_valid is True
        assert result.total_records == 2
        assert result.valid_records == 2
        assert result.invalid_records == 0
        assert len(result.errors) == 0
    
    def test_validate_invalid_pageview_records(self):
        """Test validation catches invalid pageview records"""
        # Create record with empty article name
        invalid_record = PageviewRecord(
            article="",  # Invalid: empty article name
            timestamp=datetime(2024, 1, 1, 12, 0, 0),
            device_type="desktop",
            views_human=100,
            views_bot=10,
            views_total=110
        )
        
        result = self.etl.validate_data([invalid_record])
        
        assert result.is_valid is False
        assert result.total_records == 1
        assert result.valid_records == 0
        assert result.invalid_records == 1
        assert len(result.errors) == 1
        assert "Article name is required" in result.errors[0]["error"]
    
    def test_validate_negative_views(self):
        """Test validation catches negative view counts"""
        # Create record with negative views
        invalid_record = PageviewRecord(
            article="Test_Article",
            timestamp=datetime(2024, 1, 1, 12, 0, 0),
            device_type="desktop",
            views_human=-100,  # Invalid: negative
            views_bot=10,
            views_total=-90
        )
        
        result = self.etl.validate_data([invalid_record])
        
        assert result.is_valid is False
        assert result.invalid_records == 1
        assert "negative" in result.errors[0]["error"].lower()
    
    def test_validate_valid_revision_records(self):
        """Test validation of valid revision records"""
        records = [
            RevisionRecord(
                article="Python_(programming_language)",
                revision_id=123456,
                timestamp=datetime(2024, 1, 1, 12, 0, 0),
                editor_type="registered",
                editor_id="user123",
                is_reverted=False,
                bytes_changed=100,
                edit_summary="Updated documentation"
            )
        ]
        
        result = self.etl.validate_data(records)
        
        assert result.is_valid is True
        assert result.valid_records == 1
        assert result.invalid_records == 0
    
    def test_validate_invalid_revision_id(self):
        """Test validation catches invalid revision IDs"""
        invalid_record = RevisionRecord(
            article="Test_Article",
            revision_id=-1,  # Invalid: negative revision ID
            timestamp=datetime(2024, 1, 1, 12, 0, 0),
            editor_type="registered",
            editor_id="user123",
            is_reverted=False,
            bytes_changed=100,
            edit_summary="Test"
        )
        
        result = self.etl.validate_data([invalid_record])
        
        assert result.is_valid is False
        assert result.invalid_records == 1
        assert "Revision ID must be positive" in result.errors[0]["error"]
    
    def test_validate_valid_article_content(self):
        """Test validation of valid article content"""
        records = [
            ArticleContent(
                title="Python (programming language)",
                url="https://en.wikipedia.org/wiki/Python_(programming_language)",
                summary="Python is a high-level programming language",
                infobox={"paradigm": "multi-paradigm"},
                tables=[],
                categories=["Programming languages", "Python"],
                internal_links=["Guido_van_Rossum", "Programming"],
                crawl_timestamp=datetime(2024, 1, 1, 12, 0, 0)
            )
        ]
        
        result = self.etl.validate_data(records)
        
        assert result.is_valid is True
        assert result.valid_records == 1
        assert result.invalid_records == 0
    
    # ========================================================================
    # Deduplication Tests
    # ========================================================================
    
    def test_deduplicate_pageviews(self):
        """Test deduplication of pageview records"""
        # Create duplicate records
        record1 = PageviewRecord(
            article="Test_Article",
            timestamp=datetime(2024, 1, 1, 12, 0, 0),
            device_type="desktop",
            views_human=100,
            views_bot=10,
            views_total=110
        )
        
        record2 = PageviewRecord(
            article="Test_Article",
            timestamp=datetime(2024, 1, 1, 12, 0, 0),
            device_type="desktop",
            views_human=150,  # Different views but same key
            views_bot=15,
            views_total=165
        )
        
        records = [record1, record2]
        deduplicated = self.etl.deduplicate(records)
        
        assert len(deduplicated) == 1, "Should remove duplicate"
        assert deduplicated[0] == record1, "Should keep first record"
    
    def test_deduplicate_revisions(self):
        """Test deduplication of revision records"""
        # Create duplicate records with same revision_id
        record1 = RevisionRecord(
            article="Test_Article",
            revision_id=123456,
            timestamp=datetime(2024, 1, 1, 12, 0, 0),
            editor_type="registered",
            editor_id="user123",
            is_reverted=False,
            bytes_changed=100,
            edit_summary="First"
        )
        
        record2 = RevisionRecord(
            article="Test_Article",
            revision_id=123456,  # Same revision_id
            timestamp=datetime(2024, 1, 1, 13, 0, 0),
            editor_type="registered",
            editor_id="user123",
            is_reverted=False,
            bytes_changed=200,
            edit_summary="Second"
        )
        
        records = [record1, record2]
        deduplicated = self.etl.deduplicate(records)
        
        assert len(deduplicated) == 1, "Should remove duplicate"
        assert deduplicated[0] == record1, "Should keep first record"
    
    def test_deduplicate_crawl_results(self):
        """Test deduplication of crawl results"""
        # Create duplicate records
        timestamp = datetime(2024, 1, 1, 12, 0, 0)
        
        record1 = ArticleContent(
            title="Test_Article",
            url="https://en.wikipedia.org/wiki/Test_Article",
            summary="First crawl",
            infobox={},
            tables=[],
            categories=[],
            internal_links=[],
            crawl_timestamp=timestamp
        )
        
        record2 = ArticleContent(
            title="Test_Article",
            url="https://en.wikipedia.org/wiki/Test_Article",
            summary="Second crawl",
            infobox={},
            tables=[],
            categories=[],
            internal_links=[],
            crawl_timestamp=timestamp  # Same timestamp
        )
        
        records = [record1, record2]
        deduplicated = self.etl.deduplicate(records)
        
        assert len(deduplicated) == 1, "Should remove duplicate"
        assert deduplicated[0] == record1, "Should keep first record"
    
    def test_deduplicate_empty_list(self):
        """Test deduplication handles empty list"""
        deduplicated = self.etl.deduplicate([])
        assert deduplicated == []
    
    def test_deduplicate_no_duplicates(self):
        """Test deduplication preserves unique records"""
        records = [
            PageviewRecord(
                article="Article1",
                timestamp=datetime(2024, 1, 1, 12, 0, 0),
                device_type="desktop",
                views_human=100,
                views_bot=10,
                views_total=110
            ),
            PageviewRecord(
                article="Article2",
                timestamp=datetime(2024, 1, 1, 12, 0, 0),
                device_type="desktop",
                views_human=200,
                views_bot=20,
                views_total=220
            )
        ]
        
        deduplicated = self.etl.deduplicate(records)
        
        assert len(deduplicated) == 2, "Should preserve all unique records"
    
    # ========================================================================
    # Error Quarantine Tests
    # ========================================================================
    
    def test_quarantine_invalid_record(self):
        """Test that invalid records are quarantined"""
        invalid_record = PageviewRecord(
            article="",  # Invalid
            timestamp=datetime(2024, 1, 1, 12, 0, 0),
            device_type="desktop",
            views_human=100,
            views_bot=10,
            views_total=110
        )
        
        # Mock cache.set to verify quarantine
        self.cache.set = Mock()
        
        result = self.etl.validate_data([invalid_record])
        
        assert result.invalid_records == 1
        # Verify quarantine was called
        assert self.cache.set.called, "Should quarantine invalid record"
    
    # ========================================================================
    # Edge Cases
    # ========================================================================
    
    def test_validate_empty_dataset(self):
        """Test validation of empty dataset"""
        result = self.etl.validate_data([])
        
        assert result.is_valid is True
        assert result.total_records == 0
        assert result.valid_records == 0
        assert result.invalid_records == 0
    
    def test_validate_mixed_valid_invalid(self):
        """Test validation of mixed valid and invalid records"""
        valid_record = PageviewRecord(
            article="Valid_Article",
            timestamp=datetime(2024, 1, 1, 12, 0, 0),
            device_type="desktop",
            views_human=100,
            views_bot=10,
            views_total=110
        )
        
        invalid_record = PageviewRecord(
            article="",  # Invalid
            timestamp=datetime(2024, 1, 1, 12, 0, 0),
            device_type="desktop",
            views_human=100,
            views_bot=10,
            views_total=110
        )
        
        result = self.etl.validate_data([valid_record, invalid_record])
        
        assert result.is_valid is False
        assert result.total_records == 2
        assert result.valid_records == 1
        assert result.invalid_records == 1
