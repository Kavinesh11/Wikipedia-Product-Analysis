"""
Unit Tests for CheckpointManager

Tests checkpoint save/load, resumption, and expiration functionality.

Requirements: 11.7
"""
import pytest
from unittest.mock import Mock, MagicMock, patch
from datetime import datetime, timedelta
from collections import deque

from src.processing.checkpoint_manager import CheckpointManager, CheckpointMetadata
from src.data_ingestion.crawl4ai_pipeline import CrawlCheckpoint
from src.storage.dto import ArticleContent


class TestCheckpointManager:
    """Test CheckpointManager functionality"""
    
    def test_initialization(self):
        """Test CheckpointManager initialization"""
        # Test with provided mock cache
        mock_cache = Mock()
        manager = CheckpointManager(cache=mock_cache)
        assert manager.default_ttl == 86400  # 24 hours
        assert manager.cache is mock_cache
        
        # Test with custom TTL
        manager = CheckpointManager(cache=mock_cache, default_ttl=3600)
        assert manager.default_ttl == 3600
    
    def test_save_checkpoint_success(self):
        """Test successful checkpoint save"""
        # Setup mock cache
        mock_cache = Mock()
        mock_cache.set = Mock(return_value=True)
        
        manager = CheckpointManager(cache=mock_cache, default_ttl=3600)
        
        # Create test state
        test_state = {
            "visited": ["url1", "url2"],
            "queue": ["url3", "url4"],
            "depth": 2
        }
        
        # Save checkpoint
        success = manager.save_checkpoint(
            checkpoint_id="test_checkpoint_1",
            state=test_state,
            operation_type="deep_crawl",
            progress_info={"articles_crawled": 2, "total_articles": 10}
        )
        
        # Verify
        assert success is True
        mock_cache.set.assert_called_once()
        
        # Check the call arguments
        call_args = mock_cache.set.call_args
        assert call_args[0][0] == "pipeline:checkpoint:test_checkpoint_1"
        assert call_args[1]["ttl"] == 3600
    
    def test_save_checkpoint_with_custom_ttl(self):
        """Test checkpoint save with custom TTL"""
        mock_cache = Mock()
        mock_cache.set = Mock(return_value=True)
        
        manager = CheckpointManager(cache=mock_cache, default_ttl=3600)
        
        # Save with custom TTL
        success = manager.save_checkpoint(
            checkpoint_id="test_checkpoint_2",
            state={"data": "test"},
            ttl=7200  # 2 hours
        )
        
        assert success is True
        call_args = mock_cache.set.call_args
        assert call_args[1]["ttl"] == 7200
    
    def test_save_checkpoint_failure(self):
        """Test checkpoint save failure handling"""
        mock_cache = Mock()
        mock_cache.set = Mock(return_value=False)
        
        manager = CheckpointManager(cache=mock_cache)
        
        success = manager.save_checkpoint(
            checkpoint_id="test_checkpoint_3",
            state={"data": "test"}
        )
        
        assert success is False
    
    def test_save_checkpoint_exception(self):
        """Test checkpoint save with exception"""
        mock_cache = Mock()
        mock_cache.set = Mock(side_effect=Exception("Redis error"))
        
        manager = CheckpointManager(cache=mock_cache)
        
        success = manager.save_checkpoint(
            checkpoint_id="test_checkpoint_4",
            state={"data": "test"}
        )
        
        assert success is False
    
    def test_load_checkpoint_success(self):
        """Test successful checkpoint load"""
        # Setup mock cache
        mock_cache = Mock()
        test_state = {"visited": ["url1", "url2"], "queue": ["url3"]}
        mock_cache.get = Mock(return_value={
            "metadata": {
                "checkpoint_id": "test_checkpoint_5",
                "operation_type": "deep_crawl",
                "created_at": datetime.now().isoformat(),
                "ttl_seconds": 3600,
                "progress_info": {"articles_crawled": 2}
            },
            "state": test_state
        })
        
        manager = CheckpointManager(cache=mock_cache)
        
        # Load checkpoint
        loaded_state = manager.load_checkpoint("test_checkpoint_5")
        
        # Verify
        assert loaded_state is not None
        assert loaded_state == test_state
        mock_cache.get.assert_called_once_with("pipeline:checkpoint:test_checkpoint_5")
    
    def test_load_checkpoint_not_found(self):
        """Test loading non-existent checkpoint"""
        mock_cache = Mock()
        mock_cache.get = Mock(return_value=None)
        
        manager = CheckpointManager(cache=mock_cache)
        
        loaded_state = manager.load_checkpoint("nonexistent_checkpoint")
        
        assert loaded_state is None
    
    def test_load_checkpoint_exception(self):
        """Test checkpoint load with exception"""
        mock_cache = Mock()
        mock_cache.get = Mock(side_effect=Exception("Redis error"))
        
        manager = CheckpointManager(cache=mock_cache)
        
        loaded_state = manager.load_checkpoint("test_checkpoint_6")
        
        assert loaded_state is None
    
    def test_checkpoint_exists(self):
        """Test checkpoint existence check"""
        mock_cache = Mock()
        mock_cache.exists = Mock(return_value=True)
        
        manager = CheckpointManager(cache=mock_cache)
        
        exists = manager.checkpoint_exists("test_checkpoint_7")
        
        assert exists is True
        mock_cache.exists.assert_called_once_with("pipeline:checkpoint:test_checkpoint_7")
    
    def test_checkpoint_not_exists(self):
        """Test checkpoint non-existence"""
        mock_cache = Mock()
        mock_cache.exists = Mock(return_value=False)
        
        manager = CheckpointManager(cache=mock_cache)
        
        exists = manager.checkpoint_exists("nonexistent_checkpoint")
        
        assert exists is False
    
    def test_delete_checkpoint_success(self):
        """Test successful checkpoint deletion"""
        mock_cache = Mock()
        mock_cache.delete = Mock(return_value=True)
        
        manager = CheckpointManager(cache=mock_cache)
        
        success = manager.delete_checkpoint("test_checkpoint_8")
        
        assert success is True
        mock_cache.delete.assert_called_once_with("pipeline:checkpoint:test_checkpoint_8")
    
    def test_delete_checkpoint_failure(self):
        """Test checkpoint deletion failure"""
        mock_cache = Mock()
        mock_cache.delete = Mock(return_value=False)
        
        manager = CheckpointManager(cache=mock_cache)
        
        success = manager.delete_checkpoint("test_checkpoint_9")
        
        assert success is False
    
    def test_get_checkpoint_metadata(self):
        """Test retrieving checkpoint metadata"""
        mock_cache = Mock()
        created_at = datetime.now()
        mock_cache.get = Mock(return_value={
            "metadata": {
                "checkpoint_id": "test_checkpoint_10",
                "operation_type": "deep_crawl",
                "created_at": created_at.isoformat(),
                "ttl_seconds": 3600,
                "progress_info": {"articles_crawled": 5, "total_articles": 20}
            },
            "state": {"data": "test"}
        })
        
        manager = CheckpointManager(cache=mock_cache)
        
        metadata = manager.get_checkpoint_metadata("test_checkpoint_10")
        
        assert metadata is not None
        assert metadata.checkpoint_id == "test_checkpoint_10"
        assert metadata.operation_type == "deep_crawl"
        assert metadata.ttl_seconds == 3600
        assert metadata.progress_info["articles_crawled"] == 5
    
    def test_get_checkpoint_metadata_not_found(self):
        """Test getting metadata for non-existent checkpoint"""
        mock_cache = Mock()
        mock_cache.get = Mock(return_value=None)
        
        manager = CheckpointManager(cache=mock_cache)
        
        metadata = manager.get_checkpoint_metadata("nonexistent_checkpoint")
        
        assert metadata is None


class TestCheckpointResumption:
    """Test checkpoint resumption scenarios"""
    
    def test_crawl_checkpoint_resumption(self):
        """Test resuming a crawl from checkpoint"""
        # Setup mock cache
        mock_cache = Mock()
        
        # Create checkpoint state
        seed_url = "https://en.wikipedia.org/wiki/Test"
        visited_urls = {"https://en.wikipedia.org/wiki/Article1", "https://en.wikipedia.org/wiki/Article2"}
        queue = deque([("https://en.wikipedia.org/wiki/Article3", 1)])
        crawled_articles = [
            ArticleContent(
                title="Article1",
                url="https://en.wikipedia.org/wiki/Article1",
                summary="Summary 1",
                infobox={},
                tables=[],
                categories=[],
                internal_links=[],
                crawl_timestamp=datetime.now()
            )
        ]
        depth_map = {
            "https://en.wikipedia.org/wiki/Article1": 0,
            "https://en.wikipedia.org/wiki/Article2": 1,
            "https://en.wikipedia.org/wiki/Article3": 1
        }
        
        checkpoint = CrawlCheckpoint(
            seed_url=seed_url,
            visited_urls=visited_urls,
            queue=queue,
            crawled_articles=crawled_articles,
            timestamp=datetime.now(),
            depth_map=depth_map
        )
        
        # Mock cache to return checkpoint
        mock_cache.get = Mock(return_value={
            "metadata": {
                "checkpoint_id": "crawl_checkpoint_1",
                "operation_type": "deep_crawl",
                "created_at": datetime.now().isoformat(),
                "ttl_seconds": 3600,
                "progress_info": {"visited_count": 2, "queue_count": 1}
            },
            "state": checkpoint
        })
        
        manager = CheckpointManager(cache=mock_cache)
        
        # Load checkpoint
        loaded_checkpoint = manager.load_checkpoint("crawl_checkpoint_1")
        
        # Verify state is preserved
        assert loaded_checkpoint is not None
        assert loaded_checkpoint.seed_url == seed_url
        assert len(loaded_checkpoint.visited_urls) == 2
        assert len(loaded_checkpoint.queue) == 1
        assert len(loaded_checkpoint.crawled_articles) == 1
        assert len(loaded_checkpoint.depth_map) == 3


class TestCheckpointExpiration:
    """Test checkpoint TTL and expiration"""
    
    def test_checkpoint_expires_after_ttl(self):
        """Test that checkpoint expires after TTL"""
        mock_cache = Mock()
        
        # First call returns checkpoint, second call returns None (expired)
        mock_cache.get = Mock(side_effect=[
            {
                "metadata": {
                    "checkpoint_id": "expiring_checkpoint",
                    "operation_type": "deep_crawl",
                    "created_at": datetime.now().isoformat(),
                    "ttl_seconds": 1,
                    "progress_info": {}
                },
                "state": {"data": "test"}
            },
            None  # Expired
        ])
        
        manager = CheckpointManager(cache=mock_cache)
        
        # First load succeeds
        state1 = manager.load_checkpoint("expiring_checkpoint")
        assert state1 is not None
        
        # Second load fails (expired)
        state2 = manager.load_checkpoint("expiring_checkpoint")
        assert state2 is None
    
    def test_short_ttl_checkpoint(self):
        """Test checkpoint with short TTL"""
        mock_cache = Mock()
        mock_cache.set = Mock(return_value=True)
        
        manager = CheckpointManager(cache=mock_cache)
        
        # Save with 60 second TTL
        success = manager.save_checkpoint(
            checkpoint_id="short_ttl_checkpoint",
            state={"data": "test"},
            ttl=60
        )
        
        assert success is True
        call_args = mock_cache.set.call_args
        assert call_args[1]["ttl"] == 60


class TestCheckpointEdgeCases:
    """Test edge cases and error conditions"""
    
    def test_empty_state_checkpoint(self):
        """Test checkpoint with empty state"""
        mock_cache = Mock()
        mock_cache.set = Mock(return_value=True)
        
        manager = CheckpointManager(cache=mock_cache)
        
        success = manager.save_checkpoint(
            checkpoint_id="empty_checkpoint",
            state={}
        )
        
        assert success is True
    
    def test_large_state_checkpoint(self):
        """Test checkpoint with large state"""
        mock_cache = Mock()
        mock_cache.set = Mock(return_value=True)
        
        manager = CheckpointManager(cache=mock_cache)
        
        # Create large state
        large_state = {
            "visited": [f"url_{i}" for i in range(1000)],
            "queue": [f"queued_{i}" for i in range(500)],
            "data": "x" * 10000
        }
        
        success = manager.save_checkpoint(
            checkpoint_id="large_checkpoint",
            state=large_state
        )
        
        assert success is True
    
    def test_special_characters_in_checkpoint_id(self):
        """Test checkpoint ID with special characters"""
        mock_cache = Mock()
        mock_cache.set = Mock(return_value=True)
        
        manager = CheckpointManager(cache=mock_cache)
        
        # Use checkpoint ID with hyphens and underscores
        success = manager.save_checkpoint(
            checkpoint_id="checkpoint-with_special-chars_123",
            state={"data": "test"}
        )
        
        assert success is True
    
    def test_none_progress_info(self):
        """Test checkpoint with None progress_info"""
        mock_cache = Mock()
        mock_cache.set = Mock(return_value=True)
        
        manager = CheckpointManager(cache=mock_cache)
        
        success = manager.save_checkpoint(
            checkpoint_id="no_progress_checkpoint",
            state={"data": "test"},
            progress_info=None
        )
        
        assert success is True
        
        # Verify progress_info defaults to empty dict
        call_args = mock_cache.set.call_args
        checkpoint_data = call_args[0][1]
        assert checkpoint_data["metadata"]["progress_info"] == {}
