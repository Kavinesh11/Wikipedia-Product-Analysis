"""
Checkpoint Manager for Long-Running Operations

This module provides checkpointing functionality for long-running crawl operations,
allowing them to be paused and resumed without losing progress.

Requirements: 11.7
"""

import json
import logging
from datetime import datetime
from typing import Optional, Any, Dict
from dataclasses import dataclass, asdict

from src.storage.cache import RedisCache

logger = logging.getLogger(__name__)


@dataclass
class CheckpointMetadata:
    """Metadata for a checkpoint."""
    checkpoint_id: str
    operation_type: str  # e.g., "deep_crawl", "bulk_etl"
    created_at: datetime
    ttl_seconds: int
    progress_info: Dict[str, Any]  # e.g., {"articles_crawled": 50, "total_articles": 100}


class CheckpointManager:
    """
    Manages checkpoints for long-running operations.
    
    Stores checkpoint state in Redis with configurable TTL, allowing operations
    to be resumed from the last saved state.
    
    Features:
    - Save checkpoint state to Redis
    - Load checkpoint state for resumption
    - Automatic TTL management
    - Progress tracking
    
    Requirements: 11.7
    """
    
    def __init__(
        self,
        cache: Optional[RedisCache] = None,
        default_ttl: int = 86400  # 24 hours
    ):
        """
        Initialize CheckpointManager.
        
        Args:
            cache: Redis cache instance (creates default if None)
            default_ttl: Default TTL for checkpoints in seconds (default: 24 hours)
        """
        self.cache = cache or RedisCache()
        self.default_ttl = default_ttl
        
        logger.info(
            f"CheckpointManager initialized: default_ttl={default_ttl}s"
        )
    
    def save_checkpoint(
        self,
        checkpoint_id: str,
        state: Any,
        operation_type: str = "crawl",
        ttl: Optional[int] = None,
        progress_info: Optional[Dict[str, Any]] = None
    ) -> bool:
        """
        Save checkpoint state to Redis.
        
        Stores the complete state of a long-running operation so it can be
        resumed later. The checkpoint is stored with a TTL to prevent
        indefinite storage of stale checkpoints.
        
        Args:
            checkpoint_id: Unique identifier for the checkpoint
            state: State object to checkpoint (must be serializable)
            operation_type: Type of operation (e.g., "deep_crawl", "bulk_etl")
            ttl: Time-to-live in seconds (uses default if None)
            progress_info: Optional progress information for monitoring
            
        Returns:
            True if checkpoint saved successfully, False otherwise
            
        Requirements: 11.7
        """
        try:
            ttl_seconds = ttl or self.default_ttl
            
            # Create checkpoint metadata
            metadata = CheckpointMetadata(
                checkpoint_id=checkpoint_id,
                operation_type=operation_type,
                created_at=datetime.now(),
                ttl_seconds=ttl_seconds,
                progress_info=progress_info or {}
            )
            
            # Prepare checkpoint data
            checkpoint_data = {
                "metadata": asdict(metadata),
                "state": state
            }
            
            # Generate Redis key
            key = RedisCache.PIPELINE_CHECKPOINT.format(pipeline_id=checkpoint_id)
            
            # Save to Redis with TTL
            success = self.cache.set(key, checkpoint_data, ttl=ttl_seconds)
            
            if success:
                logger.info(
                    f"Checkpoint saved: {checkpoint_id} (type={operation_type}, ttl={ttl_seconds}s)",
                    extra={
                        "checkpoint_id": checkpoint_id,
                        "operation_type": operation_type,
                        "ttl_seconds": ttl_seconds,
                        "progress_info": progress_info
                    }
                )
            else:
                logger.error(
                    f"Failed to save checkpoint: {checkpoint_id}",
                    extra={"checkpoint_id": checkpoint_id}
                )
            
            return success
        
        except Exception as e:
            logger.error(
                f"Error saving checkpoint {checkpoint_id}: {type(e).__name__}: {e}",
                extra={
                    "checkpoint_id": checkpoint_id,
                    "error_type": type(e).__name__,
                    "error": str(e)
                },
                exc_info=True
            )
            return False
    
    def load_checkpoint(
        self,
        checkpoint_id: str
    ) -> Optional[Any]:
        """
        Load checkpoint state from Redis for resumption.
        
        Retrieves the saved state of a long-running operation, allowing it
        to resume from where it left off.
        
        Args:
            checkpoint_id: Unique identifier for the checkpoint
            
        Returns:
            Checkpoint state if found, None otherwise
            
        Requirements: 11.7
        """
        try:
            # Generate Redis key
            key = RedisCache.PIPELINE_CHECKPOINT.format(pipeline_id=checkpoint_id)
            
            # Load from Redis
            checkpoint_data = self.cache.get(key)
            
            if checkpoint_data is None:
                logger.warning(
                    f"Checkpoint not found: {checkpoint_id}",
                    extra={"checkpoint_id": checkpoint_id}
                )
                return None
            
            # Extract state
            state = checkpoint_data.get("state")
            metadata = checkpoint_data.get("metadata", {})
            
            logger.info(
                f"Checkpoint loaded: {checkpoint_id} (type={metadata.get('operation_type')})",
                extra={
                    "checkpoint_id": checkpoint_id,
                    "operation_type": metadata.get("operation_type"),
                    "created_at": metadata.get("created_at"),
                    "progress_info": metadata.get("progress_info")
                }
            )
            
            return state
        
        except Exception as e:
            logger.error(
                f"Error loading checkpoint {checkpoint_id}: {type(e).__name__}: {e}",
                extra={
                    "checkpoint_id": checkpoint_id,
                    "error_type": type(e).__name__,
                    "error": str(e)
                },
                exc_info=True
            )
            return None
    
    def checkpoint_exists(self, checkpoint_id: str) -> bool:
        """
        Check if a checkpoint exists in Redis.
        
        Args:
            checkpoint_id: Unique identifier for the checkpoint
            
        Returns:
            True if checkpoint exists, False otherwise
        """
        try:
            key = RedisCache.PIPELINE_CHECKPOINT.format(pipeline_id=checkpoint_id)
            return self.cache.exists(key)
        except Exception as e:
            logger.error(
                f"Error checking checkpoint existence {checkpoint_id}: {e}",
                extra={"checkpoint_id": checkpoint_id, "error": str(e)}
            )
            return False
    
    def delete_checkpoint(self, checkpoint_id: str) -> bool:
        """
        Delete a checkpoint from Redis.
        
        Args:
            checkpoint_id: Unique identifier for the checkpoint
            
        Returns:
            True if checkpoint deleted successfully, False otherwise
        """
        try:
            key = RedisCache.PIPELINE_CHECKPOINT.format(pipeline_id=checkpoint_id)
            success = self.cache.delete(key)
            
            if success:
                logger.info(
                    f"Checkpoint deleted: {checkpoint_id}",
                    extra={"checkpoint_id": checkpoint_id}
                )
            else:
                logger.warning(
                    f"Failed to delete checkpoint: {checkpoint_id}",
                    extra={"checkpoint_id": checkpoint_id}
                )
            
            return success
        except Exception as e:
            logger.error(
                f"Error deleting checkpoint {checkpoint_id}: {e}",
                extra={"checkpoint_id": checkpoint_id, "error": str(e)},
                exc_info=True
            )
            return False
    
    def get_checkpoint_metadata(self, checkpoint_id: str) -> Optional[CheckpointMetadata]:
        """
        Get metadata for a checkpoint without loading full state.
        
        Args:
            checkpoint_id: Unique identifier for the checkpoint
            
        Returns:
            CheckpointMetadata if found, None otherwise
        """
        try:
            key = RedisCache.PIPELINE_CHECKPOINT.format(pipeline_id=checkpoint_id)
            checkpoint_data = self.cache.get(key)
            
            if checkpoint_data is None:
                return None
            
            metadata_dict = checkpoint_data.get("metadata", {})
            
            # Convert datetime string back to datetime object
            if "created_at" in metadata_dict and isinstance(metadata_dict["created_at"], str):
                metadata_dict["created_at"] = datetime.fromisoformat(metadata_dict["created_at"])
            
            return CheckpointMetadata(**metadata_dict)
        
        except Exception as e:
            logger.error(
                f"Error getting checkpoint metadata {checkpoint_id}: {e}",
                extra={"checkpoint_id": checkpoint_id, "error": str(e)}
            )
            return None
