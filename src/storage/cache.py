"""Redis Cache Utilities

Provides caching functionality with TTL support.
"""
import json
import pickle
from typing import Any, Optional, Callable
from dataclasses import asdict, is_dataclass
import redis
from redis.connection import ConnectionPool

from src.utils.logging_config import get_logger
from src.utils.config import get_config

logger = get_logger(__name__)


class RedisCache:
    """Redis cache manager with connection pooling
    
    Supports cache key patterns for metrics and dashboard data,
    serialization/deserialization for complex objects, and
    fallback to database on cache miss.
    """
    
    # Cache key patterns
    METRICS_ARTICLE_REALTIME = "metrics:article:{article_id}:realtime"
    METRICS_CLUSTER_REALTIME = "metrics:cluster:{cluster_id}:realtime"
    DASHBOARD_DEMAND_TRENDS = "dashboard:demand_trends:{hash}"
    DASHBOARD_COMPETITOR_COMPARISON = "dashboard:competitor_comparison:{hash}"
    DASHBOARD_TOPIC_HEATMAP = "dashboard:topic_heatmap:{hash}"
    ALERTS_REPUTATION = "alerts:reputation:{article_id}"
    ALERTS_HYPE = "alerts:hype:{article_id}"
    RATELIMIT_WIKIMEDIA = "ratelimit:wikimedia:{timestamp}"
    RATELIMIT_CRAWLER = "ratelimit:crawler:{timestamp}"
    PIPELINE_STATUS = "pipeline:status:{pipeline_id}"
    PIPELINE_CHECKPOINT = "pipeline:checkpoint:{pipeline_id}"
    
    # Default TTLs (in seconds)
    TTL_REALTIME_METRICS = 300  # 5 minutes
    TTL_DASHBOARD_DATA = 300  # 5 minutes
    TTL_ALERTS = 3600  # 1 hour
    TTL_RATELIMIT = 1  # 1 second
    TTL_PIPELINE = 86400  # 24 hours
    
    def __init__(self, redis_url: Optional[str] = None, max_connections: int = 50):
        """Initialize Redis cache
        
        Args:
            redis_url: Redis connection URL
            max_connections: Maximum number of connections in pool
        """
        if redis_url is None:
            config = get_config()
            redis_url = config.redis_url
        
        self.redis_url = redis_url
        self.pool: Optional[ConnectionPool] = None
        self.client: Optional[redis.Redis] = None
        self.max_connections = max_connections
        
        self._initialize_client()
    
    def _initialize_client(self) -> None:
        """Initialize Redis client with connection pooling"""
        try:
            self.pool = ConnectionPool.from_url(
                self.redis_url,
                max_connections=self.max_connections,
                decode_responses=False  # Changed to False to support pickle
            )
            
            self.client = redis.Redis(connection_pool=self.pool)
            
            # Test connection
            self.client.ping()
            
            logger.info("Redis cache initialized successfully")
            
        except Exception as e:
            logger.error(f"Failed to initialize Redis cache: {e}", exc_info=True)
            raise

    def _serialize(self, value: Any) -> bytes:
        """Serialize value for storage in Redis
        
        Supports JSON-serializable objects and dataclasses.
        Falls back to pickle for complex objects.
        
        Args:
            value: Value to serialize
            
        Returns:
            Serialized bytes
        """
        try:
            # Try JSON first for simple types
            if isinstance(value, (str, int, float, bool, type(None))):
                return json.dumps(value).encode('utf-8')
            
            # Handle dataclasses
            if is_dataclass(value):
                data = asdict(value)
                return json.dumps(data).encode('utf-8')
            
            # Handle lists and dicts
            if isinstance(value, (list, dict)):
                return json.dumps(value).encode('utf-8')
            
            # Fall back to pickle for complex objects
            return pickle.dumps(value)
            
        except Exception as e:
            logger.warning(f"JSON serialization failed, using pickle: {e}")
            return pickle.dumps(value)
    
    def _deserialize(self, data: bytes) -> Any:
        """Deserialize value from Redis
        
        Args:
            data: Serialized bytes
            
        Returns:
            Deserialized value
        """
        try:
            # Try JSON first
            return json.loads(data.decode('utf-8'))
        except (json.JSONDecodeError, UnicodeDecodeError):
            # Fall back to pickle
            return pickle.loads(data)

    def get(self, key: str, fallback: Optional[Callable[[], Any]] = None) -> Optional[Any]:
        """Get value from cache with optional database fallback
        
        Args:
            key: Cache key
            fallback: Optional function to call on cache miss (e.g., database query)
            
        Returns:
            Cached value, fallback result, or None if not found
        """
        if self.client is None:
            logger.warning("Redis client not initialized, using fallback")
            if fallback:
                return fallback()
            return None
        
        try:
            value = self.client.get(key)
            if value is not None:
                return self._deserialize(value)
            
            # Cache miss - try fallback
            if fallback:
                logger.debug(f"Cache miss for key {key}, using fallback")
                result = fallback()
                if result is not None:
                    # Cache the fallback result
                    self.set(key, result)
                return result
            
            return None
            
        except redis.RedisError as e:
            logger.error(f"Redis error getting key {key}: {e}")
            # On Redis error, try fallback
            if fallback:
                logger.info(f"Using fallback due to Redis error for key {key}")
                return fallback()
            return None
        except Exception as e:
            logger.error(f"Failed to get cache key {key}: {e}")
            return None
    
    def set(self, key: str, value: Any, ttl: Optional[int] = None) -> bool:
        """Set value in cache
        
        Args:
            key: Cache key
            value: Value to cache (supports JSON-serializable objects and dataclasses)
            ttl: Time to live in seconds (optional)
            
        Returns:
            True if successful, False otherwise
        """
        if self.client is None:
            logger.warning("Redis client not initialized, cannot cache")
            return False
        
        try:
            serialized = self._serialize(value)
            if ttl:
                self.client.setex(key, ttl, serialized)
            else:
                self.client.set(key, serialized)
            return True
        except redis.RedisError as e:
            logger.error(f"Redis error setting key {key}: {e}")
            return False
        except Exception as e:
            logger.error(f"Failed to set cache key {key}: {e}")
            return False

    def delete(self, key: str) -> bool:
        """Delete key from cache
        
        Args:
            key: Cache key
            
        Returns:
            True if key was deleted, False otherwise
        """
        if self.client is None:
            logger.warning("Redis client not initialized")
            return False
        
        try:
            result = self.client.delete(key)
            return result > 0
        except redis.RedisError as e:
            logger.error(f"Redis error deleting key {key}: {e}")
            return False
        except Exception as e:
            logger.error(f"Failed to delete cache key {key}: {e}")
            return False
    
    def exists(self, key: str) -> bool:
        """Check if key exists in cache
        
        Args:
            key: Cache key
            
        Returns:
            True if key exists, False otherwise
        """
        if self.client is None:
            logger.warning("Redis client not initialized")
            return False
        
        try:
            return self.client.exists(key) > 0
        except redis.RedisError as e:
            logger.error(f"Redis error checking key {key}: {e}")
            return False
        except Exception as e:
            logger.error(f"Failed to check cache key {key}: {e}")
            return False
    
    def flush(self) -> bool:
        """Clear all keys from cache (use with caution!)
        
        Returns:
            True if successful, False otherwise
        """
        if self.client is None:
            logger.warning("Redis client not initialized")
            return False
        
        try:
            self.client.flushdb()
            logger.warning("Redis cache flushed")
            return True
        except redis.RedisError as e:
            logger.error(f"Redis error flushing cache: {e}")
            return False
        except Exception as e:
            logger.error(f"Failed to flush cache: {e}")
            return False
    
    def close(self) -> None:
        """Close Redis connections"""
        if self.pool:
            self.pool.disconnect()
            logger.info("Redis connections closed")
    
    # Helper methods for cache key patterns
    
    def get_article_metrics(self, article_id: int, fallback: Optional[Callable[[], Any]] = None) -> Optional[Any]:
        """Get real-time article metrics from cache
        
        Args:
            article_id: Article ID
            fallback: Optional database query function
            
        Returns:
            Article metrics or None
        """
        key = self.METRICS_ARTICLE_REALTIME.format(article_id=article_id)
        return self.get(key, fallback)
    
    def set_article_metrics(self, article_id: int, metrics: Any) -> bool:
        """Set real-time article metrics in cache
        
        Args:
            article_id: Article ID
            metrics: Metrics data
            
        Returns:
            True if successful
        """
        key = self.METRICS_ARTICLE_REALTIME.format(article_id=article_id)
        return self.set(key, metrics, ttl=self.TTL_REALTIME_METRICS)
    
    def get_cluster_metrics(self, cluster_id: int, fallback: Optional[Callable[[], Any]] = None) -> Optional[Any]:
        """Get real-time cluster metrics from cache
        
        Args:
            cluster_id: Cluster ID
            fallback: Optional database query function
            
        Returns:
            Cluster metrics or None
        """
        key = self.METRICS_CLUSTER_REALTIME.format(cluster_id=cluster_id)
        return self.get(key, fallback)
    
    def set_cluster_metrics(self, cluster_id: int, metrics: Any) -> bool:
        """Set real-time cluster metrics in cache
        
        Args:
            cluster_id: Cluster ID
            metrics: Metrics data
            
        Returns:
            True if successful
        """
        key = self.METRICS_CLUSTER_REALTIME.format(cluster_id=cluster_id)
        return self.set(key, metrics, ttl=self.TTL_REALTIME_METRICS)
    
    def get_dashboard_data(self, data_type: str, hash_key: str, fallback: Optional[Callable[[], Any]] = None) -> Optional[Any]:
        """Get dashboard data from cache
        
        Args:
            data_type: Type of dashboard data (demand_trends, competitor_comparison, topic_heatmap)
            hash_key: Hash of query parameters
            fallback: Optional database query function
            
        Returns:
            Dashboard data or None
        """
        pattern_map = {
            'demand_trends': self.DASHBOARD_DEMAND_TRENDS,
            'competitor_comparison': self.DASHBOARD_COMPETITOR_COMPARISON,
            'topic_heatmap': self.DASHBOARD_TOPIC_HEATMAP
        }
        
        pattern = pattern_map.get(data_type)
        if not pattern:
            logger.error(f"Unknown dashboard data type: {data_type}")
            return None
        
        key = pattern.format(hash=hash_key)
        return self.get(key, fallback)
    
    def set_dashboard_data(self, data_type: str, hash_key: str, data: Any) -> bool:
        """Set dashboard data in cache
        
        Args:
            data_type: Type of dashboard data
            hash_key: Hash of query parameters
            data: Dashboard data
            
        Returns:
            True if successful
        """
        pattern_map = {
            'demand_trends': self.DASHBOARD_DEMAND_TRENDS,
            'competitor_comparison': self.DASHBOARD_COMPETITOR_COMPARISON,
            'topic_heatmap': self.DASHBOARD_TOPIC_HEATMAP
        }
        
        pattern = pattern_map.get(data_type)
        if not pattern:
            logger.error(f"Unknown dashboard data type: {data_type}")
            return False
        
        key = pattern.format(hash=hash_key)
        return self.set(key, data, ttl=self.TTL_DASHBOARD_DATA)


# Global cache instance
_cache_instance: Optional[RedisCache] = None


def get_cache() -> RedisCache:
    """Get global cache instance
    
    Returns:
        RedisCache instance
    """
    global _cache_instance
    
    if _cache_instance is None:
        _cache_instance = RedisCache()
    
    return _cache_instance
