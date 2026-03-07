"""Health Check System

Provides health check endpoints for monitoring system status.
Checks database connectivity, Redis connectivity, and API availability.
"""
import asyncio
import logging
from typing import Dict, Optional, List
from dataclasses import dataclass, asdict
from datetime import datetime
from enum import Enum

import aiohttp
from sqlalchemy import text
from redis.exceptions import RedisError

from src.storage.database import Database
from src.storage.cache import RedisCache


logger = logging.getLogger(__name__)


class HealthStatus(Enum):
    """Health check status values"""
    HEALTHY = "healthy"
    DEGRADED = "degraded"
    UNHEALTHY = "unhealthy"


@dataclass
class ComponentHealth:
    """Health status for a single component"""
    name: str
    status: HealthStatus
    message: str
    response_time_ms: Optional[float] = None
    last_checked: datetime = None
    
    def __post_init__(self):
        if self.last_checked is None:
            self.last_checked = datetime.now()


@dataclass
class SystemHealth:
    """Overall system health status"""
    status: HealthStatus
    components: List[ComponentHealth]
    timestamp: datetime
    version: str = "1.0.0"
    
    def to_dict(self) -> Dict:
        """Convert to dictionary for JSON serialization"""
        return {
            'status': self.status.value,
            'components': [
                {
                    'name': c.name,
                    'status': c.status.value,
                    'message': c.message,
                    'response_time_ms': c.response_time_ms,
                    'last_checked': c.last_checked.isoformat()
                }
                for c in self.components
            ],
            'timestamp': self.timestamp.isoformat(),
            'version': self.version
        }


class HealthChecker:
    """
    System health checker.
    
    Performs health checks on all system components and returns
    aggregated health status.
    
    Requirements: 14.5
    """
    
    def __init__(
        self,
        database: Optional[Database] = None,
        cache: Optional[RedisCache] = None,
        api_endpoints: Optional[List[str]] = None,
        timeout_seconds: float = 5.0
    ):
        """Initialize health checker
        
        Args:
            database: Database instance to check
            cache: Redis cache instance to check
            api_endpoints: List of API endpoints to check
            timeout_seconds: Timeout for health checks
        """
        self.database = database
        self.cache = cache
        self.api_endpoints = api_endpoints or []
        self.timeout_seconds = timeout_seconds
        
        logger.info(
            f"HealthChecker initialized: "
            f"db={database is not None}, "
            f"cache={cache is not None}, "
            f"apis={len(self.api_endpoints)}"
        )
    
    async def check_health(self) -> SystemHealth:
        """Perform health checks on all components
        
        Returns:
            SystemHealth with status of all components
        """
        logger.info("Starting health check")
        
        components = []
        
        # Check database
        if self.database:
            db_health = await self._check_database()
            components.append(db_health)
        
        # Check Redis cache
        if self.cache:
            cache_health = await self._check_cache()
            components.append(cache_health)
        
        # Check API endpoints
        for endpoint in self.api_endpoints:
            api_health = await self._check_api(endpoint)
            components.append(api_health)
        
        # Determine overall status
        overall_status = self._determine_overall_status(components)
        
        system_health = SystemHealth(
            status=overall_status,
            components=components,
            timestamp=datetime.now()
        )
        
        logger.info(f"Health check complete: {overall_status.value}")
        
        return system_health
    
    async def _check_database(self) -> ComponentHealth:
        """Check database connectivity
        
        Returns:
            ComponentHealth for database
        """
        import time
        start_time = time.monotonic()
        
        try:
            # Try to execute a simple query
            async with asyncio.timeout(self.timeout_seconds):
                # Use sync database connection
                with self.database.get_session() as session:
                    result = session.execute(text("SELECT 1"))
                    result.fetchone()
            
            response_time = (time.monotonic() - start_time) * 1000
            
            return ComponentHealth(
                name="database",
                status=HealthStatus.HEALTHY,
                message="Database connection successful",
                response_time_ms=round(response_time, 2)
            )
            
        except asyncio.TimeoutError:
            response_time = (time.monotonic() - start_time) * 1000
            logger.error("Database health check timed out")
            return ComponentHealth(
                name="database",
                status=HealthStatus.UNHEALTHY,
                message=f"Database connection timeout after {self.timeout_seconds}s",
                response_time_ms=round(response_time, 2)
            )
            
        except Exception as e:
            response_time = (time.monotonic() - start_time) * 1000
            logger.error(f"Database health check failed: {e}", exc_info=True)
            return ComponentHealth(
                name="database",
                status=HealthStatus.UNHEALTHY,
                message=f"Database connection failed: {str(e)}",
                response_time_ms=round(response_time, 2)
            )
    
    async def _check_cache(self) -> ComponentHealth:
        """Check Redis cache connectivity
        
        Returns:
            ComponentHealth for cache
        """
        import time
        start_time = time.monotonic()
        
        try:
            # Try to ping Redis
            async with asyncio.timeout(self.timeout_seconds):
                # Test set and get
                test_key = "_health_check_test"
                test_value = "ok"
                
                self.cache.set(test_key, test_value, ttl=10)
                result = self.cache.get(test_key)
                
                if result != test_value:
                    raise ValueError("Cache read/write test failed")
                
                # Clean up
                self.cache.delete(test_key)
            
            response_time = (time.monotonic() - start_time) * 1000
            
            return ComponentHealth(
                name="cache",
                status=HealthStatus.HEALTHY,
                message="Cache connection successful",
                response_time_ms=round(response_time, 2)
            )
            
        except asyncio.TimeoutError:
            response_time = (time.monotonic() - start_time) * 1000
            logger.error("Cache health check timed out")
            return ComponentHealth(
                name="cache",
                status=HealthStatus.UNHEALTHY,
                message=f"Cache connection timeout after {self.timeout_seconds}s",
                response_time_ms=round(response_time, 2)
            )
            
        except RedisError as e:
            response_time = (time.monotonic() - start_time) * 1000
            logger.error(f"Cache health check failed: {e}", exc_info=True)
            return ComponentHealth(
                name="cache",
                status=HealthStatus.UNHEALTHY,
                message=f"Cache connection failed: {str(e)}",
                response_time_ms=round(response_time, 2)
            )
            
        except Exception as e:
            response_time = (time.monotonic() - start_time) * 1000
            logger.error(f"Cache health check failed: {e}", exc_info=True)
            return ComponentHealth(
                name="cache",
                status=HealthStatus.UNHEALTHY,
                message=f"Cache error: {str(e)}",
                response_time_ms=round(response_time, 2)
            )
    
    async def _check_api(self, endpoint: str) -> ComponentHealth:
        """Check API endpoint availability
        
        Args:
            endpoint: API endpoint URL
            
        Returns:
            ComponentHealth for API
        """
        import time
        start_time = time.monotonic()
        
        try:
            async with asyncio.timeout(self.timeout_seconds):
                async with aiohttp.ClientSession() as session:
                    async with session.get(endpoint) as response:
                        status_code = response.status
            
            response_time = (time.monotonic() - start_time) * 1000
            
            # Consider 2xx and 3xx as healthy
            if 200 <= status_code < 400:
                return ComponentHealth(
                    name=f"api:{endpoint}",
                    status=HealthStatus.HEALTHY,
                    message=f"API responding (status {status_code})",
                    response_time_ms=round(response_time, 2)
                )
            else:
                return ComponentHealth(
                    name=f"api:{endpoint}",
                    status=HealthStatus.DEGRADED,
                    message=f"API returned status {status_code}",
                    response_time_ms=round(response_time, 2)
                )
                
        except asyncio.TimeoutError:
            response_time = (time.monotonic() - start_time) * 1000
            logger.error(f"API health check timed out: {endpoint}")
            return ComponentHealth(
                name=f"api:{endpoint}",
                status=HealthStatus.UNHEALTHY,
                message=f"API timeout after {self.timeout_seconds}s",
                response_time_ms=round(response_time, 2)
            )
            
        except Exception as e:
            response_time = (time.monotonic() - start_time) * 1000
            logger.error(f"API health check failed for {endpoint}: {e}", exc_info=True)
            return ComponentHealth(
                name=f"api:{endpoint}",
                status=HealthStatus.UNHEALTHY,
                message=f"API check failed: {str(e)}",
                response_time_ms=round(response_time, 2)
            )
    
    def _determine_overall_status(self, components: List[ComponentHealth]) -> HealthStatus:
        """Determine overall system status from component statuses
        
        Args:
            components: List of component health statuses
            
        Returns:
            Overall system health status
        """
        if not components:
            return HealthStatus.HEALTHY
        
        # If any component is unhealthy, system is unhealthy
        if any(c.status == HealthStatus.UNHEALTHY for c in components):
            return HealthStatus.UNHEALTHY
        
        # If any component is degraded, system is degraded
        if any(c.status == HealthStatus.DEGRADED for c in components):
            return HealthStatus.DEGRADED
        
        # All components healthy
        return HealthStatus.HEALTHY


async def health_check_endpoint(
    database: Optional[Database] = None,
    cache: Optional[RedisCache] = None,
    api_endpoints: Optional[List[str]] = None
) -> Dict:
    """Health check endpoint handler
    
    This function can be used as a FastAPI/Flask endpoint handler.
    
    Args:
        database: Database instance
        cache: Cache instance
        api_endpoints: List of API endpoints to check
        
    Returns:
        Dictionary with health status (suitable for JSON response)
    """
    checker = HealthChecker(
        database=database,
        cache=cache,
        api_endpoints=api_endpoints
    )
    
    health = await checker.check_health()
    return health.to_dict()
