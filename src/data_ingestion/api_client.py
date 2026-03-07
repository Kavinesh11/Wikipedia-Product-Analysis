"""
Wikimedia API client with rate limiting, circuit breaker, and retry logic.

This module provides a robust HTTP client for interacting with Wikimedia APIs
with automatic error handling, exponential backoff, and circuit breaker pattern.
"""

import asyncio
import time
from typing import Optional, Dict, Any, Callable
from dataclasses import dataclass
from enum import Enum
import logging
import aiohttp
from aiohttp import ClientSession, ClientTimeout, ClientError

from .rate_limiter import RateLimiter, RateLimiterConfig

logger = logging.getLogger(__name__)


class CircuitState(Enum):
    """Circuit breaker states."""
    CLOSED = "closed"  # Normal operation
    OPEN = "open"  # Failing, reject requests
    HALF_OPEN = "half_open"  # Testing if service recovered


@dataclass
class CircuitBreakerConfig:
    """Configuration for circuit breaker."""
    failure_threshold: int = 5  # Open circuit after N failures
    success_threshold: int = 2  # Close circuit after N successes in half-open
    timeout_seconds: float = 60.0  # Time to wait before trying half-open
    

@dataclass
class RetryConfig:
    """Configuration for retry logic."""
    max_retries: int = 3
    initial_delay: float = 1.0  # Initial backoff delay in seconds
    max_delay: float = 60.0  # Maximum backoff delay
    exponential_base: float = 2.0  # Exponential backoff multiplier
    jitter: bool = True  # Add random jitter to backoff


class CircuitBreaker:
    """
    Circuit breaker pattern implementation for API endpoints.
    
    Prevents cascading failures by stopping requests to failing endpoints.
    Requirements: 12.5
    """
    
    def __init__(self, endpoint: str, config: Optional[CircuitBreakerConfig] = None):
        """
        Initialize circuit breaker for an endpoint.
        
        Args:
            endpoint: API endpoint identifier
            config: Circuit breaker configuration
        """
        self.endpoint = endpoint
        self.config = config or CircuitBreakerConfig()
        
        self.state = CircuitState.CLOSED
        self.failure_count = 0
        self.success_count = 0
        self.last_failure_time: Optional[float] = None
        
        logger.info(f"CircuitBreaker initialized for {endpoint}: {self.config}")
    
    def can_request(self) -> bool:
        """
        Check if requests are allowed through the circuit.
        
        Returns:
            True if request can proceed, False if circuit is open
        """
        if self.state == CircuitState.CLOSED:
            return True
        
        if self.state == CircuitState.HALF_OPEN:
            return True
        
        # State is OPEN - check if timeout has elapsed
        if self.last_failure_time is None:
            return False
        
        elapsed = time.monotonic() - self.last_failure_time
        if elapsed >= self.config.timeout_seconds:
            # Transition to half-open for testing
            self.state = CircuitState.HALF_OPEN
            self.success_count = 0
            logger.info(f"CircuitBreaker {self.endpoint}: OPEN -> HALF_OPEN (timeout elapsed)")
            return True
        
        return False
    
    def record_success(self) -> None:
        """
        Record a successful request.
        
        In HALF_OPEN state, transitions to CLOSED after success threshold.
        In CLOSED state, resets failure count.
        """
        if self.state == CircuitState.CLOSED:
            self.failure_count = 0
            return
        
        if self.state == CircuitState.HALF_OPEN:
            self.success_count += 1
            logger.debug(
                f"CircuitBreaker {self.endpoint}: Success {self.success_count}/"
                f"{self.config.success_threshold} in HALF_OPEN"
            )
            
            if self.success_count >= self.config.success_threshold:
                self.state = CircuitState.CLOSED
                self.failure_count = 0
                self.success_count = 0
                logger.info(f"CircuitBreaker {self.endpoint}: HALF_OPEN -> CLOSED")
    
    def record_failure(self) -> None:
        """
        Record a failed request.
        
        In CLOSED state, opens circuit after failure threshold.
        In HALF_OPEN state, immediately reopens circuit.
        """
        self.last_failure_time = time.monotonic()
        
        if self.state == CircuitState.HALF_OPEN:
            # Failed during testing, reopen circuit
            self.state = CircuitState.OPEN
            self.failure_count = 0
            self.success_count = 0
            logger.warning(f"CircuitBreaker {self.endpoint}: HALF_OPEN -> OPEN (test failed)")
            return
        
        if self.state == CircuitState.CLOSED:
            self.failure_count += 1
            logger.debug(
                f"CircuitBreaker {self.endpoint}: Failure {self.failure_count}/"
                f"{self.config.failure_threshold}"
            )
            
            if self.failure_count >= self.config.failure_threshold:
                self.state = CircuitState.OPEN
                logger.warning(
                    f"CircuitBreaker {self.endpoint}: CLOSED -> OPEN "
                    f"(threshold {self.config.failure_threshold} reached)"
                )
    
    def get_state(self) -> CircuitState:
        """Get current circuit state."""
        return self.state
    
    def reset(self) -> None:
        """Reset circuit breaker to initial state."""
        self.state = CircuitState.CLOSED
        self.failure_count = 0
        self.success_count = 0
        self.last_failure_time = None
        logger.info(f"CircuitBreaker {self.endpoint}: Reset to CLOSED")


class WikimediaAPIClient:
    """
    Base HTTP client for Wikimedia APIs with robust error handling.
    
    Features:
    - Connection pooling for efficient HTTP requests
    - Rate limiting with automatic throttling
    - Exponential backoff for rate limit errors (429)
    - Circuit breaker pattern for failing endpoints
    - Request/response logging with timestamps
    
    Requirements: 1.6, 12.5, 12.6
    """
    
    def __init__(
        self,
        base_url: str = "https://wikimedia.org/api/rest_v1",
        rate_limiter: Optional[RateLimiter] = None,
        retry_config: Optional[RetryConfig] = None,
        circuit_breaker_config: Optional[CircuitBreakerConfig] = None,
        timeout: float = 30.0,
        max_connections: int = 100,
        user_agent: str = "WikipediaIntelligenceSystem/1.0"
    ):
        """
        Initialize Wikimedia API client.
        
        Args:
            base_url: Base URL for API requests
            rate_limiter: Rate limiter instance (creates default if None)
            retry_config: Retry configuration
            circuit_breaker_config: Circuit breaker configuration
            timeout: Request timeout in seconds
            max_connections: Maximum concurrent connections
            user_agent: User agent string for requests
        """
        self.base_url = base_url.rstrip('/')
        self.rate_limiter = rate_limiter or RateLimiter()
        self.retry_config = retry_config or RetryConfig()
        self.circuit_breaker_config = circuit_breaker_config or CircuitBreakerConfig()
        self.user_agent = user_agent
        
        # Circuit breakers per endpoint
        self._circuit_breakers: Dict[str, CircuitBreaker] = {}
        
        # Connection pooling configuration
        self._timeout = ClientTimeout(total=timeout)
        self._connector = aiohttp.TCPConnector(
            limit=max_connections,
            limit_per_host=30,
            ttl_dns_cache=300
        )
        
        # Session will be created lazily
        self._session: Optional[ClientSession] = None
        
        logger.info(
            f"WikimediaAPIClient initialized: base_url={base_url}, "
            f"timeout={timeout}s, max_connections={max_connections}"
        )
    
    async def _get_session(self) -> ClientSession:
        """Get or create aiohttp session with connection pooling."""
        if self._session is None or self._session.closed:
            self._session = ClientSession(
                connector=self._connector,
                timeout=self._timeout,
                headers={"User-Agent": self.user_agent}
            )
            logger.debug("Created new aiohttp session with connection pooling")
        return self._session
    
    def _get_circuit_breaker(self, endpoint: str) -> CircuitBreaker:
        """Get or create circuit breaker for endpoint."""
        if endpoint not in self._circuit_breakers:
            self._circuit_breakers[endpoint] = CircuitBreaker(
                endpoint, self.circuit_breaker_config
            )
        return self._circuit_breakers[endpoint]
    
    def _calculate_backoff_delay(self, attempt: int) -> float:
        """
        Calculate exponential backoff delay with optional jitter.
        
        Args:
            attempt: Retry attempt number (0-indexed)
            
        Returns:
            Delay in seconds
        """
        delay = min(
            self.retry_config.initial_delay * (self.retry_config.exponential_base ** attempt),
            self.retry_config.max_delay
        )
        
        if self.retry_config.jitter:
            import random
            # Add jitter: random value between 0 and delay
            delay = delay * (0.5 + random.random() * 0.5)
        
        return delay
    
    async def request(
        self,
        method: str,
        endpoint: str,
        params: Optional[Dict[str, Any]] = None,
        headers: Optional[Dict[str, str]] = None,
        json_data: Optional[Dict[str, Any]] = None,
        **kwargs
    ) -> Dict[str, Any]:
        """
        Make HTTP request with rate limiting, retries, and circuit breaker.
        
        Args:
            method: HTTP method (GET, POST, etc.)
            endpoint: API endpoint path (relative to base_url)
            params: Query parameters
            headers: Additional headers
            json_data: JSON request body
            **kwargs: Additional arguments for aiohttp request
            
        Returns:
            Response JSON data
            
        Raises:
            aiohttp.ClientError: On request failure after retries
            RuntimeError: If circuit breaker is open
        """
        # Check circuit breaker
        circuit_breaker = self._get_circuit_breaker(endpoint)
        if not circuit_breaker.can_request():
            error_msg = f"Circuit breaker OPEN for endpoint: {endpoint}"
            logger.error(error_msg)
            raise RuntimeError(error_msg)
        
        url = f"{self.base_url}/{endpoint.lstrip('/')}"
        
        # Merge headers
        request_headers = headers or {}
        
        attempt = 0
        last_exception = None
        
        while attempt <= self.retry_config.max_retries:
            try:
                # Acquire rate limit token
                await self.rate_limiter.acquire()
                
                # Log request
                request_start = time.time()
                logger.info(
                    f"API Request: {method} {url} "
                    f"(attempt {attempt + 1}/{self.retry_config.max_retries + 1})",
                    extra={
                        "method": method,
                        "url": url,
                        "endpoint": endpoint,
                        "attempt": attempt + 1,
                        "timestamp": request_start
                    }
                )
                
                # Make request
                session = await self._get_session()
                async with session.request(
                    method,
                    url,
                    params=params,
                    headers=request_headers,
                    json=json_data,
                    **kwargs
                ) as response:
                    request_duration = time.time() - request_start
                    
                    # Log response
                    logger.info(
                        f"API Response: {method} {url} -> {response.status} "
                        f"({request_duration:.3f}s)",
                        extra={
                            "method": method,
                            "url": url,
                            "endpoint": endpoint,
                            "status_code": response.status,
                            "duration_seconds": request_duration,
                            "timestamp": time.time()
                        }
                    )
                    
                    # Handle rate limit errors with exponential backoff
                    if response.status == 429:
                        retry_after = response.headers.get('Retry-After')
                        if retry_after:
                            delay = float(retry_after)
                        else:
                            delay = self._calculate_backoff_delay(attempt)
                        
                        logger.warning(
                            f"Rate limit hit (429) for {endpoint}, "
                            f"backing off {delay:.2f}s",
                            extra={
                                "endpoint": endpoint,
                                "backoff_delay": delay,
                                "attempt": attempt + 1
                            }
                        )
                        
                        if attempt < self.retry_config.max_retries:
                            await asyncio.sleep(delay)
                            attempt += 1
                            continue
                        else:
                            circuit_breaker.record_failure()
                            raise aiohttp.ClientError(
                                f"Rate limit exceeded after {self.retry_config.max_retries} retries"
                            )
                    
                    # Handle other HTTP errors
                    if response.status >= 500:
                        # Server error - retry with backoff
                        error_text = await response.text()
                        logger.error(
                            f"Server error {response.status} for {endpoint}: {error_text}",
                            extra={
                                "endpoint": endpoint,
                                "status_code": response.status,
                                "error_text": error_text
                            }
                        )
                        
                        if attempt < self.retry_config.max_retries:
                            delay = self._calculate_backoff_delay(attempt)
                            await asyncio.sleep(delay)
                            attempt += 1
                            continue
                        else:
                            circuit_breaker.record_failure()
                            raise aiohttp.ClientError(
                                f"Server error {response.status} after retries: {error_text}"
                            )
                    
                    if response.status >= 400:
                        # Client error - don't retry
                        error_text = await response.text()
                        logger.error(
                            f"Client error {response.status} for {endpoint}: {error_text}",
                            extra={
                                "endpoint": endpoint,
                                "status_code": response.status,
                                "error_text": error_text
                            }
                        )
                        circuit_breaker.record_failure()
                        raise aiohttp.ClientError(
                            f"Client error {response.status}: {error_text}"
                        )
                    
                    # Success - parse response
                    response_data = await response.json()
                    circuit_breaker.record_success()
                    return response_data
            
            except (asyncio.TimeoutError, aiohttp.ClientError) as e:
                last_exception = e
                logger.error(
                    f"Request failed for {endpoint}: {type(e).__name__}: {e}",
                    extra={
                        "endpoint": endpoint,
                        "error_type": type(e).__name__,
                        "error_message": str(e),
                        "attempt": attempt + 1
                    }
                )
                
                if attempt < self.retry_config.max_retries:
                    delay = self._calculate_backoff_delay(attempt)
                    logger.info(f"Retrying after {delay:.2f}s...")
                    await asyncio.sleep(delay)
                    attempt += 1
                else:
                    circuit_breaker.record_failure()
                    raise
        
        # Should not reach here, but just in case
        circuit_breaker.record_failure()
        raise last_exception or RuntimeError("Request failed after all retries")
    
    async def get(
        self,
        endpoint: str,
        params: Optional[Dict[str, Any]] = None,
        **kwargs
    ) -> Dict[str, Any]:
        """
        Make GET request.
        
        Args:
            endpoint: API endpoint path
            params: Query parameters
            **kwargs: Additional arguments
            
        Returns:
            Response JSON data
        """
        return await self.request("GET", endpoint, params=params, **kwargs)
    
    async def post(
        self,
        endpoint: str,
        json_data: Optional[Dict[str, Any]] = None,
        **kwargs
    ) -> Dict[str, Any]:
        """
        Make POST request.
        
        Args:
            endpoint: API endpoint path
            json_data: JSON request body
            **kwargs: Additional arguments
            
        Returns:
            Response JSON data
        """
        return await self.request("POST", endpoint, json_data=json_data, **kwargs)
    
    async def close(self) -> None:
        """Close HTTP session and cleanup resources."""
        if self._session and not self._session.closed:
            await self._session.close()
            logger.info("WikimediaAPIClient session closed")
    
    async def __aenter__(self):
        """Async context manager entry."""
        return self
    
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Async context manager exit."""
        await self.close()
    
    def get_circuit_breaker_status(self) -> Dict[str, str]:
        """
        Get status of all circuit breakers.
        
        Returns:
            Dictionary mapping endpoint to circuit state
        """
        return {
            endpoint: breaker.get_state().value
            for endpoint, breaker in self._circuit_breakers.items()
        }
    
    def reset_circuit_breaker(self, endpoint: str) -> None:
        """
        Reset circuit breaker for specific endpoint.
        
        Args:
            endpoint: Endpoint to reset
        """
        if endpoint in self._circuit_breakers:
            self._circuit_breakers[endpoint].reset()
            logger.info(f"Reset circuit breaker for endpoint: {endpoint}")
    
    def reset_all_circuit_breakers(self) -> None:
        """Reset all circuit breakers."""
        for breaker in self._circuit_breakers.values():
            breaker.reset()
        logger.info("Reset all circuit breakers")
