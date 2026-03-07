"""
Unit tests for WikimediaAPIClient.

Tests retry logic with mock failures, circuit breaker state transitions,
and request logging functionality.

Requirements: 1.6, 12.5, 12.6
"""

import asyncio
import pytest
from unittest.mock import AsyncMock, Mock, patch, MagicMock
import time
import logging
from aiohttp import ClientError, ClientResponseError, ClientTimeout
from aiohttp.web import Response

from src.data_ingestion.api_client import (
    WikimediaAPIClient,
    CircuitBreaker,
    CircuitState,
    CircuitBreakerConfig,
    RetryConfig
)
from src.data_ingestion.rate_limiter import RateLimiter, RateLimiterConfig


class TestCircuitBreaker:
    """Test CircuitBreaker class."""
    
    def test_circuit_breaker_initialization(self):
        """Test circuit breaker initializes in CLOSED state."""
        config = CircuitBreakerConfig(failure_threshold=3, timeout_seconds=30.0)
        breaker = CircuitBreaker("test_endpoint", config)
        
        assert breaker.state == CircuitState.CLOSED
        assert breaker.failure_count == 0
        assert breaker.success_count == 0
        assert breaker.can_request()
    
    def test_circuit_breaker_opens_after_threshold(self):
        """Test circuit opens after failure threshold is reached."""
        config = CircuitBreakerConfig(failure_threshold=3)
        breaker = CircuitBreaker("test_endpoint", config)
        
        # Record failures up to threshold
        breaker.record_failure()
        assert breaker.state == CircuitState.CLOSED
        
        breaker.record_failure()
        assert breaker.state == CircuitState.CLOSED
        
        breaker.record_failure()
        assert breaker.state == CircuitState.OPEN
        assert not breaker.can_request()
    
    def test_circuit_breaker_half_open_transition(self):
        """Test circuit transitions to HALF_OPEN after timeout."""
        config = CircuitBreakerConfig(failure_threshold=2, timeout_seconds=0.1)
        breaker = CircuitBreaker("test_endpoint", config)
        
        # Open the circuit
        breaker.record_failure()
        breaker.record_failure()
        assert breaker.state == CircuitState.OPEN
        
        # Wait for timeout
        time.sleep(0.15)
        
        # Should transition to HALF_OPEN
        assert breaker.can_request()
        assert breaker.state == CircuitState.HALF_OPEN
    
    def test_circuit_breaker_closes_after_success_threshold(self):
        """Test circuit closes after success threshold in HALF_OPEN."""
        config = CircuitBreakerConfig(
            failure_threshold=2,
            success_threshold=2,
            timeout_seconds=0.1
        )
        breaker = CircuitBreaker("test_endpoint", config)
        
        # Open the circuit
        breaker.record_failure()
        breaker.record_failure()
        assert breaker.state == CircuitState.OPEN
        
        # Wait and transition to HALF_OPEN
        time.sleep(0.15)
        breaker.can_request()
        assert breaker.state == CircuitState.HALF_OPEN
        
        # Record successes
        breaker.record_success()
        assert breaker.state == CircuitState.HALF_OPEN
        
        breaker.record_success()
        assert breaker.state == CircuitState.CLOSED
    
    def test_circuit_breaker_reopens_on_half_open_failure(self):
        """Test circuit reopens if request fails in HALF_OPEN state."""
        config = CircuitBreakerConfig(failure_threshold=2, timeout_seconds=0.1)
        breaker = CircuitBreaker("test_endpoint", config)
        
        # Open the circuit
        breaker.record_failure()
        breaker.record_failure()
        
        # Transition to HALF_OPEN
        time.sleep(0.15)
        breaker.can_request()
        assert breaker.state == CircuitState.HALF_OPEN
        
        # Fail in HALF_OPEN
        breaker.record_failure()
        assert breaker.state == CircuitState.OPEN
    
    def test_circuit_breaker_reset(self):
        """Test circuit breaker reset functionality."""
        config = CircuitBreakerConfig(failure_threshold=2)
        breaker = CircuitBreaker("test_endpoint", config)
        
        # Open the circuit
        breaker.record_failure()
        breaker.record_failure()
        assert breaker.state == CircuitState.OPEN
        
        # Reset
        breaker.reset()
        assert breaker.state == CircuitState.CLOSED
        assert breaker.failure_count == 0
        assert breaker.can_request()


class TestWikimediaAPIClient:
    """Test WikimediaAPIClient class."""
    
    @pytest.fixture
    def mock_rate_limiter(self):
        """Create a mock rate limiter that doesn't throttle."""
        limiter = Mock(spec=RateLimiter)
        limiter.acquire = AsyncMock(return_value=0.0)
        return limiter
    
    @pytest.fixture
    def retry_config(self):
        """Create retry config with fast retries for testing."""
        return RetryConfig(
            max_retries=3,
            initial_delay=0.01,
            max_delay=0.1,
            exponential_base=2.0,
            jitter=False
        )
    
    @pytest.fixture
    def circuit_config(self):
        """Create circuit breaker config for testing."""
        return CircuitBreakerConfig(
            failure_threshold=3,
            success_threshold=2,
            timeout_seconds=0.5
        )
    
    @pytest.mark.asyncio
    async def test_client_initialization(self, mock_rate_limiter):
        """Test client initializes with correct configuration."""
        client = WikimediaAPIClient(
            base_url="https://test.api.com",
            rate_limiter=mock_rate_limiter,
            timeout=30.0,
            max_connections=50
        )
        
        assert client.base_url == "https://test.api.com"
        assert client.rate_limiter == mock_rate_limiter
        assert client._timeout.total == 30.0
        
        await client.close()
    
    @pytest.mark.asyncio
    async def test_successful_request_logs_correctly(
        self, mock_rate_limiter, retry_config, caplog
    ):
        """Test successful request logs request and response."""
        client = WikimediaAPIClient(
            rate_limiter=mock_rate_limiter,
            retry_config=retry_config
        )
        
        # Mock successful response
        mock_response = AsyncMock()
        mock_response.status = 200
        mock_response.json = AsyncMock(return_value={"data": "test"})
        mock_response.headers = {}
        
        mock_session = AsyncMock()
        mock_session.request = AsyncMock(return_value=mock_response)
        mock_session.__aenter__ = AsyncMock(return_value=mock_response)
        mock_session.__aexit__ = AsyncMock()
        
        client._session = mock_session
        
        with caplog.at_level(logging.INFO):
            result = await client.get("/test/endpoint")
        
        assert result == {"data": "test"}
        
        # Check logging
        log_messages = [record.message for record in caplog.records]
        assert any("API Request" in msg for msg in log_messages)
        assert any("API Response" in msg and "200" in msg for msg in log_messages)
        
        await client.close()
    
    @pytest.mark.asyncio
    async def test_retry_logic_with_transient_failures(
        self, mock_rate_limiter, retry_config
    ):
        """Test retry logic succeeds after transient failures."""
        client = WikimediaAPIClient(
            rate_limiter=mock_rate_limiter,
            retry_config=retry_config
        )
        
        # Mock responses: fail twice, then succeed
        mock_response_fail = AsyncMock()
        mock_response_fail.status = 503
        mock_response_fail.text = AsyncMock(return_value="Service Unavailable")
        mock_response_fail.headers = {}
        
        mock_response_success = AsyncMock()
        mock_response_success.status = 200
        mock_response_success.json = AsyncMock(return_value={"data": "success"})
        mock_response_success.headers = {}
        
        mock_session = AsyncMock()
        call_count = 0
        
        async def mock_request(*args, **kwargs):
            nonlocal call_count
            call_count += 1
            if call_count <= 2:
                return mock_response_fail
            return mock_response_success
        
        mock_session.request = mock_request
        client._session = mock_session
        
        # Should succeed after retries
        result = await client.get("/test/endpoint")
        
        assert result == {"data": "success"}
        assert call_count == 3  # 2 failures + 1 success
        
        await client.close()
    
    @pytest.mark.asyncio
    async def test_retry_logic_exhausts_retries(
        self, mock_rate_limiter, retry_config
    ):
        """Test retry logic fails after max retries exceeded."""
        client = WikimediaAPIClient(
            rate_limiter=mock_rate_limiter,
            retry_config=retry_config
        )
        
        # Mock response that always fails
        mock_response = AsyncMock()
        mock_response.status = 500
        mock_response.text = AsyncMock(return_value="Internal Server Error")
        mock_response.headers = {}
        
        mock_session = AsyncMock()
        mock_session.request = AsyncMock(return_value=mock_response)
        client._session = mock_session
        
        # Should raise after exhausting retries
        with pytest.raises(ClientError, match="Server error 500 after retries"):
            await client.get("/test/endpoint")
        
        # Verify circuit breaker recorded failure
        breaker = client._get_circuit_breaker("/test/endpoint")
        assert breaker.failure_count > 0
        
        await client.close()
    
    @pytest.mark.asyncio
    async def test_retry_logic_with_rate_limit_429(
        self, mock_rate_limiter, retry_config
    ):
        """Test retry logic handles 429 rate limit with exponential backoff."""
        client = WikimediaAPIClient(
            rate_limiter=mock_rate_limiter,
            retry_config=retry_config
        )
        
        # Mock responses: 429 twice, then success
        mock_response_429 = AsyncMock()
        mock_response_429.status = 429
        mock_response_429.headers = {"Retry-After": "0.01"}
        
        mock_response_success = AsyncMock()
        mock_response_success.status = 200
        mock_response_success.json = AsyncMock(return_value={"data": "success"})
        mock_response_success.headers = {}
        
        call_count = 0
        
        async def mock_request(*args, **kwargs):
            nonlocal call_count
            call_count += 1
            if call_count <= 2:
                return mock_response_429
            return mock_response_success
        
        mock_session = AsyncMock()
        mock_session.request = mock_request
        client._session = mock_session
        
        start_time = time.time()
        result = await client.get("/test/endpoint")
        elapsed = time.time() - start_time
        
        assert result == {"data": "success"}
        assert call_count == 3
        # Should have delayed for rate limit
        assert elapsed >= 0.02  # At least 2 delays of 0.01s
        
        await client.close()
    
    @pytest.mark.asyncio
    async def test_retry_logic_with_timeout_error(
        self, mock_rate_limiter, retry_config
    ):
        """Test retry logic handles timeout errors."""
        client = WikimediaAPIClient(
            rate_limiter=mock_rate_limiter,
            retry_config=retry_config
        )
        
        call_count = 0
        
        async def mock_request(*args, **kwargs):
            nonlocal call_count
            call_count += 1
            if call_count <= 2:
                raise asyncio.TimeoutError("Request timeout")
            
            # Success on third attempt
            mock_response = AsyncMock()
            mock_response.status = 200
            mock_response.json = AsyncMock(return_value={"data": "success"})
            mock_response.headers = {}
            return mock_response
        
        mock_session = AsyncMock()
        mock_session.request = mock_request
        client._session = mock_session
        
        result = await client.get("/test/endpoint")
        
        assert result == {"data": "success"}
        assert call_count == 3
        
        await client.close()
    
    @pytest.mark.asyncio
    async def test_circuit_breaker_blocks_requests_when_open(
        self, mock_rate_limiter, circuit_config
    ):
        """Test circuit breaker blocks requests when open."""
        client = WikimediaAPIClient(
            rate_limiter=mock_rate_limiter,
            circuit_breaker_config=circuit_config
        )
        
        # Mock response that always fails
        mock_response = AsyncMock()
        mock_response.status = 500
        mock_response.text = AsyncMock(return_value="Internal Server Error")
        mock_response.headers = {}
        
        mock_session = AsyncMock()
        mock_session.request = AsyncMock(return_value=mock_response)
        client._session = mock_session
        
        # Make requests to open circuit (failure_threshold=3)
        for _ in range(3):
            try:
                await client.get("/test/endpoint")
            except ClientError:
                pass
        
        # Circuit should be open now
        breaker = client._get_circuit_breaker("/test/endpoint")
        assert breaker.state == CircuitState.OPEN
        
        # Next request should be blocked
        with pytest.raises(RuntimeError, match="Circuit breaker OPEN"):
            await client.get("/test/endpoint")
        
        await client.close()
    
    @pytest.mark.asyncio
    async def test_circuit_breaker_transitions_to_half_open(
        self, mock_rate_limiter
    ):
        """Test circuit breaker transitions from OPEN to HALF_OPEN after timeout."""
        circuit_config = CircuitBreakerConfig(
            failure_threshold=2,
            timeout_seconds=0.1
        )
        
        client = WikimediaAPIClient(
            rate_limiter=mock_rate_limiter,
            circuit_breaker_config=circuit_config
        )
        
        # Mock failing response
        mock_response = AsyncMock()
        mock_response.status = 500
        mock_response.text = AsyncMock(return_value="Error")
        mock_response.headers = {}
        
        mock_session = AsyncMock()
        mock_session.request = AsyncMock(return_value=mock_response)
        client._session = mock_session
        
        # Open the circuit
        for _ in range(2):
            try:
                await client.get("/test/endpoint")
            except ClientError:
                pass
        
        breaker = client._get_circuit_breaker("/test/endpoint")
        assert breaker.state == CircuitState.OPEN
        
        # Wait for timeout
        await asyncio.sleep(0.15)
        
        # Circuit should allow request (HALF_OPEN)
        assert breaker.can_request()
        assert breaker.state == CircuitState.HALF_OPEN
        
        await client.close()
    
    @pytest.mark.asyncio
    async def test_circuit_breaker_closes_after_success(
        self, mock_rate_limiter
    ):
        """Test circuit breaker closes after successful requests in HALF_OPEN."""
        circuit_config = CircuitBreakerConfig(
            failure_threshold=2,
            success_threshold=2,
            timeout_seconds=0.1
        )
        
        client = WikimediaAPIClient(
            rate_limiter=mock_rate_limiter,
            circuit_breaker_config=circuit_config
        )
        
        # Mock responses: fail to open circuit, then succeed to close it
        mock_response_fail = AsyncMock()
        mock_response_fail.status = 500
        mock_response_fail.text = AsyncMock(return_value="Error")
        mock_response_fail.headers = {}
        
        mock_response_success = AsyncMock()
        mock_response_success.status = 200
        mock_response_success.json = AsyncMock(return_value={"data": "success"})
        mock_response_success.headers = {}
        
        call_count = 0
        
        async def mock_request(*args, **kwargs):
            nonlocal call_count
            call_count += 1
            if call_count <= 2:
                return mock_response_fail
            return mock_response_success
        
        mock_session = AsyncMock()
        mock_session.request = mock_request
        client._session = mock_session
        
        # Open the circuit
        for _ in range(2):
            try:
                await client.get("/test/endpoint")
            except ClientError:
                pass
        
        breaker = client._get_circuit_breaker("/test/endpoint")
        assert breaker.state == CircuitState.OPEN
        
        # Wait for timeout to transition to HALF_OPEN
        await asyncio.sleep(0.15)
        
        # Make successful requests to close circuit
        await client.get("/test/endpoint")
        assert breaker.state == CircuitState.HALF_OPEN
        
        await client.get("/test/endpoint")
        assert breaker.state == CircuitState.CLOSED
        
        await client.close()
    
    @pytest.mark.asyncio
    async def test_request_logging_includes_timing(
        self, mock_rate_limiter, retry_config, caplog
    ):
        """Test request logging includes timing information."""
        client = WikimediaAPIClient(
            rate_limiter=mock_rate_limiter,
            retry_config=retry_config
        )
        
        # Mock response with delay
        mock_response = AsyncMock()
        mock_response.status = 200
        mock_response.json = AsyncMock(return_value={"data": "test"})
        mock_response.headers = {}
        
        async def delayed_request(*args, **kwargs):
            await asyncio.sleep(0.05)
            return mock_response
        
        mock_session = AsyncMock()
        mock_session.request = delayed_request
        client._session = mock_session
        
        with caplog.at_level(logging.INFO):
            await client.get("/test/endpoint")
        
        # Check that response log includes duration
        log_records = [r for r in caplog.records if "API Response" in r.message]
        assert len(log_records) > 0
        
        response_log = log_records[0]
        assert hasattr(response_log, "duration_seconds")
        assert response_log.duration_seconds >= 0.05
        
        await client.close()
    
    @pytest.mark.asyncio
    async def test_request_logging_includes_status_code(
        self, mock_rate_limiter, retry_config, caplog
    ):
        """Test request logging includes HTTP status code."""
        client = WikimediaAPIClient(
            rate_limiter=mock_rate_limiter,
            retry_config=retry_config
        )
        
        # Mock response
        mock_response = AsyncMock()
        mock_response.status = 201
        mock_response.json = AsyncMock(return_value={"created": True})
        mock_response.headers = {}
        
        mock_session = AsyncMock()
        mock_session.request = AsyncMock(return_value=mock_response)
        client._session = mock_session
        
        with caplog.at_level(logging.INFO):
            await client.get("/test/endpoint")
        
        # Check status code in logs
        log_records = [r for r in caplog.records if "API Response" in r.message]
        assert len(log_records) > 0
        
        response_log = log_records[0]
        assert hasattr(response_log, "status_code")
        assert response_log.status_code == 201
        assert "201" in response_log.message
        
        await client.close()
    
    @pytest.mark.asyncio
    async def test_error_logging_on_failure(
        self, mock_rate_limiter, retry_config, caplog
    ):
        """Test error logging when requests fail."""
        client = WikimediaAPIClient(
            rate_limiter=mock_rate_limiter,
            retry_config=retry_config
        )
        
        # Mock failing response
        mock_response = AsyncMock()
        mock_response.status = 404
        mock_response.text = AsyncMock(return_value="Not Found")
        mock_response.headers = {}
        
        mock_session = AsyncMock()
        mock_session.request = AsyncMock(return_value=mock_response)
        client._session = mock_session
        
        with caplog.at_level(logging.ERROR):
            try:
                await client.get("/test/endpoint")
            except ClientError:
                pass
        
        # Check error logging
        error_logs = [r for r in caplog.records if r.levelname == "ERROR"]
        assert len(error_logs) > 0
        assert any("404" in r.message for r in error_logs)
        
        await client.close()
    
    @pytest.mark.asyncio
    async def test_exponential_backoff_calculation(self, mock_rate_limiter):
        """Test exponential backoff delay calculation."""
        retry_config = RetryConfig(
            initial_delay=1.0,
            exponential_base=2.0,
            max_delay=10.0,
            jitter=False
        )
        
        client = WikimediaAPIClient(
            rate_limiter=mock_rate_limiter,
            retry_config=retry_config
        )
        
        # Test backoff delays
        delay_0 = client._calculate_backoff_delay(0)
        delay_1 = client._calculate_backoff_delay(1)
        delay_2 = client._calculate_backoff_delay(2)
        delay_3 = client._calculate_backoff_delay(3)
        
        assert delay_0 == 1.0  # 1.0 * 2^0
        assert delay_1 == 2.0  # 1.0 * 2^1
        assert delay_2 == 4.0  # 1.0 * 2^2
        assert delay_3 == 8.0  # 1.0 * 2^3
        
        # Test max delay cap
        delay_10 = client._calculate_backoff_delay(10)
        assert delay_10 == 10.0  # Capped at max_delay
        
        await client.close()
    
    @pytest.mark.asyncio
    async def test_client_context_manager(self, mock_rate_limiter):
        """Test client works as async context manager."""
        async with WikimediaAPIClient(rate_limiter=mock_rate_limiter) as client:
            assert client._session is None or not client._session.closed
        
        # Session should be closed after context exit
        # Note: session is created lazily, so it might be None
    
    @pytest.mark.asyncio
    async def test_get_circuit_breaker_status(self, mock_rate_limiter):
        """Test getting circuit breaker status for all endpoints."""
        client = WikimediaAPIClient(rate_limiter=mock_rate_limiter)
        
        # Create circuit breakers for different endpoints
        breaker1 = client._get_circuit_breaker("/endpoint1")
        breaker2 = client._get_circuit_breaker("/endpoint2")
        
        # Open one circuit
        breaker1.record_failure()
        breaker1.record_failure()
        breaker1.record_failure()
        
        status = client.get_circuit_breaker_status()
        
        assert "/endpoint1" in status
        assert "/endpoint2" in status
        assert status["/endpoint1"] == "open"
        assert status["/endpoint2"] == "closed"
        
        await client.close()
    
    @pytest.mark.asyncio
    async def test_reset_circuit_breaker(self, mock_rate_limiter):
        """Test resetting specific circuit breaker."""
        client = WikimediaAPIClient(rate_limiter=mock_rate_limiter)
        
        breaker = client._get_circuit_breaker("/test/endpoint")
        
        # Open the circuit
        breaker.record_failure()
        breaker.record_failure()
        breaker.record_failure()
        assert breaker.state == CircuitState.OPEN
        
        # Reset
        client.reset_circuit_breaker("/test/endpoint")
        assert breaker.state == CircuitState.CLOSED
        
        await client.close()
    
    @pytest.mark.asyncio
    async def test_reset_all_circuit_breakers(self, mock_rate_limiter):
        """Test resetting all circuit breakers."""
        client = WikimediaAPIClient(rate_limiter=mock_rate_limiter)
        
        # Create and open multiple circuits
        breaker1 = client._get_circuit_breaker("/endpoint1")
        breaker2 = client._get_circuit_breaker("/endpoint2")
        
        for breaker in [breaker1, breaker2]:
            breaker.record_failure()
            breaker.record_failure()
            breaker.record_failure()
        
        assert breaker1.state == CircuitState.OPEN
        assert breaker2.state == CircuitState.OPEN
        
        # Reset all
        client.reset_all_circuit_breakers()
        
        assert breaker1.state == CircuitState.CLOSED
        assert breaker2.state == CircuitState.CLOSED
        
        await client.close()
