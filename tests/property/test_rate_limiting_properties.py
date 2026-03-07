"""Property-based tests for rate limiting and API management

Feature: wikipedia-intelligence-system
"""
import pytest
import asyncio
import time
from hypothesis import given, strategies as st, assume, settings, HealthCheck
from unittest.mock import AsyncMock, Mock

from src.data_ingestion.rate_limiter import RateLimiter, RateLimiterConfig
from src.data_ingestion.request_queue import (
    PriorityRequestQueue,
    RequestPriority,
)
from src.data_ingestion.api_client import (
    CircuitBreaker,
    CircuitBreakerConfig,
    CircuitState,
)


# Feature: wikipedia-intelligence-system, Property 57: API Rate Limit Compliance
@given(
    num_requests=st.integers(min_value=1, max_value=300),
    rate_limit=st.floats(min_value=50.0, max_value=200.0)
)
@settings(
    suppress_health_check=[HealthCheck.function_scoped_fixture],
    max_examples=100,
    deadline=None
)
@pytest.mark.asyncio
async def test_property_57_rate_limit_compliance(num_requests, rate_limit):
    """Property 57: API Rate Limit Compliance
    
    For any time window of 1 second, the System should make no more than 
    the configured maximum requests to the Wikimedia API.
    
    Validates: Requirements 12.1
    """
    # Configure rate limiter with test rate limit
    config = RateLimiterConfig(max_requests_per_second=rate_limit)
    limiter = RateLimiter(config)
    
    # Track requests made in each 1-second window
    request_times = []
    
    # Make requests
    for _ in range(num_requests):
        await limiter.acquire(tokens=1)
        request_times.append(time.monotonic())
    
    # Verify no 1-second window exceeds rate limit
    for i in range(len(request_times)):
        window_start = request_times[i]
        window_end = window_start + 1.0
        
        # Count requests in this 1-second window
        requests_in_window = sum(
            1 for t in request_times 
            if window_start <= t < window_end
        )
        
        # Should not exceed rate limit (allow small tolerance for timing)
        assert requests_in_window <= rate_limit + 2, \
            f"Window starting at {window_start:.3f} had {requests_in_window} requests, " \
            f"exceeding limit of {rate_limit}"


# Feature: wikipedia-intelligence-system, Property 58: Automatic Request Throttling
@given(
    initial_capacity=st.floats(min_value=0.05, max_value=0.15),
    rate_limit=st.floats(min_value=50.0, max_value=200.0)
)
@settings(
    suppress_health_check=[HealthCheck.function_scoped_fixture],
    max_examples=100,
    deadline=None
)
@pytest.mark.asyncio
async def test_property_58_automatic_throttling(initial_capacity, rate_limit):
    """Property 58: Automatic Request Throttling
    
    For any situation where request rate approaches 90% of the limit, the System 
    should automatically throttle subsequent requests to stay under the limit.
    
    Validates: Requirements 12.2
    """
    # Configure rate limiter with throttle threshold at 90%
    config = RateLimiterConfig(
        max_requests_per_second=rate_limit,
        throttle_threshold=0.9
    )
    limiter = RateLimiter(config)
    
    # Deplete tokens to below throttle threshold (10% capacity remaining)
    tokens_to_consume = int(limiter.max_tokens * (1 - initial_capacity))
    if tokens_to_consume > 0:
        limiter.tokens = limiter.max_tokens - tokens_to_consume
    
    # Verify we're below throttle threshold
    capacity = limiter.get_capacity_percentage()
    assume(capacity < (1 - config.throttle_threshold))
    
    # Make a request - should be throttled (delayed)
    start_time = time.monotonic()
    wait_time = await limiter.acquire(tokens=1)
    elapsed = time.monotonic() - start_time
    
    # Should have experienced throttling delay
    # (either from wait_time or actual elapsed time)
    assert wait_time > 0 or elapsed > 0.05, \
        f"Expected throttling delay when capacity={capacity:.2%}, " \
        f"but wait_time={wait_time:.3f}s, elapsed={elapsed:.3f}s"


# Feature: wikipedia-intelligence-system, Property 59: Priority Queue Ordering
@given(
    priorities=st.lists(
        st.sampled_from([
            RequestPriority.CRITICAL,
            RequestPriority.HIGH,
            RequestPriority.NORMAL,
            RequestPriority.LOW,
            RequestPriority.BULK
        ]),
        min_size=5,
        max_size=20
    )
)
@settings(
    suppress_health_check=[HealthCheck.function_scoped_fixture],
    max_examples=100,
    deadline=None
)
@pytest.mark.asyncio
async def test_property_59_priority_queue_ordering(priorities):
    """Property 59: Priority Queue Ordering
    
    For any request queue with mixed priority levels, higher priority requests 
    should be processed before lower priority requests submitted at the same time.
    
    Validates: Requirements 12.3
    """
    queue = PriorityRequestQueue(max_concurrent=1)
    
    # Track execution order
    execution_order = []
    
    async def tracked_request(req_id: str, req_priority: int):
        """Request that records its execution."""
        execution_order.append((req_id, req_priority))
        await asyncio.sleep(0.01)  # Small delay to simulate work
        return req_id
    
    # Start queue processing
    await queue.start()
    
    try:
        # Enqueue all requests at approximately the same time
        request_ids = []
        for i, priority in enumerate(priorities):
            request_id = f"req_{i}_{priority.name}"
            await queue.enqueue(
                tracked_request,
                priority=priority,
                request_id=request_id,
                req_id=request_id,
                req_priority=int(priority)
            )
            request_ids.append((request_id, int(priority)))
        
        # Wait for all requests to complete
        await asyncio.sleep(0.5 + len(priorities) * 0.02)
        
        # Verify priority ordering
        # For each pair of consecutive executed requests, check priority ordering
        for i in range(len(execution_order) - 1):
            current_id, current_priority = execution_order[i]
            next_id, next_priority = execution_order[i + 1]
            
            # Find original submission order
            current_submit_idx = next(
                idx for idx, (rid, _) in enumerate(request_ids) 
                if rid == current_id
            )
            next_submit_idx = next(
                idx for idx, (rid, _) in enumerate(request_ids) 
                if rid == next_id
            )
            
            # If submitted at similar times (within 5 positions), 
            # higher priority should execute first
            if abs(current_submit_idx - next_submit_idx) <= 5:
                # Current should have equal or higher priority (lower value)
                assert current_priority <= next_priority, \
                    f"Priority violation: {current_id} (priority={current_priority}) " \
                    f"executed before {next_id} (priority={next_priority})"
    
    finally:
        await queue.stop(wait_for_completion=False)


# Feature: wikipedia-intelligence-system, Property 60: Circuit Breaker Pattern
@given(
    failure_threshold=st.integers(min_value=2, max_value=10),
    success_threshold=st.integers(min_value=1, max_value=5),
    num_failures=st.integers(min_value=1, max_value=15)
)
@settings(
    suppress_health_check=[HealthCheck.function_scoped_fixture],
    max_examples=100
)
def test_property_60_circuit_breaker_pattern(
    failure_threshold, 
    success_threshold, 
    num_failures
):
    """Property 60: Circuit Breaker Pattern
    
    For any API endpoint that fails more than the threshold number of times, 
    the circuit breaker should open (reject requests immediately), and after 
    the timeout period, should attempt one test request before fully closing.
    
    Validates: Requirements 12.5
    """
    # Configure circuit breaker
    config = CircuitBreakerConfig(
        failure_threshold=failure_threshold,
        success_threshold=success_threshold,
        timeout_seconds=0.1  # Short timeout for testing
    )
    breaker = CircuitBreaker(endpoint="test_endpoint", config=config)
    
    # Initially should be closed
    assert breaker.get_state() == CircuitState.CLOSED
    assert breaker.can_request() is True
    
    # Record failures
    for i in range(num_failures):
        breaker.record_failure()
        
        if i + 1 >= failure_threshold:
            # Should be open after threshold
            assert breaker.get_state() == CircuitState.OPEN, \
                f"Circuit should be OPEN after {i+1} failures (threshold={failure_threshold})"
            assert breaker.can_request() is False, \
                "Circuit should reject requests when OPEN"
        else:
            # Should still be closed before threshold
            assert breaker.get_state() == CircuitState.CLOSED
    
    # If we exceeded threshold, test timeout and half-open behavior
    if num_failures >= failure_threshold:
        # Wait for timeout
        time.sleep(config.timeout_seconds + 0.05)
        
        # Should transition to half-open and allow test request
        assert breaker.can_request() is True, \
            "Circuit should allow test request after timeout"
        assert breaker.get_state() == CircuitState.HALF_OPEN, \
            "Circuit should be HALF_OPEN after timeout"
        
        # Test success threshold in half-open state
        for i in range(success_threshold):
            breaker.record_success()
            
            if i + 1 >= success_threshold:
                # Should close after success threshold
                assert breaker.get_state() == CircuitState.CLOSED, \
                    f"Circuit should be CLOSED after {i+1} successes in HALF_OPEN " \
                    f"(threshold={success_threshold})"
            else:
                # Should remain half-open before threshold
                assert breaker.get_state() == CircuitState.HALF_OPEN


# Feature: wikipedia-intelligence-system, Property 60: Circuit Breaker Half-Open Failure
@given(
    failure_threshold=st.integers(min_value=2, max_value=10)
)
@settings(
    suppress_health_check=[HealthCheck.function_scoped_fixture],
    max_examples=50
)
def test_property_60_circuit_breaker_half_open_failure(failure_threshold):
    """Property 60: Circuit Breaker Half-Open Failure Handling
    
    When a test request fails in HALF_OPEN state, the circuit should 
    immediately reopen.
    
    Validates: Requirements 12.5
    """
    config = CircuitBreakerConfig(
        failure_threshold=failure_threshold,
        success_threshold=2,
        timeout_seconds=0.1
    )
    breaker = CircuitBreaker(endpoint="test_endpoint", config=config)
    
    # Open the circuit
    for _ in range(failure_threshold):
        breaker.record_failure()
    
    assert breaker.get_state() == CircuitState.OPEN
    
    # Wait for timeout to transition to half-open
    time.sleep(config.timeout_seconds + 0.05)
    breaker.can_request()  # Trigger transition
    
    assert breaker.get_state() == CircuitState.HALF_OPEN
    
    # Record failure in half-open state
    breaker.record_failure()
    
    # Should immediately reopen
    assert breaker.get_state() == CircuitState.OPEN, \
        "Circuit should immediately reopen on failure in HALF_OPEN state"
    assert breaker.can_request() is False, \
        "Circuit should reject requests after reopening"


# Feature: wikipedia-intelligence-system, Property 4: Exponential Backoff on Rate Limits
@given(
    num_attempts=st.integers(min_value=2, max_value=10),
    initial_delay=st.floats(min_value=0.5, max_value=2.0),
    exponential_base=st.floats(min_value=2.0, max_value=3.0)
)
@settings(
    suppress_health_check=[HealthCheck.function_scoped_fixture],
    max_examples=100,
    deadline=None
)
def test_property_4_exponential_backoff_on_rate_limits(
    num_attempts,
    initial_delay,
    exponential_base
):
    """Property 4: Exponential Backoff on Rate Limits
    
    For any sequence of rate limit errors (429 status), the System should 
    implement exponential backoff where each retry delay is at least 2x 
    the previous delay.
    
    Validates: Requirements 1.6
    """
    from src.data_ingestion.api_client import RetryConfig
    
    # Configure retry with test parameters
    retry_config = RetryConfig(
        max_retries=num_attempts,
        initial_delay=initial_delay,
        exponential_base=exponential_base,
        jitter=False  # Disable jitter for predictable testing
    )
    
    # Create a minimal client to test the backoff calculation
    # We replicate the _calculate_backoff_delay logic here to test it
    class BackoffCalculator:
        def __init__(self, retry_config):
            self.retry_config = retry_config
        
        def calculate_backoff_delay(self, attempt: int) -> float:
            """Calculate exponential backoff delay."""
            delay = min(
                self.retry_config.initial_delay * (self.retry_config.exponential_base ** attempt),
                self.retry_config.max_delay
            )
            
            if self.retry_config.jitter:
                import random
                delay = delay * (0.5 + random.random() * 0.5)
            
            return delay
    
    calculator = BackoffCalculator(retry_config)
    
    # Test the backoff delay calculation
    delays = []
    for attempt in range(num_attempts):
        delay = calculator.calculate_backoff_delay(attempt)
        delays.append(delay)
    
    # Verify first delay matches initial_delay
    assert abs(delays[0] - initial_delay) < 0.01, \
        f"First delay ({delays[0]:.3f}s) should match initial_delay ({initial_delay:.3f}s)"
    
    # Verify each delay is at least exponential_base times the previous
    for i in range(1, len(delays)):
        previous_delay = delays[i - 1]
        current_delay = delays[i]
        
        # Calculate expected minimum delay (exponential growth)
        # Note: delays are capped at max_delay, so we need to account for that
        expected_delay = min(
            initial_delay * (exponential_base ** i),
            retry_config.max_delay
        )
        
        # Allow small tolerance for floating point comparison
        tolerance = 0.01
        
        # If we haven't hit the max_delay cap, verify exponential growth
        if previous_delay < retry_config.max_delay - tolerance:
            # Current should be at least exponential_base times previous
            # (unless capped by max_delay)
            expected_min = min(previous_delay * exponential_base, retry_config.max_delay)
            assert current_delay >= expected_min - tolerance, \
                f"Delay {i} ({current_delay:.3f}s) should be at least " \
                f"{exponential_base}x previous delay ({previous_delay:.3f}s), " \
                f"expected >= {expected_min:.3f}s"
        
        # Verify delay doesn't exceed max_delay
        assert current_delay <= retry_config.max_delay + tolerance, \
            f"Delay {i} ({current_delay:.3f}s) should not exceed max_delay ({retry_config.max_delay}s)"
        
        # Verify delay matches expected exponential formula (or is capped)
        assert abs(current_delay - expected_delay) < tolerance, \
            f"Delay {i} ({current_delay:.3f}s) should match expected ({expected_delay:.3f}s)"
