"""
Rate limiting implementation using token bucket algorithm.

This module provides rate limiting functionality to ensure API compliance
with Wikimedia rate limits (200 requests per second maximum).
"""

import asyncio
import time
from typing import Optional
from dataclasses import dataclass
import logging

logger = logging.getLogger(__name__)


@dataclass
class RateLimiterConfig:
    """Configuration for rate limiter."""
    max_requests_per_second: float = 200.0
    throttle_threshold: float = 0.9  # Throttle at 90% of limit
    burst_size: Optional[int] = None  # Max burst, defaults to max_requests_per_second


class RateLimiter:
    """
    Token bucket rate limiter for API requests.
    
    Implements automatic throttling when approaching rate limits.
    Thread-safe and async-compatible.
    
    Requirements: 12.1, 12.2
    """
    
    def __init__(self, config: Optional[RateLimiterConfig] = None):
        """
        Initialize rate limiter with token bucket algorithm.
        
        Args:
            config: Rate limiter configuration. Defaults to Wikimedia limits.
        """
        self.config = config or RateLimiterConfig()
        
        # Token bucket parameters
        self.max_tokens = self.config.burst_size or int(self.config.max_requests_per_second)
        self.tokens = float(self.max_tokens)
        self.refill_rate = self.config.max_requests_per_second  # tokens per second
        self.last_refill = time.monotonic()
        
        # Throttling threshold
        self.throttle_threshold = self.config.throttle_threshold
        
        # Lock for thread safety
        self._lock = asyncio.Lock()
        
        logger.info(
            f"RateLimiter initialized: {self.config.max_requests_per_second} req/s, "
            f"burst={self.max_tokens}, throttle_threshold={self.throttle_threshold}"
        )
    
    def _refill_tokens(self) -> None:
        """Refill tokens based on elapsed time since last refill."""
        now = time.monotonic()
        elapsed = now - self.last_refill
        
        # Add tokens based on elapsed time
        tokens_to_add = elapsed * self.refill_rate
        self.tokens = min(self.max_tokens, self.tokens + tokens_to_add)
        self.last_refill = now
    
    async def acquire(self, tokens: int = 1) -> float:
        """
        Acquire tokens for making requests.
        
        Blocks until tokens are available. Implements automatic throttling
        when token count approaches the limit.
        
        Args:
            tokens: Number of tokens to acquire (default 1 for single request)
            
        Returns:
            Wait time in seconds (0 if no wait was needed)
            
        Raises:
            ValueError: If tokens requested exceeds max_tokens
        """
        if tokens > self.max_tokens:
            raise ValueError(
                f"Requested {tokens} tokens exceeds max burst size {self.max_tokens}"
            )
        
        async with self._lock:
            wait_time = 0.0
            
            while True:
                self._refill_tokens()
                
                # Check if we should throttle (approaching limit)
                current_capacity = self.tokens / self.max_tokens
                if current_capacity < (1 - self.throttle_threshold):
                    # We're below throttle threshold, apply throttling delay
                    throttle_delay = 0.1  # 100ms throttle delay
                    logger.debug(
                        f"Throttling: capacity={current_capacity:.2%}, "
                        f"delaying {throttle_delay}s"
                    )
                    await asyncio.sleep(throttle_delay)
                    wait_time += throttle_delay
                    continue
                
                # Check if we have enough tokens
                if self.tokens >= tokens:
                    self.tokens -= tokens
                    logger.debug(
                        f"Acquired {tokens} token(s), {self.tokens:.2f} remaining"
                    )
                    return wait_time
                
                # Calculate wait time for tokens to refill
                tokens_needed = tokens - self.tokens
                sleep_time = tokens_needed / self.refill_rate
                
                logger.debug(
                    f"Waiting {sleep_time:.3f}s for {tokens_needed:.2f} tokens"
                )
                await asyncio.sleep(sleep_time)
                wait_time += sleep_time
    
    def try_acquire(self, tokens: int = 1) -> bool:
        """
        Try to acquire tokens without blocking.
        
        Args:
            tokens: Number of tokens to acquire
            
        Returns:
            True if tokens were acquired, False otherwise
        """
        if tokens > self.max_tokens:
            return False
        
        self._refill_tokens()
        
        if self.tokens >= tokens:
            self.tokens -= tokens
            logger.debug(f"Acquired {tokens} token(s), {self.tokens:.2f} remaining")
            return True
        
        logger.debug(f"Failed to acquire {tokens} token(s), only {self.tokens:.2f} available")
        return False
    
    def get_available_tokens(self) -> float:
        """
        Get current number of available tokens.
        
        Returns:
            Number of tokens currently available
        """
        self._refill_tokens()
        return self.tokens
    
    def get_capacity_percentage(self) -> float:
        """
        Get current capacity as percentage of max.
        
        Returns:
            Capacity percentage (0.0 to 1.0)
        """
        self._refill_tokens()
        return self.tokens / self.max_tokens
    
    async def reset(self) -> None:
        """Reset the rate limiter to full capacity."""
        async with self._lock:
            self.tokens = float(self.max_tokens)
            self.last_refill = time.monotonic()
            logger.info("RateLimiter reset to full capacity")
