"""
Priority request queue for API requests.

This module provides a priority queue implementation that ensures higher priority
requests are processed before lower priority requests, supporting efficient
request management and prioritization.
"""

import asyncio
import heapq
import time
from typing import Optional, Any, Dict, Callable, Awaitable
from dataclasses import dataclass, field
from enum import IntEnum
import logging

logger = logging.getLogger(__name__)


class RequestPriority(IntEnum):
    """
    Request priority levels.
    
    Lower numeric values = higher priority (processed first).
    """
    CRITICAL = 1  # Critical requests (alerts, real-time monitoring)
    HIGH = 2  # High priority (user-facing dashboard queries)
    NORMAL = 3  # Normal priority (scheduled data collection)
    LOW = 4  # Low priority (background analytics, batch jobs)
    BULK = 5  # Bulk operations (historical data backfill)


@dataclass(order=True)
class PriorityRequest:
    """
    Request wrapper with priority and ordering.
    
    Uses priority and timestamp for ordering in the heap queue.
    Lower priority values are processed first.
    For same priority, earlier timestamps are processed first (FIFO).
    """
    priority: int = field(compare=True)
    timestamp: float = field(compare=True)
    request_id: str = field(compare=False)
    request_func: Callable[..., Awaitable[Any]] = field(compare=False)
    args: tuple = field(default_factory=tuple, compare=False)
    kwargs: Dict[str, Any] = field(default_factory=dict, compare=False)
    
    def __post_init__(self):
        """Validate priority value."""
        if not isinstance(self.priority, int) or self.priority < 1:
            raise ValueError(f"Invalid priority: {self.priority}. Must be positive integer.")


@dataclass
class QueueStats:
    """Statistics for request queue."""
    total_enqueued: int = 0
    total_processed: int = 0
    total_failed: int = 0
    current_size: int = 0
    by_priority: Dict[int, int] = field(default_factory=dict)
    avg_wait_time: float = 0.0
    avg_processing_time: float = 0.0


class PriorityRequestQueue:
    """
    Priority queue for API requests with async processing.
    
    Features:
    - Priority-based ordering (higher priority processed first)
    - FIFO ordering within same priority level
    - Async request processing with configurable concurrency
    - Queue statistics and monitoring
    - Graceful shutdown with pending request handling
    
    Requirements: 12.3
    """
    
    def __init__(
        self,
        max_concurrent: int = 10,
        max_queue_size: Optional[int] = None
    ):
        """
        Initialize priority request queue.
        
        Args:
            max_concurrent: Maximum number of concurrent request processors
            max_queue_size: Maximum queue size (None for unlimited)
        """
        self.max_concurrent = max_concurrent
        self.max_queue_size = max_queue_size
        
        # Priority queue (min-heap)
        self._queue: list[PriorityRequest] = []
        self._lock = asyncio.Lock()
        self._not_empty = asyncio.Condition(self._lock)
        self._not_full = asyncio.Condition(self._lock) if max_queue_size else None
        
        # Processing state
        self._workers: list[asyncio.Task] = []
        self._running = False
        self._shutdown_event = asyncio.Event()
        
        # Statistics
        self._stats = QueueStats()
        self._wait_times: list[float] = []
        self._processing_times: list[float] = []
        
        logger.info(
            f"PriorityRequestQueue initialized: max_concurrent={max_concurrent}, "
            f"max_queue_size={max_queue_size or 'unlimited'}"
        )
    
    async def enqueue(
        self,
        request_func: Callable[..., Awaitable[Any]],
        priority: RequestPriority = RequestPriority.NORMAL,
        request_id: Optional[str] = None,
        *args,
        **kwargs
    ) -> str:
        """
        Add request to priority queue.
        
        Args:
            request_func: Async function to execute
            priority: Request priority level
            request_id: Optional request identifier (auto-generated if None)
            *args: Positional arguments for request_func
            **kwargs: Keyword arguments for request_func
            
        Returns:
            Request ID
            
        Raises:
            asyncio.QueueFull: If queue is at max capacity
        """
        # Generate request ID if not provided
        if request_id is None:
            request_id = f"req_{int(time.time() * 1000000)}"
        
        # Create priority request
        priority_request = PriorityRequest(
            priority=int(priority),
            timestamp=time.monotonic(),
            request_id=request_id,
            request_func=request_func,
            args=args,
            kwargs=kwargs
        )
        
        # Add to queue with size check
        async with self._lock:
            # Check queue size limit
            if self.max_queue_size and len(self._queue) >= self.max_queue_size:
                if self._not_full:
                    # Wait for space if queue is full
                    logger.warning(
                        f"Queue full ({len(self._queue)}/{self.max_queue_size}), "
                        f"waiting for space..."
                    )
                    await self._not_full.wait()
                else:
                    raise asyncio.QueueFull(
                        f"Queue at max capacity: {self.max_queue_size}"
                    )
            
            # Add to heap
            heapq.heappush(self._queue, priority_request)
            
            # Update statistics
            self._stats.total_enqueued += 1
            self._stats.current_size = len(self._queue)
            priority_count = self._stats.by_priority.get(int(priority), 0)
            self._stats.by_priority[int(priority)] = priority_count + 1
            
            logger.debug(
                f"Enqueued request {request_id} with priority {priority.name} "
                f"(queue size: {len(self._queue)})"
            )
            
            # Notify waiting workers
            self._not_empty.notify()
        
        return request_id
    
    async def _dequeue(self) -> Optional[PriorityRequest]:
        """
        Remove and return highest priority request from queue.
        
        Returns:
            PriorityRequest or None if queue is empty
        """
        async with self._lock:
            if not self._queue:
                return None
            
            # Pop from heap (lowest priority value = highest priority)
            priority_request = heapq.heappop(self._queue)
            
            # Update statistics
            self._stats.current_size = len(self._queue)
            priority_count = self._stats.by_priority.get(priority_request.priority, 0)
            if priority_count > 0:
                self._stats.by_priority[priority_request.priority] = priority_count - 1
            
            # Notify if space available
            if self._not_full:
                self._not_full.notify()
            
            return priority_request
    
    async def _worker(self, worker_id: int) -> None:
        """
        Worker coroutine that processes requests from queue.
        
        Args:
            worker_id: Worker identifier for logging
        """
        logger.info(f"Worker {worker_id} started")
        
        while self._running or self._queue:
            try:
                # Wait for request
                async with self._not_empty:
                    while not self._queue and self._running:
                        await self._not_empty.wait()
                    
                    # Check if shutting down with empty queue
                    if not self._queue and not self._running:
                        break
                
                # Dequeue request
                priority_request = await self._dequeue()
                if priority_request is None:
                    continue
                
                # Calculate wait time
                wait_time = time.monotonic() - priority_request.timestamp
                self._wait_times.append(wait_time)
                
                logger.debug(
                    f"Worker {worker_id} processing request {priority_request.request_id} "
                    f"(priority: {priority_request.priority}, wait: {wait_time:.3f}s)"
                )
                
                # Execute request
                processing_start = time.monotonic()
                try:
                    result = await priority_request.request_func(
                        *priority_request.args,
                        **priority_request.kwargs
                    )
                    
                    processing_time = time.monotonic() - processing_start
                    self._processing_times.append(processing_time)
                    
                    # Update statistics
                    self._stats.total_processed += 1
                    
                    logger.debug(
                        f"Worker {worker_id} completed request {priority_request.request_id} "
                        f"(processing: {processing_time:.3f}s)"
                    )
                    
                except Exception as e:
                    processing_time = time.monotonic() - processing_start
                    self._stats.total_failed += 1
                    
                    logger.error(
                        f"Worker {worker_id} failed processing request "
                        f"{priority_request.request_id}: {type(e).__name__}: {e}",
                        exc_info=True,
                        extra={
                            "request_id": priority_request.request_id,
                            "priority": priority_request.priority,
                            "error_type": type(e).__name__,
                            "error_message": str(e),
                            "processing_time": processing_time
                        }
                    )
            
            except asyncio.CancelledError:
                logger.info(f"Worker {worker_id} cancelled")
                break
            except Exception as e:
                logger.error(
                    f"Worker {worker_id} encountered unexpected error: {e}",
                    exc_info=True
                )
        
        logger.info(f"Worker {worker_id} stopped")
    
    async def start(self) -> None:
        """
        Start queue processing workers.
        
        Creates worker tasks that process requests from the queue.
        """
        if self._running:
            logger.warning("Queue already running")
            return
        
        self._running = True
        self._shutdown_event.clear()
        
        # Create worker tasks
        self._workers = [
            asyncio.create_task(self._worker(i))
            for i in range(self.max_concurrent)
        ]
        
        logger.info(f"Started {self.max_concurrent} queue workers")
    
    async def stop(self, wait_for_completion: bool = True) -> None:
        """
        Stop queue processing.
        
        Args:
            wait_for_completion: If True, wait for pending requests to complete.
                                If False, cancel workers immediately.
        """
        if not self._running:
            logger.warning("Queue not running")
            return
        
        logger.info(
            f"Stopping queue (wait_for_completion={wait_for_completion}, "
            f"pending={len(self._queue)})"
        )
        
        self._running = False
        
        if wait_for_completion:
            # Notify all workers to check shutdown condition
            async with self._not_empty:
                self._not_empty.notify_all()
            
            # Wait for workers to finish pending requests
            await asyncio.gather(*self._workers, return_exceptions=True)
        else:
            # Cancel workers immediately
            for worker in self._workers:
                worker.cancel()
            
            # Wait for cancellation
            await asyncio.gather(*self._workers, return_exceptions=True)
        
        self._workers.clear()
        self._shutdown_event.set()
        
        logger.info("Queue stopped")
    
    def size(self) -> int:
        """
        Get current queue size.
        
        Returns:
            Number of pending requests
        """
        return len(self._queue)
    
    def is_empty(self) -> bool:
        """
        Check if queue is empty.
        
        Returns:
            True if queue has no pending requests
        """
        return len(self._queue) == 0
    
    def is_running(self) -> bool:
        """
        Check if queue is running.
        
        Returns:
            True if workers are processing requests
        """
        return self._running
    
    def get_stats(self) -> QueueStats:
        """
        Get queue statistics.
        
        Returns:
            QueueStats with current statistics
        """
        # Calculate averages
        if self._wait_times:
            self._stats.avg_wait_time = sum(self._wait_times) / len(self._wait_times)
        if self._processing_times:
            self._stats.avg_processing_time = sum(self._processing_times) / len(self._processing_times)
        
        return self._stats
    
    def reset_stats(self) -> None:
        """Reset queue statistics."""
        self._stats = QueueStats()
        self._wait_times.clear()
        self._processing_times.clear()
        logger.info("Queue statistics reset")
    
    async def __aenter__(self):
        """Async context manager entry."""
        await self.start()
        return self
    
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Async context manager exit."""
        await self.stop(wait_for_completion=True)
