"""
Unit tests for priority request queue.

Tests the priority queue implementation including priority ordering,
FIFO within same priority, concurrent processing, and statistics.
"""

import asyncio
import pytest
from unittest.mock import AsyncMock, Mock
import time

from src.data_ingestion.request_queue import (
    PriorityRequestQueue,
    RequestPriority,
    PriorityRequest,
    QueueStats
)


class TestPriorityRequest:
    """Test PriorityRequest dataclass."""
    
    def test_priority_request_creation(self):
        """Test creating a priority request."""
        async def dummy_func():
            return "result"
        
        req = PriorityRequest(
            priority=RequestPriority.HIGH,
            timestamp=time.monotonic(),
            request_id="test-1",
            request_func=dummy_func,
            args=(),
            kwargs={}
        )
        
        assert req.priority == RequestPriority.HIGH
        assert req.request_id == "test-1"
        assert req.request_func == dummy_func
    
    def test_priority_request_ordering(self):
        """Test that priority requests are ordered correctly."""
        async def dummy_func():
            return "result"
        
        # Create requests with different priorities
        high_req = PriorityRequest(
            priority=RequestPriority.HIGH,
            timestamp=time.monotonic(),
            request_id="high",
            request_func=dummy_func
        )
        
        low_req = PriorityRequest(
            priority=RequestPriority.LOW,
            timestamp=time.monotonic(),
            request_id="low",
            request_func=dummy_func
        )
        
        # High priority (lower value) should be less than low priority
        assert high_req < low_req
    
    def test_priority_request_fifo_within_priority(self):
        """Test FIFO ordering within same priority level."""
        async def dummy_func():
            return "result"
        
        # Create requests with same priority but different timestamps
        req1 = PriorityRequest(
            priority=RequestPriority.NORMAL,
            timestamp=1.0,
            request_id="first",
            request_func=dummy_func
        )
        
        req2 = PriorityRequest(
            priority=RequestPriority.NORMAL,
            timestamp=2.0,
            request_id="second",
            request_func=dummy_func
        )
        
        # Earlier timestamp should be processed first
        assert req1 < req2
    
    def test_invalid_priority_raises_error(self):
        """Test that invalid priority values raise ValueError."""
        async def dummy_func():
            return "result"
        
        with pytest.raises(ValueError, match="Invalid priority"):
            PriorityRequest(
                priority=-1,
                timestamp=time.monotonic(),
                request_id="invalid",
                request_func=dummy_func
            )


class TestPriorityRequestQueue:
    """Test PriorityRequestQueue class."""
    
    @pytest.mark.asyncio
    async def test_queue_initialization(self):
        """Test queue initialization with default parameters."""
        queue = PriorityRequestQueue(max_concurrent=5)
        
        assert queue.max_concurrent == 5
        assert queue.max_queue_size is None
        assert queue.size() == 0
        assert queue.is_empty()
        assert not queue.is_running()
    
    @pytest.mark.asyncio
    async def test_enqueue_request(self):
        """Test enqueueing a request."""
        queue = PriorityRequestQueue()
        
        async def dummy_func():
            return "result"
        
        request_id = await queue.enqueue(
            dummy_func,
            priority=RequestPriority.HIGH,
            request_id="test-1"
        )
        
        assert request_id == "test-1"
        assert queue.size() == 1
        assert not queue.is_empty()
    
    @pytest.mark.asyncio
    async def test_enqueue_auto_generates_id(self):
        """Test that request ID is auto-generated if not provided."""
        queue = PriorityRequestQueue()
        
        async def dummy_func():
            return "result"
        
        request_id = await queue.enqueue(dummy_func, priority=RequestPriority.NORMAL)
        
        assert request_id.startswith("req_")
        assert queue.size() == 1
    
    @pytest.mark.asyncio
    async def test_priority_ordering(self):
        """Test that higher priority requests are processed first."""
        queue = PriorityRequestQueue(max_concurrent=1)
        results = []
        
        async def record_result(value):
            results.append(value)
            await asyncio.sleep(0.01)
        
        # Enqueue requests in reverse priority order
        await queue.enqueue(record_result, RequestPriority.LOW, args=("low",))
        await queue.enqueue(record_result, RequestPriority.HIGH, args=("high",))
        await queue.enqueue(record_result, RequestPriority.NORMAL, args=("normal",))
        await queue.enqueue(record_result, RequestPriority.CRITICAL, args=("critical",))
        
        # Start processing
        await queue.start()
        
        # Wait for all requests to complete
        while queue.size() > 0 or queue._stats.total_processed < 4:
            await asyncio.sleep(0.01)
        
        await queue.stop()
        
        # Verify processing order: CRITICAL, HIGH, NORMAL, LOW
        assert results == ["critical", "high", "normal", "low"]
    
    @pytest.mark.asyncio
    async def test_fifo_within_priority(self):
        """Test FIFO ordering within same priority level."""
        queue = PriorityRequestQueue(max_concurrent=1)
        results = []
        
        async def record_result(value):
            results.append(value)
            await asyncio.sleep(0.01)
        
        # Enqueue multiple requests with same priority
        await queue.enqueue(record_result, RequestPriority.NORMAL, args=("first",))
        await queue.enqueue(record_result, RequestPriority.NORMAL, args=("second",))
        await queue.enqueue(record_result, RequestPriority.NORMAL, args=("third",))
        
        await queue.start()
        
        # Wait for completion
        while queue.size() > 0 or queue._stats.total_processed < 3:
            await asyncio.sleep(0.01)
        
        await queue.stop()
        
        # Verify FIFO order
        assert results == ["first", "second", "third"]
    
    @pytest.mark.asyncio
    async def test_concurrent_processing(self):
        """Test concurrent request processing."""
        queue = PriorityRequestQueue(max_concurrent=3)
        processing_times = []
        
        async def slow_task(task_id):
            start = time.monotonic()
            await asyncio.sleep(0.1)
            end = time.monotonic()
            processing_times.append((task_id, start, end))
        
        # Enqueue 6 tasks
        for i in range(6):
            await queue.enqueue(slow_task, RequestPriority.NORMAL, args=(i,))
        
        await queue.start()
        
        # Wait for completion
        while queue.size() > 0 or queue._stats.total_processed < 6:
            await asyncio.sleep(0.01)
        
        await queue.stop()
        
        # Verify 6 tasks completed
        assert len(processing_times) == 6
        
        # Verify concurrent execution (first 3 should overlap)
        first_batch = processing_times[:3]
        # Check that at least 2 of the first 3 tasks overlapped
        overlaps = 0
        for i in range(len(first_batch)):
            for j in range(i + 1, len(first_batch)):
                start_i, end_i = first_batch[i][1], first_batch[i][2]
                start_j, end_j = first_batch[j][1], first_batch[j][2]
                # Check if time ranges overlap
                if start_i < end_j and start_j < end_i:
                    overlaps += 1
        
        assert overlaps >= 1, "Expected concurrent execution"
    
    @pytest.mark.asyncio
    async def test_error_handling(self):
        """Test that errors in request processing are handled gracefully."""
        queue = PriorityRequestQueue(max_concurrent=1)
        
        async def failing_task():
            raise ValueError("Test error")
        
        async def successful_task():
            return "success"
        
        await queue.enqueue(failing_task, RequestPriority.NORMAL)
        await queue.enqueue(successful_task, RequestPriority.NORMAL)
        
        await queue.start()
        
        # Wait for completion
        while queue.size() > 0 or queue._stats.total_processed < 1:
            await asyncio.sleep(0.01)
        
        await queue.stop()
        
        # Verify stats
        stats = queue.get_stats()
        assert stats.total_failed == 1
        assert stats.total_processed == 1
    
    @pytest.mark.asyncio
    async def test_queue_statistics(self):
        """Test queue statistics tracking."""
        queue = PriorityRequestQueue(max_concurrent=2)
        
        async def dummy_task():
            await asyncio.sleep(0.01)
        
        # Enqueue requests with different priorities
        await queue.enqueue(dummy_task, RequestPriority.HIGH)
        await queue.enqueue(dummy_task, RequestPriority.NORMAL)
        await queue.enqueue(dummy_task, RequestPriority.LOW)
        
        await queue.start()
        
        # Wait for completion
        while queue.size() > 0 or queue._stats.total_processed < 3:
            await asyncio.sleep(0.01)
        
        await queue.stop()
        
        stats = queue.get_stats()
        assert stats.total_enqueued == 3
        assert stats.total_processed == 3
        assert stats.total_failed == 0
        assert stats.avg_wait_time >= 0
        assert stats.avg_processing_time >= 0
    
    @pytest.mark.asyncio
    async def test_max_queue_size(self):
        """Test queue size limit enforcement."""
        queue = PriorityRequestQueue(max_concurrent=1, max_queue_size=2)
        
        async def dummy_task():
            await asyncio.sleep(0.1)
        
        # Fill queue to max
        await queue.enqueue(dummy_task, RequestPriority.NORMAL)
        await queue.enqueue(dummy_task, RequestPriority.NORMAL)
        
        # Next enqueue should raise QueueFull
        with pytest.raises(asyncio.QueueFull):
            await queue.enqueue(dummy_task, RequestPriority.NORMAL)
    
    @pytest.mark.asyncio
    async def test_graceful_shutdown(self):
        """Test graceful shutdown waits for pending requests."""
        queue = PriorityRequestQueue(max_concurrent=1)
        completed = []
        
        async def slow_task(task_id):
            await asyncio.sleep(0.05)
            completed.append(task_id)
        
        await queue.enqueue(slow_task, RequestPriority.NORMAL, args=(1,))
        await queue.enqueue(slow_task, RequestPriority.NORMAL, args=(2,))
        
        await queue.start()
        
        # Wait a bit then stop gracefully
        await asyncio.sleep(0.02)
        await queue.stop(wait_for_completion=True)
        
        # Both tasks should complete
        assert len(completed) == 2
    
    @pytest.mark.asyncio
    async def test_immediate_shutdown(self):
        """Test immediate shutdown cancels pending requests."""
        queue = PriorityRequestQueue(max_concurrent=1)
        completed = []
        
        async def slow_task(task_id):
            await asyncio.sleep(0.1)
            completed.append(task_id)
        
        await queue.enqueue(slow_task, RequestPriority.NORMAL, args=(1,))
        await queue.enqueue(slow_task, RequestPriority.NORMAL, args=(2,))
        await queue.enqueue(slow_task, RequestPriority.NORMAL, args=(3,))
        
        await queue.start()
        
        # Wait a bit then stop immediately
        await asyncio.sleep(0.05)
        await queue.stop(wait_for_completion=False)
        
        # Not all tasks should complete
        assert len(completed) < 3
    
    @pytest.mark.asyncio
    async def test_context_manager(self):
        """Test async context manager support."""
        completed = []
        
        async def dummy_task(task_id):
            await asyncio.sleep(0.01)
            completed.append(task_id)
        
        async with PriorityRequestQueue(max_concurrent=2) as queue:
            await queue.enqueue(dummy_task, RequestPriority.NORMAL, args=(1,))
            await queue.enqueue(dummy_task, RequestPriority.NORMAL, args=(2,))
            
            # Wait for completion
            while queue.size() > 0 or queue._stats.total_processed < 2:
                await asyncio.sleep(0.01)
        
        # Queue should be stopped after context exit
        assert not queue.is_running()
        assert len(completed) == 2
    
    @pytest.mark.asyncio
    async def test_reset_statistics(self):
        """Test resetting queue statistics."""
        queue = PriorityRequestQueue()
        
        async def dummy_task():
            await asyncio.sleep(0.01)
        
        await queue.enqueue(dummy_task, RequestPriority.NORMAL)
        
        await queue.start()
        while queue.size() > 0 or queue._stats.total_processed < 1:
            await asyncio.sleep(0.01)
        await queue.stop()
        
        stats_before = queue.get_stats()
        assert stats_before.total_enqueued > 0
        
        queue.reset_stats()
        
        stats_after = queue.get_stats()
        assert stats_after.total_enqueued == 0
        assert stats_after.total_processed == 0
