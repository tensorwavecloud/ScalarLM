"""
KV Cache Manager for dynamic batch size calculation and token management.
Implements the free_kv_cache_tokens algorithm with race condition protection.
"""

import asyncio
import logging
from typing import Optional, Dict, Any
from dataclasses import dataclass, field
from datetime import datetime

logger = logging.getLogger(__name__)


@dataclass
class KVCacheStats:
    """Statistics for KV cache usage"""
    total_tokens: int
    free_tokens: int
    reserved_tokens: int
    total_requests_processed: int = 0
    total_tokens_processed: int = 0
    last_update: datetime = field(default_factory=datetime.now)
    
    @property
    def utilization_percent(self) -> float:
        """Calculate cache utilization percentage"""
        if self.total_tokens == 0:
            return 0.0
        return ((self.total_tokens - self.free_tokens) / self.total_tokens) * 100


class KVCacheManager:
    """
    Manages KV cache tokens for dynamic batch sizing and memory management.
    
    This implements the algorithm:
    1. Deduct max_sequence_length + max_output_tokens when getting work
    2. Add back unused tokens after tokenization
    3. Add back all tokens when request completes
    4. Check for newly freed tokens from vLLM engine
    """
    
    def __init__(
        self, 
        total_tokens: int,
        max_tokens_per_request: int,
        max_batch_size: int = 1024
    ):
        """
        Initialize KV Cache Manager.
        
        Args:
            total_tokens: Total KV cache tokens available
            max_tokens_per_request: Maximum tokens per single request (model max_length)
            max_batch_size: Maximum batch size allowed
        """
        self.total_tokens = total_tokens
        self.free_tokens = total_tokens
        self.max_tokens_per_request = max_tokens_per_request
        self.max_batch_size = max_batch_size
        
        # Track reserved tokens per request_id
        self.reserved_tokens: Dict[str, int] = {}
        
        # Statistics
        self.stats = KVCacheStats(
            total_tokens=total_tokens,
            free_tokens=total_tokens,
            reserved_tokens=0
        )
        
        # Lock for thread safety
        self._lock = asyncio.Lock()
        
        logger.info(
            f"KV Cache Manager initialized: total_tokens={total_tokens}, "
            f"max_tokens_per_request={max_tokens_per_request}, "
            f"max_batch_size={max_batch_size}"
        )
        
        # Validate configuration
        assert total_tokens >= max_tokens_per_request, (
            f"Total tokens ({total_tokens}) must be >= max tokens per request ({max_tokens_per_request})"
        )
    
    async def calculate_batch_size(self, requested_batch_size: Optional[int] = None) -> int:
        """
        Calculate dynamic batch size based on available KV cache tokens.
        
        Args:
            requested_batch_size: Requested batch size (will be capped by available tokens)
            
        Returns:
            Calculated batch size that fits in available KV cache
        """
        async with self._lock:
            # Calculate how many requests we can fit
            available_batch_size = self.free_tokens // self.max_tokens_per_request
            
            # Apply max batch size limit
            available_batch_size = min(available_batch_size, self.max_batch_size)
            
            # Apply requested batch size if provided
            if requested_batch_size is not None:
                available_batch_size = min(available_batch_size, requested_batch_size)
            
            logger.debug(
                f"Calculated batch size: {available_batch_size} "
                f"(free_tokens={self.free_tokens}, requested={requested_batch_size})"
            )
            
            return available_batch_size
    
    async def reserve_tokens(self, request_ids: list[str], max_tokens_override: Optional[int] = None) -> bool:
        """
        Reserve KV cache tokens for a batch of requests.
        
        Args:
            request_ids: List of request IDs to reserve tokens for
            max_tokens_override: Override max tokens per request (for variable length requests)
            
        Returns:
            True if reservation successful, False if not enough tokens
        """
        async with self._lock:
            tokens_per_request = max_tokens_override or self.max_tokens_per_request
            tokens_needed = len(request_ids) * tokens_per_request
            
            if tokens_needed > self.free_tokens:
                logger.warning(
                    f"Not enough free tokens: needed={tokens_needed}, free={self.free_tokens}"
                )
                return False
            
            # Reserve tokens
            self.free_tokens -= tokens_needed
            
            # Track reserved tokens per request
            for request_id in request_ids:
                self.reserved_tokens[request_id] = tokens_per_request
            
            # Update stats
            self.stats.free_tokens = self.free_tokens
            self.stats.reserved_tokens += tokens_needed
            self.stats.last_update = datetime.now()
            
            logger.info(
                f"Reserved {tokens_needed} tokens for {len(request_ids)} requests. "
                f"Free tokens: {self.free_tokens}/{self.total_tokens}"
            )
            
            return True
    
    async def release_tokens_partial(self, request_id: str, actual_tokens_used: int) -> int:
        """
        Release unused tokens after tokenization (when actual < reserved).
        
        Args:
            request_id: Request ID
            actual_tokens_used: Actual tokens used after tokenization
            
        Returns:
            Number of tokens released back to pool
        """
        async with self._lock:
            if request_id not in self.reserved_tokens:
                logger.warning(f"Request {request_id} not found in reserved tokens")
                return 0
            
            reserved = self.reserved_tokens[request_id]
            tokens_to_release = reserved - actual_tokens_used
            
            if tokens_to_release > 0:
                self.free_tokens += tokens_to_release
                self.reserved_tokens[request_id] = actual_tokens_used
                
                # Update stats
                self.stats.free_tokens = self.free_tokens
                self.stats.reserved_tokens -= tokens_to_release
                self.stats.last_update = datetime.now()
                
                logger.debug(
                    f"Released {tokens_to_release} unused tokens for request {request_id}. "
                    f"Free tokens: {self.free_tokens}/{self.total_tokens}"
                )
            
            return tokens_to_release
    
    async def release_tokens_complete(self, request_id: str, total_tokens_used: int) -> int:
        """
        Release all tokens when request completes.
        
        Args:
            request_id: Request ID
            total_tokens_used: Total tokens actually used by the request
            
        Returns:
            Number of tokens released back to pool
        """
        async with self._lock:
            if request_id not in self.reserved_tokens:
                logger.warning(f"Request {request_id} not found in reserved tokens")
                return 0
            
            # Release all reserved tokens
            reserved = self.reserved_tokens.pop(request_id)
            self.free_tokens += reserved
            
            # Ensure we don't exceed total capacity
            if self.free_tokens > self.total_tokens:
                logger.warning(
                    f"Free tokens ({self.free_tokens}) exceeds total ({self.total_tokens}). "
                    "Capping to total."
                )
                self.free_tokens = self.total_tokens
            
            # Update stats
            self.stats.free_tokens = self.free_tokens
            self.stats.reserved_tokens -= reserved
            self.stats.total_requests_processed += 1
            self.stats.total_tokens_processed += total_tokens_used
            self.stats.last_update = datetime.now()
            
            logger.info(
                f"Released {reserved} tokens for completed request {request_id}. "
                f"Free tokens: {self.free_tokens}/{self.total_tokens}"
            )
            
            return reserved
    
    async def add_free_tokens(self, newly_freed_tokens: int) -> None:
        """
        Add newly freed tokens reported by vLLM engine.
        
        Args:
            newly_freed_tokens: Number of tokens freed by vLLM engine
        """
        async with self._lock:
            self.free_tokens += newly_freed_tokens
            
            # Ensure we don't exceed total capacity
            if self.free_tokens > self.total_tokens:
                logger.warning(
                    f"Free tokens ({self.free_tokens}) exceeds total ({self.total_tokens}). "
                    "Capping to total."
                )
                self.free_tokens = self.total_tokens
            
            # Update stats
            self.stats.free_tokens = self.free_tokens
            self.stats.last_update = datetime.now()
            
            logger.debug(f"Added {newly_freed_tokens} freed tokens. Free tokens: {self.free_tokens}")
    
    async def get_stats(self) -> KVCacheStats:
        """Get current KV cache statistics"""
        async with self._lock:
            return KVCacheStats(
                total_tokens=self.stats.total_tokens,
                free_tokens=self.stats.free_tokens,
                reserved_tokens=self.stats.reserved_tokens,
                total_requests_processed=self.stats.total_requests_processed,
                total_tokens_processed=self.stats.total_tokens_processed,
                last_update=self.stats.last_update
            )
    
    async def reset(self) -> None:
        """Reset KV cache manager to initial state (useful for testing)"""
        async with self._lock:
            self.free_tokens = self.total_tokens
            self.reserved_tokens.clear()
            self.stats = KVCacheStats(
                total_tokens=self.total_tokens,
                free_tokens=self.total_tokens,
                reserved_tokens=0
            )
            logger.info("KV Cache Manager reset to initial state")


# Global instance
_kv_cache_manager: Optional[KVCacheManager] = None


def initialize_kv_cache_manager(
    total_tokens: int,
    max_tokens_per_request: int,
    max_batch_size: int = 1024
) -> KVCacheManager:
    """Initialize the global KV cache manager"""
    global _kv_cache_manager
    _kv_cache_manager = KVCacheManager(total_tokens, max_tokens_per_request, max_batch_size)
    return _kv_cache_manager


async def get_kv_cache_manager() -> KVCacheManager:
    """Get the global KV cache manager instance"""
    if _kv_cache_manager is None:
        raise RuntimeError("KV Cache Manager not initialized. Call initialize_kv_cache_manager() first.")
    return _kv_cache_manager