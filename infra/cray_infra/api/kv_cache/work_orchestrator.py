"""
Work Orchestrator - Eliminates race conditions by design.

Instead of separate get_work() and get_adaptors() calls that create race conditions,
this provides a single atomic operation that handles everything in one transaction.
"""

import asyncio
import logging
from typing import Optional, List, Dict, Any, Set
from dataclasses import dataclass
from datetime import datetime

from cray_infra.api.kv_cache.kv_cache_manager import get_kv_cache_manager
from cray_infra.api.work_queue.inference_work_queue import get_inference_work_queue
from cray_infra.api.fastapi.routers.request_types.get_work_response import GetWorkResponse
from cray_infra.util.get_config import get_config

logger = logging.getLogger(__name__)


@dataclass
class WorkResult:
    """Result of atomic work acquisition"""
    requests: List[GetWorkResponse]
    adaptors_loaded: List[str]
    kv_tokens_reserved: int
    batch_id: str
    timestamp: datetime
    
    @property
    def request_ids(self) -> List[str]:
        return [r.request_id for r in self.requests]


class WorkOrchestrator:
    """
    Manages work acquisition atomically to prevent race conditions.
    
    The key insight: By combining get_work + get_adaptors + reserve_tokens
    into a single atomic operation, we eliminate race conditions by design.
    No complex locking needed!
    """
    
    def __init__(self):
        """Initialize the work orchestrator"""
        self.loaded_adaptors: Set[str] = set()
        self.total_batches_processed = 0
        self.total_requests_processed = 0
        
        # Single lock for atomic operations - much simpler!
        self._atomic_lock = asyncio.Lock()
        
        logger.info("Work Orchestrator initialized")
    
    async def get_work_atomic(
        self, 
        requested_batch_size: Optional[int] = None
    ) -> Optional[WorkResult]:
        """
        Atomically get work, load adaptors, and reserve KV cache tokens.
        
        This single method replaces:
        - get_work()
        - get_adaptors() 
        - reserve_tokens()
        
        By doing everything in one atomic operation, race conditions are impossible.
        
        Args:
            requested_batch_size: Optional requested batch size
            
        Returns:
            WorkResult if work available, None otherwise
        """
        async with self._atomic_lock:
            try:
                # 1. Calculate dynamic batch size based on KV cache
                kv_manager = await get_kv_cache_manager()
                batch_size = await kv_manager.calculate_batch_size(requested_batch_size)
                
                if batch_size == 0:
                    logger.info("No KV cache tokens available")
                    return None
                
                # 2. Get work from queue
                work_requests = await self._get_work_from_queue(batch_size)
                
                if not work_requests:
                    logger.debug("No work available in queue")
                    return None
                
                # 3. Reserve KV cache tokens
                request_ids = [req.request_id for req in work_requests]
                reserved = await kv_manager.reserve_tokens(request_ids)
                
                if not reserved:
                    logger.error("Failed to reserve KV cache tokens")
                    await self._return_work_to_queue(work_requests)
                    return None
                
                # 4. Determine and load required adaptors
                required_adaptors = self._extract_required_adaptors(work_requests)
                newly_loaded_adaptors = await self._load_adaptors_atomic(required_adaptors)
                
                # 5. Create atomic result with everything needed
                config = get_config()
                reserved_tokens = len(work_requests) * config.get("max_model_length", 2048)
                
                result = WorkResult(
                    requests=work_requests,
                    adaptors_loaded=list(self.loaded_adaptors),
                    kv_tokens_reserved=reserved_tokens,
                    batch_id=f"batch_{self.total_batches_processed}",
                    timestamp=datetime.now()
                )
                
                # Update statistics
                self.total_batches_processed += 1
                self.total_requests_processed += len(work_requests)
                
                logger.info(
                    f"Atomic work acquisition successful: "
                    f"batch_id={result.batch_id}, "
                    f"requests={len(work_requests)}, "
                    f"new_adaptors={len(newly_loaded_adaptors)}, "
                    f"reserved_tokens={reserved_tokens}"
                )
                
                return result
                
            except Exception as e:
                logger.error(f"Error in atomic work acquisition: {e}")
                return None
    
    async def _get_work_from_queue(self, batch_size: int) -> List[GetWorkResponse]:
        """
        Get work from the inference work queue.
        
        Args:
            batch_size: Number of requests to get
            
        Returns:
            List of work requests
        """
        inference_queue = await get_inference_work_queue()
        requests = []
        
        try:
            # Get first request (blocking)
            first_request, request_id = await inference_queue.get()
            
            if first_request is None:
                return []
            
            requests.append(
                GetWorkResponse(
                    prompt=first_request["prompt"],
                    request_id=request_id,
                    model=first_request.get("model"),
                    request_type=first_request.get("request_type", "generate"),
                    max_tokens=first_request.get("max_tokens"),
                )
            )
            
            # Get remaining requests (non-blocking)
            for _ in range(batch_size - 1):
                try:
                    request, request_id = await inference_queue.get_nowait()
                    if request is None:
                        break
                    
                    requests.append(
                        GetWorkResponse(
                            prompt=request["prompt"],
                            request_id=request_id,
                            model=request.get("model"),
                            request_type=request.get("request_type", "generate"),
                            max_tokens=request.get("max_tokens"),
                        )
                    )
                except asyncio.QueueEmpty:
                    break
        
        except Exception as e:
            logger.error(f"Error getting work from queue: {e}")
        
        logger.debug(f"Retrieved {len(requests)} requests from queue")
        return requests
    
    async def _return_work_to_queue(self, work_requests: List[GetWorkResponse]) -> None:
        """
        Return work to the queue (used when reservation fails).
        
        Args:
            work_requests: Work requests to return to queue
        """
        inference_queue = await get_inference_work_queue()
        
        for request in work_requests:
            await inference_queue.put({
                "prompt": request.prompt,
                "model": request.model,
                "request_type": request.request_type,
                "max_tokens": request.max_tokens,
            })
        
        logger.info(f"Returned {len(work_requests)} requests to queue")
    
    def _extract_required_adaptors(self, work_requests: List[GetWorkResponse]) -> Set[str]:
        """Extract adaptor requirements from work requests"""
        required_adaptors = set()
        
        for request in work_requests:
            # Extract adaptor info from request
            # This depends on how adaptors are specified in requests
            if hasattr(request, 'adaptor') and request.adaptor:
                required_adaptors.add(request.adaptor)
            elif hasattr(request, 'model') and request.model:
                # Model might indicate adaptor requirement
                if 'lora' in str(request.model).lower():
                    required_adaptors.add(request.model)
        
        return required_adaptors
    
    async def _load_adaptors_atomic(self, required_adaptors: Set[str]) -> List[str]:
        """
        Load required adaptors atomically.
        
        Since this is called within the atomic lock, no race condition possible!
        """
        newly_loaded = []
        
        for adaptor in required_adaptors:
            if adaptor not in self.loaded_adaptors:
                try:
                    # In real implementation, this would call vLLM to load the adaptor
                    await self._load_single_adaptor(adaptor)
                    self.loaded_adaptors.add(adaptor)
                    newly_loaded.append(adaptor)
                    logger.info(f"Loaded adaptor: {adaptor}")
                except Exception as e:
                    logger.error(f"Failed to load adaptor {adaptor}: {e}")
        
        return newly_loaded
    
    async def _load_single_adaptor(self, adaptor: str) -> None:
        """Load a single adaptor to vLLM engine"""
        # This would make the actual call to vLLM to load the adaptor
        # For now, simulate with a small delay
        await asyncio.sleep(0.01)
        logger.debug(f"Loaded adaptor to vLLM: {adaptor}")
    
    async def complete_work(
        self,
        request_id: str,
        total_tokens_used: int
    ) -> None:
        """
        Mark work as complete and release KV cache tokens.
        
        Args:
            request_id: Request ID
            total_tokens_used: Total tokens actually used
        """
        kv_manager = await get_kv_cache_manager()
        released = await kv_manager.release_tokens_complete(request_id, total_tokens_used)
        
        logger.debug(f"Work completed: request_id={request_id}, released_tokens={released}")
    
    async def update_tokenization(
        self,
        request_id: str,
        actual_tokens: int
    ) -> None:
        """
        Update token reservation after tokenization.
        
        Called when actual token count is known (less than reserved max).
        
        Args:
            request_id: Request ID
            actual_tokens: Actual tokens after tokenization
        """
        kv_manager = await get_kv_cache_manager()
        released = await kv_manager.release_tokens_partial(request_id, actual_tokens)
        
        logger.debug(f"Updated tokenization: request_id={request_id}, released_tokens={released}")
    
    async def get_stats(self) -> Dict[str, Any]:
        """Get manager statistics"""
        kv_manager = await get_kv_cache_manager()
        kv_stats = await kv_manager.get_stats()
        
        return {
            "work_manager": {
                "total_batches_processed": self.total_batches_processed,
                "total_requests_processed": self.total_requests_processed,
                "loaded_adaptors": len(self.loaded_adaptors),
                "loaded_adaptor_list": list(self.loaded_adaptors),
            },
            "kv_cache": {
                "total_tokens": kv_stats.total_tokens,
                "free_tokens": kv_stats.free_tokens,
                "reserved_tokens": kv_stats.reserved_tokens,
                "utilization_percent": kv_stats.utilization_percent,
            }
        }


# Global instance
_work_orchestrator: Optional[WorkOrchestrator] = None


def initialize_work_orchestrator() -> WorkOrchestrator:
    """Initialize the global work orchestrator"""
    global _work_orchestrator
    _work_orchestrator = WorkOrchestrator()
    return _work_orchestrator


async def get_work_orchestrator() -> WorkOrchestrator:
    """Get the global work orchestrator instance"""
    if _work_orchestrator is None:
        raise RuntimeError("Work Orchestrator not initialized. Call initialize_work_orchestrator() first.")
    return _work_orchestrator