"""
Work Router - Single endpoint for race-condition-free work acquisition.

This provides a unified endpoint that eliminates race conditions by design
through atomic work and adaptor acquisition.
"""

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
from typing import List, Optional, Dict, Any

from cray_infra.api.kv_cache.work_orchestrator import (
    get_work_orchestrator,
    initialize_work_orchestrator,
    WorkResult
)

router = APIRouter(prefix="/v1/work", tags=["work"])

class WorkRequest(BaseModel):
    """Request for work acquisition"""
    requested_batch_size: Optional[int] = None
    worker_id: Optional[str] = None


class WorkResponse(BaseModel):
    """Response containing work and all necessary context"""
    requests: List[Dict[str, Any]]
    adaptors_loaded: List[str]
    kv_tokens_reserved: int
    batch_id: str

    @classmethod
    def from_result(cls, result: WorkResult) -> "WorkResponse":
        """Create response from WorkResult"""
        return cls(
            requests=[
                {
                    "prompt": r.prompt,
                    "request_id": r.request_id,
                    "model": r.model,
                    "request_type": r.request_type,
                    "max_tokens": r.max_tokens,
                }
                for r in result.requests
            ],
            adaptors_loaded=result.adaptors_loaded,
            kv_tokens_reserved=result.kv_tokens_reserved,
            batch_id=result.batch_id,
        )


class WorkCompleteRequest(BaseModel):
    """Request to mark work as complete"""
    request_id: str
    total_tokens_used: int


class TokenizationUpdateRequest(BaseModel):
    """Request to update tokenization"""
    request_id: str
    actual_tokens: int


@router.post("/get_work_and_adaptors", response_model=WorkResponse)
async def get_work_and_adaptors(request: WorkRequest):
    """
    Get work, load adaptors, and reserve KV cache tokens in one operation.
    
    This unified endpoint eliminates race conditions by handling everything
    in a single transaction.
    
    Benefits:
    - No race conditions possible
    - Simpler client code (one call instead of multiple)
    - Better performance (one network round-trip)
    - Easier to reason about and debug
    """
    try:
        work_orchestrator = await get_work_orchestrator()
    except RuntimeError:
        # Initialize if not already done
        initialize_work_orchestrator()
        work_orchestrator = await get_work_orchestrator()
    
    # Get work atomically
    result = await work_orchestrator.get_work_atomic(
        requested_batch_size=request.requested_batch_size
    )
    
    if result is None:
        raise HTTPException(
            status_code=204,  # No Content
            detail="No work available or insufficient KV cache"
        )
    
    return WorkResponse.from_result(result)


@router.post("/complete_work")
async def complete_work(request: WorkCompleteRequest):
    """
    Mark work as complete and release KV cache tokens.
    
    Call this when a request has finished processing.
    """
    try:
        work_orchestrator = await get_work_orchestrator()
        await work_orchestrator.complete_work(
            request_id=request.request_id,
            total_tokens_used=request.total_tokens_used
        )
        return {"status": "success", "request_id": request.request_id}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/update_tokenization")
async def update_tokenization(request: TokenizationUpdateRequest):
    """
    Update token reservation after tokenization.
    
    Call this after tokenization when you know the actual token count.
    """
    try:
        work_orchestrator = await get_work_orchestrator()
        await work_orchestrator.update_tokenization(
            request_id=request.request_id,
            actual_tokens=request.actual_tokens
        )
        return {"status": "success", "request_id": request.request_id}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/stats")
async def get_stats():
    """Get work manager statistics"""
    try:
        work_orchestrator = await get_work_orchestrator()
        stats = await work_orchestrator.get_stats()
        return stats
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

