"""
HTTP-based vLLM engine implementation.
Communicates with vLLM server via HTTP API calls.
"""

import aiohttp
import asyncio
from typing import List, Dict, Any, Optional
import logging

from .engine_interface import VLLMEngineInterface

logger = logging.getLogger(__name__)


class HTTPVLLMEngine(VLLMEngineInterface):
    """HTTP-based vLLM engine that communicates via REST API."""
    
    def __init__(self, base_url: str, timeout: float = 30.0):
        self.base_url = base_url.rstrip('/')
        self.timeout = timeout
        self.session: Optional[aiohttp.ClientSession] = None
        
    async def _get_session(self) -> aiohttp.ClientSession:
        """Get or create HTTP session."""
        if self.session is None or self.session.closed:
            timeout = aiohttp.ClientTimeout(total=self.timeout)
            self.session = aiohttp.ClientSession(timeout=timeout)
        return self.session
    
    async def generate_embeddings(self, prompt: str, model: str) -> List[float]:
        """Generate embeddings via HTTP API."""
        session = await self._get_session()
        
        payload = {
            "input": prompt,
            "model": model
        }
        
        try:
            async with session.post(
                f"{self.base_url}/v1/embeddings",
                json=payload
            ) as response:
                if response.status != 200:
                    error_text = await response.text()
                    raise RuntimeError(f"vLLM embeddings failed ({response.status}): {error_text}")
                
                data = await response.json()
                
                # Extract embedding from response
                embeddings = data.get("data", [])
                if not embeddings:
                    raise RuntimeError("No embeddings returned from vLLM")
                
                return embeddings[0].get("embedding", [])
                
        except aiohttp.ClientError as e:
            logger.error(f"HTTP error during embeddings: {e}")
            raise RuntimeError(f"HTTP error during embeddings: {e}")
        except Exception as e:
            logger.error(f"Error generating embeddings: {e}")
            raise
    
    async def generate_completion(
        self, 
        prompt: str, 
        model: str, 
        max_tokens: Optional[int] = None,
        **kwargs
    ) -> str:
        """Generate completion via HTTP API."""
        session = await self._get_session()
        
        payload = {
            "prompt": prompt,
            "model": model
        }
        
        if max_tokens is not None:
            payload["max_tokens"] = max_tokens
            
        # Add any additional parameters
        payload.update(kwargs)
        
        try:
            async with session.post(
                f"{self.base_url}/v1/completions",
                json=payload
            ) as response:
                if response.status != 200:
                    error_text = await response.text()
                    raise RuntimeError(f"vLLM completion failed ({response.status}): {error_text}")
                
                data = await response.json()
                
                # Extract completion from response
                choices = data.get("choices", [])
                if not choices:
                    raise RuntimeError("No completions returned from vLLM")
                
                return choices[0].get("text", "")
                
        except aiohttp.ClientError as e:
            logger.error(f"HTTP error during completion: {e}")
            raise RuntimeError(f"HTTP error during completion: {e}")
        except Exception as e:
            logger.error(f"Error generating completion: {e}")
            raise
    
    async def generate_chat_completion(
        self,
        messages: List[Dict[str, str]],
        model: str,
        max_tokens: Optional[int] = None,
        **kwargs
    ) -> str:
        """Generate chat completion via HTTP API."""
        session = await self._get_session()
        
        payload = {
            "messages": messages,
            "model": model
        }
        
        if max_tokens is not None:
            payload["max_tokens"] = max_tokens
            
        # Add any additional parameters
        payload.update(kwargs)
        
        try:
            async with session.post(
                f"{self.base_url}/v1/chat/completions",
                json=payload
            ) as response:
                if response.status != 200:
                    error_text = await response.text()
                    raise RuntimeError(f"vLLM chat completion failed ({response.status}): {error_text}")
                
                data = await response.json()
                
                # Extract completion from response
                choices = data.get("choices", [])
                if not choices:
                    raise RuntimeError("No chat completions returned from vLLM")
                
                message = choices[0].get("message", {})
                return message.get("content", "")
                
        except aiohttp.ClientError as e:
            logger.error(f"HTTP error during chat completion: {e}")
            raise RuntimeError(f"HTTP error during chat completion: {e}")
        except Exception as e:
            logger.error(f"Error generating chat completion: {e}")
            raise
    
    async def health_check(self) -> bool:
        """Check vLLM server health via HTTP."""
        session = await self._get_session()
        
        try:
            async with session.get(f"{self.base_url}/health") as response:
                return response.status == 200
        except Exception as e:
            logger.warning(f"Health check failed: {e}")
            return False
    
    async def cleanup(self):
        """Clean up HTTP session."""
        if self.session and not self.session.closed:
            await self.session.close()
    
    async def get_free_kv_cache_tokens(self) -> int:
        """Get free KV cache tokens via HTTP API."""
        session = await self._get_session()
        
        try:
            async with session.get(
                f"{self.base_url}/v1/kv_cache/free_tokens"
            ) as response:
                if response.status != 200:
                    error_text = await response.text()
                    logger.warning(f"Failed to get free tokens via HTTP ({response.status}): {error_text}")
                    return 0
                
                data = await response.json()
                free_tokens = data.get("free_tokens", 0)
                logger.debug(f"Retrieved {free_tokens} free tokens via HTTP API")
                return free_tokens
                
        except Exception as e:
            logger.error(f"Error getting free KV cache tokens via HTTP: {e}")
            return 0
    
    @property
    def engine_type(self) -> str:
        return "http"
    
    def __repr__(self):
        return f"HTTPVLLMEngine(base_url='{self.base_url}')"