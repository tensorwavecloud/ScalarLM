"""
Direct vLLM engine implementation.
Calls vLLM engine methods directly without HTTP overhead.
"""

from typing import List, Dict, Any, Optional
import logging
import asyncio

from .engine_interface import VLLMEngineInterface

logger = logging.getLogger(__name__)


class DirectVLLMEngine(VLLMEngineInterface):
    """Direct vLLM engine that calls methods directly."""
    
    def __init__(self, engine, model_config):
        """
        Initialize with direct access to vLLM engine.
        
        Args:
            engine: AsyncLLMEngine instance
            model_config: Model configuration for serving layers
        """
        self.engine = engine
        self.model_config = model_config
        self._serving_embedding = None
        self._serving_completion = None
        self._serving_chat = None
        self._stats_task = None
        
        # Start stats logging task
        self._start_stats_logging()
        
    def _start_stats_logging(self):
        """Start periodic stats logging task."""
        try:
            # Only start if we're in an async context
            loop = asyncio.get_running_loop()
            self._stats_task = loop.create_task(self._stats_logging_loop())
            logger.info("✓ Started stats logging task for DirectVLLMEngine")
        except RuntimeError:
            # Not in async context, will start later if needed
            logger.debug("Not in async context, stats logging will start when first async call is made")
            pass
    
    async def _stats_logging_loop(self):
        """Periodic stats logging loop."""
        try:
            while True:
                await asyncio.sleep(10.0)  # Log every 10 seconds (matches VLLM_LOG_STATS_INTERVAL default)
                try:
                    await self.engine.do_log_stats()
                except Exception as e:
                    logger.warning(f"Failed to log stats: {e}")
        except asyncio.CancelledError:
            logger.info("Stats logging task cancelled")
            raise
        except Exception as e:
            logger.error(f"Stats logging task error: {e}")
        
    async def _get_serving_embedding(self):
        """Get or create OpenAI embedding serving layer."""
        if self._serving_embedding is None:
            try:
                from vllm.entrypoints.openai.serving_embedding import OpenAIServingEmbedding
                self._serving_embedding = OpenAIServingEmbedding(
                    engine_client=self.engine,
                    model_config=self.model_config,
                    chat_template=None,
                    response_role="assistant",
                    lora_modules=None,
                    prompt_adapters=None,
                    request_logger=None,
                )
            except ImportError as e:
                logger.error(f"Failed to import OpenAIServingEmbedding: {e}")
                raise RuntimeError("vLLM serving components not available")
        return self._serving_embedding
    
    async def _get_serving_completion(self):
        """Get or create OpenAI completion serving layer."""
        if self._serving_completion is None:
            try:
                from vllm.entrypoints.openai.serving_completion import OpenAIServingCompletion
                # Try to initialize with minimal parameters
                self._serving_completion = OpenAIServingCompletion(
                    engine_client=self.engine,
                    model_config=self.model_config,
                )
            except ImportError as e:
                logger.error(f"Failed to import OpenAIServingCompletion: {e}")
                raise RuntimeError("vLLM serving components not available")
        return self._serving_completion
    
    async def _get_serving_chat(self):
        """Get or create OpenAI chat serving layer."""
        if self._serving_chat is None:
            try:
                from vllm.entrypoints.openai.serving_chat import OpenAIServingChat
                self._serving_chat = OpenAIServingChat(
                    engine_client=self.engine,
                    model_config=self.model_config,
                    chat_template=None,
                    response_role="assistant",
                    lora_modules=None,
                    prompt_adapters=None,
                    request_logger=None,
                    tool_parser=None,
                )
            except ImportError as e:
                logger.error(f"Failed to import OpenAIServingChat: {e}")
                raise RuntimeError("vLLM serving components not available")
        return self._serving_chat
    
    async def generate_embeddings(self, prompt: str, model: str) -> List[float]:
        """Generate embeddings using direct vLLM engine calls."""
        try:
            # Ensure stats task is running
            await self._ensure_stats_task_started()
            serving_embedding = await self._get_serving_embedding()
            
            # Create embedding request
            from vllm.entrypoints.openai.protocol import EmbeddingRequest
            request = EmbeddingRequest(
                input=prompt,
                model=model,
                encoding_format="float",
                dimensions=None,
                user=None,
                extra_body={}
            )
            
            # Generate embedding
            response = await serving_embedding.create_embedding(request, raw_request=None)
            
            # Extract embedding data
            if hasattr(response, 'data') and response.data:
                return response.data[0].embedding
            else:
                raise RuntimeError("No embedding data returned from vLLM")
                
        except Exception as e:
            logger.error(f"Error in direct embedding generation: {e}")
            raise RuntimeError(f"Direct embedding generation failed: {e}")
    
    async def _ensure_stats_task_started(self):
        """Ensure stats logging task is running."""
        if self._stats_task is None or self._stats_task.done():
            try:
                loop = asyncio.get_running_loop()
                self._stats_task = loop.create_task(self._stats_logging_loop())
                logger.info("✓ Started stats logging task for DirectVLLMEngine")
            except Exception as e:
                logger.warning(f"Failed to start stats logging task: {e}")

    async def generate_completion(
        self, 
        prompt: str, 
        model: str, 
        max_tokens: Optional[int] = None,
        **kwargs
    ) -> str:
        """Generate completion using direct vLLM engine calls."""
        try:
            # Ensure stats task is running
            await self._ensure_stats_task_started()
            # Bypass serving layer and call engine directly
            from vllm import SamplingParams
            
            # Create sampling parameters compatible with vLLM v1
            sampling_params = SamplingParams(
                max_tokens=max_tokens or 100,
                temperature=kwargs.get('temperature', 1.0),
                top_p=kwargs.get('top_p', 1.0),
                stop=kwargs.get('stop'),
                presence_penalty=kwargs.get('presence_penalty', 0.0),
                frequency_penalty=kwargs.get('frequency_penalty', 0.0),
                # use_beam_search not supported in v1 - removed
                top_k=kwargs.get('top_k', -1),
                # ignore_eos not supported in v1 - removed  
                skip_special_tokens=kwargs.get('skip_special_tokens', True),
                spaces_between_special_tokens=kwargs.get('spaces_between_special_tokens', True),
            )
            
            # Generate completion using the engine directly
            # In v1, engine.generate() returns an async generator, not a single result
            request_id = f"direct_{id(prompt)}"
            
            # Collect all results from the async generator
            results = []
            async for result in self.engine.generate(prompt, sampling_params, request_id=request_id):
                results.append(result)
            
            # Extract the generated text from the final result
            if results and len(results) > 0:
                final_result = results[-1]  # Get the final result
                outputs = final_result.outputs
                if outputs and len(outputs) > 0:
                    return outputs[0].text
                else:
                    raise RuntimeError("No output text generated")
            else:
                raise RuntimeError("No results returned from engine")
                
        except Exception as e:
            logger.error(f"Error in direct completion generation: {e}")
            raise RuntimeError(f"Direct completion generation failed: {e}")
    
    async def generate_chat_completion(
        self,
        messages: List[Dict[str, str]],
        model: str,
        max_tokens: Optional[int] = None,
        **kwargs
    ) -> str:
        """Generate chat completion using direct vLLM engine calls."""
        try:
            # Ensure stats task is running
            await self._ensure_stats_task_started()
            serving_chat = await self._get_serving_chat()
            
            # Convert messages to proper format
            from vllm.entrypoints.openai.protocol import ChatCompletionRequest
            chat_messages = []
            for msg in messages:
                chat_messages.append({
                    "role": msg.get("role", "user"),
                    "content": msg.get("content", "")
                })
            
            request = ChatCompletionRequest(
                model=model,
                messages=chat_messages,
                max_tokens=max_tokens or 100,
                temperature=kwargs.get('temperature', 1.0),
                top_p=kwargs.get('top_p', 1.0),
                n=1,
                stream=False,
                stop=kwargs.get('stop'),
                presence_penalty=kwargs.get('presence_penalty', 0.0),
                frequency_penalty=kwargs.get('frequency_penalty', 0.0),
                logit_bias=None,
                user=None,
                response_format=kwargs.get('response_format'),
                tools=None,
                tool_choice="none",
                parallel_tool_calls=False,
                guided_json=kwargs.get('guided_json'),
                guided_regex=kwargs.get('guided_regex'),
                guided_choice=kwargs.get('guided_choice'),
                guided_grammar=kwargs.get('guided_grammar'),
                guided_decoding_backend=kwargs.get('guided_decoding_backend'),
                guided_whitespace_pattern=kwargs.get('guided_whitespace_pattern'),
                priority=kwargs.get('priority', 0),
                extra_body={}
            )
            
            # Generate chat completion
            response = await serving_chat.create_chat_completion(request, raw_request=None)
            
            # Extract response content
            if hasattr(response, 'choices') and response.choices:
                message = response.choices[0].message
                return message.content or ""
            else:
                raise RuntimeError("No chat completion choices returned from vLLM")
                
        except Exception as e:
            logger.error(f"Error in direct chat completion generation: {e}")
            raise RuntimeError(f"Direct chat completion generation failed: {e}")
    
    async def health_check(self) -> bool:
        """Check engine health directly."""
        try:
            # Try to check engine health
            if hasattr(self.engine, 'check_health'):
                await self.engine.check_health()
                return True
            else:
                # If no health check method, assume healthy if engine exists
                return self.engine is not None
        except Exception as e:
            logger.warning(f"Direct engine health check failed: {e}")
            return False
    
    async def cleanup(self):
        """Clean up engine resources."""
        try:
            # Clean up serving layers
            self._serving_embedding = None
            self._serving_completion = None
            self._serving_chat = None
            
            # Note: We don't shutdown the engine here as it might be shared
            # Engine lifecycle should be managed at a higher level
            logger.info("Direct engine cleanup completed")
        except Exception as e:
            logger.warning(f"Error during direct engine cleanup: {e}")
    
    async def get_free_kv_cache_tokens(self) -> int:
        """Get free KV cache tokens via direct method call."""
        try:
            # Call the method we implemented in the vLLM fork directly
            free_tokens = await self.engine.get_free_kv_cache_tokens()
            logger.debug(f"Retrieved {free_tokens} free tokens via direct method")
            return free_tokens
        except Exception as e:
            logger.error(f"Error getting free KV cache tokens via direct method: {e}")
            return 0
    
    @property
    def engine_type(self) -> str:
        return "direct"
    
    def __repr__(self):
        return f"DirectVLLMEngine(engine={type(self.engine).__name__})"