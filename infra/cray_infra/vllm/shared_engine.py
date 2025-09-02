"""
Shared vLLM Engine implementation.
Provides direct access to a shared vLLM AsyncLLMEngine instance that exists in the same process.
"""

import logging
import weakref
from typing import List, Dict, Any, Optional
from .engine_interface import VLLMEngineInterface

logger = logging.getLogger(__name__)

# Global registry for shared engine instances
_shared_engines: Dict[str, Any] = {}


def register_shared_engine(name: str, engine, model_config) -> None:
    """
    Register a vLLM engine instance for shared access.
    
    Args:
        name: Name/key for the engine
        engine: AsyncLLMEngine instance  
        model_config: Model configuration
    """
    global _shared_engines
    _shared_engines[name] = {
        'engine': engine,
        'model_config': model_config
    }
    logger.info(f"Registered shared vLLM engine: {name}")


def unregister_shared_engine(name: str) -> None:
    """
    Unregister a shared engine instance.
    
    Args:
        name: Name/key of the engine to remove
    """
    global _shared_engines
    if name in _shared_engines:
        del _shared_engines[name]
        logger.info(f"Unregistered shared vLLM engine: {name}")


def get_shared_engine_names() -> List[str]:
    """Get list of registered shared engine names."""
    return list(_shared_engines.keys())


class SharedVLLMEngine(VLLMEngineInterface):
    """
    Provides direct access to a shared vLLM AsyncLLMEngine instance.
    
    This engine type is used when multiple components need to access the same
    vLLM engine instance within the same process, providing true direct method
    access without HTTP overhead.
    """
    
    def __init__(self, engine_name: str = "default"):
        """
        Initialize connection to shared vLLM engine.
        
        Args:
            engine_name: Name of the registered shared engine
        """
        self.engine_name = engine_name
        self._engine = None
        self._model_config = None
        self._serving_embedding = None
        self._serving_completion = None  
        self._serving_chat = None
        
    def _get_engine_info(self) -> Dict[str, Any]:
        """Get the shared engine and model config."""
        global _shared_engines
        
        if self.engine_name not in _shared_engines:
            available = list(_shared_engines.keys())
            raise RuntimeError(
                f"Shared engine '{self.engine_name}' not found. "
                f"Available engines: {available}. "
                f"Register engine using register_shared_engine() first."
            )
        
        engine_info = _shared_engines[self.engine_name]
        return engine_info['engine'], engine_info['model_config']
        
    @property
    def engine(self):
        """Get the vLLM engine instance."""
        if self._engine is None:
            self._engine, self._model_config = self._get_engine_info()
        return self._engine
    
    @property 
    def model_config(self):
        """Get the model configuration."""
        if self._model_config is None:
            self._engine, self._model_config = self._get_engine_info()
        return self._model_config
        
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
        """Generate embeddings using direct shared engine access."""
        try:
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
                raise RuntimeError("No embedding data returned from shared vLLM engine")
                
        except Exception as e:
            logger.error(f"Error in shared engine embedding generation: {e}")
            raise RuntimeError(f"Shared engine embedding generation failed: {e}")
    
    async def generate_completion(
        self, 
        prompt: str, 
        model: str, 
        max_tokens: Optional[int] = None,
        **kwargs
    ) -> str:
        """Generate completion using direct shared engine access."""
        try:
            # Use the engine directly like DirectVLLMEngine
            from vllm import SamplingParams
            
            # Create sampling parameters
            sampling_params = SamplingParams(
                max_tokens=max_tokens or 100,
                temperature=kwargs.get('temperature', 1.0),
                top_p=kwargs.get('top_p', 1.0),
                stop=kwargs.get('stop'),
                presence_penalty=kwargs.get('presence_penalty', 0.0),
                frequency_penalty=kwargs.get('frequency_penalty', 0.0),
                use_beam_search=kwargs.get('use_beam_search', False),
                top_k=kwargs.get('top_k', -1),
                ignore_eos=kwargs.get('ignore_eos', False),
                skip_special_tokens=kwargs.get('skip_special_tokens', True),
                spaces_between_special_tokens=kwargs.get('spaces_between_special_tokens', True),
            )
            
            # Generate completion using the shared engine directly
            results = await self.engine.generate(prompt, sampling_params, request_id=f"shared_{id(prompt)}")
            
            # Extract the generated text
            if results and len(results) > 0:
                outputs = results[0].outputs
                if outputs and len(outputs) > 0:
                    return outputs[0].text
                else:
                    raise RuntimeError("No output text generated")
            else:
                raise RuntimeError("No results returned from shared engine")
                
        except Exception as e:
            logger.error(f"Error in shared engine completion generation: {e}")
            raise RuntimeError(f"Shared engine completion generation failed: {e}")
    
    async def generate_chat_completion(
        self,
        messages: List[Dict[str, str]],
        model: str,
        max_tokens: Optional[int] = None,
        **kwargs
    ) -> str:
        """Generate chat completion using direct shared engine access."""
        try:
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
                raise RuntimeError("No chat completion choices returned from shared vLLM engine")
                
        except Exception as e:
            logger.error(f"Error in shared engine chat completion generation: {e}")
            raise RuntimeError(f"Shared engine chat completion generation failed: {e}")
    
    async def health_check(self) -> bool:
        """Check shared engine health directly."""
        try:
            # Try to access the shared engine
            engine = self.engine
            
            # Check if engine has health check method
            if hasattr(engine, 'check_health'):
                await engine.check_health()
                return True
            else:
                # If no health check method, assume healthy if engine exists
                return engine is not None
        except Exception as e:
            logger.warning(f"Shared engine health check failed: {e}")
            return False
    
    async def get_free_kv_cache_tokens(self) -> int:
        """Get free KV cache tokens via direct method call on shared engine."""
        try:
            # Call the method we implemented in the vLLM fork directly
            free_tokens = await self.engine.get_free_kv_cache_tokens()
            logger.debug(f"Retrieved {free_tokens} free tokens via shared engine direct method")
            return free_tokens
        except Exception as e:
            logger.error(f"Error getting free KV cache tokens via shared engine: {e}")
            return 0
    
    async def cleanup(self):
        """Clean up shared engine resources."""
        try:
            # Clean up serving layers
            self._serving_embedding = None
            self._serving_completion = None
            self._serving_chat = None
            
            # Clear references to shared engine (but don't shut it down)
            self._engine = None
            self._model_config = None
            
            logger.info("Shared engine cleanup completed")
        except Exception as e:
            logger.warning(f"Error during shared engine cleanup: {e}")
    
    @property
    def engine_type(self) -> str:
        return "shared"
    
    def __repr__(self):
        return f"SharedVLLMEngine(engine_name={self.engine_name})"