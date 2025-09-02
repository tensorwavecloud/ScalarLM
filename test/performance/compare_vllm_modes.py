#!/usr/bin/env python3
"""
Performance comparison test for HTTP vs Direct vLLM access modes
Tests both vllm_use_http: True and vllm_use_http: False configurations
"""

import asyncio
import time
import statistics
import aiohttp
import logging
import sys
import os
from typing import List, Dict, Any

# Add infra to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..', 'infra'))

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class PerformanceComparator:
    def __init__(self):
        self.api_base = "http://localhost:8000"
        self.results = {}
        
    async def test_generation_performance(self, use_http: bool, prompts: List[str], iterations: int = 3) -> Dict[str, Any]:
        """Test generation performance for HTTP vs Direct engine access"""
        mode = "HTTP" if use_http else "Direct"
        logger.info(f"Testing {mode} generation performance (vllm_use_http={use_http}) with {len(prompts)} prompts, {iterations} iterations")
        
        times = []
        errors = []
        
        for i in range(iterations):
            start_time = time.time()
            
            try:
                if use_http:
                    await self._test_http_engine_access(prompts[0])
                else:
                    await self._test_direct_engine_access(prompts[0])
                    
                end_time = time.time()
                iteration_time = end_time - start_time
                times.append(iteration_time)
                logger.info(f"  Iteration {i+1}: {iteration_time:.3f}s")
                
            except Exception as e:
                logger.error(f"  Iteration {i+1} FAILED: {e}")
                errors.append(str(e))
        
        if not times:
            return {
                "mode": mode,
                "use_http": use_http,
                "type": "generation",
                "status": "FAILED",
                "errors": errors,
                "iterations": iterations,
                "successful_iterations": 0
            }
        
        return {
            "mode": mode,
            "use_http": use_http,
            "type": "generation",
            "status": "SUCCESS",
            "iterations": iterations,
            "successful_iterations": len(times),
            "times": times,
            "avg_time": statistics.mean(times),
            "min_time": min(times),
            "max_time": max(times),
            "std_dev": statistics.stdev(times) if len(times) > 1 else 0,
            "errors": errors
        }
    
    
    async def _test_http_engine_access(self, prompt: str):
        """Test HTTP engine access using engine factory"""
        try:
            from cray_infra.vllm.engine_factory import create_vllm_engine
            
            # Create HTTP engine
            config = {
                "vllm_engine_type": "http",
                "vllm_api_url": "http://localhost:8001"
            }
            
            engine = create_vllm_engine(config)
            
            # Test engine functionality
            health = await engine.health_check()
            if not health:
                raise Exception("HTTP engine health check failed")
            
            # Test KV cache token retrieval (key functionality we're testing)
            free_tokens = await engine.get_free_kv_cache_tokens()
            logger.debug(f"HTTP engine free tokens: {free_tokens}")
            
            # Test generation
            result = await engine.generate_completion(
                prompt,
                "default",
                max_tokens=20
            )
            
            await engine.cleanup()
            return {"text": result, "free_tokens": free_tokens}
            
        except Exception as e:
            logger.error(f"HTTP engine test failed: {e}")
            raise
    
    async def _test_direct_engine_access(self, prompt: str):
        """Test direct engine access using engine factory"""
        try:
            from cray_infra.vllm.engine_factory import create_vllm_engine
            
            # Create direct engine
            config = {
                "vllm_engine_type": "direct",
                "model": "microsoft/DialoGPT-small",
                "max_model_length": 512,
                "gpu_memory_utilization": 0.3,
                "enable_lora": False  # Disable for GPT-2 models
            }
            
            engine = create_vllm_engine(config)
            
            # Test engine functionality
            health = await engine.health_check()
            if not health:
                raise Exception("Direct engine health check failed")
            
            # Test KV cache token retrieval (key functionality we're testing)
            free_tokens = await engine.get_free_kv_cache_tokens()
            logger.debug(f"Direct engine free tokens: {free_tokens}")
            
            # Test generation
            result = await engine.generate_completion(
                prompt,
                "microsoft/DialoGPT-small",
                max_tokens=20
            )
            
            await engine.cleanup()
            return {"text": result, "free_tokens": free_tokens}
            
        except Exception as e:
            logger.error(f"Direct engine test failed: {e}")
            raise
    
    
    
    def print_results(self, results: List[Dict[str, Any]]):
        """Print performance comparison results"""
        print("\n" + "="*80)
        print("vLLM ENGINE MODE COMPARISON RESULTS")
        print("="*80)
        print("HTTP MODE (vllm_use_http=True):   HTTPVLLMEngine → External vLLM server")  
        print("DIRECT MODE (vllm_use_http=False): DirectVLLMEngine → In-process engine")
        print("="*80)
        
        for result in results:
            mode = result["mode"]
            status = result["status"]
            
            print(f"\n{mode} MODE (vllm_use_http={result['use_http']}):")
            print("-" * 50)
            
            if status == "SUCCESS":
                print(f"✅ SUCCESS - {result['successful_iterations']}/{result['iterations']} iterations")
                print(f"   Average Time: {result['avg_time']:.3f}s")
                print(f"   Min Time:     {result['min_time']:.3f}s") 
                print(f"   Max Time:     {result['max_time']:.3f}s")
                print(f"   Std Dev:      {result['std_dev']:.3f}s")
            else:
                print(f"❌ FAILED - {result['successful_iterations']}/{result['iterations']} iterations")
                for i, error in enumerate(result.get('errors', [])[:3]):  # Show first 3 errors
                    print(f"   Error {i+1}: {error}")
        
        # Calculate speedup if both modes successful
        successful_results = [r for r in results if r["status"] == "SUCCESS"]
        if len(successful_results) == 2:
            http_result = next((r for r in successful_results if r["use_http"]), None)
            direct_result = next((r for r in successful_results if not r["use_http"]), None)
            
            if http_result and direct_result:
                speedup = http_result["avg_time"] / direct_result["avg_time"]
                print(f"\n🏁 PERFORMANCE COMPARISON:")
                print(f"   Direct mode is {speedup:.2f}x {'faster' if speedup > 1 else 'slower'} than HTTP mode")
        
        print("\n" + "="*80)
        

async def main():
    """Run the performance comparison"""
    print("🧪 vLLM Engine Mode Performance Comparison")
    print("Testing both vllm_use_http=True and vllm_use_http=False configurations")
    print("="*80)
    
    comparator = PerformanceComparator()
    
    # Test prompts
    test_prompts = [
        "Hello, how are you today?",
        "What is machine learning?", 
        "Explain the concept of recursion."
    ]
    
    results = []
    
    try:
        logger.info("🚀 Starting vLLM engine mode comparison tests...")
        
        # Test HTTP Mode (vllm_use_http=True)
        logger.info("\n1️⃣ Testing HTTP Engine Mode (vllm_use_http=True)")
        http_result = await comparator.test_generation_performance(
            use_http=True, 
            prompts=test_prompts[:1], 
            iterations=3
        )
        results.append(http_result)
        
        # Test Direct Mode (vllm_use_http=False)  
        logger.info("\n2️⃣ Testing Direct Engine Mode (vllm_use_http=False)")
        direct_result = await comparator.test_generation_performance(
            use_http=False,
            prompts=test_prompts[:1], 
            iterations=3
        )
        results.append(direct_result)
        
    except Exception as e:
        logger.error(f"Performance test failed: {e}")
        import traceback
        traceback.print_exc()
        return
    
    # Print results
    comparator.print_results(results)
    
    # Print summary
    successful_tests = sum(1 for r in results if r["status"] == "SUCCESS")
    total_tests = len(results)
    
    print(f"\n📊 TEST SUMMARY:")
    print(f"   Total Tests: {total_tests}")
    print(f"   Successful: {successful_tests}")
    print(f"   Failed: {total_tests - successful_tests}")
    
    if successful_tests == total_tests:
        print("   🎉 All engine modes working correctly!")
    else:
        print("   ⚠️ Some engine modes failed - check logs above")
    
    return successful_tests == total_tests

if __name__ == "__main__":
    asyncio.run(main())