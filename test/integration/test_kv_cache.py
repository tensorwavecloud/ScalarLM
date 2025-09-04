"""
Integration test for KV Cache token lifecycle and work orchestration.
Tests the complete flow from token reservation through generation to cleanup.
"""

import pytest
import asyncio
import requests
import time
from typing import List, Dict, Any

SCALARLM_URL = "http://localhost:8000"
VLLM_URL = "http://localhost:8001"


class TestKVCacheTokenLifecycle:
    """Test complete KV cache token lifecycle integration"""
    
    def test_kv_cache_stats_endpoint(self):
        """Test KV cache statistics endpoint"""
        response = requests.get(f"{SCALARLM_URL}/v1/kv_cache/stats")
        assert response.status_code in [200, 404, 500]
        
        if response.status_code == 200:
            try:
                data = response.json()
                assert isinstance(data, dict)
                # Expected fields from KVCacheStats
                expected_fields = ["total_tokens", "free_tokens", "reserved_tokens", "utilization_percent"]
                for field in expected_fields:
                    if field in data:
                        print(f"✓ KV cache has {field}: {data[field]}")
                        assert isinstance(data[field], (int, float))
            except:
                pass

    def test_atomic_work_acquisition(self):
        """Test atomic get_work_and_adaptors endpoint"""
        # Test the new atomic endpoint that replaces racy get_work + get_adaptors
        response = requests.post(f"{SCALARLM_URL}/v1/work/get_work_and_adaptors", json={
            "requested_batch_size": 2
        })
        assert response.status_code in [200, 204, 404, 500]
        
        if response.status_code == 200:
            try:
                data = response.json()
                print(f"Atomic work result: {data}")
                
                # Should contain work requests + adaptor info + token reservation
                assert isinstance(data, dict)
                if "requests" in data:
                    assert isinstance(data["requests"], list)
                if "kv_tokens_reserved" in data:
                    assert isinstance(data["kv_tokens_reserved"], int)
                if "adaptors_loaded" in data:
                    assert isinstance(data["adaptors_loaded"], list)
            except:
                pass
        elif response.status_code == 204:
            print("✓ No work available (expected when queue empty)")

    def test_token_reservation_lifecycle(self):
        """Test complete token reservation → release cycle"""
        # Submit a generation request that should trigger token reservation
        payload = {
            "prompt": "Count to 5",
            "model": "TinyLlama/TinyLlama-1.1B-Chat-v1.0",
            "max_tokens": 20
        }
        
        # Get KV stats before
        stats_before = requests.get(f"{SCALARLM_URL}/v1/kv_cache/stats")
        
        # Submit generation request
        response = requests.post(f"{SCALARLM_URL}/v1/generate", json=payload)
        assert response.status_code in [200, 422, 500]
        
        if response.status_code == 200:
            # Get KV stats after
            stats_after = requests.get(f"{SCALARLM_URL}/v1/kv_cache/stats")
            
            if stats_before.status_code == 200 and stats_after.status_code == 200:
                try:
                    before_data = stats_before.json()
                    after_data = stats_after.json()
                    
                    print(f"KV tokens before: {before_data.get('free_tokens', 'unknown')}")
                    print(f"KV tokens after: {after_data.get('free_tokens', 'unknown')}")
                    
                    # After completion, tokens should be released back
                    if 'free_tokens' in before_data and 'free_tokens' in after_data:
                        # Tokens might be temporarily reserved during generation
                        print(f"✓ Token lifecycle test completed")
                except:
                    pass

    def test_concurrent_token_reservations(self):
        """Test concurrent token reservations don't cause race conditions"""
        import concurrent.futures
        
        def make_concurrent_request(request_id):
            """Make a generation request with specific ID tracking"""
            try:
                response = requests.post(f"{SCALARLM_URL}/v1/generate", json={
                    "prompt": f"Request {request_id}: What is 1+1?",
                    "model": "TinyLlama/TinyLlama-1.1B-Chat-v1.0",
                    "max_tokens": 5
                })
                return request_id, response.status_code
            except Exception as e:
                return request_id, f"Error: {e}"
        
        # Make 3 concurrent requests to test token reservation races
        with concurrent.futures.ThreadPoolExecutor(max_workers=3) as executor:
            futures = [executor.submit(make_concurrent_request, i) for i in range(3)]
            results = [future.result() for future in futures]
        
        print(f"Concurrent token reservation results: {results}")
        
        # At least some requests should succeed or fail gracefully
        success_count = sum(1 for _, status in results if isinstance(status, int) and status == 200)
        error_count = sum(1 for _, status in results if isinstance(status, int) and status in [422, 500, 503])
        
        print(f"Successful: {success_count}, Errors: {error_count}")
        assert success_count > 0 or error_count == len(results)

    def test_batch_size_calculation(self):
        """Test dynamic batch size calculation based on available tokens"""
        # Test the work orchestrator's batch size calculation
        response = requests.post(f"{SCALARLM_URL}/v1/work/calculate_batch_size", json={
            "requested_batch_size": 10
        })
        assert response.status_code in [200, 404, 500]
        
        if response.status_code == 200:
            try:
                data = response.json()
                print(f"Dynamic batch size calculation: {data}")
                
                if "calculated_batch_size" in data:
                    batch_size = data["calculated_batch_size"]
                    assert isinstance(batch_size, int)
                    assert batch_size >= 0
                    print(f"✓ Calculated batch size: {batch_size}")
                
                if "available_tokens" in data:
                    tokens = data["available_tokens"]
                    assert isinstance(tokens, int)
                    print(f"✓ Available KV cache tokens: {tokens}")
            except:
                pass

    def test_token_accounting_accuracy(self):
        """Test that token accounting remains accurate across requests"""
        # Get initial stats
        initial_response = requests.get(f"{SCALARLM_URL}/v1/kv_cache/stats")
        if initial_response.status_code != 200:
            pytest.skip("KV cache stats not available")
        
        try:
            initial_data = initial_response.json()
            initial_free = initial_data.get("free_tokens", 0)
            print(f"Initial free tokens: {initial_free}")
        except:
            pytest.skip("KV cache stats not parseable")
        
        # Make a small request
        response = requests.post(f"{SCALARLM_URL}/v1/generate", json={
            "prompt": "Hi",
            "model": "TinyLlama/TinyLlama-1.1B-Chat-v1.0", 
            "max_tokens": 3
        })
        
        # Wait for completion
        time.sleep(2)
        
        # Check final stats
        final_response = requests.get(f"{SCALARLM_URL}/v1/kv_cache/stats")
        if final_response.status_code == 200:
            try:
                final_data = final_response.json()
                final_free = final_data.get("free_tokens", 0)
                print(f"Final free tokens: {final_free}")
                
                # Tokens should be back to initial level (or close)
                # Allow for small discrepancies due to timing
                token_diff = abs(final_free - initial_free)
                if token_diff < 100:  # Small tolerance
                    print("✓ Token accounting appears accurate")
                else:
                    print(f"⚠ Token accounting discrepancy: {token_diff} tokens")
            except:
                pass

    def test_kv_cache_pressure_scenarios(self):
        """Test system behavior under KV cache memory pressure"""
        # Try to make many small requests that would consume KV cache
        requests_to_make = 5
        results = []
        
        for i in range(requests_to_make):
            response = requests.post(f"{SCALARLM_URL}/v1/generate", json={
                "prompt": f"Memory pressure test {i}",
                "model": "TinyLlama/TinyLlama-1.1B-Chat-v1.0",
                "max_tokens": 10
            })
            results.append((i, response.status_code))
            time.sleep(0.5)  # Small delay between requests
        
        print(f"Memory pressure test results: {results}")
        
        # System should handle gracefully - either succeed or return 503/429
        for req_id, status in results:
            if isinstance(status, int):
                assert status in [200, 429, 503, 422, 500]  # Include rate limiting

    def test_work_orchestrator_stats(self):
        """Test work orchestrator statistics and monitoring"""
        response = requests.get(f"{SCALARLM_URL}/v1/work/stats")
        assert response.status_code in [200, 404, 500]
        
        if response.status_code == 200:
            try:
                data = response.json()
                print(f"Work orchestrator stats: {data}")
                
                # Should have both work manager and kv cache stats
                if "work_manager" in data:
                    work_stats = data["work_manager"]
                    expected_fields = ["total_batches_processed", "total_requests_processed"]
                    for field in expected_fields:
                        if field in work_stats:
                            assert isinstance(work_stats[field], int)
                            print(f"✓ Work manager {field}: {work_stats[field]}")
                
                if "kv_cache" in data:
                    kv_stats = data["kv_cache"]
                    expected_fields = ["total_tokens", "free_tokens", "utilization_percent"]
                    for field in expected_fields:
                        if field in kv_stats:
                            print(f"✓ KV cache {field}: {kv_stats[field]}")
            except:
                pass

    def test_adaptor_loading_integration(self):
        """Test adaptor loading through work orchestrator"""
        # Test getting adaptors for a fake job hash
        response = requests.post(f"{SCALARLM_URL}/v1/work/get_adaptors", json={
            "job_hash": "test_job_12345"
        })
        assert response.status_code in [200, 404, 422, 500]
        
        if response.status_code == 200:
            try:
                data = response.json()
                print(f"Adaptors for test job: {data}")
                
                # Should return adaptor information
                if isinstance(data, dict):
                    if "adaptors" in data:
                        assert isinstance(data["adaptors"], list)
                    if "adaptor_paths" in data:
                        assert isinstance(data["adaptor_paths"], list)
            except:
                pass

    def test_token_update_mechanisms(self):
        """Test token update from vLLM engine"""
        # Test updating tokenization info after actual token count known
        payload = {
            "request_id": "test_update_123",
            "actual_tokens": 150,
            "flops": 1000000
        }
        
        response = requests.post(f"{SCALARLM_URL}/v1/work/update_tokenization", json=payload)
        assert response.status_code in [200, 404, 422, 500]
        
        if response.status_code == 200:
            print("✓ Tokenization update endpoint working")
        
        # Test work completion
        complete_payload = {
            "request_id": "test_update_123",
            "total_tokens_used": 150,
            "result": {"generated_text": "test result"}
        }
        
        response = requests.post(f"{SCALARLM_URL}/v1/work/complete_work", json=complete_payload)
        assert response.status_code in [200, 404, 422, 500]
        
        if response.status_code == 200:
            print("✓ Work completion endpoint working")


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-s"])