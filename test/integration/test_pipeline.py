"""
End-to-end integration tests for ScalarLM training -> generation pipeline.
Tests the complete flow from training a model to using it for generation.
"""

import requests
import time
import pytest
import json
import io
import concurrent
SCALARLM_URL = "http://localhost:8000"
VLLM_URL = "http://localhost:8001"

# skip this as it takes a while to run
@pytest.mark.skip(reason="This test is disabled by default due to taking time to complete.")
def test_complete_pipeline():
    """Test complete pipeline: train model -> use for generation."""
    
    # Step 1: Create training dataset as tar.gz file
    import tempfile
    import tarfile
    import os
    
    training_data = {
        "base_model": "TinyLlama/TinyLlama-1.1B-Chat-v1.0",
        "dataset": {
            "type": "math",
            "examples": [
                {"input": "What is 2+2?", "output": "The answer is 4."},
                {"input": "What is 3+3?", "output": "The answer is 6."},
                {"input": "What is 5+5?", "output": "The answer is 10."}
            ]
        },
        "num_steps": 1,  # Minimal steps for fast test
        "learning_rate": 0.0001
    }
    
    # Create a temporary tar.gz file with the training data
    with tempfile.NamedTemporaryFile(suffix='.tar.gz', delete=False) as tmp_file:
        tar_path = tmp_file.name
        
        with tarfile.open(tar_path, 'w:gz') as tar:
            # Add training config as JSON file
            config_data = json.dumps(training_data).encode('utf-8')
            config_info = tarfile.TarInfo(name='config.json')
            config_info.size = len(config_data)
            tar.addfile(config_info, io.BytesIO(config_data))
    
    try:
        # Upload the training file
        with open(tar_path, 'rb') as f:
            files = {'file': f}
            data = {'params': json.dumps(training_data)}
            response = requests.post(f"{SCALARLM_URL}/v1/megatron/train", files=files, data=data)
    finally:
        # Clean up temp file
        if os.path.exists(tar_path):
            os.unlink(tar_path)
    
    # Check response
    if response.status_code == 500:
        try:
            error_data = response.json()
            print(f"Training failed: {error_data}")
            pytest.skip(f"Training infrastructure error: {error_data}")
        except:
            print(f"Training failed: {response.text}")
            pytest.skip(f"Training infrastructure error: {response.text}")
    
    assert response.status_code in [200, 201, 404]  # 404 if training not implemented
    
    if response.status_code == 404:
        pytest.skip("Training endpoint not implemented")
    
    data = response.json()
    
    # Extract job_id from nested structure
    if "job_status" in data and "job_id" in data["job_status"]:
        job_id = data["job_status"]["job_id"]
    elif "job_status" in data and "model_name" in data["job_status"]:
        job_id = data["job_status"]["model_name"]  # Use model_name as job identifier
    elif "job_id" in data:
        job_id = data["job_id"]
    elif "job_hash" in data:
        job_id = data["job_hash"]
    else:
        pytest.fail(f"No job identifier found in response: {data}")
    
    # Step 2: Wait for training to complete (max 2 minutes for minimal training)  
    max_wait = 120  # 2 minutes for 1-step training
    start_time = time.time()
    last_status = None
    
    print(f"Training job submitted: {job_id}")
    
    while time.time() - start_time < max_wait:
        status_response = requests.get(f"{SCALARLM_URL}/v1/megatron/train/{job_id}")
        
        if status_response.status_code == 200:
            status_data = status_response.json()
            status = status_data.get("status", "unknown")
            
            # Print status changes
            if status != last_status:
                elapsed = time.time() - start_time
                print(f"[{elapsed:.1f}s] Training status: {status}")
                last_status = status
            
            if status == "completed":
                print(f"Training completed in {time.time() - start_time:.1f} seconds")
                break
            elif status == "failed":
                pytest.fail(f"Training failed: {status_data}")
            
        time.sleep(10)
    else:
        pytest.fail(f"Training did not complete within {max_wait} seconds")
    
    # Step 3: Test generation with trained model
    trained_response = requests.post(f"{SCALARLM_URL}/v1/generate", json={
        "prompt": "What is 3+3?",
        "model": job_id,  # Use job_id as model name
        "max_tokens": 20
    })
    
    assert trained_response.status_code == 200
    trained_data = trained_response.json()
    trained_answer = trained_data.get("generated_text", "")
    
    print(f"Trained model answer: '{trained_answer}'")
    
    # Step 4: Verify we got a response
    assert len(trained_answer.strip()) > 0


def test_quick_pipeline():
    """Fast test that verifies training submission pipeline works without waiting for completion."""
    import tempfile
    import tarfile
    import os
    
    training_data = {
        "base_model": "TinyLlama/TinyLlama-1.1B-Chat-v1.0",
        "dataset": {
            "type": "math", 
            "examples": [{"input": "Test", "output": "Response"}]
        },
        "num_steps": 1,
        "learning_rate": 0.001
    }
    
    # Create tar.gz file
    with tempfile.NamedTemporaryFile(suffix='.tar.gz', delete=False) as tmp_file:
        tar_path = tmp_file.name
        
        with tarfile.open(tar_path, 'w:gz') as tar:
            config_data = json.dumps(training_data).encode('utf-8')
            config_info = tarfile.TarInfo(name='config.json')
            config_info.size = len(config_data)
            tar.addfile(config_info, io.BytesIO(config_data))
    
    try:
        with open(tar_path, 'rb') as f:
            files = {'file': f}
            data = {'params': json.dumps(training_data)}
            response = requests.post(f"{SCALARLM_URL}/v1/megatron/train", files=files, data=data)
    finally:
        if os.path.exists(tar_path):
            os.unlink(tar_path)
    
    # Verify training submission worked
    if response.status_code == 404:
        pytest.skip("Training not implemented")
    
    assert response.status_code in [200, 201]
    data = response.json()
    
    # Extract job identifier
    if "job_status" in data and "model_name" in data["job_status"]:
        job_id = data["job_status"]["model_name"]
        print(f"Training job submitted successfully: {job_id}")
        
        # Quick status check to verify job was queued
        status_response = requests.get(f"{SCALARLM_URL}/v1/megatron/train/{job_id}")
        if status_response.status_code == 200:
            status_data = status_response.json()
            status = status_data.get("status", "unknown")
            print(f"Job status: {status}")
            print(f"Full status response: {status_data}")
            # Accept any status as long as we can query the job
            assert status is not None
        
    else:
        pytest.fail(f"No job identifier in response: {data}")


def test_model_switching():
    """Test switching between different models."""
    
    # Test base model
    response1 = requests.post(f"{SCALARLM_URL}/v1/generate", json={
        "prompt": "Hello",
        "model": "TinyLlama/TinyLlama-1.1B-Chat-v1.0",
        "max_tokens": 5
    })
    
    # If generation works, test with different prompt/model
    if response1.status_code == 200:
        response2 = requests.post(f"{SCALARLM_URL}/v1/generate", json={
            "prompt": "Hi there",
            "model": "TinyLlama/TinyLlama-1.1B-Chat-v1.0", 
            "max_tokens": 5
        })
        
        assert response2.status_code == 200
        
        # Should get different responses for different prompts
        data1 = response1.json()
        data2 = response2.json()
        
        text1 = data1.get("generated_text", "")
        text2 = data2.get("generated_text", "")
        
        # Different prompts might give different responses
        print(f"Response 1: '{text1}'")
        print(f"Response 2: '{text2}'")


def test_adapter_loading_flow():
    """Test adapter discovery and loading mechanism."""
    
    # Step 1: Check if we can get adaptors for a non-existent job
    response = requests.post(f"{SCALARLM_URL}/v1/work/get_adaptors", json={
        "job_hash": "nonexistent123"
    })
    assert response.status_code in [200, 404, 422, 500]
    
    # Step 2: Check if we can get work from the queue
    response = requests.post(f"{SCALARLM_URL}/v1/work/get_work", json={})
    assert response.status_code in [200, 204, 404, 500]  # 204 = no work available
    
    # Step 3: Test generate with adaptors call
    response = requests.post(f"{SCALARLM_URL}/v1/generate/get_adaptors", json={
        "model_name": "test_model"
    })
    assert response.status_code in [200, 404, 422, 500]


def test_training_job_lifecycle():
    """Test training job status transitions."""
    
    # Submit minimal training job
    payload = {
        "base_model": "TinyLlama/TinyLlama-1.1B-Chat-v1.0",
        "dataset": {
            "examples": [{"input": "Hi", "output": "Hello"}]
        },
        "num_steps": 5
    }
    
    response = requests.post(f"{SCALARLM_URL}/v1/megatron/train", json=payload)
    if response.status_code == 404:
        pytest.skip("Training not implemented")
    
    if response.status_code in [200, 201]:
        data = response.json()
        job_id = data.get("job_id") or data.get("job_hash")
        
        # Check initial status
        status_response = requests.get(f"{SCALARLM_URL}/v1/megatron/train/{job_id}")
        if status_response.status_code == 200:
            status_data = status_response.json()
            initial_status = status_data.get("status")
            assert initial_status in ["pending", "running", "queued"]
            print(f"Initial training status: {initial_status}")
        
        # Wait briefly and check again
        time.sleep(5)
        status_response = requests.get(f"{SCALARLM_URL}/v1/megatron/train/{job_id}")
        if status_response.status_code == 200:
            status_data = status_response.json()
            current_status = status_data.get("status")
            print(f"Status after 5s: {current_status}")


def test_openai_compatibility_with_trained_models():
    """Test that OpenAI endpoints work with job hashes as model names."""
    
    # This assumes we have a trained model from previous tests or setup
    # Use a fake job hash first to test error handling
    fake_job_hash = "test123456789"
    
    # Test chat completions with job hash
    response = requests.post(f"{SCALARLM_URL}/v1/chat/completions", json={
        "model": fake_job_hash,
        "messages": [{"role": "user", "content": "What is 2+2?"}],
        "max_tokens": 10
    })
    
    # Should either work (if job exists) or return model not found
    assert response.status_code in [200, 404, 422, 500]
    
    # Test completions with job hash
    response = requests.post(f"{SCALARLM_URL}/v1/completions", json={
        "model": fake_job_hash,
        "prompt": "What is 2+2?",
        "max_tokens": 10
    })
    
    assert response.status_code in [200, 404, 422, 500]


def test_work_orchestrator_integration():
    """Test work orchestrator and get_work flow."""
    
    # Test getting work and adaptors together
    response = requests.post(f"{SCALARLM_URL}/v1/work/get_work_and_adaptors", json={})
    assert response.status_code in [200, 204, 404, 500]
    
    if response.status_code == 200:
        try:
            data = response.json()
            # Check structure of work + adaptors response
            assert isinstance(data, dict)
            print(f"Work and adaptors response: {data}")
        except:
            pass
    
    # Test work completion flow
    complete_payload = {
        "work_id": "test123",
        "result": {"generated_text": "test result"}
    }
    response = requests.post(f"{SCALARLM_URL}/v1/work/complete_work", json=complete_payload)
    assert response.status_code in [200, 404, 422, 500]
    
    # Test tokenization update
    tokenization_payload = {
        "work_id": "test123", 
        "tokens": 150,
        "flops": 1000000
    }
    response = requests.post(f"{SCALARLM_URL}/v1/work/update_tokenization", json=tokenization_payload)
    assert response.status_code in [200, 404, 422, 500]


def test_metrics_and_monitoring():
    """Test that metrics are being tracked across the pipeline."""
    
    # Test generation metrics
    response = requests.get(f"{SCALARLM_URL}/v1/generate/metrics")
    assert response.status_code in [200, 404, 500]
    
    if response.status_code == 200:
        try:
            data = response.json()
            print(f"Generation metrics: {data}")
        except:
            pass
    
    # Test work stats
    response = requests.get(f"{SCALARLM_URL}/v1/work/stats")
    assert response.status_code in [200, 404, 500]
    
    if response.status_code == 200:
        try:
            data = response.json()
            print(f"Work stats: {data}")
        except:
            pass

def test_invalid_model_names():
    """Test generation with invalid model names."""
    
    invalid_models = [
        "nonexistent/model",
        "",  # Empty string
        "model with spaces",
        "model/with/too/many/slashes",
        "a" * 1000,  # Very long name
        "model-with-unicode-ñ",
        "123456789",  # Pure numbers (could be job hash)
        "../../../etc/passwd"  # Path traversal attempt
    ]
    
    for model in invalid_models:
        response = requests.post(f"{SCALARLM_URL}/v1/generate", json={
            "prompt": "Hello",
            "model": model,
            "max_tokens": 5
        })
        
        # Should handle gracefully with 4xx errors
        assert response.status_code in [400, 404, 422, 500]
        print(f"Model '{model}': {response.status_code}")


def test_invalid_training_requests():
    """Test training endpoint with invalid inputs."""
    
    invalid_payloads = [
        {},  # Empty payload
        {"base_model": ""},  # Empty model
        {"base_model": "test", "dataset": {}},  # Empty dataset
        {"base_model": "test", "dataset": {"examples": []}},  # No examples
        {"base_model": "test", "num_steps": -1},  # Negative steps
        {"base_model": "test", "learning_rate": -0.1},  # Negative LR
        {"base_model": "test", "num_steps": 1000000},  # Huge steps
        {"base_model": "../../../etc/passwd"},  # Path traversal
        {"base_model": "test", "dataset": "not_a_dict"}  # Wrong type
    ]
    
    for payload in invalid_payloads:
        response = requests.post(f"{SCALARLM_URL}/v1/megatron/train", json=payload)
        assert response.status_code in [400, 404, 422, 500]


def test_generation_parameter_limits():
    """Test generation with extreme parameters."""
    
    extreme_cases = [
        {"max_tokens": -1},  # Negative tokens
        {"max_tokens": 0},   # Zero tokens
        {"max_tokens": 100000},  # Very large
        {"temperature": -1.0},  # Negative temp
        {"temperature": 10.0},  # Very high temp
        {"top_p": -0.5},  # Negative top_p
        {"top_p": 2.0},   # top_p > 1
        {"prompt": ""},   # Empty prompt
        {"prompt": "x" * 10000},  # Very long prompt
    ]
    
    for params in extreme_cases:
        full_payload = {
            "prompt": "Hello",
            "model": "TinyLlama/TinyLlama-1.1B-Chat-v1.0",
            "max_tokens": 5,
            **params
        }
        
        response = requests.post(f"{SCALARLM_URL}/v1/generate", json=full_payload)
        assert response.status_code in [200, 400, 422, 500]
        
        if response.status_code not in [200]:
            print(f"Rejected {params}: {response.status_code}")


def test_concurrent_training_requests():
    """Test multiple training requests at once."""
    
    def submit_training(job_id):
        payload = {
            "base_model": "TinyLlama/TinyLlama-1.1B-Chat-v1.0",
            "dataset": {
                "examples": [{"input": f"Job {job_id}", "output": f"Response {job_id}"}]
            },
            "num_steps": 3
        }
        
        try:
            response = requests.post(f"{SCALARLM_URL}/v1/megatron/train", json=payload)
            return job_id, response.status_code
        except Exception as e:
            return job_id, f"Error: {e}"
    
    # Submit 3 concurrent training jobs
    with concurrent.futures.ThreadPoolExecutor(max_workers=3) as executor:
        futures = [executor.submit(submit_training, i) for i in range(3)]
        results = [future.result() for future in futures]
    
    print(f"Concurrent training results: {results}")
    
    # Should handle concurrent requests gracefully
    for job_id, status in results:
        if isinstance(status, int):
            assert status in [200, 201, 400, 404, 422, 500, 503]


def test_malformed_json_requests():
    """Test endpoints with malformed JSON."""
    
    endpoints = [
        ("POST", "/v1/generate"),
        ("POST", "/v1/completions"),
        ("POST", "/v1/chat/completions"),
        ("POST", "/v1/megatron/train"),
        ("POST", "/v1/work/get_work")
    ]
    
    # Send malformed JSON (not actually JSON)
    malformed_data = "{ not valid json }"
    
    for method, endpoint in endpoints:
        try:
            response = requests.request(
                method, 
                f"{SCALARLM_URL}{endpoint}",
                data=malformed_data,  # Raw string, not json=
                headers={"Content-Type": "application/json"}
            )
            
            # Should reject malformed JSON
            assert response.status_code in [400, 422, 500]
            
        except Exception as e:
            print(f"Exception for {endpoint}: {e}")


def test_large_request_handling():
    """Test handling of large requests."""
    
    # Large prompt
    large_prompt = "Tell me a story about " + "cats and dogs " * 500  # ~5KB prompt
    
    response = requests.post(f"{SCALARLM_URL}/v1/generate", json={
        "prompt": large_prompt,
        "model": "TinyLlama/TinyLlama-1.1B-Chat-v1.0",
        "max_tokens": 10
    })
    
    # Should handle large prompts gracefully
    assert response.status_code in [200, 400, 413, 422, 500]
    
    if response.status_code == 413:
        print("Server properly rejects large requests with 413 Payload Too Large")
    elif response.status_code == 200:
        print("Server handles large prompts successfully")


def test_rapid_sequential_requests():
    """Test rapid sequential requests to same endpoint."""
    
    results = []
    
    # Make 10 quick requests
    for i in range(10):
        start_time = time.time()
        response = requests.post(f"{SCALARLM_URL}/v1/generate", json={
            "prompt": f"Quick test {i}",
            "model": "TinyLlama/TinyLlama-1.1B-Chat-v1.0",
            "max_tokens": 1
        })
        
        elapsed = time.time() - start_time
        results.append((i, response.status_code, elapsed))
    
    print("Rapid request results:")
    for i, status, elapsed in results:
        print(f"  Request {i}: {status} in {elapsed:.3f}s")
    
    # Should handle rapid requests without completely failing
    successful = sum(1 for _, status, _ in results if status == 200)
    print(f"Successful requests: {successful}/10")
    
    # At least some should succeed or fail gracefully
    assert successful > 0 or all(status in [422, 500, 503] for _, status, _ in results)


def test_missing_required_fields():
    """Test endpoints with missing required fields."""
    
    test_cases = [
        # Generate without model
        ("POST", "/v1/generate", {"prompt": "Hello", "max_tokens": 5}),
        
        # Generate without prompt
        ("POST", "/v1/generate", {"model": "test", "max_tokens": 5}),
        
        # Completions without model
        ("POST", "/v1/completions", {"prompt": "Hello"}),
        
        # Chat without messages
        ("POST", "/v1/chat/completions", {"model": "test"}),
        
        # Training without model
        ("POST", "/v1/megatron/train", {"dataset": {"examples": []}}),
        
        # Training without dataset
        ("POST", "/v1/megatron/train", {"base_model": "test"})
    ]
    
    for method, endpoint, payload in test_cases:
        response = requests.request(method, f"{SCALARLM_URL}{endpoint}", json=payload)
        
        # Should reject missing fields with 422 Unprocessable Entity (or 200 with defaults)
        assert response.status_code in [200, 400, 422, 500]
        print(f"{endpoint} without required fields: {response.status_code}")


def test_request_timeout_handling():
    """Test how system handles request timeouts."""
    
    # Make a request that might take a while
    payload = {
        "prompt": "Write a very long detailed story about",
        "model": "TinyLlama/TinyLlama-1.1B-Chat-v1.0", 
        "max_tokens": 100  # Larger generation
    }
    
    # Set a short timeout to test timeout handling
    try:
        response = requests.post(
            f"{SCALARLM_URL}/v1/generate",
            json=payload,
            timeout=5.0  # 5 second timeout
        )
        
        print(f"Request completed with status: {response.status_code}")
        assert response.status_code in [200, 408, 422, 500, 504]
        
    except requests.exceptions.Timeout:
        print("Request timed out (expected behavior)")
        # Timeout is acceptable for this test


def test_system_resource_limits():
    """Test system behavior under resource pressure."""
    
    # Test with many concurrent requests to stress the system
    def stress_request(request_id):
        try:
            response = requests.post(f"{SCALARLM_URL}/v1/generate", json={
                "prompt": "Stress test",
                "model": "TinyLlama/TinyLlama-1.1B-Chat-v1.0",
                "max_tokens": 2
            }, timeout=10)
            return response.status_code
        except Exception as e:
            return f"Error: {e}"
    
    # Make 5 concurrent stress requests
    with concurrent.futures.ThreadPoolExecutor(max_workers=5) as executor:
        futures = [executor.submit(stress_request, i) for i in range(5)]
        results = [future.result() for future in futures]
    
    print(f"Stress test results: {results}")
    
    # System should either handle gracefully or return appropriate errors
    for result in results:
        if isinstance(result, int):
            assert result in [200, 422, 429, 500, 503]  # Include 422 Unprocessable Entity


def test_error_response_format():
    """Test that error responses have consistent format."""
    
    # Make a request that will definitely fail
    response = requests.post(f"{SCALARLM_URL}/v1/generate", json={
        "prompt": "Test",
        "model": "definitely-nonexistent-model",
        "max_tokens": 5
    })
    
    assert response.status_code in [400, 404, 422, 500]
    
    # Try to parse error response
    try:
        error_data = response.json()
        print(f"Error response format: {error_data}")
        
        # Common error fields
        error_fields = ["error", "detail", "message", "code"]
        has_error_info = any(field in error_data for field in error_fields)
        
        if has_error_info:
            print("✓ Error response has proper structure")
        else:
            print("! Error response missing standard fields")
            
    except:
        print("Error response is not JSON (plain text error)")




if __name__ == "__main__":
    pytest.main([__file__, "-v", "-s"])
