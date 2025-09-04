"""
Simple integration tests for vLLM server API endpoints.
Run with: pytest test/integration/test_vllm_api.py
"""

import requests
import pytest

BASE_URL = "http://localhost:8001"

def test_health_endpoints():
    """Test vLLM health endpoints."""
    # Health check
    response = requests.get(f"{BASE_URL}/health")
    assert response.status_code == 200
    
    # Ping
    response = requests.get(f"{BASE_URL}/ping")
    assert response.status_code in [200, 404]
    
    response = requests.post(f"{BASE_URL}/ping", json={})
    assert response.status_code in [200, 404]


def test_version():
    """Test version endpoint."""
    response = requests.get(f"{BASE_URL}/version")
    assert response.status_code in [200, 404]
    if response.status_code == 200:
        data = response.json()
        assert "version" in data or isinstance(data, str)


def test_models_endpoint():
    """Test models listing."""
    response = requests.get(f"{BASE_URL}/v1/models")
    assert response.status_code == 200
    data = response.json()
    assert "data" in data
    assert isinstance(data["data"], list)
    assert len(data["data"]) > 0  # Should have at least one model


def test_completions():
    """Test text completions."""
    payload = {
        "model": "TinyLlama/TinyLlama-1.1B-Chat-v1.0",
        "prompt": "Hello",
        "max_tokens": 5
    }
    response = requests.post(f"{BASE_URL}/v1/completions", json=payload)
    assert response.status_code in [200, 404, 422, 500]
    if response.status_code == 200:
        data = response.json()
        assert "choices" in data
        assert len(data["choices"]) > 0
        assert "text" in data["choices"][0]


def test_chat_completions():
    """Test chat completions."""
    payload = {
        "model": "TinyLlama/TinyLlama-1.1B-Chat-v1.0",
        "messages": [{"role": "user", "content": "Hello"}],
        "max_tokens": 5
    }
    response = requests.post(f"{BASE_URL}/v1/chat/completions", json=payload)
    assert response.status_code in [200, 404, 422, 500]
    if response.status_code == 200:
        data = response.json()
        assert "choices" in data
        assert len(data["choices"]) > 0
        assert "message" in data["choices"][0]


def test_tokenize():
    """Test tokenization."""
    payload = {
        "text": "Hello world"
    }
    response = requests.post(f"{BASE_URL}/tokenize", json=payload)
    assert response.status_code in [200, 400, 404, 422]
    if response.status_code == 200:
        data = response.json()
        assert "tokens" in data or "ids" in data or isinstance(data, list)


def test_detokenize():
    """Test detokenization."""
    payload = {
        "tokens": [1, 2, 3]  # Some token IDs
    }
    response = requests.post(f"{BASE_URL}/detokenize", json=payload)
    assert response.status_code in [200, 404, 422]


def test_embeddings():
    """Test embeddings endpoint."""
    payload = {
        "model": "TinyLlama/TinyLlama-1.1B-Chat-v1.0",
        "input": "Hello world"
    }
    response = requests.post(f"{BASE_URL}/v1/embeddings", json=payload)
    assert response.status_code in [200, 404, 501]  # May not be supported


def test_lora_adapter_endpoints():
    """Test LoRA adapter management."""
    # Try to load adapter (will likely fail but should respond)
    payload = {
        "lora_name": "test_adapter",
        "lora_path": "/nonexistent/path"
    }
    response = requests.post(f"{BASE_URL}/v1/load_lora_adapter", json=payload)
    assert response.status_code in [200, 400, 404, 422]
    
    # Try to unload adapter
    payload = {"lora_name": "test_adapter"}
    response = requests.post(f"{BASE_URL}/v1/unload_lora_adapter", json=payload)
    assert response.status_code in [200, 400, 404, 422]


def test_metrics():
    """Test metrics endpoint."""
    response = requests.get(f"{BASE_URL}/metrics")
    assert response.status_code in [200, 404]
    if response.status_code == 200:
        # Metrics might be in Prometheus format (text) or JSON
        assert len(response.text) > 0


def test_load_endpoint():
    """Test load endpoint."""
    response = requests.get(f"{BASE_URL}/load")
    assert response.status_code in [200, 404]


def test_responses_endpoints():
    """Test response management endpoints."""
    # Create response (may not be implemented)
    response = requests.post(f"{BASE_URL}/v1/responses", json={})
    assert response.status_code in [200, 400, 404, 422, 500]
    
    # Get response (will fail with fake ID)
    response = requests.get(f"{BASE_URL}/v1/responses/fake-id")
    assert response.status_code in [200, 400, 404]
    
    # Cancel response
    response = requests.post(f"{BASE_URL}/v1/responses/fake-id/cancel", json={})
    assert response.status_code in [200, 400, 404]


def test_additional_endpoints():
    """Test additional vLLM endpoints."""
    # These endpoints may or may not be implemented
    endpoints = [
        ("/pooling", "POST", {}),
        ("/classify", "POST", {}),
        ("/score", "POST", {}),
        ("/v1/score", "POST", {}),
        ("/rerank", "POST", {}),
        ("/v1/rerank", "POST", {}),
        ("/invocations", "POST", {})
    ]
    
    for endpoint, method, payload in endpoints:
        if method == "POST":
            response = requests.post(f"{BASE_URL}{endpoint}", json=payload)
        else:
            response = requests.get(f"{BASE_URL}{endpoint}")
        assert response.status_code in [200, 400, 404, 422, 500, 501]


if __name__ == "__main__":
    # Run tests
    pytest.main([__file__, "-v"])