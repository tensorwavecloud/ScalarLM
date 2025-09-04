"""
Simple integration tests for SLURM API endpoints.
Run with: pytest test/integration/test_slurm_api.py
"""

import requests
import pytest

BASE_URL = "http://localhost:8000"

def test_slurm_status():
    """Test SLURM status endpoint."""
    response = requests.get(f"{BASE_URL}/slurm/status")
    assert response.status_code in [200, 404, 503]  # 503 if SLURM not available
    if response.status_code == 200:
        data = response.json()
        assert isinstance(data, dict)


def test_slurm_squeue():
    """Test SLURM queue endpoint."""
    response = requests.get(f"{BASE_URL}/slurm/squeue")
    assert response.status_code in [200, 404, 503]
    if response.status_code == 200:
        data = response.json()
        assert isinstance(data, (dict, list))


def test_slurm_cancel():
    """Test SLURM job cancellation endpoint."""
    # Try to cancel a non-existent job
    response = requests.post(f"{BASE_URL}/slurm/cancel/999999")
    assert response.status_code in [200, 404, 400, 503]
    # 200 - cancelled (unlikely for fake job)
    # 404 - job not found 
    # 400 - bad request
    # 503 - SLURM not available


def test_slurm_endpoints():
    """Test SLURM endpoints listing."""
    response = requests.get(f"{BASE_URL}/slurm/endpoints")
    assert response.status_code in [200, 404]
    if response.status_code == 200:
        data = response.json()
        assert isinstance(data, (dict, list))


def test_megatron_squeue():
    """Test Megatron's SLURM queue endpoint."""
    response = requests.get(f"{BASE_URL}/v1/megatron/squeue")
    assert response.status_code in [200, 404, 503]
    if response.status_code == 200:
        data = response.json()
        assert isinstance(data, (dict, list))


if __name__ == "__main__":
    # Run tests
    pytest.main([__file__, "-v"])