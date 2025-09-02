import unittest
import logging
import requests
import pytest

logger = logging.getLogger(__name__)


class TestVLLMHealth(unittest.TestCase):
    """
    Integration test for vLLM health endpoints.
    Requires a running vLLM container (e.g., via `./scalarlm up cpu`).
    """

    @classmethod
    def setUpClass(cls):
        """Check if vLLM container is running before running tests."""
        cls.base_url = "http://localhost:8000"
        
        try:
            # Quick connectivity test using the actual vLLM health endpoint
            response = requests.get(f"{cls.base_url}/v1/health", timeout=5)
            cls.container_available = True
            logger.info(f"✅ vLLM container detected at {cls.base_url}")
        except requests.exceptions.RequestException as e:
            cls.container_available = False
            logger.warning(f"⚠️  No vLLM container running at {cls.base_url}: {e}")

    def setUp(self):
        """Skip tests if container is not available."""
        if not self.container_available:
            self.skipTest("vLLM container not running. Start with: ./scalarlm up cpu")

    def test_v1_health_endpoint(self):
        """Test the /v1/health endpoint returns 200."""
        logger.debug("Testing /v1/health endpoint")
        
        response = requests.get(f"{self.base_url}/v1/health", timeout=10)
        self.assertEqual(response.status_code, 200)
        logger.info("✅ /v1/health endpoint responding")

    def test_openai_models_endpoint(self):
        """Test the /v1/models endpoint (OpenAI compatibility)."""
        logger.debug("Testing /v1/models endpoint")
        
        response = requests.get(f"{self.base_url}/v1/models", timeout=10)
        self.assertEqual(response.status_code, 200)
        
        data = response.json()
        self.assertIn("data", data)
        self.assertIsInstance(data["data"], list)
        logger.info(f"✅ /v1/models endpoint responding with {len(data['data'])} models")


# Allow running this test file directly
if __name__ == "__main__":
    unittest.main()
