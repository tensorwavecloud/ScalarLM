import logging
import subprocess
import unittest

logger = logging.getLogger(__name__)


class TestSlurm(unittest.TestCase):
    def test_srun(self):
        run_command = [
            "srun",
            "hostname",
        ]
        result = subprocess.run(run_command, stdout=subprocess.PIPE)

        logger.debug(f"result: {result}")

        self.assertTrue(result.returncode == 0)
