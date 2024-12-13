from cray_infra.training.training_job_status import TrainingJobStatus
from cray_infra.training.training_harness import TrainingHarness
from cray_infra.training.print_logo import print_logo

from cray_megatron.megatron.tokenformer.tokenformer_trainer import TokenformerTrainer

import traceback
import sys

import logging

logger = logging.getLogger(__name__)


class MegatronTrainer:
    def __init__(self, training_harness: TrainingHarness):
        self.training_harness = training_harness

    def train(self):
        try:
            self.train_loop()
        except Exception as e:
            print_exception()
            self.training_harness.update_status(
                status=TrainingJobStatus.FAILED, metadata={"error": str(e)}
            )
            raise e

    def train_loop(self):
        self.training_harness.update_status(status=TrainingJobStatus.TRAINING)

        print_logo()

        TokenformerTrainer(self.training_harness).train()

        self.training_harness.update_status(status=TrainingJobStatus.COMPLETED)


def print_exception():
    exc_type, exc_value, exc_traceback = sys.exc_info()
    traceback.print_exception(exc_type, exc_value, exc_traceback)
