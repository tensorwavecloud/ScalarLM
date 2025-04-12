from cray_infra.training.training_harness import TrainingHarness
from cray_infra.training.training_job_status import TrainingJobStatus
from cray_infra.training.print_logo import print_logo

from cray_megatron.megatron.training_loop import TrainingLoop, get_max_steps

import sys

import logging

logger = logging.getLogger(__name__)


class MegatronTrainer:
    def __init__(self, training_harness: TrainingHarness):
        self.training_harness = training_harness

    def train(self):
        self.train_loop()

    def train_loop(self):
        self.training_harness.update_status(
            status=TrainingJobStatus.TRAINING, metadata={"max_steps": get_max_steps()}
        )

        print_logo()

        TrainingLoop(self.training_harness).train()

        self.training_harness.update_status(status=TrainingJobStatus.COMPLETED)
