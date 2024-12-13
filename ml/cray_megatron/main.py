from cray_infra.training.training_harness import TrainingHarness
from cray_megatron.megatron.megatron_trainer import MegatronTrainer

import logging

logging.basicConfig(level=logging.DEBUG)


def main():
    trainer = MegatronTrainer(TrainingHarness())
    trainer.train()


main()
