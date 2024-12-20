from cray_infra.training.training_harness import TrainingHarness
from cray_megatron.megatron.megatron_trainer import MegatronTrainer

import logging



def main():
    setup_logging()

    trainer = MegatronTrainer(TrainingHarness())
    trainer.train()

def setup_logging():
    logging.basicConfig(level=logging.DEBUG)

    logging.getLogger("filelock").setLevel(logging.WARNING)


main()
