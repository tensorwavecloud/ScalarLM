from cray_infra.training.training_harness import TrainingHarness
from cray_infra.training.training_job_status import TrainingJobStatus
from cray_infra.huggingface.get_hf_token import get_hf_token

import traceback
import sys
import os
from gpu_aware_mpi import finalize_mpi


def print_exception():
    exc_type, exc_value, exc_traceback = sys.exc_info()
    traceback.print_exception(exc_type, exc_value, exc_traceback)


try:
    from cray_megatron.megatron.megatron_trainer import MegatronTrainer
except Exception as e:
    print_exception()

import signal
import logging


def main():

    harness = TrainingHarness()

    os.environ["HUGGING_FACE_HUB_TOKEN"] = get_hf_token()

    try:
        setup_logging()
        setup_signal_handler(harness)

        trainer = MegatronTrainer(training_harness=harness)
        trainer.train()
    except Exception as e:
        print_exception()
        harness.update_status(
            status=TrainingJobStatus.FAILED, metadata={"error": str(e)}
        )
        raise e

    finalize_mpi()

def setup_logging():
    logging.basicConfig(level=logging.DEBUG)

    logging.getLogger("filelock").setLevel(logging.WARNING)
    logging.getLogger("cray_infra.training.distribution_strategy.fsdp.fsdp").setLevel(
        logging.INFO
    )

def setup_signal_handler(harness):
    def signal_handler(sig, frame):
        logger.warn("Received signal: ", sig)
        harness.update_status(status=TrainingJobStatus.QUEUED)

        sys.exit(0)

    signal.signal(signal.SIGCONT, signal_handler)


main()
