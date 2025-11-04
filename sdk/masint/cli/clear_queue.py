from masint.api.supermassive_intelligence import SupermassiveIntelligence

import traceback

import logging

logger = logging.getLogger(__name__)


def clear_queue():

    logger.info(f"Clearing inference queue...")

    try:
        smi = SupermassiveIntelligence()
        response = smi.clear_queue()
        logger.info(f"Inference queue cleared: {response}")
    except Exception as e:
        logger.error(f"Failed to clear inference queue.")
        logger.error(e)
        logger.error(traceback.format_exc())
