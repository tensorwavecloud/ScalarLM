from masint.api.supermassive_intelligence import SupermassiveIntelligence

import traceback

import logging

logger = logging.getLogger(__name__)


def cancel(model_name):

    logger.info(f"Cancelling model: {model_name}")

    try:
        smi = SupermassiveIntelligence()
        response = smi.cancel(model_name)
        logger.info(f"Cancel response: {response}")
    except Exception as e:
        logger.error(f"Failed to cancel model: {model_name}")
        logger.error(e)
        logger.error(traceback.format_exc())

