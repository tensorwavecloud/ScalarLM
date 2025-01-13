from masint import SupermassiveIntelligence

import logging

logger = logging.getLogger(__name__)


def ls():
    logger.debug(f"Listing models")

    try:
        llm = SupermassiveIntelligence()

        models = llm.list_models()
    except Exception as e:
        logger.error(f"Failed to list models")
        logger.error(e)

    print_models(models["models"])


def print_models(models):
    for model in models:
        print(model)
