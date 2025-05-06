import scalarlm
import logging
import os
import time
import random

logger = logging.getLogger(__name__)

logging.basicConfig(level=logging.DEBUG)

count = 3
selected = random.randint(1, count)

TINY_BASE_MODEL = "masint/tiny-random-llama"
TEST_QUESTION = f"What is {selected} + {selected}?"
TEST_ANSWER = f"The answer is {selected + selected}."


def create_training_set():
    dataset = []
    for i in range(1, count + 1):
        dataset.append(
            {
                "input": f"What is {i} + {i}?",
                "output": "The answer is " + str(i + i) + ".",
            }
        )

    return dataset


def get_dataset():
    dataset = []
    dataset.append(TEST_QUESTION)
    return dataset


def run_test():
    # 0. VLLM Health up
    llm = scalarlm.SupermassiveIntelligence()

    results = llm.health()

    logger.info(results)

    dataset = get_dataset()

    # 1. Call generate on base model
    base_model_generate_results = llm.generate(
        prompts=dataset, model_name=TINY_BASE_MODEL
    )
    logger.info(
        f"Base model on prompt {dataset} returned {base_model_generate_results}"
    )

    # 2. Train a base model with small dataset
    training_response = llm.train(
        create_training_set(),
        train_args={"max_steps": (count * 50), "learning_rate": 3e-3, "gpus": 1, "max_gpus": 1},
    )
    logger.info(training_response)

    job_hash = os.path.basename(training_response["job_status"]["job_directory"])
    logger.debug(f"Created a training job: {job_hash}")

    training_status = training_response["job_status"]["status"]
    tuned_model_name = training_response["job_status"]["model_name"]
    logger.debug(f"Created a trained model: {tuned_model_name}")

    # 3. Wait till training is complete
    while True:
        training_response = llm.get_training_job(job_hash)
        training_status = training_response["job_status"]["status"]
        logger.debug(f"Training job {job_hash} has status {training_status}")

        if training_status == "COMPLETED":
            break

        time.sleep(10)

    training_response = llm.get_training_job(job_hash)
    # logger.info(f"Training status {training_response}.")

    # 4. Wait ~30 seconds to allow for auto-registration of the new pretrained model
    while training_response["deployed"] == False:
        logger.debug(f"Waiting for model {tuned_model_name} to be deployed.")
        time.sleep(10)
        training_response = llm.get_training_job(job_hash)

    logger.info(f"Model {tuned_model_name} is deployed.")

    # 4. Generate response on pretrained model
    pretrained_model_generate_results = llm.generate(
        prompts=dataset, model_name=tuned_model_name
    )
    logger.info(
        f"Trained model on prompt {dataset} returned {pretrained_model_generate_results}"
    )
    # 5. Compare and make sure based model and pretrained model have different responses
    assert base_model_generate_results != pretrained_model_generate_results
    # 6. Make sure pretrained model gives the expected answer
    assert pretrained_model_generate_results == [TEST_ANSWER]


def main():
    run_test()


# Ensure the main function is only executed when the script is run directly
if __name__ == "__main__":
    main()
