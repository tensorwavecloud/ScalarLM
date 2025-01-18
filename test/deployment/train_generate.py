import masint
import logging
import os
import time

logger = logging.getLogger(__name__)

logging.basicConfig(level=logging.DEBUG)

TINY_BASE_MODEL = "masint/tiny-random-llama"
TEST_QUESTION = "What is 0 + 0"

def create_training_set():
    dataset = []
    count = 1
    for i in range(count):
        dataset.append(
            {
                "input": f"What is {i} + {i}",
                "output": "The answer is " + str(i+i),
            }
        )

    return dataset

def get_dataset():
    dataset = []
    dataset.append(TEST_QUESTION)
    return dataset


def run_test():
    # 0. VLLM Health up
    llm = masint.SupermassiveIntelligence()

    results = llm.health()

    logger.info(results)

    dataset = get_dataset()

    # 1. Call generate on base model
    base_model_generate_results = llm.generate(
        prompts=dataset,
        model_name=TINY_BASE_MODEL
       )
    logger.info(
        f"Base model on prompt {dataset} returned {base_model_generate_results}"
        )

    # 2. Train a base model with small dataset
    training_response = llm.train(create_training_set(), train_args={"max_steps": 100})
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

        time.sleep(1)

    training_response = llm.get_training_job(job_hash)
    logger.info(f"Training status {training_response}.")

    # 4. Wait ~30 seconds to allow the registration of the new pretrained model
    time.sleep(30)	

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
    assert pretrained_model_generate_results == ['The answer is 0']    

def main():
    run_test()
    

# Ensure the main function is only executed when the script is run directly
if __name__ == "__main__":
    main()
