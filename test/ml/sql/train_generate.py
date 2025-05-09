import scalarlm
import logging
import os
import time
import json
import argparse

logger = logging.getLogger(__name__)

logging.basicConfig(level=logging.INFO)

def get_directory():
    # Get the directory of the current file
    current_dir = os.path.dirname(os.path.abspath(__file__))

    return current_dir

def get_datasets(num_examples=1):
    # Load data from file
    path = os.path.join(get_directory(), 'data.json')
    data = []
    with open(path, 'r') as file:
        data = json.load(file)
    
    dataset = [example['input'] for example in data[:num_examples]]
    gold_dataset = [example['output'] for example in data[:num_examples]]
    
    train_dataset = []
    count = 0
    for example in data:
        for _ in range(10):
            train_dataset.append(example)
        count += 1
        if count >= num_examples:
            break

    return dataset, train_dataset, gold_dataset

def parse_args():
    
    parser = argparse.ArgumentParser(description="Parse number of GPUs and examples.")
    parser.add_argument('--num_gpus', type=int, default=1, help='Number of GPUs to use (default: 1)')
    parser.add_argument('--num_examples', type=int, default=1, help='Number of examples to process (default: 1)')
    return parser.parse_args()

def run_test():
    
    args = parse_args()
    # 0. VLLM Health up
    llm = scalarlm.SupermassiveIntelligence()

    results = llm.health()

    logger.info(results)

    dataset, train_dataset, gold_dataset = get_datasets(args.num_examples)

    # 1. Call generate on base model
    base_model_generate_results = llm.generate(prompts=dataset, max_tokens=6000)
    logger.info(f"Base model on prompt {dataset} returned {base_model_generate_results}")

    # 2. Train a base model with small dataset
    training_response = llm.train(
        train_dataset,
        train_args={"max_steps": 5 * len(train_dataset), "learning_rate": 4e-4, "max_token_block_size": 4096, "gpus": args.num_gpus},
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

    # 4. Wait to allow for auto-registration of the new trained model
    while training_response["deployed"] == False:
        logger.debug(f"Waiting for model {tuned_model_name} to be deployed.")
        time.sleep(10)
        training_response = llm.get_training_job(job_hash)

    logger.info(f"Model {tuned_model_name} is deployed.")

    # 4. Generate response on trained model
    trained_model_generate_results = llm.generate(
        prompts=dataset, model_name=tuned_model_name, max_tokens=6000
    )
    logger.info(
        f"Trained model on prompt {dataset} returned {trained_model_generate_results}"
    )
    # 5. Make sure base model gives different answer than the trained model
    assert base_model_generate_results != trained_model_generate_results
    # 6. Make sure trained model gives the expected answers
    assert trained_model_generate_results == gold_dataset


# Ensure the main function is only executed when the script is run directly
if __name__ == "__main__":
    run_test()
