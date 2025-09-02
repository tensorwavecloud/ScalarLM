import scalarlm
import logging
import os
import time
import random
import argparse
import shutil
from pathlib import Path

logger = logging.getLogger(__name__)

logging.basicConfig(level=logging.DEBUG)

count = 3
selected = random.randint(1, count)

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
    llm = scalarlm.SupermassiveIntelligence(api_url="http://localhost:8000")

    results = llm.health()

    logger.info(results)

    dataset = get_dataset()

    # 1. Call generate on base model
    base_model_generate_results = llm.generate(
        prompts=dataset
    )
    logger.info(
        f"Base model on prompt {dataset} returned {base_model_generate_results}"
    )

    # 2. Check SLURM queue status for resource contention
    import requests
    
    # Check if there are any running/pending jobs
    try:
        slurm_response = requests.get("http://localhost:8000/slurm/status")
        slurm_data = slurm_response.json()
        queue_output = slurm_data.get("queue", {}).get("squeue_output", "")
        
        # Parse queue output to check for running/pending jobs
        lines = queue_output.strip().split('\n')
        running_jobs = []
        pending_jobs = []
        
        for line in lines[1:]:  # Skip header
            if line.strip():
                parts = line.split()
                if len(parts) >= 5:
                    job_id = parts[0]
                    job_name = parts[2]
                    state = parts[4]
                    if state == "RUNNING":
                        running_jobs.append(f"Job {job_id} ({job_name})")
                    elif state == "PENDING":
                        pending_jobs.append(f"Job {job_id} ({job_name})")
        
        if running_jobs or pending_jobs:
            logger.warning("SLURM QUEUE STATUS:")
            if running_jobs:
                logger.warning(f"  Running jobs: {', '.join(running_jobs)}")
            if pending_jobs:
                logger.warning(f"  Pending jobs: {', '.join(pending_jobs)}")
            
            if pending_jobs:
                logger.error("❌ STOPPING TEST: There are pending jobs in the queue, indicating resource contention.")
                logger.error("   Wait for current jobs to complete or cancel them before running this test.")
                logger.error("   To check status: curl http://localhost:8000/slurm/status")
                return
            
            logger.warning("⚠️  Note: There are running jobs. New training job may queue behind them.")
        
    except Exception as e:
        logger.warning(f"Could not check SLURM status: {e}")

    # 3. Train a base model with small dataset
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

    # 4. Wait till training is complete
    while True:
        training_response = llm.get_training_job(job_hash)
        training_status = training_response["job_status"]["status"]
        logger.debug(f"Training job {job_hash} has status {training_status}")

        if training_status == "COMPLETED":
            break

        time.sleep(10)

    training_response = llm.get_training_job(job_hash)
    # logger.info(f"Training status {training_response}.")

    # 5. Wait to allow for auto-registration of the new pretrained model
    while training_response["deployed"] == False:
        logger.debug(f"Waiting for model {tuned_model_name} to be deployed.")
        time.sleep(10)
        training_response = llm.get_training_job(job_hash)

    logger.info(f"Model {tuned_model_name} is deployed.")

    # 6. Generate response on pretrained model
    pretrained_model_generate_results = llm.generate(
        prompts=dataset, model_name=tuned_model_name
    )
    logger.info(
        f"Trained model on prompt {dataset} returned {pretrained_model_generate_results}"
    )
    # 7. Make sure pretrained model gives the expected answer
    assert pretrained_model_generate_results == [TEST_ANSWER]


def main():
    parser = argparse.ArgumentParser(description='Train and generate test')
    parser.add_argument('--force-recreate', action='store_true', 
                       help='Delete existing trained models before training')
    args = parser.parse_args()
    
    if args.force_recreate:
        # Delete the specific cached model that this test would use
        # We need to predict what the model hash will be by looking for existing models
        # that match our training dataset
        training_job_dir = os.environ.get("TRAINING_JOB_DIRECTORY", "/app/cray/jobs")
        if os.path.exists(training_job_dir):
            # Look for any existing training job that might be from this test
            # We'll identify it by finding models trained on our specific dataset
            for item in os.listdir(training_job_dir):
                item_path = os.path.join(training_job_dir, item)
                if os.path.isdir(item_path) and len(item) == 64:  # SHA256 hash length
                    # Check if this looks like a model from our test (simple heuristic)
                    pt_files = list(Path(item_path).glob("*.pt"))
                    if pt_files:
                        logger.info(f"🗑️  --force-recreate: Found existing model {item}, removing {item_path}")
                        shutil.rmtree(item_path)
                        break  # Only remove the first one we find
                    
    run_test()


# Ensure the main function is only executed when the script is run directly
if __name__ == "__main__":
    main()
