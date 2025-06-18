import scalarlm

scalarlm.api_url = "https://llama70b.cray-lm.com"


def main():
    setup_logging()

    dataset = load_dataset()

    model_name = None
    max_iterations = 10
    target_accuracy = 0.95

    accuracy = eval_llm(dataset, model_name)

    for i in range(max_iterations):
        trajectories = try_reasoning(dataset, model_name)
        selected_trajectories = judge_trajectories(trajectories)

        model_name = train_llm(selected_trajectories)

        accuracy = eval_llm(dataset, model_name)

        if accuracy >= target_accuracy:
            break


import json
import sqlite3
import os
import time
import logging
import copy


logger = logging.getLogger(__name__)

def setup_logging():
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(levelname)s - %(message)s",
    )


def load_dataset():
    path = os.path.join(get_directory(), "mini-bird.json")

    with open(path, "r") as f:
        data = json.load(f)

    return data


def eval_llm(dataset, model_name):
    logger.info(f"Evaluating model {model_name} on {len(dataset)} examples")

    prompts = make_text2sql_prompts(dataset)

    llm = scalarlm.SupermassiveIntelligence()

    responses = llm.generate(prompts=prompts, model_name=model_name, max_tokens=512)

    accuracy = score_responses(responses, dataset)

    logger.info(f"Model {model_name} has accuracy {accuracy*100:.2f}% on {len(dataset)} examples")

    return accuracy


def make_text2sql_prompts(examples):

    prompts = []

    for example in examples:
        prompt = make_text2sql_prompt(example)
        prompts.append(prompt)

    return prompts


def make_text2sql_prompt(example):

    prompt = make_base_prompt(example)

    prompt += "The question is:\n"
    prompt += f"`{example['question']}`\n"

    prompt += "<|eot_id|><|start_header_id|>assistant<|end_header_id|>"

    return prompt


def get_table_info(sqlite3_connection):
    # Get all table names and column names
    cursor = sqlite3_connection.cursor()
    cursor.execute("SELECT name FROM sqlite_master WHERE type='table';")
    tables = cursor.fetchall()

    table_info = {}
    for table in tables:
        table_name = table[0]
        cursor.execute(f"PRAGMA table_info('{table_name}')")
        columns = cursor.fetchall()

        for index, column in enumerate(columns):

            columns[index] = list(column)

            if column[1].find("-") != -1:
                columns[index][1] = '"' + column[1] + '"'
            elif column[1].find(" ") != -1:
                columns[index][1] = "'" + column[1] + "'"

        table_info[table_name] = [
            {"column": column[1], "type": column[2]} for column in columns
        ]

    return table_info


def make_create_table_statement(table_name, columns):
    column_statements = []
    for column in columns:
        column_statements.append(f"{column['column']} {column['type']}")

    return f"CREATE TABLE {table_name} ({', '.join(column_statements)})"


def score_responses(responses, dataset):
    correct_query_count = 0

    for response, example in zip(responses, dataset):
        sql_query = extract_sql(response)

        sql_results, sql_failed = execute_sql(sql_query, example["db_id"])

        expected_sql = example["SQL"]

        expected_sql_results, expected_sql_failed = execute_sql(
            expected_sql, example["db_id"]
        )

        if sql_failed or expected_sql_failed:
            continue

        if sql_results == expected_sql_results:
            correct_query_count += 1

    accuracy = correct_query_count / len(responses)

    return accuracy


def extract_sql(response):
    # If there is a ```sql block, grab the content
    if "```sql" in response:
        response = response.split("```sql")[-1]
        response = response.split("```")[0]

    # Grab the final query if there are multiple separated by ;
    response = response.split(";")[0]

    return response.strip()


def execute_sql(query, database_name):

    database_path = os.path.join(get_directory(), database_name + ".sqlite")

    failed = False
    try:
        connection = sqlite3.connect(database_path)
        cursor = connection.cursor()
        cursor.execute(query)
    except Exception as e:
        logger.debug(f"Error executing query: {query}")
        logger.debug(f"Error message: {str(e)}")
        result = str(e)
        failed = True

    result = str(cursor.fetchall())

    return limit_length(result), failed

def get_directory():
    # Get the directory of the current file
    current_dir = os.path.dirname(os.path.abspath(__file__))

    return current_dir


def limit_length(string):
    max_len = 100
    if len(string) > max_len:
        return string[:max_len] + "..."

    return string


def try_reasoning(dataset, model_name):
    logger.info(f"Adding reasoning to model {model_name}")

    prompts = make_reasoning_prompts(dataset)

    llm = scalarlm.SupermassiveIntelligence()

    responses = llm.generate(prompts=prompts, model_name=model_name, max_tokens=512)

    trajectories = make_trajectories(responses, dataset)

    return trajectories


def make_reasoning_prompts(examples):
    prompts = []

    for example in examples:
        prompt = make_reasoning_prompt(example)
        prompts.append(prompt)

    return prompts


def make_reasoning_prompt(example):
    prompt = make_base_prompt(example)

    prompt += "Think step by step to answer this question.\n"
    prompt += f"First, explain in plain English how you would answer the question in at most 3 sentences.\n"
    prompt += f"Second, generate a SQL query to answer this question: `{example['question']}`\n"
    prompt += "Use simple and clean sqlite3 syntax.\n"
    prompt += (
        " i.e. Use CAST(numerator AS REAL) / denominator when computing averages.\n"
    )
    prompt += " i.e. Include AS in FROM and JOIN clauses to alias tables.\n"
    prompt += " i.e. Answer the question with only the requested column.\n"
    prompt += " i.e. Use JOIN to combine tables when necessary.\n"
    prompt += (
        " i.e. Escape column names with space with backticks, e.g. `column name`.\n"
    )
    prompt += "Write the SQL query in a ```sql block.\n"
    prompt += "<|eot_id|><|start_header_id|>assistant<|end_header_id|>"

    return prompt


def make_base_prompt(example):
    prompt = "<|begin_of_text|><|start_header_id|>user<|end_header_id|>"

    prompt += "\n"
    prompt += "The database schema is as follows:\n"

    database_path = os.path.join(get_directory(), example["db_id"] + ".sqlite")
    connection = sqlite3.connect(database_path)
    table_info = get_table_info(connection)

    table_prompt = ""

    for table, columns in table_info.items():
        create_table_statement = make_create_table_statement(table, columns)
        table_prompt += create_table_statement + "\n"

    # Limit the table max length to 2048 characters
    if len(table_prompt) > 2048:
        table_prompt = table_prompt[:2048]

    prompt += table_prompt

    if len(example["evidence"]) > 0:
        prompt += f"`{example['evidence']}`\n"

    prompt += "\n"

    return prompt


def make_trajectories(responses, dataset):
    trajectories = []

    for response, example in zip(responses, dataset):
        sql_query = extract_sql(response)
        reasoning = response.split("```sql")[0].strip()

        sql_results, sql_failed = execute_sql(sql_query, example["db_id"])

        expected_sql = example["SQL"]

        expected_sql_results, expected_sql_failed = execute_sql(
            expected_sql, example["db_id"]
        )

        example["generated_sql"] = sql_query
        example["reasoning"] = reasoning

        example["results_match"] = sql_results == expected_sql_results
        example["sql_failed"] = sql_failed
        example["expected_sql_failed"] = expected_sql_failed

        trajectories.append(example)

    return trajectories


def judge_trajectories(trajectories):
    selected_trajectories = []

    for trajectory in trajectories:
        if trajectory["sql_failed"]:
            continue

        if trajectory["expected_sql_failed"]:
            continue

        if not trajectory["results_match"]:
            continue

        selected_trajectories.append(trajectory)

    logger.info(f"Selected {len(selected_trajectories)} trajectories for training")

    training_data = make_training_data(selected_trajectories)

    return training_data


def make_training_data(trajectories):
    training_data = []

    for trajectory in trajectories:
        training_example = copy.deepcopy(trajectory)

        training_example["input"] = make_training_input(training_example)
        training_example["output"] = make_training_output(training_example)

        training_data.append(training_example)

    return training_data


def make_training_input(example):
    input = make_base_prompt(example)
    input += "\n"
    input += "The question is:\n"
    input += f"`{example['question']}`\n"
    input += "\n"
    input += "<|eot_id|><|start_header_id|>assistant<|end_header_id|>"
    return input


def make_training_output(example):
    output = example["reasoning"] + "\n"
    output += "```sql\n"
    output += example["generated_sql"] + "\n"
    output += "```\n"
    output += "<|eot_id|>"
    return output


def train_llm(dataset):
    llm = scalarlm.SupermassiveIntelligence()

    dataset_size = len(dataset)
    max_steps = dataset_size * 10

    status = llm.train(
        dataset,
        train_args={
            "max_steps": max_steps,
            "learning_rate": 4e-4,
            "gpus": 2,
            "timeout": 60 * 60 * 4 * 3,
            "max_token_block_size": 4096,
            "steps_per_checkpoint": 1000,
        },
    )

    job_hash = os.path.basename(status["job_status"]["job_directory"])
    tuned_model_name = status["job_status"]["model_name"]

    logger.info(
        f"Launched training job {job_hash} has status {status['job_status']['status']}"
    )

    status = wait_for_training_to_complete(llm, job_hash)

    logger.info(f"Training job {job_hash} finished with {status}")

    return tuned_model_name


def wait_for_training_to_complete(llm, job_hash):
    while True:
        training_response = llm.get_training_job(job_hash)
        training_status = training_response["job_status"]["status"]
        logger.debug(f"Training job {job_hash} has status {training_status}")

        if training_status == "FAILED":
            raise RuntimeError(f"Training job {job_hash} has failed, please check the logs")

        if training_status == "COMPLETED":
            break

        time.sleep(10)

    logger.info(f"Training job {job_hash} has completed successfully, waiting for model to be registered")

    # 4. Wait for deployment of the pre-trained model
    training_response = llm.get_training_job(job_hash)

    while training_response["deployed"] is False:
        logger.debug(f"Model {job_hash} has not been registered yet, sleeping for 10 seconds")
        time.sleep(10)
        training_response = llm.get_training_job(job_hash)

    logger.info(f"Model {job_hash} has been registered successfully")

    return training_response



main()
