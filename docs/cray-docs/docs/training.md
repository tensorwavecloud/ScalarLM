# Training

## Training jobs

You can also use the Python client to submit training jobs to the Cray server.

```python

def get_dataset():
    dataset = []

    count = 5

    for i in range(count):
        dataset.append(
            {"input": f"What is {i} + {i}?", "output": str(i + i)}
        )

    return dataset


llm = masint.SupermassiveIntelligence()

dataset = get_dataset()

status = llm.train(dataset, train_args={"max_steps": 200, "learning_rate": 3e-3})

print(status)
```

You get a command line output like this:

```console
(environment) gregorydiamos@Air-Gregory cray % python test/deployment/train.py
{'job_id': '1', 'status': 'QUEUED', 'message': 'Training job launched', 'dataset_id': 'dataset', 'job_directory': '/app/cray/jobs/69118a251a074f9f9d37a2ddc903243e428d30c3c31ad019cbf62ac777e42e6e', 'model_name': '69118a251a074f9f9d37a2ddc903243e428d30c3c31ad019cbf62ac777e42e6e'}
```

