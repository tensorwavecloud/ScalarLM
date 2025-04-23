# ScalarLM Frequently Asked Questions

## Training and Fine-tuning

#### Do I need special templates (special tokens) for training or inference prompts?
Yes. The training and inference prompts follow the same format as the prompt template for the model published on HuggingFace. Each deployment is tied to a specific model, refer to the model card for the prompt template. For example, [this](https://www.llama.com/docs/model-cards-and-prompt-formats/llama3_1/) is the template for the ScalarLM deployment that supports Llama8B Instruct. 

### How can I monitor fine-tuning progress, such as loss function curves?
Use the `scalarlm plot` CLI command after setting your API URL. We provide a pip-installable tool called `scalarlm` for this purpose.

## Job Management

### What happens if I launch 5 fine‑tuning jobs at once?
Jobs are queued automatically by the framework's built‑in scheduler—you don't need to implement your own queue.

## Training Parameters

### How do I change the "chunking" (token‑block) size?
Pass `max_token_block_size` in `train_args`:
```python
llm.train(
    dataset,
    train_args={
        "max_steps": 200,
        "learning_rate": 3e-4,
        "max_token_block_size": 1024
    }
)
```

### Can I change the quantization / dtype?
Yes — also via `train_args`:
```python
llm.train(
    ...,
    train_args={
        "max_steps": count*50,
        "learning_rate": 3e-3,
        "gpus": 1,
        "dtype": "float32"
    }
)
```
At inference, VLLM will use the dtype defined in the default configuration at the time of deployment. When this type differs from the dtype of trained models, VLLM automatically converts the type when loading the model.

### Which GPUs parameters should I use for multi‑GPU training?
You can use
- `gpus`: total GPUs to request (e.g. `"gpus": 2`)
- Do not use `max_gpus` (it's only for debugging)

### Are there additional parameters needed for inference on multi-GPU?
No additional parameters are needed. Multi-GPU is configured in the deployment.

## Monitoring & Loss

### How can I view fine‑tuning progress and loss curves?
Use the `scalarlm plot` CLI command after setting your API URL.

### How do I install the CLI for monitoring fine‑tuning progress?
You can install it simply via pip
```
pip install scalarlm
```
Then set your API endpoint:
```
export MAS_INT_API_URL="the-hosted-ip-for-craylm"

scalarlm [-h] {logs,plot,ls,squeue} ...

The command line interface for ScalarLM

positional arguments:
  {logs,plot,ls,squeue}
    logs                View logs
    plot                Plot the results of a model
    ls                  List models
    squeue              View the squeue

options:
  -h, --help            show this help message and exit
```

### How do I install and configure the framework?
You can directly install the ScalarLM client from PyPI:
```bash
pip install scalarlm
```
The client requirements are listed in github.com/cray-lm/cray-lm/blob/main/sdk/pyproject.toml.

If you want to modify hyperparameters during training, set your PYTHONPATH to the checked-out repository:
```bash
export PYTHONPATH=/path/to/checkout/cray/cray/sdk
```
Then you can modify files in the ml directory, such as the optimizer configuration at github.com/cray-lm/cray-lm/blob/main/ml/cray_megatron/megatron/training_loop.py.

### Can I change the loss function?
Yes — you can swap in a custom loss in the training loop (here)[https://github.com/cray‑lm/cray‑lm/blob/493250c3b93a3113a9dc9cf04993e795515cf746/ml/cray_megatron/megatron/training_loop.py#L105]

## Caching & Performance

### Does ScalarLM cache inference results?
No. Inference is sufficiently fast that no cache is provided.

### Any performance limitations or known issues?
Current limitations include:
- Each deployment is tied to a single base model.
- Large‑model training may not yet be fully optimized; benchmarks and speedups are in progress.

## Advanced Topics

### Can I implement RLHF?
Yes. Use the ScalarLM inference endpoint to score or rank data with your reward model, then feed the selected data back into the training endpoint to update the model.

### Is early stopping available?
The framework itself doesn't expose early-stop parameters, but since it's built on PyTorch/Hugging Face, you can integrate the Hugging Face early stopping callback into your training loop. You'll need to write new code to enable this and determine how to configure the relevant hyperparameters on the callback. See huggingface.co/docs/transformers/en/main_classes/callback for details.

### Where can I see a full list of fine‑tuning parameters?
Instead of providing a single config file listing all possible training parameters, ScalarLM lets you modify and write new code in the `ml/` directory to add/enhance/update training parameters as needed. This gives the user the highest level of autonomy and flexibility. ScalarLM is designed to allow users to experiment as seamlessly as possible.

### Can I use any Hugging Face model?
In principle yes, but each deployment must be explicitly configured. 

### How to set inference temperature?
We recommend not changing it, because higher temperature means higher error. 
If you must change it, it is a parameter to vllm. The [quickstart page](https://docs.vllm.ai/en/v0.6.1/getting_started/quickstart.html) shows examples of how to set it.
