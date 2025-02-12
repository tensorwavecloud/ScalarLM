# Craylm

Craylm has three high level APIs:

* **completions** provides OpenAI client compatibility
* **generate** provides a simple interface for generating text
* **train** provides a simple interface for submitting training jobs

![Craylm overview](assets/cray-arch.png)


Inference is performed by vLLM workers that are orchestrated by pulling requests from a queue.

Training is performed by Megatron-LM workers that are orchestrated by SLURM.

Trained models are automatically registered with the inference workers.
