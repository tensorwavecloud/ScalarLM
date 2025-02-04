# Welcome to Craylm

Craylm is a fully open source, CC-0 Licensed, integrated LLM inference and training platform.

Craylm builds on top of the vLLM inference engine, the Megatron-LM training framework, and the HuggingFace model hub. It unifies the capabilities of these tools into a single platform, enabling users to easily perform LLM inference and training, and build higher lever applications such as Agents.

Craylm is designed for high peformance. It inherits the distributed training capabilities of Megatron-LM and the optimized inference engine of vLLM. Craylm is also designed to be easy to use. It provides an OpenAI compatible server and a simple command line interface for users to interact with the platform.

Craylm is inspired by the work of Seymour Roger Cray (September 28, 1925 – October 5, 1996), an American electrical engineer and supercomputer architect who designed a series of computers that were the fastest in the world for decades, and founded Cray Research, which built many of these machines. Called "the father of supercomputing", Cray has been credited with creating the supercomputer industry.

Read the [documentation](https://docs.cray-lm.com) and [blog](https://blog.cray-lm.com) for more information.

# Docker builds

Check out prebuilt docker containers for different targets:

| Target | Container                   | Latest Release v0.5      |
-------- | --------------------------- | ------------------------ | 
| NVIDIA | gdiamos/cray-nvidia:latest  | gdiamos/cray-nvidia:v0.5 |
| ARM    | gdiamos/cray-arm:latest     | gdiamos/cray-arm:v0.5    |
| AMD    | gdiamos/cray-amd:latest     | gdiamos/cray-amd:v0.5    |
| x86    | gdiamos/cray-cpu:latest     | gdiamos/cray-cpu:v0.5    |

For example, to launch a development server on a modern macbook, e.g. m2

```bash
docker run -it -p 8000:8000 --entrypoint /app/cray/scripts/start_one_server.sh gdiamos/cray-arm:v0.5
```



