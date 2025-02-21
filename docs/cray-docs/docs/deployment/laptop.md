# Laptop

ScalarLM can be run on your laptop for development purposes. It requires Docker.

Clone the [ScalarLM repository](https://github.com/tensorwavecloud/scalarlm) and start the server.

```
git clone git@github.com:tensorwavelm/scalarlm.git
cd scalarlm
./cray up
```

You should see the server come up.


```console
(environment) gregorydiamos@Air-Gregory cray % ./cray up
+++ dirname ./cray
++ cd .
++ pwd
+ LOCAL_DIRECTORY=/Users/gregorydiamos/checkout/cray
+ /Users/gregorydiamos/checkout/cray/cmd/bashly.sh generate
++++ dirname /Users/gregorydiamos/checkout/cray/cmd/bashly.sh
+++ cd /Users/gregorydiamos/checkout/cray/cmd
+++ pwd
++ LOCAL_DIRECTORY=/Users/gregorydiamos/checkout/cray/cmd
+++ id -u
+++ id -g
++ docker run --rm -it --user 501:20 --volume /Users/gregorydiamos/checkout/cray/cmd:/app/cmd --volume /Users/gregorydiamos/checkout/cray/cmd/../scripts:/app/scripts --volume /Users/gregorydiamos/checkout/cray/cmd/bashly-settings.yml:/app/bashly-settings.yml dannyben/bashly generate
creating user files in cmd
skipped cmd/build_image_command.sh (exists)
skipped cmd/depot_build_command.sh (exists)
skipped cmd/up_command.sh (exists)
skipped cmd/test_command.sh (exists)
skipped cmd/deploy_command.sh (exists)
skipped cmd/serve_command.sh (exists)
skipped cmd/llm_plot_command.sh (exists)
skipped cmd/llm_logs_command.sh (exists)
skipped cmd/llm_ls_command.sh (exists)
skipped cmd/llm_squeue_command.sh (exists)
skipped cmd/diffusion_command.sh (exists)
created /app/scripts/cray
run /app/scripts/cray --help to test your bash script
+ /Users/gregorydiamos/checkout/cray/scripts/cray up
[+] Running 0/14.2s (10/37)                                                                                                                                                              docker:desktop-linux
 => [vllm internal] load build definition from Dockerfile                                                                                                                                                0.0s
[+] Building 217.0s (39/39) FINISHED                                                                                                                                                     docker:desktop-linux
 => [vllm internal] load build definition from Dockerfile                                                                                                                                                0.0s
 => => transferring dockerfile: 4.35kB                                                                                                                                                                   0.0s
 => [vllm internal] load metadata for docker.io/library/ubuntu:24.04                                                                                                                                     0.3s
 => [vllm internal] load .dockerignore                                                                                                                                                                   0.0s
 => => transferring context: 2B                                                                                                                                                                          0.0s
 => CACHED [vllm cpu 1/6] FROM docker.io/library/ubuntu:24.04@sha256:80dd3c3b9c6cecb9f1667e9290b3bc61b78c2678c02cbdae5f0fea92cc6734ab                                                                    0.0s
 => => resolve docker.io/library/ubuntu:24.04@sha256:80dd3c3b9c6cecb9f1667e9290b3bc61b78c2678c02cbdae5f0fea92cc6734ab                                                                                    0.0s
 => [vllm internal] load build context                                                                                                                                                                   0.2s
 => => transferring context: 723.40kB                                                                                                                                                                    0.2s
 => [vllm cpu 2/6] RUN --mount=type=cache,target=/var/cache/apt     apt-get update -y     && apt-get install -y python3 python3-pip python3-venv     openmpi-bin libopenmpi-dev libpmix-dev             60.7s
 => [vllm cpu 3/6] RUN python3 -m venv /app/.venv                                                                                                                                                        2.6s
 => [vllm cpu 4/6] RUN . /app/.venv/bin/activate                                                                                                                                                         0.1s
 => [vllm cpu 5/6] RUN pip install uv                                                                                                                                                                    1.4s
 => [vllm cpu 6/6] RUN uv pip install torch==2.4.0 --index-url https://download.pytorch.org/whl/cpu                                                                                                      6.3s
 => [vllm vllm  1/17] RUN --mount=type=cache,target=/var/cache/apt     apt-get update -y     && apt-get install -y curl ccache git vim numactl gcc-12 g++-12 libomp-dev libnuma-dev     && apt-get ins  74.9s
 => [vllm vllm  2/17] COPY ./requirements.txt /app/cray/requirements.txt                                                                                                                                 0.1s
 => [vllm vllm  3/17] COPY ./test/requirements-pytest.txt /app/cray/requirements-pytest.txt                                                                                                              0.0s
 => [vllm vllm  4/17] COPY ./infra/requirements-vllm-build.txt /app/cray/requirements-vllm-build.txt                                                                                                     0.0s
 => [vllm vllm  5/17] RUN uv pip install --no-compile --no-cache-dir -r /app/cray/requirements.txt                                                                                                       8.6s
 => [vllm vllm  6/17] RUN uv pip install --no-compile --no-cache-dir -r /app/cray/requirements-vllm-build.txt                                                                                            1.6s
 => [vllm vllm  7/17] RUN uv pip install --no-compile --no-cache-dir -r /app/cray/requirements-pytest.txt                                                                                                0.5s
 => [vllm vllm  8/17] WORKDIR /app/cray                                                                                                                                                                  0.0s
 => [vllm vllm  9/17] COPY ./infra/cray_infra/vllm /app/cray/infra/cray_infra/vllm                                                                                                                       0.1s
 => [vllm vllm 10/17] COPY ./infra/setup.py /app/cray/infra/cray_infra/setup.py                                                                                                                          0.0s
 => [vllm vllm 11/17] COPY ./infra/CMakeLists.txt /app/cray/infra/cray_infra/CMakeLists.txt                                                                                                              0.0s
 => [vllm vllm 12/17] COPY ./infra/cmake /app/cray/infra/cray_infra/cmake                                                                                                                                0.0s
 => [vllm vllm 13/17] COPY ./infra/csrc /app/cray/infra/cray_infra/csrc                                                                                                                                  0.0s
 => [vllm vllm 14/17] COPY ./infra/requirements-vllm.txt /app/cray/infra/cray_infra/requirements.txt                                                                                                     0.0s
 => [vllm vllm 15/17] WORKDIR /app/cray/infra/cray_infra                                                                                                                                                 0.0s
 => [vllm vllm 16/17] RUN --mount=type=cache,target=/root/.cache/pip     --mount=type=cache,target=/root/.cache/ccache     MAX_JOBS=8 TORCH_CUDA_ARCH_LIST="7.5 8.6" VLLM_TARGET_DEVICE=cpu     python  39.0s
 => [vllm vllm 17/17] WORKDIR /app/cray                                                                                                                                                                  0.0s
 => [vllm infra  1/10] RUN apt-get update -y      && apt-get install -y slurm-wlm libslurm-dev     build-essential     less curl wget net-tools vim iputils-ping     && rm -rf /var/lib/apt/lists/*     15.0s
 => [vllm infra  2/10] COPY ./infra/slurm_src /app/cray/infra/slurm_src                                                                                                                                  0.0s
 => [vllm infra  3/10] RUN /app/cray/infra/slurm_src/compile.sh                                                                                                                                          0.2s
 => [vllm infra  4/10] RUN mkdir -p /app/cray/jobs                                                                                                                                                       0.1s
 => [vllm infra  5/10] COPY ./infra /app/cray/infra                                                                                                                                                      0.9s
 => [vllm infra  6/10] COPY ./sdk /app/cray/sdk                                                                                                                                                          0.0s
 => [vllm infra  7/10] COPY ./test /app/cray/test                                                                                                                                                        0.0s
 => [vllm infra  8/10] COPY ./cray /app/cray/cray                                                                                                                                                        0.0s
 => [vllm infra  9/10] COPY ./ml /app/cray/ml                                                                                                                                                            0.0s
 => [vllm infra 10/10] COPY ./scripts /app/cray/scripts                                                                                                                                                  0.0s
 => [vllm] exporting to image                                                                                                                                                                            4.0s
 => => exporting layers                                                                                                                                                                                  4.0s
 => => writing image sha256:2e9c8cea0daed2da4c4bd5bec4c875c1e4a773395b95cd4ddbd7823479c4ef83                                                                                                             0.0s
[+] Running 2/2o docker.io/library/cray-vllm                                                                                                                                                             0.0s
 ✔ Service vllm           Built                                                                                                                                                                        217.1s
 ✔ Container cray-vllm-1  Recreated                                                                                                                                                                      0.2s
Attaching to vllm-1
vllm-1  | +++ dirname /app/cray/scripts/start_one_server.sh
vllm-1  | ++ cd /app/cray/scripts
vllm-1  | ++ pwd
vllm-1  | + LOCAL_DIRECTORY=/app/cray/scripts
vllm-1  | + /app/cray/scripts/start_slurm.sh
vllm-1  | +++ dirname /app/cray/scripts/start_slurm.sh
vllm-1  | ++ cd /app/cray/scripts
vllm-1  | ++ pwd
vllm-1  | + LOCAL_DIRECTORY=/app/cray/scripts
vllm-1  | + python /app/cray/scripts/../infra/cray_infra/slurm/discovery/discover_clusters.py
vllm-1  | + slurmctld
vllm-1  | + slurmd
vllm-1  | + python -m cray_infra.one_server.main
vllm-1  | INFO 01-09 18:47:16 importing.py:10] Triton not installed; certain GPU-related functions will not be available.
vllm-1  | INFO:     Will watch for changes in these directories: ['/app/cray/infra/cray_infra']
vllm-1  | INFO:     Uvicorn running on http://0.0.0.0:8000 (Press CTRL+C to quit)
vllm-1  | INFO:     Started reloader process [38] using WatchFiles
vllm-1  | INFO 01-09 18:47:19 importing.py:10] Triton not installed; certain GPU-related functions will not be available.
vllm-1  | DEBUG:asyncio:Using selector: EpollSelector
vllm-1  | DEBUG:cray_infra.one_server.start_cray_server:Starting servers: ['api']
vllm-1  | DEBUG:cray_infra.one_server.start_cray_server:Starting API server
vllm-1  | INFO:persistqueue.serializers.pickle:Selected pickle protocol: '4'
vllm-1  | INFO:persistqueue:DBUtils may not be installed, install via 'pip install persist-queue[extra]'
vllm-1  | INFO:     Started server process [51]
vllm-1  | INFO:     Waiting for application startup.
vllm-1  | INFO:cray_infra.training.register_megatron_models:Registering Megatron models
vllm-1  | INFO:     Application startup complete.
vllm-1  | INFO:     Uvicorn running on http://0.0.0.0:8000 (Press CTRL+C to quit)
vllm-1  | ERROR:cray_infra.one_server.wait_for_vllm:Error getting health: Cannot connect to host localhost:8001 ssl:default [Connect call failed ('127.0.0.1', 8001)]
vllm-1  | INFO:cray_infra.training.register_megatron_models:VLLM is not ready. Skipping model registration
vllm-1  | INFO:cray_infra.training.restart_megatron_jobs:Restarting Megatron jobs
vllm-1  | INFO:cray_infra.training.restart_megatron_jobs:Slurm jobs running: []
vllm-1  | DEBUG:persistqueue.sqlbase:Initializing Sqlite3 Queue with path /app/cray/inference_work_queue.sqlite
vllm-1  | INFO:cray_infra.generate.clear_acked_requests_from_queue:Cleared 0 acked requests from the queue.
```




