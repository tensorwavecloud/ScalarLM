# ScalarLM Architecture Documentation

## 1. High-Level System Architecture

```mermaid
graph TB
    subgraph "User Layer"
        CLI[CLI/Scripts]
        API[REST API]
        SDK[Python SDK]
        OpenAI[OpenAI Compatible]
    end

    subgraph "ScalarLM Core - Port 8000"
        Router[FastAPI Router<br/>All /v1/* endpoints]
        WorkQueue[Inference Work Queue<br/>SQLite-based]
        GenWorker[Generate Worker<br/>Async processing]
        TrainWorker[Training Worker<br/>SLURM jobs]
    end

    subgraph "vLLM Engine Layer"
        VLLMServer[vLLM Server<br/>Port 8001]
        EngineFactory[Engine Factory<br/>HTTP or Direct mode]
        TokenMgr[Tokenformer Manager<br/>Checkpoint loader]
    end

    subgraph "Model Storage"
        BaseModel[Base Models<br/>HuggingFace models]
        TrainedModels[Trained Checkpoints<br/>/app/cray/jobs/{hash}/*.pt]
        Adapters[Tokenformer Adapters<br/>Weight modifications]
    end

    subgraph "Infrastructure"
        Docker[Docker Containers]
        SLURM[SLURM Scheduler]
        K8s[Kubernetes<br/>Optional]
    end

    CLI --> Router
    API --> Router
    SDK --> Router
    OpenAI --> Router

    Router --> WorkQueue
    WorkQueue --> GenWorker
    Router --> TrainWorker

    GenWorker --> EngineFactory
    EngineFactory --> VLLMServer
    
    GenWorker --> TokenMgr
    TokenMgr --> TrainedModels
    TokenMgr --> Adapters
    
    VLLMServer --> BaseModel
    VLLMServer --> Adapters

    TrainWorker --> SLURM
    SLURM --> TrainedModels
    
    Docker --> VLLMServer
    Docker --> Router
```

## 2. Technology Overview

```mermaid
graph LR
    subgraph "What is ScalarLM?"
        Desc[Distributed LLM Training & Serving Platform]
    end

    subgraph "Core Technologies"
        vLLM[vLLM<br/>High-performance<br/>LLM inference engine]
        FastAPI[FastAPI<br/>Modern Python<br/>web framework]
        PyTorch[PyTorch<br/>Deep learning<br/>framework]
        LoRA[LoRA<br/>Efficient model<br/>fine-tuning]
    end

    subgraph "Infrastructure"
        Docker[Docker<br/>Container<br/>platform]
        SLURM[SLURM<br/>Job scheduler<br/>for HPC]
        MPI[MPI<br/>Multi-node<br/>communication]
    end

    subgraph "Key Concepts"
        Checkpoint[Checkpoints<br/>.pt files with<br/>model weights]
        Adapter[Adapters<br/>Small model<br/>modifications]
        FLOP[FLOPs<br/>Computational<br/>cost tracking]
    end

    Desc --> vLLM
    Desc --> FastAPI
    vLLM --> PyTorch
    vLLM --> LoRA
    LoRA --> Adapter
    PyTorch --> Checkpoint
    Docker --> vLLM
    SLURM --> MPI
    vLLM --> FLOP
```

## 3. Training Pipeline (Mid-Level)

```mermaid
flowchart TD
    Start[User submits<br/>training request] --> API["/v1/megatron/train endpoint"]
    
    API --> Upload[Upload training data<br/>multipart/form-data]
    Upload --> Validate{Validate<br/>parameters}
    Validate -->|Invalid| Error[Return 422 error]
    Validate -->|Valid| Hash[Generate job hash<br/>from parameters]
    
    Hash --> JobDir[Create job directory<br/>/app/cray/jobs/{hash}]
    JobDir --> SLURM[Submit SLURM job]
    
    SLURM --> Response[Return job_status<br/>and job_hash]
    
    subgraph "SLURM Execution"
        Worker[Training Worker] --> LoadBase[Load base model<br/>e.g. TinyLlama]
        LoadBase --> PrepData[Prepare training<br/>dataset from JSONL]
        
        PrepData --> Train[Train with LoRA<br/>15-150 steps]
        
        Train --> Save[Save checkpoint<br/>checkpoint_15.pt]
        Save --> Metrics[Log metrics<br/>loss, FLOPs, time]
    end
    
    Metrics --> Complete[Job completes]
    Complete --> Ready[Model ready as<br/>adapter via hash]
```

## 4. Generation Pipeline (Mid-Level)

```mermaid
flowchart LR
    subgraph "Work Queue Flow"
        User[User Request] --> Router["/v1/generate"]
        Router --> Queue[SQLite Work Queue]
        Queue --> GetWork[Worker pulls work<br/>via get_work]
    end

    subgraph "Adapter Discovery"
        GetWork --> CheckModel{Is model<br/>a job hash?}
        CheckModel -->|Yes| GetAdaptors[get_adaptors()<br/>finds .pt files]
        CheckModel -->|No| UseBase[Use base model]
        GetAdaptors --> LoadLoRA[LoadLoRAAdapterRequest<br/>to vLLM]
        LoadLoRA --> Tokenformer[Load via<br/>Tokenformer system]
    end

    subgraph "Generation"
        Tokenformer --> Generate[vLLM generates<br/>via HTTP/Direct]
        UseBase --> Generate
        Generate --> FLOPs[Calculate FLOPs]
        FLOPs --> FinishWork[finish_work()<br/>marks complete]
    end

    subgraph "Response"
        FinishWork --> Queue2[Update queue]
        Queue2 --> GetResults[Client calls<br/>get_results()]
        GetResults --> Response[Return response<br/>with metrics]
    end
```

## 5. Low-Level Training Code Path

```mermaid
sequenceDiagram
    participant User
    participant API as megatron_router.py
    participant Upload as upload_training_data.py
    participant Launch as launch_training_job.py
    participant SLURM as SLURM Scheduler
    participant Worker as Training Script
    participant FS as File System

    User->>API: POST /v1/megatron/train
    API->>Upload: upload_training_data(request)
    Upload->>Upload: save multipart file
    Upload->>Upload: parse config JSON
    Upload-->>API: training_data_path, params
    
    API->>Launch: launch_training_job(job_config)
    Launch->>Launch: generate_hash(params)
    Launch->>FS: mkdir /app/cray/jobs/{hash}
    Launch->>FS: save job_config.json
    
    Launch->>SLURM: sbatch train_job.sh
    SLURM-->>Launch: job_id
    Launch-->>API: job_status
    
    API-->>User: TrainResponse(job_hash, status)
    
    Note over SLURM,Worker: Asynchronous SLURM execution
    
    SLURM->>Worker: Execute training
    Worker->>Worker: Load base model
    Worker->>Worker: Setup LoRA config
    
    loop Training Steps (15)
        Worker->>Worker: forward_pass()
        Worker->>Worker: calculate_loss()
        Worker->>Worker: backward_pass()
        Worker->>Worker: optimizer.step()
    end
    
    Worker->>FS: torch.save(checkpoint_15.pt)
    Worker->>SLURM: Job complete
```

## 6. Low-Level Generation Code Path

```mermaid
sequenceDiagram
    participant User
    participant Router as generate_router.py
    participant Queue as Work Queue
    participant Worker as create_generate_worker.py
    participant GetWork as get_work.py
    participant Adaptors as get_adaptors.py
    participant VLLM as vLLM Server:8001
    participant Tokenformer as tokenformer.py

    User->>Router: POST /v1/generate
    Router->>Queue: Add work item
    Router-->>User: {"request_id": "req_123"}
    
    Note over Worker: Worker polling loop
    Worker->>Router: POST /v1/generate/get_work
    Router->>GetWork: get_work(batch_size)
    GetWork->>Queue: Get next item
    GetWork->>Adaptors: get_adaptors(request)
    
    alt Model is job hash
        Adaptors->>Adaptors: scan /app/cray/jobs/{hash}/*.pt
        Adaptors-->>GetWork: ["checkpoint_15.pt"]
        GetWork-->>Worker: work + new_adaptors
        
        Worker->>Worker: add_new_adaptor()
        Worker->>VLLM: LoadLoRAAdapterRequest
        VLLM->>Tokenformer: from_local_checkpoint()
        Tokenformer->>Tokenformer: torch.load(checkpoint.pt)
        Tokenformer->>Tokenformer: extract tokenformer weights
        Tokenformer-->>VLLM: adapter loaded
    else Base model
        GetWork-->>Worker: work item (no adaptors)
    end
    
    Worker->>VLLM: create_completion(request)
    VLLM-->>Worker: generated_text
    
    Worker->>Worker: compute_flop_count()
    Worker->>Router: POST /v1/generate/finish_work
    Router->>Queue: Mark complete
    
    User->>Router: POST /v1/generate/get_results
    Router->>Queue: Get results
    Queue-->>Router: completed result
    Router-->>User: {"response": "...", "flop_count": 150000000}
```

## 7. Component Relationships

```mermaid
graph TD
    subgraph "API Layer - All /v1/*"
        FastAPI[FastAPI Application<br/>main.py]
        GenRouter[Generate Router<br/>generate_router.py]
        OpenAIv1[OpenAI v1 Router<br/>openai_v1_router.py]
        MegatronRouter[Megatron Router<br/>megatron_router.py]
        SLURMRouter[SLURM Router<br/>slurm_router.py]
        HealthRouter[Health Router<br/>health_router.py]
    end

    subgraph "Work Queue System"
        GenWorker[Generate Worker<br/>create_generate_worker.py]
        GetWork[Get Work<br/>get_work.py]
        GetAdaptors[Get Adaptors<br/>get_adaptors.py]
        FinishWork[Finish Work<br/>finish_work.py]
        WorkQueue[SQLite Queue<br/>inference_work_queue.py]
    end

    subgraph "Adapter Management"
        TokenMgr[Tokenformer Manager<br/>tokenformer.py]
        ModelState[Model State Manager<br/>Manages weight swapping]
        AdapterCache[Adapter Cache<br/>LRU caching]
    end

    subgraph "vLLM Integration"
        EngineFactory[Engine Factory<br/>engine_factory.py]
        HTTPEngine[HTTP Engine<br/>http_engine.py - Port 8001]
        DirectEngine[Direct Engine<br/>direct_engine.py]
        CreateVLLM[Create vLLM<br/>create_vllm.py]
    end

    subgraph "Storage"
        Jobs[Job Checkpoints<br/>/app/cray/jobs/{hash}/*.pt]
        Config[Configuration<br/>default_config.py]
    end

    FastAPI --> GenRouter
    FastAPI --> OpenAIv1
    FastAPI --> MegatronRouter
    FastAPI --> SLURMRouter
    FastAPI --> HealthRouter

    GenRouter --> GetWork
    GenRouter --> FinishWork
    GetWork --> WorkQueue
    GetWork --> GetAdaptors
    
    GenWorker --> GetWork
    GenWorker --> GetAdaptors
    GenWorker --> EngineFactory
    
    GetAdaptors --> Jobs
    GenWorker --> TokenMgr
    
    TokenMgr --> ModelState
    TokenMgr --> AdapterCache
    TokenMgr --> Jobs
    
    EngineFactory --> HTTPEngine
    EngineFactory --> DirectEngine
    CreateVLLM --> HTTPEngine
    
    Config --> EngineFactory
    Config --> GenWorker
```

## 8. Docker Build Process

```mermaid
flowchart TD
    subgraph "Multi-Stage Build"
        Base[Base Image Selection<br/>CPU/NVIDIA/AMD]
        Base --> VLLM[vLLM Build Stage]
        VLLM --> Infra[Infrastructure Stage]
        Infra --> Final[Final Image]
    end

    subgraph "Base Stage Details"
        CPU[Ubuntu 24.04<br/>+ Python 3.12<br/>+ PyTorch CPU]
        NVIDIA[NVIDIA PyTorch 24.07<br/>+ CUDA 12.x<br/>+ Flash Attention]
        AMD[ROCm Base<br/>+ HIP<br/>+ Custom kernels]
    end

    subgraph "vLLM Build v0.10.0+"
        Source[vLLM Source<br/>Local copy or fork]
        Source --> BuildExt[Build C++ extensions<br/>cmake/ninja]
        BuildExt --> InstallVLLM[pip install vllm]
        InstallVLLM --> Adapters[Install ScalarLM adapters]
    end

    subgraph "Final Setup"
        CopyCode[Copy ScalarLM code]
        CopyCode --> SetEnv[Set environment vars<br/>VLLM_USE_V1=1 for CPU]
        SetEnv --> SLURM[Install SLURM]
        SLURM --> Ready[Container Ready<br/>Ports 8000, 8001]
    end

    Base --> CPU
    Base --> NVIDIA
    Base --> AMD
    
    CPU --> VLLM
    NVIDIA --> VLLM
    AMD --> VLLM
    
    VLLM --> Source
    Adapters --> Infra
    Infra --> CopyCode
    Ready --> Final
```

## 9. Key File Locations

```mermaid
graph TD
    subgraph Configuration
        Config["infra/cray_infra/util/default_config.py<br/>Main configuration"]
        EnvFile[".env.example<br/>Environment variables"]
    end

    subgraph "API Layer /v1/*"
        MainAPI["infra/cray_infra/api/fastapi/main.py<br/>FastAPI app - Port 8000"]
        Routers["infra/cray_infra/api/fastapi/routers/<br/>generate, megatron, openai_v1, slurm"]
    end

    subgraph "Worker & Queue"
        Workers1["infra/cray_infra/one_server/<br/>create_generate_worker.py, create_vllm.py"]
        Generate["infra/cray_infra/api/fastapi/generate/<br/>get_work.py, finish_work.py, get_adaptors.py"]
    end

    subgraph "vLLM Adapters"
        VLLMWrapper["infra/cray_infra/vllm/<br/>http_engine.py, direct_engine.py"]
        Adapters["infra/cray_infra/adapters/<br/>tokenformer.py, model state management"]
    end

    subgraph "Training System"
        Training["infra/cray_infra/training/<br/>launch_training_job.py, SLURM integration"]
    end

    subgraph "Runtime Data"
        Jobs["/app/cray/jobs/{hash}/<br/>checkpoint_15.pt files"]
        WorkDB["/app/cray/inference_work_queue.sqlite<br/>Work queue database"]
    end

    Config --> MainAPI
    MainAPI --> Routers
    Routers --> Generate
    Generate --> Workers1
    Workers1 --> VLLMWrapper
    VLLMWrapper --> Adapters
    Adapters --> Jobs
    Generate --> WorkDB
    Training --> Jobs
```

## Notes for Emacs Users

To work with these Mermaid diagrams in Emacs:

1. Install `mermaid-mode` from MELPA:
   ```elisp
   M-x package-install RET mermaid-mode RET
   ```

2. For live preview, install `mermaid-cli`:
   ```bash
   npm install -g @mermaid-js/mermaid-cli
   ```

3. Configure Emacs:
   ```elisp
   (setq mermaid-output-format ".svg")
   (setq mermaid-tmp-dir "/tmp/mermaid")
   ```

4. Commands:
   - `C-c C-c` - Compile current diagram
   - `C-c C-o` - Open compiled diagram
   - `C-c C-f` - Compile file
   - `C-c C-r` - Compile region

5. For inline preview in Org-mode:
   ```org
   #+BEGIN_SRC mermaid :file diagram.svg
   graph TD
       A --> B
   #+END_SRC
   ```
