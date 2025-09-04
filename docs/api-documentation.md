# ScalarLM API Documentation

## API Architecture Overview

```mermaid
graph TB
    subgraph "Client Access"
        HTTP[HTTP Clients]
        SDK[Python SDK]
        CLI[CLI Tools]
    end

    subgraph "ScalarLM API Layer - Port 8000"
        FastAPI[FastAPI Server]
        
        subgraph "API Routers - All under /v1"
            GenRoute["Generate Router /v1/generate/*"]
            OpenAIRoute["OpenAI v1 Router /v1/*"]
            MegatronRoute["Megatron Router /v1/megatron/*"]
            SLURMRoute["SLURM Router /slurm/*"]
            HealthRoute["Health Router /v1/health"]
        end

        subgraph "Work Queue System"
            WorkQueue["Inference Work Queue"]
            AdapterMgr["Adapter Manager"]
        end
    end

    subgraph "vLLM Server - Port 8001"
        VLLMOpenAI[OpenAI Compatible API]
        
        subgraph "vLLM Endpoints"
            Models["/v1/models"]
            Complete["/v1/completions"]
            Chat["/v1/chat/completions"]
            LoRA["/v1/load_lora_adapter"]
            Unload["/v1/unload_lora_adapter"]
        end
    end

    HTTP --> FastAPI
    SDK --> FastAPI
    CLI --> FastAPI
    
    FastAPI --> GenRoute
    FastAPI --> OpenAIRoute
    FastAPI --> MegatronRoute
    FastAPI --> SLURMRoute
    FastAPI --> HealthRoute
    
    GenRoute --> WorkQueue
    WorkQueue --> AdapterMgr
    AdapterMgr --> VLLMOpenAI
    OpenAIRoute --> VLLMOpenAI
    
    VLLMOpenAI --> Models
    VLLMOpenAI --> Complete
    VLLMOpenAI --> Chat
    VLLMOpenAI --> LoRA
    VLLMOpenAI --> Unload
```

## ScalarLM API Endpoints (Port 8000)

### 1. Generation Endpoints (Work Queue Based)

```mermaid
graph LR
    subgraph "Generate Router /v1/generate"
        Gen["POST /v1/generate - Submit work"]
        GetWork["POST /v1/generate/get_work - Workers pull work"]
        FinishWork["POST /v1/generate/finish_work - Complete work"]
        GetResults["POST /v1/generate/get_results - Get results"]
        GetAdaptors["POST /v1/generate/get_adaptors - List adapters"]
        Metrics["GET /v1/generate/metrics - System metrics"]
    end

    subgraph "Work Queue Flow"
        Submit[Client submits request]
        Queue[Added to work queue]
        Worker[Worker pulls work]
        Process[Process with vLLM]
        Complete[Mark complete]
        Retrieve[Client gets results]
    end

    Submit --> Gen
    Gen --> Queue
    Queue --> GetWork
    GetWork --> Worker
    Worker --> Process
    Process --> FinishWork
    FinishWork --> Complete
    Complete --> GetResults
    GetResults --> Retrieve
```

#### POST `/v1/generate`
```json
// Request - Submit work to queue
{
  "prompt": "What is 3+3?",
  "model": "TinyLlama/TinyLlama-1.1B-Chat-v1.0",  // or job hash like "a1b2c3d4e5f6"
  "max_tokens": 100,
  "request_type": "generate"
}

// Response
{
  "request_id": "req_12345",
  "status": "queued"
}
```

#### POST `/v1/generate/get_work`
```json
// Request - Worker pulls work
{
  "batch_size": 4,
  "loaded_adaptor_count": 2
}

// Response
{
  "requests": [
    {
      "prompt": "What is 3+3?",
      "request_id": "req_12345",
      "model": "TinyLlama/TinyLlama-1.1B-Chat-v1.0",
      "request_type": "generate",
      "max_tokens": 100
    }
  ],
  "new_adaptors": {
    "new_adaptors": ["job_hash_123", "job_hash_456"]
  }
}
```

#### POST `/v1/generate/finish_work`
```json
// Request - Worker completes work
{
  "requests": [
    {
      "request_id": "req_12345",
      "response": "The answer is 6.",
      "token_count": 15,
      "flop_count": 150000000
    }
  ]
}
```

#### POST `/v1/generate/get_results`
```json
// Request - Client retrieves results
{
  "request_ids": ["req_12345"]
}

// Response
{
  "results": [
    {
      "request_id": "req_12345",
      "response": "The answer is 6.",
      "token_count": 15,
      "flop_count": 150000000,
      "status": "completed"
    }
  ]
}
```

### 2. Training Endpoints (Megatron Router)

```mermaid
graph TD
    subgraph "Megatron Training API /v1/megatron"
        Train["POST /v1/megatron/train - Start training job"]
        JobInfo["GET /v1/megatron/train/{job_hash} - Get job info"]
        Logs["GET /v1/megatron/train/logs/{model_name} - Stream logs"]
        ListModels["GET /v1/megatron/list_models - List trained models"]
        SQueue["GET /v1/megatron/squeue - SLURM queue status"]
        GPUCount["GET /v1/megatron/gpu_count - Available GPUs"]
        NodeCount["GET /v1/megatron/node_count - Available nodes"]
    end

    subgraph "Training Request"
        BaseModel[base_model: str]
        Dataset[dataset: multipart/form-data]
        Config[job_config: dict]
        DataPath[training_data_path: str]
    end

    subgraph "Job Response"
        JobHash[job_hash: str]
        JobStatus[job_status: dict]
        Deployed[deployed: bool]
        JobConfig[job_config: dict]
    end

    Train --> Dataset
    Train --> Config
    Train --> JobHash
    JobInfo --> JobStatus
    ListModels --> JobHash
```

#### POST `/v1/megatron/train`
```bash
# Training request with multipart form data
curl -X POST http://localhost:8000/v1/megatron/train \
  -F "file=@training_data.jsonl" \
  -F 'config={
    "model": "TinyLlama/TinyLlama-1.1B-Chat-v1.0",
    "num_steps": 15,
    "learning_rate": 0.0001,
    "batch_size": 4,
    "lora_r": 8,
    "lora_alpha": 16
  }'

# Response
{
  "job_status": {
    "job_id": "12345",
    "status": "PENDING",
    "slurm_state": "PD"
  },
  "job_config": {
    "model": "TinyLlama/TinyLlama-1.1B-Chat-v1.0",
    "num_steps": 15,
    "job_hash": "a1b2c3d4e5f6"
  },
  "deployed": false
}
```

#### GET `/v1/megatron/train/{job_hash}`
```json
// Response
{
  "job_hash": "a1b2c3d4e5f6",
  "status": "RUNNING",
  "checkpoint_path": "/app/cray/jobs/a1b2c3d4e5f6/checkpoint_15.pt",
  "current_step": 10,
  "total_steps": 15,
  "loss": 0.234
}
```

### 3. SLURM Management Endpoints

```mermaid
graph LR
    subgraph "SLURM Router /slurm"
        Status["GET /slurm/status - Cluster status"]
        Queue["GET /slurm/squeue - Job queue"]
        Endpoints["GET /slurm/endpoints - List routes"]
    end

    subgraph "Status Response"
        QueueInfo[queue: SLURM jobs]
        Resources[resources: GPUs/nodes]
        ClusterStatus[status: active/error]
    end

    subgraph "Resources"
        GPUs[gpu_count: int]
        Nodes[node_count: int]
        Jobs[jobs: list]
    end

    Status --> QueueInfo
    Status --> Resources
    Queue --> Jobs
    Resources --> GPUs
    Resources --> Nodes
```

#### GET `/slurm/status`
```json
// Response
{
  "queue": {
    "jobs": [
      {
        "job_id": "12345",
        "name": "train_a1b2c3d4",
        "state": "RUNNING",
        "time": "0:05:23",
        "nodes": 1
      }
    ]
  },
  "resources": {
    "gpu_count": 8,
    "node_count": 2
  },
  "status": "active"
}
```

### 4. Health & Metrics Endpoints

```mermaid
graph TD
    subgraph "Health Router /v1"
        Health["GET /v1/health - System health"]
        Metrics["GET /v1/generate/metrics - Generation metrics"]
    end

    subgraph "Health Response"
        Status[status: healthy/unhealthy]
        Components[components: dict]
        Timestamp[timestamp: str]
    end

    subgraph "Metrics Response"
        QueueSize[queue_size: int]
        ActiveRequests[active_requests: int]
        ModelStats[model_stats: dict]
        Performance[performance: dict]
    end

    Health --> Status
    Health --> Components
    Metrics --> QueueSize
    Metrics --> ActiveRequests
    Metrics --> ModelStats
```

#### GET `/v1/health`
```json
// Response
{
  "status": "healthy",
  "components": {
    "vllm_server": "healthy",
    "work_queue": "healthy",
    "slurm": "active"
  },
  "timestamp": "2024-01-27T10:00:00Z"
}
```

## OpenAI Compatible API (Port 8000 → 8001)

### OpenAI v1 Endpoints (ScalarLM Proxy)

```mermaid
graph TD
    subgraph "OpenAI v1 Router - ScalarLM"
        ChatComp["POST /v1/chat/completions - Chat completion"]
        Comp["POST /v1/completions - Text completion"]
        ModList["GET /v1/models - List available models"]
    end

    subgraph "Proxy Flow"
        Client[Client Request]
        ScalarLM[ScalarLM:8000]
        vLLM[vLLM:8001]
        Response[Response]
    end

    subgraph "Features"
        Streaming[SSE Streaming]
        System[System Messages]
        Multi[Multi-turn Chat]
        Adapter[LoRA Adapter Support]
    end

    Client --> ScalarLM
    ScalarLM --> vLLM
    vLLM --> Response
    ChatComp --> Streaming
    ChatComp --> System
    ChatComp --> Multi
    Comp --> Adapter
```

#### POST `/v1/chat/completions`
```json
// Request (proxied to vLLM)
{
  "model": "TinyLlama/TinyLlama-1.1B-Chat-v1.0",  // or job hash
  "messages": [
    {"role": "system", "content": "You are a helpful assistant."},
    {"role": "user", "content": "What is 3+3?"}
  ],
  "temperature": 0.7,
  "max_tokens": 100,
  "stream": false
}

// Response (from vLLM)
{
  "id": "chatcmpl-123",
  "object": "chat.completion",
  "created": 1706352000,
  "model": "TinyLlama/TinyLlama-1.1B-Chat-v1.0",
  "choices": [{
    "index": 0,
    "message": {
      "role": "assistant",
      "content": "3 + 3 equals 6."
    },
    "finish_reason": "stop"
  }],
  "usage": {
    "prompt_tokens": 20,
    "completion_tokens": 10,
    "total_tokens": 30
  }
}
```

#### POST `/v1/completions`
```json
// Request (proxied to vLLM)
{
  "model": "TinyLlama/TinyLlama-1.1B-Chat-v1.0",
  "prompt": "The capital of France is",
  "max_tokens": 10,
  "temperature": 0.0,
  "stream": true
}

// Response (SSE stream from vLLM)
data: {"id":"cmpl-1","object":"text_completion","created":1706352000,"model":"TinyLlama/TinyLlama-1.1B-Chat-v1.0","choices":[{"text":" Paris","index":0,"logprobs":null,"finish_reason":null}]}
data: {"id":"cmpl-1","object":"text_completion","created":1706352000,"model":"TinyLlama/TinyLlama-1.1B-Chat-v1.0","choices":[{"text":".","index":0,"logprobs":null,"finish_reason":"stop"}]}
data: [DONE]
```

## vLLM Server Native Endpoints (Port 8001)

### vLLM OpenAI API

```mermaid
graph TB
    subgraph "vLLM Native API"
        VModels["GET /v1/models - Model registry"]
        VChat["POST /v1/chat/completions - Chat endpoint"]
        VComplete["POST /v1/completions - Completion endpoint"]
    end

    subgraph "vLLM Adapter Management"
        LoadLoRA["POST /v1/load_lora_adapter - Load adapter"]
        UnloadLoRA["POST /v1/unload_lora_adapter - Unload adapter"]
    end

    subgraph "Adapter Loading"
        AdapterName[lora_name: job_hash]
        AdapterPath[lora_path: /app/cray/jobs/path]
        Tokenformer[Tokenformer weights]
    end

    LoadLoRA --> AdapterName
    LoadLoRA --> AdapterPath
    AdapterPath --> Tokenformer
```

#### POST `/v1/load_lora_adapter` (vLLM Server)
```json
// Request (from generate worker)
{
  "lora_name": "a1b2c3d4e5f6",
  "lora_path": "/app/cray/jobs/a1b2c3d4e5f6"
}

// Note: The adapter loading uses ScalarLM's Tokenformer system
// to load .pt checkpoint files and apply weights to base model
```

## API Authentication & Headers

```mermaid
graph LR
    subgraph "Authentication"
        APIKey["API Key - X-API-Key header"]
        Bearer["Bearer Token - Authorization header"]
        Session["Session Cookie - scalarlm_session"]
    end

    subgraph "Common Headers"
        ContentType[Content-Type: application/json]
        Accept[Accept: application/json]
        UserAgent[User-Agent: scalarlm-client/1.0]
    end

    subgraph "Response Headers"
        RateLimit[X-RateLimit-Remaining]
        RequestID[X-Request-ID]
        ProcessTime[X-Process-Time-Ms]
    end
```

## Request Flow Diagram

```mermaid
sequenceDiagram
    participant Client
    participant ScalarLM as ScalarLM API 8000
    participant Queue as Work Queue
    participant Worker as Generate Worker
    participant vLLM as vLLM Server 8001
    participant Storage as Job Storage

    Client->>ScalarLM: POST /v1/generate
    ScalarLM->>Queue: Add to queue
    ScalarLM-->>Client: Return request_id
    
    Worker->>ScalarLM: POST /v1/generate/get_work
    ScalarLM->>Queue: Get next work item
    Queue-->>ScalarLM: Work item + adapters
    ScalarLM-->>Worker: Work response
    
    alt Has new adapter (job hash)
        Worker->>ScalarLM: POST /v1/generate/get_adaptors
        ScalarLM->>Storage: Find checkpoint_*.pt
        Storage-->>ScalarLM: Adapter path
        ScalarLM-->>Worker: Adapter info
        Worker->>vLLM: POST /v1/load_lora_adapter
        vLLM->>Storage: Load Tokenformer weights
        vLLM-->>Worker: Adapter loaded
    end
    
    Worker->>vLLM: POST /v1/completions or /v1/chat/completions
    vLLM-->>Worker: Generated text
    
    Worker->>Worker: Calculate FLOPs
    Worker->>ScalarLM: POST /v1/generate/finish_work
    ScalarLM->>Queue: Mark complete
    
    Client->>ScalarLM: POST /v1/generate/get_results
    ScalarLM->>Queue: Get results
    Queue-->>ScalarLM: Completed results
    ScalarLM-->>Client: Response with metrics
```

## Error Responses

```mermaid
graph TD
    subgraph "HTTP Status Codes"
        OK200["200 OK - Success"]
        Created201["201 Created - Resource created"]
        BadRequest400["400 Bad Request - Invalid input"]
        Unauthorized401["401 Unauthorized - Auth required"]
        NotFound404["404 Not Found - Resource missing"]
        RateLimit429["429 Too Many Requests - Rate limited"]
        ServerError500["500 Internal Error - Server fault"]
        ServiceUnavail503["503 Service Unavailable - Overloaded"]
    end

    subgraph "Error Response Format"
        Error["JSON error object with code, message, details"]
    end

    BadRequest400 --> Error
    NotFound404 --> Error
    ServerError500 --> Error
```

## WebSocket Endpoints (Future)

```mermaid
graph LR
    subgraph "WebSocket API"
        WSConnect["WS /ws/generate - Real-time generation"]
        WSChat["WS /ws/chat - Interactive chat"]
        WSMetrics["WS /ws/metrics - Live metrics"]
    end

    subgraph "WebSocket Events"
        Connect[connection.open]
        Message[message.receive]
        Token[token.generated]
        Complete[generation.complete]
        Disconnect[connection.close]
    end

    WSConnect --> Connect
    WSConnect --> Token
    WSChat --> Message
    WSMetrics --> Complete
```

## SDK Usage Examples

### Python SDK (Work Queue Based)
```python
from scalarlm import Client
import asyncio

# Initialize client
client = Client(base_url="http://localhost:8000")

# Submit generation request (async)
async def generate_text():
    # Submit work to queue
    request_id = await client.generate(
        prompt="What is 3+3?",
        model="TinyLlama/TinyLlama-1.1B-Chat-v1.0",
        max_tokens=100
    )
    
    # Poll for results
    result = await client.get_results(request_id)
    print(result.response)
    print(f"FLOPs used: {result.flop_count}")

# Train model using Megatron
async def train_model():
    with open("training_data.jsonl", "rb") as f:
        job = await client.megatron.train(
            file=f,
            config={
                "model": "TinyLlama/TinyLlama-1.1B-Chat-v1.0",
                "num_steps": 15,
                "learning_rate": 0.0001
            }
        )
    print(f"Training job hash: {job.job_hash}")
    
    # Use trained model (by job hash)
    request_id = await client.generate(
        prompt="What is 5+5?",
        model=job.job_hash,  # Use job hash as model
        max_tokens=100
    )
```

### cURL Examples
```bash
# Health check
curl http://localhost:8000/v1/health

# List models via OpenAI API
curl http://localhost:8000/v1/models

# Submit generation work
curl -X POST http://localhost:8000/v1/generate \
  -H "Content-Type: application/json" \
  -d '{
    "prompt": "What is 3+3?",
    "model": "TinyLlama/TinyLlama-1.1B-Chat-v1.0",
    "max_tokens": 100,
    "request_type": "generate"
  }'
# Returns: {"request_id": "req_12345", "status": "queued"}

# Get results
curl -X POST http://localhost:8000/v1/generate/get_results \
  -H "Content-Type: application/json" \
  -d '{
    "request_ids": ["req_12345"]
  }'

# OpenAI-compatible chat completion (direct proxy)
curl -X POST http://localhost:8000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "TinyLlama/TinyLlama-1.1B-Chat-v1.0",
    "messages": [
      {"role": "user", "content": "What is 3+3?"}
    ],
    "stream": false
  }'

# Check SLURM status
curl http://localhost:8000/slurm/status
```

## Rate Limiting & Quotas

```mermaid
graph TD
    subgraph "Rate Limits"
        Global[Global: 1000 req/min]
        PerUser[Per User: 100 req/min]
        PerModel[Per Model: 50 req/min]
    end

    subgraph "Resource Quotas"
        Tokens[Max Tokens: 4096]
        Context[Max Context: 2048]
        Batch[Max Batch: 32]
        Concurrent[Max Concurrent: 10]
    end

    subgraph "Response Headers"
        Limit[X-RateLimit-Limit: 100]
        Remaining[X-RateLimit-Remaining: 95]
        Reset[X-RateLimit-Reset: 1706352060]
    end

    Global --> Limit
    PerUser --> Remaining
    PerModel --> Reset
```