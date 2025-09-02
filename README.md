# ScalarLM - Advanced LLM Platform with Clean vLLM Integration

[![License: Apache-2.0](https://img.shields.io/badge/License-Apache%202.0-blue.svg)](https://opensource.org/licenses/Apache-2.0)
[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)

ScalarLM is a fully open source, integrated LLM inference and training platform built on top of vLLM with a revolutionary **clean architecture** that enhances vLLM models through external adapters without any coupling.

## 📋 Core Dependencies

ScalarLM is built on top of these core components:
- **vLLM** - High-performance LLM inference engine (REQUIRED)
- **PyTorch** - Deep learning framework
- **Transformers** - Model implementations and utilities
- **FastAPI** - API server framework

## 🏗️ Clean Architecture Overview

ScalarLM introduces a **zero-coupling** adapter system that enhances vLLM models with advanced features like **Tokenformer** while maintaining perfect separation of concerns:

- ✅ **vLLM remains completely pure** - No ScalarLM dependencies or imports
- ✅ **External enhancement** - ScalarLM functionality injected via adapters  
- ✅ **Perfect compatibility** - Works with any vLLM version
- ✅ **Independent evolution** - Both systems can be updated separately

## 🚀 Quick Start

### Prerequisites

- Python 3.8+ 
- PyTorch 2.0+
- vLLM (installed in step 2 below)
- CUDA 11.8+ (optional but recommended, for GPU acceleration)

### 1. Installation

```bash
# Clone the repository
git clone https://github.com/scalarlm/scalarlm.git
cd scalarlm

# Create virtual environment (recommended)
python -m venv .venv
source .venv/bin/activate  # On Windows: .venv\\Scripts\\activate

# Upgrade pip (required for modern packaging)
python -m pip install --upgrade pip

# Install ScalarLM
pip install -e .
```

### 2. Install vLLM Fork (Required)

ScalarLM **requires our vLLM fork** to ensure compatibility and access to any ScalarLM-specific enhancements:

```bash
# Install ScalarLM's vLLM fork (REQUIRED)
make install-vllm

# OR manually if you have the vLLM fork in ../vllm
cd ../vllm && pip install -e .
```

**⚠️ Important:** Do NOT use standard PyPI vLLM (`pip install vllm`) as it may be missing:
- ScalarLM-specific optimizations
- Bug fixes not yet upstream
- Compatibility patches
- Specific commit versions tested with ScalarLM

**The fork is mandatory** for reliable ScalarLM operation.

### 3. Validation

```bash
# Validate installation
make validate-setup

# Run integration tests
make integration-test

# Run demo
make demo
```

## 📦 What's New: Clean Architecture

ScalarLM has been completely redesigned with a **clean architecture** that solves dependency management issues:

### Before: Embedded vLLM (Problems)
- ❌ Copied vLLM 0.6.0 directly into ScalarLM
- ❌ Hard to update vLLM versions  
- ❌ Tight coupling between systems
- ❌ Maintenance nightmare

### After: Clean Architecture (Solution)
- ✅ **Zero coupling** - vLLM has no knowledge of ScalarLM
- ✅ **External enhancement** - ScalarLM adapters enhance vLLM models
- ✅ **Version independence** - Use any vLLM version
- ✅ **Clean separation** - Both systems evolve independently

### Key Components

```python
# 1. Pure vLLM (completely unaware of ScalarLM)
from vllm import LLM
vllm_model = LLM(model="meta-llama/Llama-2-7b-hf")

# 2. ScalarLM enhancement (external injection)
from scalarlm_vllm_adapters import enhance_vllm_model, initialize_clean_scalarlm_integration

initialize_clean_scalarlm_integration()
enhanced_model = enhance_vllm_model(vllm_model.llm_engine.model_executor)

# 3. Enhanced functionality (Tokenformer, etc.)
if enhanced_model.supports_tokenformer:
    # Add Tokenformer adapters
    tokenformer_manager = enhanced_model.tokenformer_manager
    # ... use enhanced features
```

## 🛠️ Development Setup

### Complete Development Environment

```bash
# Full development setup (installs required vLLM fork + runs tests)
make setup-dev

# Or step by step:
make install-vllm    # Install vLLM fork (REQUIRED)
make install-dev     # Install ScalarLM in dev mode  
make validate-setup  # Validate installation (checks vLLM is installed)
make integration-test # Run tests
```

### Available Make Commands

```bash
make help               # Show all available commands
make install            # Production install
make install-dev        # Development install
make install-vllm       # Install vLLM fork
make build              # Build packages
make clean              # Clean build artifacts

# Testing
make test               # Run pytest tests
make integration-test   # Test clean architecture
make demo               # Run clean architecture demo
make validate-setup     # Validate installation

# Code Quality
make lint               # Run linting (ruff, mypy)
make format             # Format code (black, ruff)

# Migration (for existing users)
make migration-check    # Check migration readiness
make migration          # Migrate from embedded vLLM
```

## 🏃‍♂️ Running ScalarLM

### Quick Start with `scalarlm` CLI

```bash
# Start ScalarLM server (simplest way)
./scalarlm up

# Or use the scalarlm command directly
scalarlm up

# Start with specific configuration
scalarlm up --host 0.0.0.0 --port 8000 --model meta-llama/Llama-2-7b-hf

# View available commands
./scalarlm --help
```

### Available CLI Commands

```bash
./scalarlm up              # Start ScalarLM server
./scalarlm benchmark       # Run performance benchmarks
./scalarlm llm-logs        # View LLM logs
./scalarlm llm-ls          # List available models
./scalarlm llm-plot        # Plot training metrics
./scalarlm llm-squeue      # View training queue status
./scalarlm test            # Run tests
./scalarlm build-image     # Build Docker image
```

### Alternative Server Start Methods

```bash
# Using Make commands
make serve              # Production server
make serve-debug        # Debug mode with reload

# Direct Python execution
scalarlm-server --host 0.0.0.0 --port 8000

# With environment variables
SCALARLM_HOST=0.0.0.0 SCALARLM_PORT=8000 scalarlm-server
```

### Using the Clean Architecture

```python
import torch
from scalarlm_vllm_adapters import (
    initialize_clean_scalarlm_integration,
    enhance_vllm_model, 
    TokenformerModel
)

# Initialize ScalarLM integration
initialize_clean_scalarlm_integration(config={
    "enable_tokenformer": True,
    "tokenformer_cache_capacity": 8
})

# Load your vLLM model (pure vLLM, no ScalarLM knowledge)
# ... model loading code ...

# Enhance with ScalarLM functionality
enhanced_model = enhance_vllm_model(
    model=your_vllm_model,
    device=torch.device("cuda"), 
    model_id="your-model-name"
)

# Use enhanced features
if enhanced_model.supports_tokenformer:
    # Load Tokenformer adapter
    tokenformer = TokenformerModel.from_local_checkpoint(
        "path/to/tokenformer", 
        device
    )
    enhanced_model.tokenformer_manager.add_adapter(tokenformer)
    enhanced_model.tokenformer_manager.activate_adapter(tokenformer.id)
```

## 🐳 Docker Support

### Prebuilt Containers

| Target | Container | Latest Release |
|--------|-----------|----------------|
| NVIDIA | `gdiamos/scalarlm-nvidia:latest` | `gdiamos/scalarlm-nvidia:v0.93` |
| ARM    | `gdiamos/scalarlm-arm:latest`    | `gdiamos/scalarlm-arm:v0.93`    |
| AMD    | `gdiamos/scalarlm-amd:latest`    | `gdiamos/scalarlm-amd:v0.93`    |
| x86    | `gdiamos/scalarlm-cpu:latest`    | `gdiamos/scalarlm-cpu:v0.93`    |

### Quick Docker Start

```bash
# Development server (M1/M2 Mac)
docker run -it -p 8000:8000 gdiamos/scalarlm-arm:latest

# NVIDIA GPU server
docker run -it -p 8000:8000 --gpus all gdiamos/scalarlm-nvidia:latest

# Build your own with clean architecture (NEW: uses vLLM fork)
make docker-build        # CPU version
make docker-build-gpu    # NVIDIA GPU version  
make docker-build-amd    # AMD GPU version

# Run with automatic vLLM fork integration
make docker-run          # CPU version
make docker-run-gpu      # NVIDIA GPU version
make docker-run-amd      # AMD GPU version

# Or use ./scalarlm up command (FIXED: now works with fork)
./scalarlm up cpu        # CPU version
./scalarlm up nvidia     # NVIDIA GPU version  
./scalarlm up amd        # AMD GPU version
```

## ⚙️ Configuration

### Environment Variables

```bash
# Core Settings
export SCALARLM_HOST="0.0.0.0"           # Server host (default: localhost)
export SCALARLM_PORT="8000"              # Server port (default: 8000)
export SCALARLM_MODEL="meta-llama/Llama-2-7b-hf"  # Default model

# Performance Settings
export SCALARLM_GPU_MEMORY_UTILIZATION="0.9"  # GPU memory usage
export SCALARLM_MAX_MODEL_LEN="2048"          # Maximum model length
export SCALARLM_TENSOR_PARALLEL_SIZE="1"      # Tensor parallelism

# Tokenformer Settings
export SCALARLM_ENABLE_TOKENFORMER="true"     # Enable Tokenformer
export SCALARLM_TOKENFORMER_CACHE_CAPACITY="8" # Cache capacity
```

### Configuration Files

ScalarLM looks for configuration in these locations (in order):
1. `./scalarlm.yaml` - Local project config
2. `~/.scalarlm/config.yaml` - User config
3. `/etc/scalarlm/config.yaml` - System config

Example `scalarlm.yaml`:
```yaml
server:
  host: 0.0.0.0
  port: 8000
  
model:
  name: meta-llama/Llama-2-7b-hf
  max_length: 2048
  
tokenformer:
  enabled: true
  cache_capacity: 8
  
performance:
  gpu_memory_utilization: 0.9
  tensor_parallel_size: 1
```

## 📂 Project Structure

```
scalarlm/
├── scalarlm_vllm_adapters/          # Clean architecture adapters
│   ├── __init__.py                  # Main adapter interface
│   ├── clean_integration.py         # Zero-coupling integration
│   ├── tokenformer_clean.py         # Tokenformer adapter
│   ├── adapter_commons.py           # Adapter base classes
│   └── ...
├── examples/
│   └── clean_architecture_demo.py   # Demo of clean architecture
├── scripts/
│   ├── validate_setup.py            # Setup validation
│   └── migrate_to_vllm_fork.py      # Migration tool
├── tests/
│   ├── test_clean_architecture.py   # Integration tests
│   └── test_architecture_basic.py   # Basic tests
├── infra/                           # ScalarLM infrastructure
├── ml/                              # Training and ML components
├── deployment/                      # Deployment configurations
├── pyproject.toml                   # Modern Python packaging
├── Makefile                         # Development workflow
└── README.md                        # This file
```

## 🔄 Migration from Embedded vLLM

If you're upgrading from an older ScalarLM version with embedded vLLM:

```bash
# Check migration readiness
make migration-check

# Run automated migration
make migration

# Review migration report
cat MIGRATION_REPORT.md
```

## 🧪 Testing

### Test Levels

1. **Basic Architecture Test** - No heavy dependencies
   ```bash
   python test_architecture_basic.py
   ```

2. **Full Integration Test** - With PyTorch
   ```bash  
   python test_clean_architecture.py
   ```

3. **Demo Test** - Complete workflow
   ```bash
   python examples/clean_architecture_demo.py
   ```

### Running All Tests

```bash
make test                # pytest tests
make integration-test    # Clean architecture tests
make validate-setup      # Setup validation
```

## 📊 Features

### Core Features
- 🚀 **High-performance inference** via vLLM
- 🎯 **Advanced training** with Megatron-LM integration
- 🔌 **OpenAI-compatible API** for easy integration
- 📈 **Distributed training** capabilities
- 🎛️ **Tokenformer adapters** for enhanced performance

### Clean Architecture Benefits
- 🏗️ **Zero coupling** between vLLM and ScalarLM
- 🔄 **Version independence** - use any vLLM version
- 🛡️ **Robust dependency management**
- 🔧 **Easy maintenance** and updates
- 📦 **Modern packaging** with pyproject.toml

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Make your changes
4. Run tests (`make test integration-test`)
5. Commit your changes (`git commit -m 'Add amazing feature'`)
6. Push to the branch (`git push origin feature/amazing-feature`)
7. Open a Pull Request

### Development Guidelines

- Follow the clean architecture principles
- Maintain zero coupling between vLLM and ScalarLM
- Add tests for new features
- Update documentation as needed
- Use the provided Makefile for development tasks

## 📚 Documentation

- [Full Documentation](https://www.scalarlm.com)
- [Clean Architecture Guide](CLEAN_ARCHITECTURE_SOLUTION.md)
- [Migration Guide](MIGRATION_GUIDE.md) 
- [Blog](https://www.scalarlm.com/blog)

## 🆘 Troubleshooting

### Common Issues

**Installation fails with old pip:**
```bash
# Upgrade pip first
python -m pip install --upgrade pip
pip install -e .
```

**vLLM not found (REQUIRED):**
```bash
# vLLM is required for ScalarLM to function. Install it using:

# Option 1: Install ScalarLM's fork (recommended)
make install-vllm

# Option 2: Install standard vLLM
pip install vllm>=0.6.0

# Verify installation
python -c "import vllm; print(f'vLLM {vllm.__version__} installed')"
```

**Import errors:**
```bash
# Validate setup
make validate-setup
# Check Python path
python -c "import scalarlm_vllm_adapters; print('OK')"
```

**Docker issues: ✅ FIXED**
```bash
# Docker now automatically uses ScalarLM's vLLM fork
# No more "No module named 'infra.vllm.entrypoints'" errors

# If you encounter issues, rebuild with latest fixes:
make docker-build

# Verify vLLM fork is properly installed:
make check-vllm-fork
```

### Getting Help

- [GitHub Issues](https://github.com/scalarlm/scalarlm/issues)
- [Documentation](https://www.scalarlm.com)
- [Community Forum](https://github.com/scalarlm/scalarlm/discussions)

## 📄 License

ScalarLM is licensed under the Apache License 2.0. See [LICENSE](LICENSE) for details.

## 🙏 Acknowledgments

ScalarLM is inspired by the work of **Seymour Roger Cray** (1925-1996), "the father of supercomputing", who created the supercomputer industry and designed the fastest computers in the world for decades.

Built with:
- [vLLM](https://github.com/vllm-project/vllm) - High-performance LLM inference
- [Megatron-LM](https://github.com/NVIDIA/Megatron-LM) - Large-scale training
- [HuggingFace](https://huggingface.co/) - Model hub and transformers
- [PyTorch](https://pytorch.org/) - Deep learning framework

---

**Ready to get started?** Run `make setup-dev` to set up your development environment!