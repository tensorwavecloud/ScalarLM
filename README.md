# ScalarLM - Advanced LLM Platform with Clean vLLM Integration

[![License: Apache-2.0](https://img.shields.io/badge/License-Apache%202.0-blue.svg)](https://opensource.org/licenses/Apache-2.0)
[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)

ScalarLM is a fully open source, integrated LLM inference and training platform built on top of vLLM, Huggingface, and Megatron-LM

## 📋 Core Dependencies

ScalarLM is built on top of these core components:
- **vLLM** - High-performance LLM inference engine
- **Megatron-LM** - Training harness, distribution strategy
- **PyTorch** - Deep learning framework
- **Transformers** - Model implementations and utilities
- **FastAPI** - API server framework

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

# Start it
./scalarlm up

```

## 📦 What's New in v1.0: Clean Architecture

ScalarLM has been completely redesigned with a **clean architecture** that solves dependency management issues:

### After: Clean Architecture (Solution)
- ✅ **Zero coupling** - vLLM has no knowledge of ScalarLM
- ✅ **External enhancement** - ScalarLM adapters enhance vLLM models
- ✅ **Version independence** - Use any vLLM version
- ✅ **Clean separation** - Both systems evolve independently


## 🏃‍♂️ Running ScalarLM

### Quick Start with `scalarlm` CLI

```bash
# Start ScalarLM server (simplest way)
./scalarlm up

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

## 🐳 Docker Support

### Prebuilt Containers

| Target | Container | Latest Release |
|--------|-----------|----------------|
| NVIDIA BLACKWELL | `gdiamos/scalarlm-nvidia-12.0:latest` | `gdiamos/scalarlm-nvidia-12.0:v0.99` |
| NVIDIA HOPPER    | `gdiamos/scalarlm-nvidia-8.0:latest`  | `gdiamos/scalarlm-nvidia-8.0:v0.99`  |
| NVIDIA HOPPER    | `gdiamos/scalarlm-nvidia-8.6:latest`  | `gdiamos/scalarlm-nvidia-8.6:v0.99`  |
| NVIDIA ADA       | `gdiamos/scalarlm-nvidia-7.5:latest`  | `gdiamos/scalarlm-nvidia-7.5:v0.99`  |
| ARM              | `gdiamos/scalarlm-arm:latest`         | `gdiamos/scalarlm-arm:v0.99`         |
| AMD              | `gdiamos/scalarlm-amd:latest`         | `gdiamos/scalarlm-amd:v0.99`         |
| x86              | `gdiamos/scalarlm-cpu:latest`         | `gdiamos/scalarlm-cpu:v0.99`         |

### Quick Docker Start

```bash
# Or use ./scalarlm up command
./scalarlm up cpu        # CPU version
./scalarlm up nvidia     # NVIDIA GPU version
./scalarlm up amd        # AMD GPU version
```

## ⚙️ Configuration

### Environment Variables

```bash
# Core Settings
export SCALARLM_MODEL="meta-llama/Llama-2-7b-hf"  # Default model

# Performance Settings
export SCALARLM_GPU_MEMORY_UTILIZATION="0.9"     # GPU memory usage
export SCALARLM_MAX_MODEL_LENGTH="2048"          # Maximum model length
```

### Configuration Files

ScalarLM looks for configuration in these locations (in order):
1. `/app/cray/cray-config.yaml` - Local project config (in the container)

Example `cray-config.yaml`:
```yaml
model: meta-llama/Llama-2-7b-hf
max_model_length: 2048

gpu_memory_utilization: 0.9
```

## 📂 Project Structure

```
scalarlm/
├── tests/                          # Unit and integration tests
├── infra/                           # ScalarLM infrastructure
├── ml/                              # Training and ML components
├── deployment/                      # Deployment configurations
└── README.md                        # This file
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
- [Blog](https://www.scalarlm.com/blog)


### Getting Help

- [GitHub Issues](https://github.com/scalarlm/scalarlm/issues)
- [Documentation](https://www.scalarlm.com)
- [Community Forum](https://github.com/scalarlm/scalarlm/discussions)

## 📄 License

ScalarLM is licensed under the CC-0 License. See [LICENSE](LICENSE) for details.

## 🙏 Acknowledgments

ScalarLM is inspired by the work of **Seymour Roger Cray** (1925-1996), "the father of supercomputing", who created the supercomputer industry and designed the fastest computers in the world for decades.

Built with:
- [vLLM](https://github.com/vllm-project/vllm) - High-performance LLM inference
- [Megatron-LM](https://github.com/NVIDIA/Megatron-LM) - Large-scale training
- [HuggingFace](https://huggingface.co/) - Model hub and transformers
- [PyTorch](https://pytorch.org/) - Deep learning framework

---

**Ready to get started?** Run `./scalarlm up` to set up your development environment!
