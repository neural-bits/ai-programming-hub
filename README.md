# AI Programming Hub
This repository contains the code for the NeuralBits newsletter articles, covering new programming languages and 

## Categories
### Languages
|ID| 📝&nbsp; Article  | 💻&nbsp;Code | Details | Complexity | Tech Stack |
|--|---------|-----------------|---------|------------|----------------------|
|001| [Mojo Programming Language](https://neuralbits.substack.com/p/python-flexibility-and-c-performance)| [Here](https://github.com/neural-bits/ai-programming-hub/tree/main/001-Mojo-vs-Python) | Learn about Mojo for AI, run a benchmark against Python | 🟩🟩⬜ |Python, Mojo, Jupyter|
|002| [Rust to Python Bindings](https://neuralbits.substack.com/p/lets-build-andrej-karpathys-bpetokenizer)| [Here](https://github.com/neural-bits/ai-programming-hub/tree/main/002-Rust-bindings-to-Python) | Build the BPE Tokenizer in Rust, generate Python Bindings |  🟩🟩⬜ |Python, Rust, PyO3, Maturin|


---

## 📁 Repo Structure

```
ai-programming-hub/
├── tensor-internals/           # Memory layout, tensor streaming, sparsity
├── custom-gpu-kernels/         # Mojo, Triton, Cupy, async CUDA
├── llm-systems/                # Context management, routing, speculative decoding
├── embeddings-at-scale/        # Embedding infra, indexing, retrieval, routing
├── compiler-ai/                # MLIR, Mojo, Triton optimization & fusion
├── deployment-ops/             # Model serving, tokenizer latency, quantization
├── high-performance-ml/        # Benchmarking, cost/latency, parallelism
├── multi-agent-systems/        # Agent memory, task decomposition, evals
├── real-world-projects/        # End-to-end systems (Vision AI, Perception Based, RAG, agents)
```

---


### 📦 tensor-internals/

**Focus:** Low-level tensor mechanics most ML engineers ignore.

- Tensor memory layout: strides, alignment, NHWC/NCHW
- Streaming tensor pipelines for massive model data
- Zero-copy memory: DLPack, pinned memory, memory-mapped IO
- Sparse tensor formats: CSR, block-sparse, pruning efficiency
- How hardware constraints affect tensor op performance

---

### ⚙️ custom-gpu-kernels/

**Focus:** Writing and profiling GPU kernels in Cupy, Triton, and Mojo.

- Custom matrix multiplication kernels (Cupy, Mojo)
- Mojo vs Triton: writing and benchmarking GPU ops
- Async CUDA streams and multi-kernel execution
- Kernel fusion and memory coalescing
- GPU profiling: NVTX, Nsight, PyTorch profiler

---

### 🧠 llm-systems/

**Focus:** Building LLMs into real systems — not just toy prompts.

- LLM inference optimization: prefill, caching, throughput
- Speculative decoding explained and demoed
- Context routing between multiple LLMs (e.g., Whisper + BLIP + GPT)
- Planning and routing using tools or sub-agents
- Streaming LLM output with throttled token budgets

---

### 🔍 embeddings-at-scale/

**Focus:** Working with embeddings as infrastructure components.

- Embedding-based query routing (vector-aware proxies)
- Chunking, reranking, hybrid retrieval with compression
- Embedding dimensionality tradeoffs (recall vs latency vs cost)
- Trainable vector similarity scoring
- End-to-end vector DB-backed retrieval pipelines (FAISS, HNSW)

---

### 🧬 compiler-ai/

**Focus:** AI-native compilers for maximum performance.

- Write a custom kernel with Mojo
- Build and benchmark attention kernels with Triton
- Introduction to MLIR: graph IR, tensor fusion
- Compare compiler pipelines: ONNX, TVM, MLIR, Mojo
- How tensor compilers schedule, align, and fuse ops

---

### 🚀 deployment-ops/

**Focus:** Techniques to scale and serve AI systems reliably.

- Quantized model serving with ONNX/TensorRT
- Model sharding, tensor parallelism (via DeepSpeed or custom code)
- GPU tokenizer caching + optimization for latency
- FastAPI + Triton + GPU-aware REST APIs
- Real-time model profiling + memory + token usage

---

### 🔧 high-performance-ml/

**Focus:** Performance tradeoffs and profiling for real production use.

- Pipeline vs data parallelism (and when each fails)
- Custom allocators: Pytorch, CUDA, Mojo memory strategies
- Profiling throughput: tokens/sec, latency breakdown
- Real cost-aware benchmarking (cloud cost/token)
- Python vs Rust vs Mojo performance comparison

---

### 🤖 multi-agent-systems/

**Focus:** LLM-based agents that think, retrieve, and act.

- Implement Tree-of-Thought, ReAct, Reflexion
- Use vector DBs as persistent memory for agents
- Routing logic between agents and sub-tools
- Plan/decompose/execute agent loops
- Evaluate agents using real-world tasks (latency, failure recovery)

---

### 🧪 real-world-projects/

Each folder is a minimal but complete build of a real-world AI system.

- `rag_service_pipeline/`: Retrieval + rerank + OpenAI + local embeddings
- `gpu_tensor_playground/`: Run benchmarks for matrix kernels (Cupy, Mojo, Triton)
- `ai-agent_with_tools/`: Agent with tool use, planner, memory store
- `embedding_routing_gateway/`: Proxy service using semantic embeddings
- `low-latency_inference_server/`: Triton + quant + tokenizer optimizations

---
Stuff we'll  need:
- Python 3.10+
- CUDA-capable GPU
- Mojo SDK (https://www.modular.com/mojo)
- Triton, FAISS, Cupy, MLIR, PyTorch

---

## Maybe good topics?

- `ai-on-edge/`: Jetson Nano + WASM browser inference
- `compiler-ai/`: Build an MLIR optimization pass from scratch (too adv?)
- `projects/`: Embed agents in edge environments (might be cool)

---

# TODO: add contrib, and clearer roadmap etc