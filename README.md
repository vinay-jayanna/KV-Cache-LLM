# KV Cache in Transformers – Optimizing LLM Inference

This repository provides practical code examples and performance benchmarks for understanding and implementing **Key-Value (KV) Caching** in Transformer-based Large Language Models (LLMs).

KV Caching enables fast, efficient, and scalable inference by **storing and reusing Key and Value tensors**, eliminating the need to recompute them at every step.

---

## 🔍 Technical Deep Dive

For a complete walkthrough of KV Caching, including how it works, why it matters, and how it's applied in production-grade AI systems, read the accompanying article on LinkedIn:

👉 [KV Caching in Transformers: Optimizing Large Language Model Inference](https://www.linkedin.com/pulse/kv-cache-hidden-optimization-behind-real-time-ai-vinay-jayanna-cvfec)

---

## 📁 Code Structure

- `naive_transformer_inference.py`  
  Demonstrates token-by-token inference without KV caching, recomputing K and V at every step.

- `kv_cache_transformer_inference.py`  
  Optimized inference using KV caching, storing previously computed K and V values to avoid redundancy.

- `benchmark_inference.py`  
  Compares execution time of naive vs. KV cache implementations.

- `huggingface_kv_cache_inference.py`  
  Shows how to use KV cache with Hugging Face Transformers in real-world LLM inference.

- `huggingface_kv_cache_benchmark.py`  
  Benchmark script to measure inference latency with and without caching using Hugging Face APIs.

- `gpu_memory_profiling.py`  
  Measures GPU memory usage with cached K and V tensors.

- `tensorrt_llm_kv_cache_example.py`  
  Simulated example showing how KV cache is applied in NVIDIA TensorRT-LLM optimized inference.

---

## 🚀 Setup Instructions

Install required libraries:

```bash
pip install torch torchvision torchaudio
pip install transformers
pip install nvidia-pytrt-llm  # If using TensorRT-LLM with NVIDIA GPUs
```

---

## 🧪 How to Run

Run naive transformer inference:

```bash
python naive_transformer_inference.py
```

Run optimized inference with KV cache:

```bash
python kv_cache_transformer_inference.py
```

Benchmark both approaches:

```bash
python benchmark_inference.py
```

Run inference with Hugging Face Transformers:

```bash
python huggingface_kv_cache_inference.py
```

Compare Hugging Face performance:

```bash
python huggingface_kv_cache_benchmark.py
```

---

## 📊 Performance Insight

- **Without KV Cache:** O(n²) time complexity due to recomputation of attention at each step.
- **With KV Cache:** Reduces to O(n), enabling faster, scalable inference for long sequences.

---

## 📄 License

This project is licensed under the MIT License. Feel free to use and extend it.

---

Contributions, improvements, and feedback are welcome. If this helped you, consider leaving a star on the repo!
