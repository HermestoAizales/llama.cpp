# Fused MoE — Research & Benchmark Plan

> **Status**: Draft — 2026-05-26
> **Goal**: Theoretical and practical justification for our Fused MoE approach + benchmark plan vs. Megablocks baseline

---

## 1. Scientific Background

### 1.1 Mixture of Experts (MoE) Inference Challenge

MoE models (Mixtral, DeepSeek-V3, Qwen3-MoE) activate only a subset of experts per token (top-k routing). At inference:

- **Memory bandwidth bottleneck**: Expert weights (~10-40GB) exceed VRAM capacity
- **Underutilization**: GPU idle during PCIe weight transfers
- **Kernel launch overhead**: 20-30+ separate `mul_mat` calls per layer

### 1.2 Key Papers

| Paper | Year | Contribution |
|-------|------|-------------|
| **Megablocks** (Brandon et al., Stanford) | 2022 | Grouped GEMM for MoE, sorted-token dispatch, HIPT baseline |
| **DeepSpeed-MEI** (Rajbhandari et al.) | 2022 | Expert prefetching, overlapping compute/transfer |
| **FastMoE** (He et al.) | 2021 | Dynamic expert routing with PyTorch Dispatch |
| **ScatteredMoE** | 2024 | Memory-efficient MoE with fragmented expert storage |
| **TritonMoE** (bassrehab, arxiv 2605.23911) | 2026 | Fused dispatch kernel in pure Triton, 89-131% of Megablocks |
| **SmoothQuant+MoE** | 2023 | Quantization-aware MoE inference |
| **DeepSeek-V3 Technical Report** | 2024 | Multi-Token Prediction + MoE, auxiliary-load-balancing loss |

### 1.3 Relevant Optimizations (from literature)

1. **Token Dispatch Fusion**: Group tokens by expert → single batched GEMM per expert (Megablocks)
2. **Expert Weight Prefetch**: Overlap PCIe transfer of next expert's weights with current expert's compute
3. **Fused Gate+Up Projection**: Compute gate and up projections in one kernel, keep intermediate in registers/SharedMem
4. **Shared Expert**: One expert always active (DeepSeek-V3), reduces per-token memory
5. **Quantized Expert Weights**: Q4/Q8 experts to fit more in VRAM

---

## 2. Our Fused MoE Implementation

### 2.1 Architecture (`ggml/src/ggml-cuda/fused-moe.cu`)

```
Standard path (24+ kernel launches):
  for each (token, expert):
    mul_mat(gate_weight[expert], hidden) → gate_act
    mul_mat(up_weight[expert], hidden) → up_val
    SiLU(gate_act) → gate_out
    gate_out * up_val → intermediate
    mul_mat(down_weight[expert], intermediate) → output

Fused path (5 kernel launches):
  1. Fused gate_up projection (all experts, tiled)
  2. SwiGLU activation (fused with projection)
  3. Down projection
  4. Residual scaling
  5. Final accumulation
```

### 2.2 Key Optimizations

| Optimization | Expected Impact |
|-------------|----------------|
| Async weight prefetch (dedicated CUDA streams) | Hide PCIe latency behind attention compute |
| VRAM expert cache (LRU eviction) | Avoid repeated transfers for frequently used experts |
| Fused gate_up + SwiGLU | ~35% global memory traffic reduction (matches TritonMoE finding) |
| Supports Q4_K, Q5_K, Q6_K, Q8_0 | Reduced VRAM footprint → more experts cached |
| Multi-architecture support | Qwen35/36, DeepSeek V3, Llama4, GroveMoE, Step35 |

### 2.3 Limitations (Known)

- Fallback to standard path for quantized types (not all kernels fused)
- Cache hit rate depends on routing locality
- PCIe Gen4 x16 has ~25 GB/s, HBM3 has ~3 TB/s → 100x bandwidth gap

---

## 3. Comparison with Existing Approaches

### 3.1 vs. Megablocks (Stanford)

| Aspect | Megablocks | Our Fused MoE |
|--------|-----------|--------------|
| Language | CUDA | CUDA |
| Dispatch | Sorted-token dispatch + grouped GEMM | Per-token async prefetch |
| Prefetch | Expert-parallel, model-parallel overlap | Dedicated prefetch streams |
| Quantization | FP16/BF16 | Q4_K, Q5_K, Q6_K, Q8_0, F16, BF16 |
| Portability | NVIDIA + HIP (AMD) | NVIDIA only |

### 3.2 vs. TritonMoE (bassrehab)

| Aspect | TritonMoE | Our Fused MoE |
|--------|----------|--------------|
| Language | Pure Triton | CUDA C++ |
| Portability | NVIDIA + AMD (MI300X) | NVIDIA only |
| Performance | 89-131% of Megablocks @ batch ≤512 | Unknown (needs benchmark) |
| Fused Gate+Up | ✅ (35% memory traffic reduction) | ✅ |
| Quantization | F16/BF16 | Q4_K, Q5_K, Q6_K, Q8_0 |

### 3.3 vs. DeepSpeed-MEI

| Aspect | DeepSpeed-MEI | Our Fused MoE |
|--------|--------------|--------------|
| Framework | PyTorch | llama.cpp (GGML) |
| Prefetch | Expert-parallel | Dedicated streams |
| Quantization | FP16 | Q4_K, Q5_K, Q6_K, Q8_0 |

---

## 4. Benchmark Plan

### 4.1 Target Models

| Model | Architecture | n_expert | n_expert_used | Total Expert Size |
|-------|-------------|----------|---------------|-------------------|
| Mixtral-8x7B | Mistral MoE | 8 | 2 | ~47 GB (F16) |
| Qwen3-35B-A3B | Qwen MoE | 128 | 8 | ~35 GB (F16) |
| DeepSeek-V3 | DeepSeek MoE | 256 | 8 | ~373 GB (F16) |

**Recommended**: Mixtral-8x7B (most cited, good DeepSeek-V3 proxy)

### 4.2 Metrics

- **Tokens/second** (primary metric)
- **Expert cache hit rate**
- **PCIe transfer volume** (bytes/sec)
- **GPU utilization** (%)
- **VRAM usage** (peak)

### 4.3 Test Configurations

| Configuration | Description |
|--------------|-------------|
| **Baseline** | Standard `mul_mat_id` (no fused MoE) |
| **Fused MoE** | Our implementation, prefetch=2 streams, Q4_K experts |
| **Megablocks** (if available) | Stanford's CUDA-optimized MoE library |

### 4.4 Environment

- **GPU**: A100 80GB or RTX 4090 24GB
- **CPU**: AMD EPYC or Ryzen 7950X (128 GB RAM)
- **OS**: Ubuntu 24.04, CUDA 12.4, cuBLAS

### 4.5 Test Matrix

| Variable | Values |
|----------|--------|
| Batch size | 1, 16, 64, 128, 256, 512 |
| Context length | 512, 2048, 8192 |
| Expert cache size | 0, 2, 4, 8, 16, all |
| Quantization | F16, Q4_K, Q8_0 |
| Routing | Top-2 (Mixtral), Top-8 (DeepSeek-V3) |

---

## 5. Open Research Questions

1. **What is the theoretical peak speedup from expert prefetching?**
   - PCIe Gen4 x16: ~25 GB/s
   - Expert weight size (Mixtral Q4): ~734 MB per expert = ~30ms transfer
   - A100 FP16 TFLOPS: 312 → token FFN time ~0.5ms
   - **Prediction**: Need ~60 experts in flight to saturate GPU
   - **Reality**: Only 2-8 experts active → prefetching helps but is not miracle

2. **Does the Triton approach (portable) outperform CUDA for small batches?**
   - TritonMoE: 89-131% of Megablocks @ batch ≤512
   - Megablocks baseline: performance at batch=1 is typically 20-40% of peak
   - Our CUDA kernel: unknown

3. **Is quantization-aware MoE dispatch beneficial?**
   - Q4_K experts: 4x smaller → 4x more experts fit in VRAM
   - Dequantization overhead in kernel
   - **Tradeoff**: Cache size vs. per-token dequant cost

---

## 6. Related Work & Code Repositories

| Repository | Description |
|-----------|-------------|
| [Megablocks](https://github.com/stanford-futuredata/megablocks) | Stanford's CUDA MoE library, state-of-the-art |
| [triton-kernels](https://github.com/bassrehab/triton-kernels) | Fused MoE dispatch in pure Triton |
| [DeepSpeed-MEI](https://github.com/microsoft/DeepSpeed) | Microsoft's MoE inference optimizer |
| [FastMoE](https://github.com/laekov/fastmoe) | Dynamic expert routing |
| [llama.cpp Fused MoE](ggml/src/ggml-cuda/fused-moe.cu) | Our implementation |

---

## 7. Roadmap

### Phase 1: Benchmark Infrastructure (this week)
1. Set up Mixtral-8x7B Q4_K inference
2. Measure baseline TPS (no fused MoE) across batch sizes
3. Measure fused MoE TPS with different cache sizes
4. Document cache hit rates

### Phase 2: Megablocks Integration Prototype (next week)
1. Integrate Megablocks library
2. Compare Megablocks vs. Fused MoE on same model
3. Identify performance gaps

### Phase 3: Optimization (weekend)
1. Based on Phase 2 results, identify top 3 bottlenecks
2. Implement fixes
3. Re-benchmark

### Phase 4: Triton Port (future)
1. Evaluate TritonMoE for AMD support
2. Implement Triton dispatch kernel
3. Cross-platform benchmarks

---

## 8. CLI Flags Reference

| Flag | Type | Default | Description |
|------|------|---------|-------------|
| `--fused-moe` | bool | off | Enable fused MoE dispatch |
| `--moe-prefetch-streams` | int | 2 | Number of async prefetch streams |
| `--moe-max-vram-mb` | int | 0 | Max VRAM for expert cache (0=auto) |
| `--n-cpu-moe` | int | 0 | Offload N MoE layers to CPU (0=all GPU) |
