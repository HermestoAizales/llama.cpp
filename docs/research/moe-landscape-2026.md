# MoE Inference Optimization Landscape — May 2026

> **Research date**: 2026-05-26
> **Sources**: arXiv, ddgr (DuckDuckGo), GitHub, vLLM/SGLang docs
> **Goal**: Identify relevant MoE optimization techniques for our llama.cpp fork

---

## 1. Key Papers (May 2026)

### 1.1 TritonMoE — Cross-Platform Fused MoE Dispatch (arxiv 2605.23911)

**Authors**: bassrehab (Subhadip Mitra)
**Published**: 2026-05-26

**Core idea**: Pure Triton fused MoE kernel replacing CUDA. Full forward pass (router scoring, token permutation, expert GEMMs, weighted output) in portable Triton.

**Key results**:
- **89-131% of Megablocks** throughput at batch ≤ 512 tokens (Mixtral-8x7B, DeepSeek-V3, Qwen2-MoE)
- **35% global memory traffic reduction** via fused gate+up GEMM with in-register SiLU
- **Cross-platform**: NVIDIA A100 + AMD MI300X with zero code changes
- Fixed-tile scheduling underperforms Megablocks at 64+ experts with extreme routing skew
- Code: https://github.com/bassrehab/triton-kernels

**Relevance to our fork**:
- Confirms fused gate+up as highest-impact optimization (matches our approach)
- Triton path could add AMD support without CUDA rewrite
- Our CUDA kernel should target >89% of Megablocks as minimum bar

---

### 1.2 ReMoE — Expert Reuse via Router Fine-Tuning (arxiv 2605.27081)

**Published**: 2026-05-26

**Core idea**: Fine-tune router to increase temporal expert reuse. Biases routing toward recently selected experts, matching cache locality.

**Key results**:
- **26% expert reuse improvement** on DeepSeek + Qwen models
- **8.4% throughput gain** under vLLM GPU-CPU expert offloading
- **43.6-49.8% TPOT reduction** on Jetson Orin NX (1.77-1.99× decode speedup)
- No added inference-time computation

**Relevance to our fork**:
- Complementary to our cache-based approach
- Router fine-tuning could improve our cache hit rates
- Particularly relevant for memory-constrained setups

---

### 1.3 GEMQ — Global Expert-Level Mixed-Precision Quantization (arxiv 2605.23078)

**Published**: 2026-05-21

**Core idea**: Mixed-precision quantization at expert level with global linear programming for importance estimation + router fine-tuning adaptation.

**Key results**:
- Significant memory reduction with minimal accuracy degradation
- Outperforms layer-wise importance estimation
- Integrated progressive quantization framework

**Relevance to our fork**:
- We support Q4_K, Q5_K, Q6_K, Q8_0 but use uniform quantization
- Expert-level mixed-precision could further reduce memory → more experts cached
- Integration opportunity: GEMQ + our fused dispatch

---

### 1.4 MobileMoE — On-Device MoE Scaling (arxiv 2605.27358)

**Published**: 2026-05-26

**Core idea**: MoE at sub-billion scale for on-device. Moderate sparsity with fine-grained + shared experts.

**Key results**:
- 0.3-0.9B active / 1.3-5.3B total parameters
- **2-4× fewer FLOPs** vs dense on-device LLMs (same quality)
- **1.8-3.8× faster prefill**, **2.2-3.4× faster decode** on smartphone (INT4)
- On-device MoE scaling law derived

**Relevance to our fork**:
- Validates MoE for edge/CPU inference
- llama.cpp already targets edge; MobileMoE techniques could apply
- Shared expert pattern reduces cache pressure

---

### 1.5 StreamIndex — Memory-Bounded Compressed Sparse Attention (arxiv 2605.02568)

**Published**: 2026-05-26

**Core idea**: Triton implementation of DeepSeek-V4's Compressed Sparse Attention (CSA) with streaming top-k. Never materializes full score tensor.

**Key results**:
- 65K seq length: OOM → 6.21 GB peak HBM (256 GB → 6.21 GB)
- Up to 1M seq length on single H200
- Bit-exact recall ≥ 0.9980 across design space

**Relevance to our fork**:
- DeepSeek-V4 uses MoE + CSA together
- Long-context MoE requires efficient attention too
- Triton kernel portable NVIDIA+AMD

---

### 1.6 DeepSeek-V4 Technical Report (April 2026)

**Core facts**:
- DeepSeek-V4-Pro: 1.6T total, 49B activated per token
- DeepSeek-V4-Flash: 284B total, 13B activated
- 1M context length support
- Key innovations: **Hash routing** (replacing learned router), compressed sparse attention, **auxiliary-loss-free load balancing**, FP4 quantization-aware training for MoE experts

**Relevance to our fork**:
- Hash routing has zero compute overhead vs learned router — could simplify our dispatch
- FP4 expert weights would be 2× smaller than our Q4_K
- Auxiliary-loss-free balancing means inference router is simpler

---

## 2. Framework Landscape

| Framework | MoE Approach | Language | Platform |
|-----------|-------------|----------|----------|
| **vLLM** | FusedMoE Triton kernel (GroupedTopk) | Triton | NVIDIA |
| **SGLang** | fused_moe_triton with TMA tuning | Triton | NVIDIA |
| **Megablocks** | Block-sparse grouped GEMM | CUDA + HIP | NVIDIA + AMD (HIP) |
| **DeepSpeed-MEI** | Expert prefetch + model parallel | PyTorch | NVIDIA |
| **llama.cpp (our fork)** | Fused gate_up + async prefetch + VRAM cache | CUDA | NVIDIA |
| **TritonMoE** | Full dispatch fusion | Triton | NVIDIA + AMD |

---

## 3. Optimization Techniques Summary

| Technique | Impact | Complexity | Our Status |
|-----------|--------|------------|------------|
| Fused gate+up GEMM | -35% memory traffic | Medium | ✅ Implemented |
| Async expert prefetch | Hides PCIe latency | Medium | ✅ Implemented |
| VRAM expert cache (LRU) | Reduces repeated transfers | Low | ✅ Implemented |
| Triton portability | AMD support | High | ❌ NVIDIA only |
| Hash routing | Zero router overhead | Low | ❌ Learned router |
| Expert quantization (FP4/Q4) | -50-75% expert memory | Medium | ✅ Q4_K/Q5_K/Q6_K |
| Router fine-tuning (ReMoE) | +26% cache hit rate | Training | ❌ Not implemented |
| Global mixed-precision (GEMQ) | Pareto-optimal compression | Medium | ❌ Uniform quant |
| Shared experts | -1 expert always cached | Architecture | ❌ Not implemented |
| Sparse attention (StreamIndex) | Enables 1M+ context | High | ❌ Standard attention |

---

## 4. Recommendations for llama.cpp Fork

### 4.1 Short-term (do now)
1. **Benchmark our CUDA kernel vs baseline** (no fused MoE) on Mixtral-8x7B Q4_K
2. **Fix `backend_sampling` removal** broke speculative decoding path — need proper fix
3. **Add TritonMoE as optional backend** — adds AMD support, validates our approach

### 4.2 Medium-term (next month)
1. **Implement hash routing** option (DropIn replacement for learned router)
2. **Add FP4 expert weight support** (2× smaller than Q4_K)
3. **Expert-level mixed-precision** (GEMQ-style, using our cache architecture)
4. **Integration: ReMoE router fine-tuning** (add Llama.cpp training recipe)

### 4.3 Long-term
1. **Triton backend for full portability** (one kernel, NVIDIA + AMD)
2. **Sparse attention + MoE** for million-token DeepSeek-V4-scale contexts
3. **Shared expert pattern** reduces cache pressure
4. **Custom MoE architecture training** (MobileMoE-style for edge)

---

## 5. Benchmark Hardware Requirements

| GPU | VRAM | Mixtral-8x7B Q4_K fits? | DeepSeek-V3 Q4_K fits? |
|-----|------|------------------------|----------------------|
| RTX 4090 | 24 GB | ✅ (with offload) | ❌ |
| A100 40GB | 40 GB | ✅ | ❌ |
| A100 80GB | 80 GB | ✅ | ✅ (partial cache) |
| H100 80GB | 80 GB | ✅ | ✅ (partial cache) |
| H200 141GB | 141 GB | ✅ | ✅ (fits fully) |
| AMD MI300X | 192 GB | ✅ | ✅ (fits fully) |
