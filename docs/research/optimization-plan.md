# Optimization Plan — Low-Hanging Fruit

> **Created**: 2026-05-26
> **Branch**: `feature/optimization-low-hanging-fruit`
> **Priority**: Impact × ease of implementation

---

## Architecture Understanding

### MoE Dispatch Chain
```
Model (qwen35moe.cpp)
  → llama-graph.cpp (generic graph construction)
    → ggml backend ops (GGML_OP_MUL_MAT_ID)
      → mmid.cu (standard MoE, 24+ kernel launches)
      → mmf.cu (fused MoE template, 5 kernel launches) [EXISTS but DISABLED]
      → fused-moe.cu (our fork kernel, 5 launches + async prefetch) [EXISTS but DISABLED]
```

### Current State
- **mmid.cu**: Standard path, works, slow (24+ kernel launches per MoE layer)
- **mmf.cu**: Upstream fused template — exists, compiled, but NOT wired to cparams
- **fused-moe.cu**: Our fork kernel — exists (682 LOC), compiled, but NEVER called

---

## Phase 1: Quick Wins (this session)

### 1.1 ✅ Fix cparams transfer (DONE 2026-05-26)
All fork flags transferred from `common_params` to `llama_cparams` in `common_context_params_to_llama()`.

### 1.2 ✅ Wire Up Fused MoE Dispatch (DONE 2026-05-27)
- `ggml_cuda_fused_moe_forward()` dispatched in `ggml_cuda_mul_mat_id()` when `g_fused_moe_enabled[device]`
- `down_exps` MUL_MAT_ID skipped in graph when `fused_moe && gate_up_exps` (fused kernel handles down)
- Dynamic function loading via `ggml_backend_reg_get_proc_address()`

### 1.3 ✅ Add `n_cpu_moe` Internal Parameter (DONE 2026-05-27)
Implemented: `n_cpu_moe` in `llama_cparams`, `llama_model_params`, `llama_context_params`, load_tensors() dispatch keeps last N MoE layers on CPU.

### 1.4 ✅ Bounded KV Cache Eviction + Async Checkpoint (DONE 2026-05-27)
- Eviction logic (`evict_bounded()`) implemented and called in `init_batch()`
- Checkpoint extraction uses `ggml_backend_tensor_get_async` + `std::async` background thread
- Thread-safe checkpoint buffer access via `std::mutex`
if (cparams.kv_cache_bounded > 0 && kv->n_res_checkpoints > kv->max_bounded) {
    kv->evict_bounded(n_evict);
}
```

**Expected impact**: Enables 128K+ context on 24GB GPUs (vs OOM at ~32K)
**Complexity**: ~40 LOC
**Risk**: Very low

---

## Phase 2: Optimize Existing Paths (next session)

### 2.1 Eliminate Redundant Memory Copies in MoE
Expert weight prefetch copies weights even when they're already cached.

**Fix**: Check cache before copy in `ggml_cuda_fused_moe_prefetch()`.

### 2.2 Increase Default `moe_max_vram_mb` from 0 (auto) to 80% of free VRAM
Users expect "just works" — auto should intelligently use available VRAM.

### 2.3 Expose `--moe-prefetch-streams` and `--moe-max-vram-mb` in CLI
These cparams exist but have no CLI flags.

### 2.4 Fix `backend_sampling` Speculative Decoding
We removed the `backend_sampling` field from `common_params_speculative_draft` but it's still referenced in speculative decoding code. Need proper fix or revert.

---

## Phase 3: HISA & Advanced Optimizations (future)

### 3.1 Wire up HISA in `llama-graph.cpp`
**Problem**: HISA kernels exist for CUDA/CPU/Metal but are NEVER called.

**Complexity**: ~200 LOC — needs careful integration with attention dispatch

### 3.2 Layer-Adaptive HISA Sparsity
**Paper**: PyramidKV (Cai et al. 2024) — deeper layers can be sparser

### 3.3 TritonMoE Backend for AMD Support
**Paper**: TritonMoE (arxiv 2605.23911) — pure Triton, zero code changes for AMD

---

## Benchmark Plan

### Test Models
| Model | Configuration |
|-------|--------------|
| Mixtral-8x7B Q4_K | n_expert=8, n_used=2, ~24GB |
| Qwen3-35B-A3B Q4_K | n_expert=128, n_used=8, ~35GB |

### Metrics
- **Decode TPS** (tokens/sec) at batch=1, ctx=2048
- **Prefill TPS** at ctx=2048
- **Peak VRAM** usage
- **Expert cache hit rate** (fused MoE)

### Configurations
1. Baseline (no fused MoE, no n_cpu_moe)
2. Fused MoE only
3. Fused MoE + n_cpu_moe=4
4. Fused MoE + n_cpu_moe=8
5. All of the above + bounded_kv=4096

### Test Matrix
```
For each config:
  batch_size ∈ {1, 4, 16, 64}
  ctx_length ∈ {512, 2048, 8192}
  quant ∈ {Q4_K, Q5_K, Q6_K}
```

---

## CLI Flags

| Flag | Type | Default | Description |
|------|------|---------|-------------|
| `--fused-moe` | flag | off | Enable fused MoE dispatch |
| `--moe-prefetch-streams` | int | 2 | Number of async prefetch streams |
| `--moe-max-vram-mb` | int | 0 (=auto) | Max VRAM for expert cache |
| `--n-cpu-moe` | int | 0 | Offload last N MoE layers to CPU |
| `--hisa` | flag | off | Enable HISA sparse attention |
| `--kv-cache-bounded` | int | 0 (=unlimited) | Max tokens in active KV cache |
