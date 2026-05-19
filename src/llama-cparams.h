#pragma once

#include "llama.h"

#include <cstdint>

#define LLAMA_MAX_SEQ 256

struct llama_cparams {
    uint32_t n_ctx;           // context size used during inference
    uint32_t n_ctx_seq;       // context for a single sequence
    uint32_t n_batch;
    uint32_t n_ubatch;
    uint32_t n_seq_max;
    uint32_t n_rs_seq;        // number of recurrent-state snapshots per seq for rollback
    int32_t  n_threads;       // number of threads to use for generation
    int32_t  n_threads_batch; // number of threads to use for batch processing

    float rope_freq_base;
    float rope_freq_scale;

    uint32_t n_ctx_orig_yarn;
    // These hyperparameters are not exposed in GGUF, because all
    // existing YaRN models use the same values for them.
    float yarn_ext_factor;
    float yarn_attn_factor;
    float yarn_beta_fast;
    float yarn_beta_slow;

    bool embeddings;
    bool embeddings_pre_norm;        // also extract the hidden state before the final output norm
    bool embeddings_pre_norm_masked; // extract for only rows where batch.logits != 0
    bool causal_attn;
    bool offload_kqv;
    bool flash_attn;
    bool auto_fa;
    bool fused_gdn_ar;       // use fused gated delta net (autoregressive)
    bool fused_gdn_ch;       // use fused gated delta net (chunked)
    bool auto_fgdn;
    bool no_perf;
    bool warmup;
    bool op_offload;
    bool kv_unified;  // use a unified buffer across the input sequences when computing the attention
    bool pipeline_parallel;
<<<<<<< HEAD
    bool fused_moe;
    int32_t moe_prefetch_streams;
    int32_t moe_max_vram_mb;
    bool hisa;              // use Hierarchical Indexed Sparse Attention
    int32_t hisa_block_size; // block size for HISA sparse attention
    int32_t hisa_min_tokens; // minimum KV tokens before HISA activates (0 = always on)
    float hisa_sparsity;    // fraction of blocks to select (0.0=all, 0.5=half, 0.9=top 10%)
    bool hisa_sink_protect; // protect attention sink tokens (block 0) from eviction
    float hisa_sparsity_scale; // layer-adaptive sparsity scale (0.0=uniform, 1.0=pyramid)
    bool hisa_per_head; // per-head sparse attention: select Top-K blocks per KV-head
    int32_t kv_cache_bounded; // bounded KV cache: max active tokens (0 = unlimited)
    bool pipeline_partial;  // enable pipeline parallelism for partial offload

    enum llama_context_type ctx_type;
    enum llama_pooling_type pooling_type;

    ggml_backend_sched_eval_callback cb_eval;
    void * cb_eval_user_data;
};
