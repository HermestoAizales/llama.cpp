#pragma once

#include <string>
#include <vector>
#include <cstdint>

#include "ggml.h"

// ---------------------------------------------------------------------------
// A single benchmark configuration to test.
// Covers every runtime-relevant parameter that affects speed vs quality.
// ---------------------------------------------------------------------------
struct optimizer_config {
    std::string label;           // human-readable label for display

    // --- Model offload ---
    int         n_gpu_layers;    // -1 = auto, -2 = all, >= 0 = exact count
    int         split_mode;      // 0=none, 1=layer, 2=row, 3=tensor  (-1 = don't override)

    // --- KV cache ---
    int         cache_type_k;    // ggml_type as int (-1 = don't override)
    int         cache_type_v;    // ggml_type as int (-1 = don't override)

    // --- Batch / compute ---
    int         n_batch;
    int         n_ubatch;
    int         n_threads;       // 0 = auto (hardware concurrency)
    int         n_threads_batch; // 0 = same as n_threads

    // --- Parallel / context ---
    int         n_parallel;      // number of parallel sequences
    bool        pipeline_partial;// pipeline parallelism for partial offload

    // --- IO ---
    bool        use_mmap;        // memory-map model file
    bool        use_direct_io;   // O_DIRECT (Linux) / unbuffered
    bool        use_mlock;       // mlock model in RAM

    // --- Offload toggles ---
    bool        offload_kqv;     // offload KV cache to GPU
    bool        op_offload;      // offload individual ops to GPU
    bool        no_extra_bufts;  // disable weight repacking buffer types

    // --- Fit ---
    int         fit_target_mib;  // per-device headroom in MiB (0 = default 1 GiB)

    // --- MoE offload --
    // -1 = don't override, 0 = all MoE on CPU (--cpu-moe), > 0 = first N layers MoE on CPU
    int         n_cpu_moe;

    // --- Fused MoE --
    // -1 = don't override, 0 = off, 1 = on
    int         fused_moe;
    int         moe_prefetch_streams;  // 0 = don't override
    int         moe_max_vram_mb;       // 0 = don't override (auto)

    // -- Speculative decoding --
    // If enabled, the optimizer will test with a small draft model (if available)
    // or n-gram speculation.  Empty = don't use speculation.
    std::string spec_type;       // "" = none, "ngram" = n-gram spec, "draft" = external draft
    int         spec_ngram_size; // ngram size if spec_type == "ngram"
};

// ---------------------------------------------------------------------------
// Result of a single benchmark run.
// ---------------------------------------------------------------------------
struct optimizer_result {
    optimizer_config config;
    float   gen_tps;             // generation tokens/sec
    float   prompt_tps;          // prompt processing tokens/sec (0 if not measured)
    bool    success;
    std::string error;
};

// ---------------------------------------------------------------------------
// User preferences collected via interactive prompts.
// ---------------------------------------------------------------------------
struct optimizer_user_params {
    std::string model_path;
    int         desired_ctx         = 0;  // 0 = model train ctx
    int         n_parallel          = 1;

    enum class priority { speed, balanced, quality };
    priority    optimization_goal   = priority::balanced;

    // Use-case hints
    bool        is_chat             = false;  // short prompts, many turns
    bool        is_batch            = false;  // long prompts, throughput matters
    bool        is_long_context     = false;  // ctx > 8K

    // Hardware hints (auto-detected, but user can override)
    bool        has_gpu             = false;
    int         n_gpu_layers_hint   = -1;  // -1 = auto, else exact
    bool        force_cpu           = false;

    // IO hints
    bool        model_on_ssd        = true;
    bool        model_on_nfs        = false;
    bool        prefer_mmap         = true;   // false = direct IO may be better

    // Quality floor
    float       min_acceptable_tps  = 0.0f;  // 0 = no floor
    bool        allow_quant_cache   = true;   // allow Q8_0/Q4_0 KV cache

    // --- MoE ---
    bool        allow_moe_cpu       = false;  // allow CPU offload for MoE experts

    // --- Fused MoE --
    bool        allow_fused_moe     = false;  // test fused MoE kernel (gate_up + silu + down)
    int         fused_moe_streams   = 2;      // prefetch streams for fused MoE
    int         moe_max_vram_mb     = 0;      // VRAM budget for expert cache (0=auto)

    // --- Speculative decoding ---
    bool        allow_speculative   = false;  // test n-gram speculation
    int         spec_ngram_size     = 3;      // ngram size to test

    // --- Attention ---
    bool        flash_attn          = true;   // use Flash Attention
    bool        swa_full            = false;  // full SWA cache

    // --- NUMA ---
    bool        use_numa            = false;  // NUMA optimization

    // --- Context / batching ---
    bool        ctx_shift           = false;  // infinite chat ctx shift
    bool        cont_batching       = true;   // continuous batching
};

// ---------------------------------------------------------------------------
// The optimizer: generates configs, benchmarks, prints report.
// ---------------------------------------------------------------------------
class optimizer {
public:
    optimizer();
    ~optimizer();

    // Interactive CLI: asks user questions, returns true on success.
    bool interactive_setup(optimizer_user_params & out);

    // Generate benchmark configurations based on user params + detected hw.
    std::vector<optimizer_config> generate_configs(const optimizer_user_params & up) const;

    // Run a single benchmark.
    optimizer_result benchmark_single(const optimizer_config & cfg, const optimizer_user_params & up) const;

    // Run all benchmarks, return results.
    std::vector<optimizer_result> run_benchmarks(const optimizer_user_params & up) const;

    // Select best result.
    optimizer_result select_best(const std::vector<optimizer_result> & results,
                                 const optimizer_user_params & up) const;

    // Print a copy-paste-ready report of the best configuration.
    void print_report(const optimizer_result & best,
                      const optimizer_user_params & up) const;

    // Print a full comparison table of all results.
    void print_comparison_table(const std::vector<optimizer_result> & results) const;

private:
    // --- Detected hardware info ---
    int         m_n_layers        = 0;
    int         m_n_ctx_train     = 0;
    int         m_n_gpu_devices   = 0;
    int         m_n_cpu_threads   = 0;
    int64_t     m_model_size_bytes = 0;
    bool        m_has_gpu         = false;
    bool        m_is_moe          = false;

    // --- Tunables ---
    mutable int m_benchmark_tokens = 30;
    int         m_warmup_tokens   = 5;
    int         m_prompt_bench_tokens = 128; // tokens for prompt-bench
};
