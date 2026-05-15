#pragma once

#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <cstdint>

// Fused MoE with async weight prefetch
// Supports: Q4_K, Q5_K, Q6_K, Q8_0, Q2_K, F16, BF16
// Architectures: Qwen35/36 MoE, DeepSeek V3, Llama4, GroveMoE, Step35

// Expert cache entry - tracks an expert's weights in VRAM
struct moe_expert_cache_entry {
    int32_t expert_id;          // global expert index
    int32_t last_used;          // timestamp for LRU eviction
    bool    valid;              // entry contains valid data
    // device pointers for cached weights (allocated in VRAM ring buffer)
    void *  gate_up_weight;     // [n_ff*2, n_embd] or nullptr if separate gate/up
    void *  down_weight;        // [n_embd, n_ff]
    size_t  weight_size;        // total bytes occupied
};

// Prefetch stream state for async weight transfer
struct moe_prefetch_state {
    cudaStream_t stream;        // dedicated CUDA stream for prefetch
    int32_t      expert_id;     // expert being prefetched (-1 = idle)
    bool         active;        // transfer in progress
};

// Fused MoE kernel configuration
struct moe_kernel_config {
    int64_t n_embd;             // hidden size
    int64_t n_ff;               // feed-forward intermediate size
    int64_t n_expert;           // total number of experts
    int64_t n_expert_used;      // experts activated per token (top-k)
    int64_t n_tokens;           // batch size (number of tokens)
    int64_t n_ff_per_expert;    // FF dim per expert (for shared experts)
    bool    has_gate;           // separate gate projection (vs merged gate_up)
    bool    has_merged_gate_up; // merged gate_up projection
    bool    has_shared_expert;  // shared expert (Step35 style)
    bool    weight_before_ffn;  // apply weights before FFN (Llama4 style)
    bool    norm_weights;       // normalize expert weights (DeepSeek style)
    float   w_scale;            // expert weight scale factor
    ggml_type weight_type;      // quantization type of expert weights
};

// Host-side MoE expert cache manager
struct moe_expert_cache {
    // VRAM ring buffer for expert weights
    void *  buffer;             // device memory buffer
    size_t  buffer_size;        // total buffer size in bytes
    size_t  buffer_used;        // currently used bytes
    int64_t buffer_pos;         // ring buffer write position

    // Cache entries
    moe_expert_cache_entry * entries;   // [n_expert]
    int32_t                  n_entries;
    int32_t                  timestamp; // global LRU timestamp

    // Prefetch streams
    moe_prefetch_state * prefetch_streams;
    int32_t              n_prefetch_streams;

    // Statistics
    int64_t cache_hits;
    int64_t cache_misses;
    int64_t prefetch_hits;      // prefetched expert was used
    int64_t prefetch_misses;    // prefetched expert was evicted before use
};

// Initialize expert cache
void moe_expert_cache_init(
    moe_expert_cache * cache,
    int64_t            n_expert,
    size_t             max_vram_bytes,
    int32_t            n_streams);

// Free expert cache
void moe_expert_cache_free(moe_expert_cache * cache);

// Check if expert weights are cached
bool moe_expert_cache_lookup(moe_expert_cache * cache, int32_t expert_id);

// Prefetch expert weights asynchronously (non-blocking)
void moe_expert_cache_prefetch(
    moe_expert_cache *      cache,
    int32_t                 expert_id,
    const void *            weight_data,    // host pointer to expert weights
    size_t                  weight_size,
    int32_t                 stream_idx);

// Wait for prefetch to complete
void moe_expert_cache_prefetch_wait(
    moe_expert_cache * cache,
    int32_t            stream_idx);

// Evict least-recently-used expert to make room
void moe_expert_cache_evict_lru(moe_expert_cache * cache);

// Get cached expert weight pointer
void * moe_expert_cache_get_weight(
    moe_expert_cache * cache,
    int32_t            expert_id);

// Print cache statistics
void moe_expert_cache_print_stats(const moe_expert_cache * cache);
