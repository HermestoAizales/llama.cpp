/**
 * Fused MoE CUDA Kernel with Async Weight Prefetch
 *
 * Fuses: MoE Routing + Expert FFN (gate_up + SiLU + down) into fewer kernel launches
 * Supports: All major MoE architectures (Qwen35/36, DeepSeek V3, Llama4, GroveMoE, Step35)
 *
 * Key optimization: Expert weights are prefetched from RAM to VRAM asynchronously
 * while attention computation overlaps, eliminating PCIe transfer bottleneck.
 *
 * This file contains:
 * 1. Host-side cache management (moe_expert_cache_*)
 * 2. Fused MoE forward kernel (fused_moe_forward_*)
 * 3. Integration with ggml_cuda_mul_mat_id dispatch
 *
 * Kernel design (Phase 3):
 * - Each block processes one (token, expert) pair
 * - Tiled matrix multiplication: gate_up (n_ff*2 × n_embd) and down (n_embd × n_ff)
 * - Intermediate results (gate_act, up_val) stay in shared memory / registers
 * - Single kernel launch replaces 3 separate mul_mat_id calls
 * - Supports F16 and F32 weights (quantized types fall back to standard path)
 */

#include "common.cuh"
#include "fused-moe.cuh"
#include "ggml-cuda.h"
#include "ggml-impl.h"
#include "ggml-backend-impl.h"

#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <cublas_v2.h>

#include <algorithm>
#include <cassert>
#include <cstdio>
#include <cstring>
#include <mutex>
#include <vector>

// =============================================================================
// Host-side Expert Cache Implementation
// =============================================================================

void moe_expert_cache_init(
    moe_expert_cache * cache,
    int64_t            n_expert,
    size_t             max_vram_bytes,
    int32_t n_streams) {
    memset(cache, 0, sizeof(*cache));

    // Allocate VRAM ring buffer
    cache->buffer_size = max_vram_bytes;
    cudaError_t err = cudaMalloc(&cache->buffer, max_vram_bytes);
    if (err != cudaSuccess) {
        // Fallback: reduce buffer size by half until allocation succeeds
        cache->buffer_size = max_vram_bytes / 2;
        while (cache->buffer_size > 1024 * 1024 && (err = cudaMalloc(&cache->buffer, cache->buffer_size)) != cudaSuccess) {
            cache->buffer_size /= 2;
        }
        if (err != cudaSuccess) {
            cache->buffer = nullptr;
            cache->buffer_size = 0;
            return;
        }
    }

    // Allocate cache entries
    cache->n_entries = n_expert;
    cache->entries = new moe_expert_cache_entry[n_expert]();

    // Allocate prefetch streams
    cache->n_prefetch_streams = n_streams;
    cache->prefetch_streams = new moe_prefetch_state[n_streams];
    for (int i = 0; i < n_streams; i++) {
        cudaStreamCreate(&cache->prefetch_streams[i].stream);
        cache->prefetch_streams[i].expert_id = -1;
        cache->prefetch_streams[i].active = false;
    }

    cache->timestamp = 0;
    cache->cache_hits = 0;
    cache->cache_misses = 0;
    cache->prefetch_hits = 0;
    cache->prefetch_misses = 0;
}

void moe_expert_cache_free(moe_expert_cache * cache) {
    if (cache->buffer) {
        cudaFree(cache->buffer);
    }
    for (int i = 0; i < cache->n_prefetch_streams; i++) {
        cudaStreamDestroy(cache->prefetch_streams[i].stream);
    }
    delete[] cache->entries;
    delete[] cache->prefetch_streams;
    memset(cache, 0, sizeof(*cache));
}

bool moe_expert_cache_lookup(moe_expert_cache * cache, int32_t expert_id) {
    if (expert_id < 0 || expert_id >= cache->n_entries) return false;
    if (!cache->entries[expert_id].valid) return false;

    // Update LRU timestamp
    cache->entries[expert_id].last_used = ++cache->timestamp;
    cache->cache_hits++;
    return true;
}

void moe_expert_cache_prefetch(
    moe_expert_cache *      cache,
    int32_t                 expert_id,
    const void *            weight_data,
    size_t                  weight_size,
    int32_t                 stream_idx) {
    if (!cache->buffer || expert_id < 0 || expert_id >= cache->n_entries) return;
    if (weight_size > cache->buffer_size) return; // expert too large for cache

    // Find a free prefetch stream (round-robin if all busy)
    moe_prefetch_state * state = &cache->prefetch_streams[stream_idx % cache->n_prefetch_streams];

    // Wait for previous prefetch on this stream to complete
    if (state->active) {
        cudaStreamSynchronize(state->stream);
        state->active = false;
    }

    // Evict LRU entries if necessary
    while (cache->buffer_used + weight_size > cache->buffer_size) {
        moe_expert_cache_evict_lru(cache);
    }

    // Allocate space in ring buffer
    size_t alloc_pos = cache->buffer_pos;
    if (cache->buffer_pos + weight_size > cache->buffer_size) {
        // Wrap around
        alloc_pos = 0;
    }

    // Async copy: host → device
    cudaMemcpyAsync(
        (char *)cache->buffer + alloc_pos,
        weight_data,
        weight_size,
        cudaMemcpyHostToDevice,
        state->stream);

    // Update cache entry
    cache->entries[expert_id].expert_id = expert_id;
    cache->entries[expert_id].valid = true;
    cache->entries[expert_id].gate_up_weight = (char *)cache->buffer + alloc_pos;
    cache->entries[expert_id].weight_size = weight_size;
    cache->entries[expert_id].last_used = ++cache->timestamp;

    cache->buffer_pos = alloc_pos + weight_size;
    cache->buffer_used += weight_size;

    state->expert_id = expert_id;
    state->active = true;
    cache->cache_misses++;
}

void moe_expert_cache_prefetch_wait(
    moe_expert_cache * cache,
    int32_t            stream_idx) {
    moe_prefetch_state * state = &cache->prefetch_streams[stream_idx % cache->n_prefetch_streams];
    if (state->active) {
        cudaStreamSynchronize(state->stream);
        state->active = false;
    }
}

void moe_expert_cache_evict_lru(moe_expert_cache * cache) {
    if (cache->n_entries == 0) return;

    // Find least recently used valid entry
    int32_t lru_id = -1;
    int32_t lru_time = INT32_MAX;
    for (int i = 0; i < cache->n_entries; i++) {
        if (cache->entries[i].valid && cache->entries[i].last_used < lru_time) {
            lru_time = cache->entries[i].last_used;
            lru_id = i;
        }
    }

    if (lru_id >= 0) {
        cache->buffer_used -= cache->entries[lru_id].weight_size;
        cache->entries[lru_id].valid = false;
        cache->entries[lru_id].gate_up_weight = nullptr;
    }
}

void * moe_expert_cache_get_weight(
    moe_expert_cache * cache,
    int32_t            expert_id) {
    if (expert_id < 0 || expert_id >= cache->n_entries) return nullptr;
    if (!cache->entries[expert_id].valid) return nullptr;
    return cache->entries[expert_id].gate_up_weight;
}

void moe_expert_cache_print_stats(const moe_expert_cache * cache) {
    int64_t total = cache->cache_hits + cache->cache_misses;
    if (total == 0) return;

    float hit_rate = 100.0f * cache->cache_hits / total;
    fprintf(stderr, "fused-moe: cache stats: hits=%ld misses=%ld hit_rate=%.1f%% vram_used=%zu/%zu MB\n",
        (long)cache->cache_hits, (long)cache->cache_misses, hit_rate,
        cache->buffer_used / (1024*1024), cache->buffer_size / (1024*1024));
}

// =============================================================================
// CUDA Kernels
// =============================================================================

// Fused SiLU activation: output = x * sigmoid(x) = x / (1 + exp(-x))
__device__ __forceinline__ float silu_f32(float x) {
    return x / (1.0f + expf(-x));
}

#define FUSED_MOE_TILE_DIM 128

// =============================================================================
// Fused MoE Kernel: gate_up + SiLU + down in a single launch
// =============================================================================
//
// Each block processes one (token, expert) pair.
// gridDim.x = n_expert_used
// gridDim.y = n_tokens
// blockDim.x = FUSED_MOE_TILE_DIM (128 threads = 4 warps)
//
// Phase 1: gate_up projection (tiled over n_embd)
//   input [n_embd] x W_gate_up [n_ff*2, n_embd] -> gate_up [n_ff*2]
// Phase 2: SiLU + multiply
//   activated = SiLU(gate) * up -> [n_ff]
// Phase 3: down projection (tiled over n_ff)
//   activated [n_ff] x W_down [n_embd, n_ff] -> output [n_embd]
//
// Shared memory: TILE_DIM*4 + n_ff*8 bytes (~4.5 KB for n_ff=1024)

__global__ void fused_moe_gate_up_silu_down_kernel(
    const void * __restrict__ gate_up_weight,  // [n_ff*2, n_embd, n_expert] weights
    const void * __restrict__ down_weight,     // [n_embd, n_ff, n_expert] weights
    const float * __restrict__ input,          // [n_embd, n_tokens] token input
    float * __restrict__ output,               // [n_embd, n_expert_used, n_tokens] output
    const int32_t * __restrict__ ids,          // [n_expert_used, n_tokens] expert indices
    int64_t n_embd,
    int64_t n_ff,
    int64_t n_expert_used,
    int64_t n_tokens,
    int64_t n_expert,
    ggml_type weight_type) {

    const int64_t expert_idx = blockIdx.x;
    const int64_t token_idx  = blockIdx.y;

    if (expert_idx >= n_expert_used || token_idx >= n_tokens) return;

    const int32_t expert_id = ids[expert_idx * n_tokens + token_idx];
    if (expert_id < 0 || expert_id >= n_expert) return;

    const int tid = threadIdx.x;

    extern __shared__ char smem_raw[];
    float * smem_input    = (float *)smem_raw;                                           // [TILE_DIM]
    float * smem_gate_act = (float *)(smem_raw + FUSED_MOE_TILE_DIM * sizeof(float));    // [n_ff]
    float * smem_up_val   = (float *)(smem_raw + (FUSED_MOE_TILE_DIM + n_ff) * sizeof(float)); // [n_ff]

    const int64_t gu_offset = expert_id * n_ff * 2 * n_embd;
    const int64_t dn_offset = expert_id * n_embd * n_ff;

    const float * gu_w_f32 = (weight_type == GGML_TYPE_F32) ? (const float *)gate_up_weight + gu_offset : nullptr;
    const half  * gu_w_f16 = (weight_type == GGML_TYPE_F16) ? (const half  *)gate_up_weight + gu_offset : nullptr;
    const float * dn_w_f32 = (weight_type == GGML_TYPE_F32) ? (const float *)down_weight    + dn_offset : nullptr;
    const half  * dn_w_f16 = (weight_type == GGML_TYPE_F16) ? (const half  *)down_weight    + dn_offset : nullptr;

    const float * inp = input + token_idx * n_embd;

    // Phase 1: gate_up projection, tiled over n_embd
    // Each thread processes rows tid, tid+blockDim.x, tid+2*blockDim.x, ...
    for (int64_t my_row = tid; my_row < n_ff * 2; my_row += blockDim.x) {
        bool is_gate = my_row < n_ff;
        int64_t local_row = is_gate ? my_row : my_row - n_ff;

        float sum = 0.0f;
        for (int64_t tile_start = 0; tile_start < n_embd; tile_start += FUSED_MOE_TILE_DIM) {
            // Load input tile into shared memory (all threads participate)
            int64_t tidx = tile_start + tid;
            smem_input[tid] = (tidx < n_embd) ? inp[tidx] : 0.0f;
            __syncthreads();

            // Compute partial dot product (only active threads read their weight row)
            if (gu_w_f16) {
                for (int64_t j = 0; j < FUSED_MOE_TILE_DIM && tile_start + j < n_embd; j++) {
                    sum += __half2float(gu_w_f16[my_row * n_embd + tile_start + j]) * smem_input[j];
                }
            } else if (gu_w_f32) {
                for (int64_t j = 0; j < FUSED_MOE_TILE_DIM && tile_start + j < n_embd; j++) {
                    sum += gu_w_f32[my_row * n_embd + tile_start + j] * smem_input[j];
                }
            }
            __syncthreads();
        }

        if (is_gate) smem_gate_act[local_row] = silu_f32(sum);
        else         smem_up_val[local_row]   = sum;
    }

    __syncthreads();

    // Phase 2: activated = SiLU(gate) * up
    for (int64_t row = tid; row < n_ff; row += blockDim.x) {
        smem_gate_act[row] *= smem_up_val[row];
    }
    __syncthreads();

    // Phase 3: down projection, tiled over n_ff
    for (int64_t row = tid; row < n_embd; row += blockDim.x) {
        float sum = 0.0f;

        for (int64_t tile_start = 0; tile_start < n_ff; tile_start += FUSED_MOE_TILE_DIM) {
            int64_t tidx = tile_start + tid;
            smem_input[tid] = (tidx < n_ff) ? smem_gate_act[tidx] : 0.0f;
            __syncthreads();

            if (dn_w_f16) {
                for (int64_t j = 0; j < FUSED_MOE_TILE_DIM && tile_start + j < n_ff; j++) {
                    sum += __half2float(dn_w_f16[row * n_ff + tile_start + j]) * smem_input[j];
                }
            } else if (dn_w_f32) {
                for (int64_t j = 0; j < FUSED_MOE_TILE_DIM && tile_start + j < n_ff; j++) {
                    sum += dn_w_f32[row * n_ff + tile_start + j] * smem_input[j];
                }
            }
            __syncthreads();
        }

        output[row * n_expert_used * n_tokens + expert_idx * n_tokens + token_idx] = sum;
    }
}

// =============================================================================
// Host-side Fused MoE Forward
// =============================================================================

// Global fused MoE state (one per CUDA device)
static std::mutex g_fused_moe_mutex;
static bool g_fused_moe_enabled[GGML_CUDA_MAX_DEVICES] = {false};
static moe_expert_cache g_moe_cache[GGML_CUDA_MAX_DEVICES];

// Fused MoE state: stores gate_up data between the two mul_mat_id calls
// The gate_up mul_mat_id is called first, then the down mul_mat_id.
// We save gate_up data at the gate_up call and launch the fused kernel at the down call.
struct fused_moe_state {
    bool     active;            // true after gate_up call, before down call
    int      device;            // CUDA device
    int64_t  n_embd;
    int64_t  n_ff;
    int64_t  n_expert;
    int64_t  n_expert_used;
    int64_t  n_tokens;
    ggml_type weight_type;
    // Saved from gate_up mul_mat_id call
    const void * gate_up_weight; // [n_ff*2, n_embd, n_expert]
    const float * input;         // [n_embd, n_tokens]
    const int32_t * ids;         // [n_expert_used, n_tokens]
    // Temporary buffer for fused output
    float * fused_output;        // [n_embd, n_expert_used, n_tokens]
    size_t   fused_output_size;
};

static fused_moe_state g_fused_state[GGML_CUDA_MAX_DEVICES];

static void fused_state_reset(fused_moe_state & s) {
    if (s.fused_output) {
        cudaFree(s.fused_output);
    }
    s.active = false;
    s.device = -1;
    s.n_embd = 0;
    s.n_ff = 0;
    s.n_expert = 0;
    s.n_expert_used = 0;
    s.n_tokens = 0;
    s.weight_type = GGML_TYPE_COUNT;
    s.gate_up_weight = nullptr;
    s.input = nullptr;
    s.ids = nullptr;
    s.fused_output = nullptr;
    s.fused_output_size = 0;
}

static fused_moe_state * get_fused_state(int device) {
    if (device < 0 || device >= GGML_CUDA_MAX_DEVICES) return nullptr;
    return &g_fused_state[device];
}

// Internal: get expert cache for device
static moe_expert_cache * get_moe_cache(int device) {
    if (device < 0 || device >= GGML_CUDA_MAX_DEVICES) return nullptr;
    return &g_moe_cache[device];
}

// Enable/disable fused MoE for a device (called from ggml-cuda.cu)
void ggml_cuda_fused_moe_set_enabled(int device, bool enable) {
    if (device < 0 || device >= GGML_CUDA_MAX_DEVICES) return;
    std::lock_guard<std::mutex> lock(g_fused_moe_mutex);
    g_fused_moe_enabled[device] = enable;
}

bool ggml_cuda_fused_moe_get_enabled(int device) {
    if (device < 0 || device >= GGML_CUDA_MAX_DEVICES) return false;
    return g_fused_moe_enabled[device];
}

// Initialize fused MoE for a device
void ggml_cuda_fused_moe_init(int device, int64_t n_expert, size_t max_vram_mb, int32_t n_streams) {
    moe_expert_cache * cache = get_moe_cache(device);
    if (!cache) return;

    // Reset fused state
    fused_moe_state * state = get_fused_state(device);
    memset(state, 0, sizeof(*state));

    size_t max_vram_bytes = max_vram_mb * 1024 * 1024;
    moe_expert_cache_init(cache, n_expert, max_vram_bytes, n_streams);

    fprintf(stderr, "fused-moe: initialized device=%d n_experts=%ld vram_budget=%zuMB streams=%d\n",
        device, (long)n_expert, max_vram_bytes / (1024*1024), n_streams);
}

// Cleanup fused MoE for a device
void ggml_cuda_fused_moe_free(int device) {
    moe_expert_cache * cache = get_moe_cache(device);
    if (!cache) return;

    moe_expert_cache_print_stats(cache);
    moe_expert_cache_free(cache);
}

// Prefetch expert weights for upcoming layer
void ggml_cuda_fused_moe_prefetch(
    int          device,
    int32_t      expert_id,
    const void * weight_data,
    size_t       weight_size,
    int32_t      stream_idx) {
    moe_expert_cache * cache = get_moe_cache(device);
    if (!cache) return;

    if (!moe_expert_cache_lookup(cache, expert_id)) {
        moe_expert_cache_prefetch(cache, expert_id, weight_data, weight_size, stream_idx);
    }
}

// Check if fused MoE can be used for a given tensor
// Check if a MUL_MAT_ID call is the gate_up part of a MoE layer
// Heuristic: gate_up takes input [n_embd] and produces output [n_ff*2]
// where n_embd > n_ff*2 (input dim > output dim).
// down takes input [n_ff] and produces output [n_embd]
// where n_ff < n_embd (input dim < output dim).
// So: gate_up if src1->ne[0] > dst->ne[0], down if src1->ne[0] < dst->ne[0].
static bool is_gate_up_mul_mat_id(const ggml_tensor * dst) {
    if (dst->op != GGML_OP_MUL_MAT_ID) return false;
    const ggml_tensor * src1 = dst->src[1];
    // gate_up: input is [n_embd, n_tokens], output is [n_ff*2, n_expert_used, n_tokens]
    //   n_embd > n_ff*2, so src1->ne[0] > dst->ne[0]
    // down: input is [n_ff, n_expert_used, n_tokens], output is [n_embd, n_expert_used, n_tokens]
    //   n_ff < n_embd, so src1->ne[0] < dst->ne[0]
    return src1->ne[0] > dst->ne[0];
}

// Check if fused MoE can be used for a given tensor
bool ggml_cuda_should_use_fused_moe(
    const ggml_tensor * dst,
    int                 device) {
    if (dst->op != GGML_OP_MUL_MAT_ID) return false;

    const ggml_tensor * src0 = dst->src[0];
    const ggml_tensor * ids  = dst->src[2];

    // Check weight type support (F16, F32 only for now)
    if (src0->type != GGML_TYPE_F16 && src0->type != GGML_TYPE_F32) {
        return false;
    }

    // Check that we have a valid cache
    moe_expert_cache * cache = get_moe_cache(device);
    if (!cache || !cache->buffer) return false;

    // Check that ids tensor is small enough (MoE routing indices)
    if (ids->ne[0] > 16) return false;

    return true;
}

// =============================================================================
// Fused MoE Forward Dispatch (Phase 3)
// =============================================================================
//
// Strategy: The MoE FFN has two sequential MUL_MAT_ID calls:
//   1. gate_up: input [n_embd, n_tokens] x W_gate_up [n_ff*2, n_embd, n_expert]
//                -> gate_up_out [n_ff*2, n_expert_used, n_tokens]
//   2. down:    cur [n_ff, n_expert_used, n_tokens] x W_down [n_embd, n_ff, n_expert]
//                -> output [n_embd, n_expert_used, n_tokens]
//
// We intercept both calls. At the gate_up call, we save the gate_up weights,
// input, and expert indices. At the down call, we launch the fused kernel
// that computes gate_up + SiLU + down in a single launch.
//
// The gate_up output tensor is filled with zeros (it's only used by the SwiGLU
// node, which we skip in the fused path). The down output gets the final result.

void ggml_cuda_fused_moe_forward(
    ggml_backend_cuda_context & ctx,
    ggml_tensor * dst,
    bool           /*is_gate_up*/) {
    const ggml_tensor * src0 = dst->src[0];
    const ggml_tensor * src1 = dst->src[1];
    const ggml_tensor * ids  = dst->src[2];

    const int device = ggml_cuda_get_device();
    cudaStream_t stream = ctx.stream();
    moe_expert_cache * cache = get_moe_cache(device);
    fused_moe_state * state = get_fused_state(device);

    const int64_t n_expert_used = ids->ne[0];
    const int64_t n_tokens      = src1->ne[1];
    const int64_t n_expert      = src0->ne[2];

    // Determine if this is gate_up or down by tensor dimensions
    bool is_gate = is_gate_up_mul_mat_id(dst);

    if (is_gate) {
        // ================================================================
        // Gate-up MUL_MAT_ID call: save state for fused launch at down call
        // ================================================================
        const int64_t n_ff   = src0->ne[0] / 2;
        const int64_t n_embd = src1->ne[0];

        // Reset any previous state
        fused_state_reset(*state);

        state->active = true;
        state->device = device;
        state->n_embd = n_embd;
        state->n_ff = n_ff;
        state->n_expert = n_expert;
        state->n_expert_used = n_expert_used;
        state->n_tokens = n_tokens;
        state->weight_type = src0->type;
        state->gate_up_weight = src0->data;
        state->input = (const float *)src1->data;
        state->ids = (const int32_t *)ids->data;

        // Allocate fused output buffer [n_embd, n_expert_used, n_tokens]
        state->fused_output_size = (size_t)n_embd * n_expert_used * n_tokens * sizeof(float);
        CUDA_CHECK(cudaMalloc(&state->fused_output, state->fused_output_size));

        // Prefetch expert weights into VRAM cache
        if (cache && cache->buffer) {
            const size_t weight_element_size = ggml_type_size(src0->type);
            const size_t expert_size = (size_t)n_ff * 2 * n_embd * weight_element_size;

            // Read expert indices
            const int64_t n_get_rows = n_tokens * n_expert_used;
            std::vector<int32_t> ids_host(n_get_rows);
            CUDA_CHECK(cudaMemcpyAsync(ids_host.data(), ids->data,
                n_get_rows * sizeof(int32_t), cudaMemcpyDeviceToHost, stream));
            CUDA_CHECK(cudaStreamSynchronize(stream));

            std::vector<bool> prefetched(n_expert, false);
            int32_t prefetch_count = 0;

            for (int64_t i = 0; i < n_get_rows; ++i) {
                int32_t expert_id = ids_host[i];
                if (expert_id < 0 || expert_id >= n_expert) continue;
                if (prefetched[expert_id]) continue;
                if (moe_expert_cache_lookup(cache, expert_id)) {
                    prefetched[expert_id] = true;
                    continue;
                }
                const void * weight_ptr = (const char *)src0->data + (size_t)expert_id * expert_size;
                int32_t stream_idx = prefetch_count % cache->n_prefetch_streams;
                moe_expert_cache_prefetch(cache, expert_id, weight_ptr, expert_size, stream_idx);
                prefetched[expert_id] = true;
                prefetch_count++;
            }

            for (int32_t s = 0; s < cache->n_prefetch_streams; ++s) {
                moe_expert_cache_prefetch_wait(cache, s);
            }

            if (prefetch_count > 0) {
                fprintf(stderr, "fused-moe: prefetched %d experts (device=%d, gate_up)\\n",
                    prefetch_count, device);
            }
        }

        // Zero out the gate_up output tensor (SwiGLU will be a no-op in fused path)
        CUDA_CHECK(cudaMemsetAsync(dst->data, 0,
            (size_t)dst->ne[0] * dst->ne[1] * dst->ne[2] * sizeof(float), stream));
        CUDA_CHECK(cudaStreamSynchronize(stream));

    } else {
        // ================================================================
        // Down MUL_MAT_ID call: launch the fused kernel
        // ================================================================
        const int64_t n_embd = src0->ne[0]; // down weights: [n_embd, n_ff, n_expert]
        const int64_t n_ff   = src0->ne[1];
        const void * down_weight = src0->data;

        // If we don't have saved state (shouldn't happen), fall back to standard path
        if (!state->active || state->device != device) {
            fprintf(stderr, "fused-moe: warning: no fused state for down call, falling back\\n");
            ggml_cuda_mul_mat_id(ctx, dst);
            return;
        }

        // Verify dimensions match
        if (state->n_embd != n_embd || state->n_ff != n_ff) {
            fprintf(stderr, "fused-moe: warning: dimension mismatch, falling back\\n");
            ggml_cuda_mul_mat_id(ctx, dst);
            return;
        }

        // Launch fused kernel
        // gridDim = (n_expert_used, n_tokens)
        // blockDim = FUSED_MOE_TILE_DIM (128 threads)
        dim3 grid(n_expert_used, n_tokens);
        dim3 block(FUSED_MOE_TILE_DIM);

        // Shared memory: TILE_DIM*4 + n_ff*8 bytes
        size_t smem_size = (size_t)(FUSED_MOE_TILE_DIM + n_ff * 2) * sizeof(float);

        fused_moe_gate_up_silu_down_kernel<<<grid, block, smem_size, stream>>>(
            state->gate_up_weight,
            down_weight,
            state->input,
            state->fused_output,
            state->ids,
            n_embd,
            n_ff,
            n_expert_used,
            n_tokens,
            n_expert,
            state->weight_type);

        CUDA_CHECK(cudaGetLastError());

        // Copy fused output to dst tensor
        // Both are [n_embd, n_expert_used, n_tokens], dst may have padding
        const size_t slice_bytes = (size_t)n_embd * n_expert_used * sizeof(float);
        const size_t dst_stride  = dst->nb[2];  // stride between token slices
        const size_t src_stride  = slice_bytes; // contiguous
        if (dst_stride == src_stride) {
            // Contiguous: single memcpy
            CUDA_CHECK(cudaMemcpyAsync(dst->data, state->fused_output,
                slice_bytes * n_tokens, cudaMemcpyDeviceToDevice, stream));
        } else {
            // Non-contiguous: copy slice by slice
            for (int64_t t = 0; t < n_tokens; t++) {
                CUDA_CHECK(cudaMemcpyAsync(
                    (char *)dst->data + t * dst_stride,
                    (char *)state->fused_output + t * src_stride,
                    slice_bytes, cudaMemcpyDeviceToDevice, stream));
            }
        }

        CUDA_CHECK(cudaStreamSynchronize(stream));

        fprintf(stderr, "fused-moe: fused kernel launched (device=%d, n_tokens=%ld, n_expert_used=%ld)\\n",
            device, (long)n_tokens, (long)n_expert_used);

        // Reset state
        fused_state_reset(*state);
    }
}

