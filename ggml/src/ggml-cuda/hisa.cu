#include "hisa.cuh"

// ============================================================
// Kernel 1: hisa_block_pool
// Mean-pool B consecutive rows of K into one block representation.
// src:  [d, n_kv, n_head_kv, n_batch]
// dst:  [d, n_blocks, n_head_kv, n_batch]
// n_blocks = n_kv / B
// ============================================================

static __global__ void hisa_block_pool_f32(
        const float * __restrict__ src,
        float * __restrict__ dst,
        const int64_t d,
        const int64_t n_kv,
        const int64_t n_blocks,
        const int32_t block_size,
        const size_t src_nb1, const size_t src_nb2, const size_t src_nb3,
        const size_t dst_nb1, const size_t dst_nb2, const size_t dst_nb3) {

    const int64_t iblk = blockIdx.x;
    const int64_t ih   = blockIdx.y;
    const int64_t ib   = blockIdx.z;

    if (iblk >= n_blocks || ib >= gridDim.z) {
        return;
    }

    const int64_t src_row_base = iblk * block_size;
    const char * src_base = (const char *)src + ib * src_nb3 + ih * src_nb2;
    char * dst_base = (char *)dst + ib * dst_nb3 + ih * dst_nb2;

    for (int64_t j = threadIdx.x; j < d; j += blockDim.x) {
        float sum = 0.0f;
        for (int32_t b = 0; b < block_size; b++) {
            const float * src_ptr = (const float *)(src_base + (src_row_base + b) * src_nb1 + j * sizeof(float));
            sum += *src_ptr;
        }
        float * dst_ptr = (float *)(dst_base + iblk * dst_nb1 + j * sizeof(float));
        *dst_ptr = sum / (float)block_size;
    }
}

static __global__ void hisa_block_pool_f16(
        const half * __restrict__ src,
        half * __restrict__ dst,
        const int64_t d,
        const int64_t n_kv,
        const int64_t n_blocks,
        const int32_t block_size,
        const size_t src_nb1, const size_t src_nb2, const size_t src_nb3,
        const size_t dst_nb1, const size_t dst_nb2, const size_t dst_nb3) {

    const int64_t iblk = blockIdx.x;
    const int64_t ih   = blockIdx.y;
    const int64_t ib   = blockIdx.z;

    if (iblk >= n_blocks || ib >= gridDim.z) {
        return;
    }

    const int64_t src_row_base = iblk * block_size;
    const char * src_base = (const char *)src + ib * src_nb3 + ih * src_nb2;
    char * dst_base = (char *)dst + ib * dst_nb3 + ih * dst_nb2;

    for (int64_t j = threadIdx.x; j < d; j += blockDim.x) {
        float sum = 0.0f;
        for (int32_t b = 0; b < block_size; b++) {
            const half * src_ptr = (const half *)(src_base + (src_row_base + b) * src_nb1 + j * sizeof(half));
            sum += __half2float(*src_ptr);
        }
        half * dst_ptr = (half *)(dst_base + iblk * dst_nb1 + j * sizeof(half));
        *dst_ptr = __float2half(sum / (float)block_size);
    }
}

void ggml_cuda_op_hisa_block_pool(ggml_backend_cuda_context & ctx, ggml_tensor * dst) {
    const ggml_tensor * src = dst->src[0];
    const float * src_d = (const float *)src->data;
    float * dst_d = (float *)dst->data;
    cudaStream_t stream = ctx.stream();

    GGML_ASSERT(src->type == GGML_TYPE_F32);
    GGML_ASSERT(dst->type == GGML_TYPE_F32);

    const int32_t block_size = ggml_get_op_params_i32(dst, 0);

    const int64_t d        = src->ne[0];
    const int64_t n_kv     = src->ne[1];
    const int64_t n_heads  = src->ne[2];
    const int64_t n_batch  = src->ne[3];
    const int64_t n_blocks = dst->ne[1];

    GGML_ASSERT(n_kv == n_blocks * block_size);

    const dim3 block_dims(256, 1, 1);
    const dim3 grid_dims(n_blocks, n_heads, n_batch);

    if (n_blocks > 0 && n_heads > 0 && n_batch > 0 && d > 0) {
        hisa_block_pool_f32<<<grid_dims, block_dims, 0, stream>>>(
            src_d, dst_d,
            d, n_kv, n_blocks, block_size,
            src->nb[1], src->nb[2], src->nb[3],
            dst->nb[1], dst->nb[2], dst->nb[3]);
    }
}

// ============================================================
// Kernel 2: hisa_block_gather
// Gather full blocks of B rows from K/V by block index list.
// src:           [d, n_kv, n_head_kv, n_batch]
// block_indices: [m, n_tokens, n_head_q, n_batch]     (I32)
// dst:           [d, m*B, n_head_kv, n_batch]
// Uses first query token (dim[1]=0) for selection.
// GQA: ih_kv -> ih_q = ih_kv * (block_indices->ne[2] / n_heads_kv)
// ============================================================

static __global__ void hisa_block_gather_f32(
        const float * __restrict__ src,
        const int32_t * __restrict__ block_indices,
        float * __restrict__ dst,
        const int64_t d,
        const int64_t m,
        const int32_t block_size,
        const int64_t n_heads_kv,
        const int64_t gqa_ratio,
        const size_t src_nb1, const size_t src_nb2, const size_t src_nb3,
        const size_t dst_nb1, const size_t dst_nb2, const size_t dst_nb3,
        const size_t idx_nb0, const size_t idx_nb2, const size_t idx_nb3) {

    const int64_t i = (int64_t)blockIdx.x * blockDim.x + threadIdx.x;
    const int64_t total = d * m * block_size;

    if (i >= total) {
        return;
    }

    int64_t tmp = i;
    const int64_t j  = tmp % d;
    tmp /= d;
    const int64_t b  = tmp % block_size;
    tmp /= block_size;
    const int64_t im = tmp;

    const int64_t ih = blockIdx.y;
    const int64_t ib = blockIdx.z;

    if (ih >= n_heads_kv) {
        return;
    }

    const int64_t ih_q = ih * gqa_ratio;

    const int64_t idx_offset = im * idx_nb0
                             + 0 * 0
                             + ih_q * idx_nb2
                             + ib * idx_nb3;
    const int32_t blk_idx = *(const int32_t *)((const char *)block_indices + idx_offset);

    const int32_t src_row = blk_idx * block_size + (int32_t)b;
    const int64_t dst_row = im * block_size + b;

    const float * src_ptr = (const float *)((const char *)src
        + ib * src_nb3
        + ih * src_nb2
        + src_row * src_nb1
        + j * sizeof(float));

    float * dst_ptr = (float *)((char *)dst
        + ib * dst_nb3
        + ih * dst_nb2
        + dst_row * dst_nb1
        + j * sizeof(float));

    *dst_ptr = *src_ptr;
}

static __global__ void hisa_block_gather_f16(
        const half * __restrict__ src,
        const int32_t * __restrict__ block_indices,
        half * __restrict__ dst,
        const int64_t d,
        const int64_t m,
        const int32_t block_size,
        const int64_t n_heads_kv,
        const int64_t gqa_ratio,
        const size_t src_nb1, const size_t src_nb2, const size_t src_nb3,
        const size_t dst_nb1, const size_t dst_nb2, const size_t dst_nb3,
        const size_t idx_nb0, const size_t idx_nb2, const size_t idx_nb3) {

    const int64_t i = (int64_t)blockIdx.x * blockDim.x + threadIdx.x;
    const int64_t total = d * m * block_size;

    if (i >= total) {
        return;
    }

    int64_t tmp = i;
    const int64_t j  = tmp % d;
    tmp /= d;
    const int64_t b  = tmp % block_size;
    tmp /= block_size;
    const int64_t im = tmp;

    const int64_t ih = blockIdx.y;
    const int64_t ib = blockIdx.z;

    if (ih >= n_heads_kv) {
        return;
    }

    const int64_t ih_q = ih * gqa_ratio;

    const int64_t idx_offset = im * idx_nb0
                             + 0 * 0
                             + ih_q * idx_nb2
                             + ib * idx_nb3;
    const int32_t blk_idx = *(const int32_t *)((const char *)block_indices + idx_offset);

    const int32_t src_row = blk_idx * block_size + (int32_t)b;
    const int64_t dst_row = im * block_size + b;

    const half * src_ptr = (const half *)((const char *)src
        + ib * src_nb3
        + ih * src_nb2
        + src_row * src_nb1
        + j * sizeof(half));

    half * dst_ptr = (half *)((char *)dst
        + ib * dst_nb3
        + ih * dst_nb2
        + dst_row * dst_nb1
        + j * sizeof(half));

    *dst_ptr = *src_ptr;
}

void ggml_cuda_op_hisa_block_gather(ggml_backend_cuda_context & ctx, ggml_tensor * dst) {
    const ggml_tensor * src           = dst->src[0];
    const ggml_tensor * block_indices = dst->src[1];

    const float * src_d = (const float *)src->data;
    const int32_t * idx_d = (const int32_t *)block_indices->data;
    float * dst_d = (float *)dst->data;
    cudaStream_t stream = ctx.stream();

    GGML_ASSERT(src->type == GGML_TYPE_F32);
    GGML_ASSERT(dst->type == GGML_TYPE_F32);
    GGML_ASSERT(block_indices->type == GGML_TYPE_I32);

    const int32_t block_size = ggml_get_op_params_i32(dst, 0);

    const int64_t d          = src->ne[0];
    const int64_t n_heads_kv = src->ne[2];
    const int64_t n_batch    = src->ne[3];
    const int64_t m          = block_indices->ne[0];  // number of selected blocks

    GGML_ASSERT(dst->ne[1] == m * block_size);

    // GQA mapping ratio
    const int64_t gqa_ratio = n_heads_kv > 0 ? block_indices->ne[2] / n_heads_kv : 1;

    const int64_t total_per_head_batch = d * m * block_size;
    const int64_t n_threads = 256;
    const int64_t n_blocks_x = (total_per_head_batch + n_threads - 1) / n_threads;

    const dim3 block_dims(n_threads, 1, 1);
    const dim3 grid_dims(n_blocks_x, n_heads_kv, n_batch);

    if (total_per_head_batch > 0 && n_heads_kv > 0 && n_batch > 0) {
        hisa_block_gather_f32<<<grid_dims, block_dims, 0, stream>>>(
            src_d, idx_d, dst_d,
            d, m, block_size, n_heads_kv, gqa_ratio,
            src->nb[1], src->nb[2], src->nb[3],
            dst->nb[1], dst->nb[2], dst->nb[3],
            block_indices->nb[0], block_indices->nb[2], block_indices->nb[3]);
    }
}

// ============================================================
// Kernel 3: hisa_gather
// Gather individual rows from K/V by index list (token-level refinement).
// src:     [d, n_cand, n_head_kv, n_batch]
// indices: [budget, n_tokens, n_head_q, n_batch] (I32)
// dst:     [d, budget, n_head_kv, n_batch]
// Uses first query token (dim[1]=0) for selection.
// GQA: ih_kv -> ih_q = ih_kv * (indices->ne[2] / n_heads_kv)
// ============================================================

static __global__ void hisa_gather_f32(
        const float * __restrict__ src,
        const int32_t * __restrict__ indices,
        float * __restrict__ dst,
        const int64_t d,
        const int64_t budget,
        const int64_t n_heads_kv,
        const int64_t gqa_ratio,
        const size_t src_nb1, const size_t src_nb2, const size_t src_nb3,
        const size_t dst_nb1, const size_t dst_nb2, const size_t dst_nb3,
        const size_t idx_nb0, const size_t idx_nb2, const size_t idx_nb3) {

    const int64_t i = (int64_t)blockIdx.x * blockDim.x + threadIdx.x;
    const int64_t total = d * budget;

    if (i >= total) {
        return;
    }

    const int64_t j  = i % d;
    const int64_t is = i / d;

    const int64_t ih = blockIdx.y;
    const int64_t ib = blockIdx.z;

    if (ih >= n_heads_kv) {
        return;
    }

    const int64_t ih_q = ih * gqa_ratio;

    const int64_t idx_offset = is * idx_nb0
                             + 0 * 0
                             + ih_q * idx_nb2
                             + ib * idx_nb3;
    const int32_t idx = *(const int32_t *)((const char *)indices + idx_offset);

    const float * src_ptr = (const float *)((const char *)src
        + ib * src_nb3
        + ih * src_nb2
        + idx * src_nb1
        + j * sizeof(float));

    float * dst_ptr = (float *)((char *)dst
        + ib * dst_nb3
        + ih * dst_nb2
        + is * dst_nb1
        + j * sizeof(float));

    *dst_ptr = *src_ptr;
}

static __global__ void hisa_gather_f16(
        const half * __restrict__ src,
        const int32_t * __restrict__ indices,
        half * __restrict__ dst,
        const int64_t d,
        const int64_t budget,
        const int64_t n_heads_kv,
        const int64_t gqa_ratio,
        const size_t src_nb1, const size_t src_nb2, const size_t src_nb3,
        const size_t dst_nb1, const size_t dst_nb2, const size_t dst_nb3,
        const size_t idx_nb0, const size_t idx_nb2, const size_t idx_nb3) {

    const int64_t i = (int64_t)blockIdx.x * blockDim.x + threadIdx.x;
    const int64_t total = d * budget;

    if (i >= total) {
        return;
    }

    const int64_t j  = i % d;
    const int64_t is = i / d;

    const int64_t ih = blockIdx.y;
    const int64_t ib = blockIdx.z;

    if (ih >= n_heads_kv) {
        return;
    }

    const int64_t ih_q = ih * gqa_ratio;

    const int64_t idx_offset = is * idx_nb0
                             + 0 * 0
                             + ih_q * idx_nb2
                             + ib * idx_nb3;
    const int32_t idx = *(const int32_t *)((const char *)indices + idx_offset);

    const half * src_ptr = (const half *)((const char *)src
        + ib * src_nb3
        + ih * src_nb2
        + idx * src_nb1
        + j * sizeof(half));

    half * dst_ptr = (half *)((char *)dst
        + ib * dst_nb3
        + ih * dst_nb2
        + is * dst_nb1
        + j * sizeof(half));

    *dst_ptr = *src_ptr;
}

void ggml_cuda_op_hisa_gather(ggml_backend_cuda_context & ctx, ggml_tensor * dst) {
    const ggml_tensor * src     = dst->src[0];
    const ggml_tensor * indices = dst->src[1];

    const float * src_d = (const float *)src->data;
    const int32_t * idx_d = (const int32_t *)indices->data;
    float * dst_d = (float *)dst->data;
    cudaStream_t stream = ctx.stream();

    GGML_ASSERT(src->type == GGML_TYPE_F32);
    GGML_ASSERT(dst->type == GGML_TYPE_F32);
    GGML_ASSERT(indices->type == GGML_TYPE_I32);

    const int64_t d          = src->ne[0];
    const int64_t n_heads_kv = src->ne[2];
    const int64_t n_batch    = src->ne[3];
    const int64_t budget     = indices->ne[0];  // number of selected tokens

    GGML_ASSERT(dst->ne[1] == budget);

    // GQA mapping ratio
    const int64_t gqa_ratio = n_heads_kv > 0 ? indices->ne[2] / n_heads_kv : 1;

    const int64_t total_per_head_batch = d * budget;
    const int64_t n_threads = 256;
    const int64_t n_blocks_x = (total_per_head_batch + n_threads - 1) / n_threads;

    const dim3 block_dims(n_threads, 1, 1);
    const dim3 grid_dims(n_blocks_x, n_heads_kv, n_batch);

    if (total_per_head_batch > 0 && n_heads_kv > 0 && n_batch > 0) {
        hisa_gather_f32<<<grid_dims, block_dims, 0, stream>>>(
            src_d, idx_d, dst_d,
            d, budget, n_heads_kv, gqa_ratio,
            src->nb[1], src->nb[2], src->nb[3],
            dst->nb[1], dst->nb[2], dst->nb[3],
            indices->nb[0], indices->nb[2], indices->nb[3]);
    }
}

// ============================================================
// Kernel 4: hisa_gather_mask
// Gather rows from kq_mask using the two-level HISA index mapping.
// Maps top_budget_indices -> topm_indices -> absolute KV position,
// then copies the corresponding mask rows.
// kq_mask:             [n_kv, T, 1, S]  (F16, since flash_attn uses F16 mask)
// topm_indices:        [m, T, Hq, S]    (I32)
// top_budget_indices:  [budget, T, Hq, S](I32)
// dst:                 [budget, T, 1, S] (F16)
// Uses head 0, token 0 for index lookup (consistent with other HISA ops).
// ============================================================

static __global__ void hisa_gather_mask_f16(
        const half * __restrict__ kq_mask,
        const int32_t * __restrict__ topm_indices,
        const int32_t * __restrict__ top_budget_indices,
        half * __restrict__ dst,
        const int block_size,
        const int n_kv, const int T, const int budget, const int S,
        const size_t mask_nb0, const size_t mask_nb1, const size_t mask_nb2, const size_t mask_nb3,
        const size_t dst_nb0,  const size_t dst_nb1,  const size_t dst_nb2,  const size_t dst_nb3,
        const size_t topm_nb0, const size_t topm_nb1, const size_t topm_nb2, const size_t topm_nb3,
        const size_t topb_nb0, const size_t topb_nb1, const size_t topb_nb2, const size_t topb_nb3) {

    // Each thread handles one output element: dst[j, t, 0, s]
    const int idx = blockIdx.x * blockDim.x + threadIdx.x;
    const int total = budget * T * S;
    if (idx >= total) return;

    // Decompose linear index into (j, t, s)
    int s = idx / (budget * T);
    int remainder = idx - s * budget * T;
    int j = remainder / T;
    int t = remainder - j * T;

    // Step 1: Read top_budget_indices[j, 0, 0, s] (head 0, token 0)
    const int32_t cand_idx = *(const int32_t *)((const char *)top_budget_indices
        + j  * topb_nb0
        + 0  * topb_nb1
        + 0  * topb_nb2
        + s  * topb_nb3);

    // Step 2: Decompose into block ordinal and offset
    const int32_t block_ord = cand_idx / block_size;
    const int32_t block_off = cand_idx % block_size;

    // Step 3: Read topm_indices[block_ord, 0, 0, s] to get absolute block index
    const int32_t block_idx = *(const int32_t *)((const char *)topm_indices
        + block_ord * topm_nb0
        + 0         * topm_nb1
        + 0         * topm_nb2
        + s         * topm_nb3);

    // Step 4: Compute absolute KV position
    const int32_t abs_pos = block_idx * block_size + block_off;

    // Step 5: Copy mask value: kq_mask[abs_pos, t, 0, s] -> dst[j, t, 0, s]
    const half * src_ptr = (const half *)((const char *)kq_mask
        + s       * mask_nb3
        + 0       * mask_nb2
        + t       * mask_nb1
        + abs_pos * mask_nb0);

    half * dst_ptr = (half *)((char *)dst
        + s * dst_nb3
        + 0 * dst_nb2
        + t * dst_nb1
        + j * dst_nb0);

    *dst_ptr = *src_ptr;
}

// F32 variant for CPU fallback / testing
static __global__ void hisa_gather_mask_f32(
        const float * __restrict__ kq_mask,
        const int32_t * __restrict__ topm_indices,
        const int32_t * __restrict__ top_budget_indices,
        float * __restrict__ dst,
        const int block_size,
        const int n_kv, const int T, const int budget, const int S,
        const size_t mask_nb0, const size_t mask_nb1, const size_t mask_nb2, const size_t mask_nb3,
        const size_t dst_nb0,  const size_t dst_nb1,  const size_t dst_nb2,  const size_t dst_nb3,
        const size_t topm_nb0, const size_t topm_nb1, const size_t topm_nb2, const size_t topm_nb3,
        const size_t topb_nb0, const size_t topb_nb1, const size_t topb_nb2, const size_t topb_nb3) {

    const int idx = blockIdx.x * blockDim.x + threadIdx.x;
    const int total = budget * T * S;
    if (idx >= total) return;

    int s = idx / (budget * T);
    int remainder = idx - s * budget * T;
    int j = remainder / T;
    int t = remainder - j * T;

    const int32_t cand_idx = *(const int32_t *)((const char *)top_budget_indices
        + j  * topb_nb0
        + 0  * topb_nb1
        + 0  * topb_nb2
        + s  * topb_nb3);

    const int32_t block_ord = cand_idx / block_size;
    const int32_t block_off = cand_idx % block_size;

    const int32_t block_idx = *(const int32_t *)((const char *)topm_indices
        + block_ord * topm_nb0
        + 0         * topm_nb1
        + 0         * topm_nb2
        + s         * topm_nb3);

    const int32_t abs_pos = block_idx * block_size + block_off;

    const float * src_ptr = (const float *)((const char *)kq_mask
        + s       * mask_nb3
        + 0       * mask_nb2
        + t       * mask_nb1
        + abs_pos * mask_nb0);

    float * dst_ptr = (float *)((char *)dst
        + s * dst_nb3
        + 0 * dst_nb2
        + t * dst_nb1
        + j * dst_nb0);

    *dst_ptr = *src_ptr;
}

void ggml_cuda_op_hisa_gather_mask(ggml_backend_cuda_context & ctx, ggml_tensor * dst) {
    const ggml_tensor * kq_mask            = dst->src[0];
    const ggml_tensor * topm_indices        = dst->src[1];
    const ggml_tensor * top_budget_indices  = dst->src[2];
    const int block_size = ggml_get_op_params_i32(dst, 0);

    const int64_t n_kv    = kq_mask->ne[0];
    const int64_t T       = kq_mask->ne[1];
    const int64_t budget  = top_budget_indices->ne[0];
    const int64_t S       = kq_mask->ne[3];

    cudaStream_t stream = ctx.stream();

    const int64_t total = budget * T * S;
    const int n_threads = 256;
    const int n_blocks_cuda = (total + n_threads - 1) / n_threads;

    if (total == 0) return;

    if (kq_mask->type == GGML_TYPE_F16) {
        GGML_ASSERT(dst->type == GGML_TYPE_F16);

        hisa_gather_mask_f16<<<n_blocks_cuda, n_threads, 0, stream>>>(
            (const half *)kq_mask->data,
            (const int32_t *)topm_indices->data,
            (const int32_t *)top_budget_indices->data,
            (half *)dst->data,
            block_size,
            (int)n_kv, (int)T, (int)budget, (int)S,
            kq_mask->nb[0], kq_mask->nb[1], kq_mask->nb[2], kq_mask->nb[3],
            dst->nb[0],      dst->nb[1],      dst->nb[2],      dst->nb[3],
            topm_indices->nb[0], topm_indices->nb[1], topm_indices->nb[2], topm_indices->nb[3],
            top_budget_indices->nb[0], top_budget_indices->nb[1], top_budget_indices->nb[2], top_budget_indices->nb[3]
        );
    } else {
        GGML_ASSERT(kq_mask->type == GGML_TYPE_F32);
        GGML_ASSERT(dst->type == GGML_TYPE_F32);

        hisa_gather_mask_f32<<<n_blocks_cuda, n_threads, 0, stream>>>(
            (const float *)kq_mask->data,
            (const int32_t *)topm_indices->data,
            (const int32_t *)top_budget_indices->data,
            (float *)dst->data,
            block_size,
            (int)n_kv, (int)T, (int)budget, (int)S,
            kq_mask->nb[0], kq_mask->nb[1], kq_mask->nb[2], kq_mask->nb[3],
            dst->nb[0],      dst->nb[1],      dst->nb[2],      dst->nb[3],
            topm_indices->nb[0], topm_indices->nb[1], topm_indices->nb[2], topm_indices->nb[3],
            top_budget_indices->nb[0], top_budget_indices->nb[1], top_budget_indices->nb[2], top_budget_indices->nb[3]
        );
    }
}
