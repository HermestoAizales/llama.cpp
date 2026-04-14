#include "hisa.cuh"

static __global__ void hisa_block_pool_f32(const float * __restrict__ src, float * __restrict__ dst, const int64_t d, const int64_t n_kv, const int64_t n_blocks, const int32_t block_size, const size_t src_nb1, const size_t src_nb2, const size_t src_nb3, const size_t dst_nb1, const size_t dst_nb2, const size_t dst_nb3) {
    // Shared memory cache for block data
    __shared__ float s_data[256];
    // Warp-level reduction using shuffle
    unsigned active_mask = __activemask();
    
    const int64_t iblk = blockIdx.x; const int64_t ih = blockIdx.y; const int64_t ib = blockIdx.z;
    if (iblk >= n_blocks || ib >= gridDim.z) return;
    const int64_t src_row_base = iblk * block_size;
    const char * src_base = (const char *)src + ib * src_nb3 + ih * src_nb2;
    char * dst_base = (char *)dst + ib * dst_nb3 + ih * dst_nb2;
    
    // Each thread accumulates its own sum
    float thread_sum = 0.0f;
    
    // Coalesced load of block data into shared memory
    int tid = threadIdx.x;
    if (tid < block_size) {
        s_data[tid] = *((const float *)(src_base + (src_row_base + tid) * src_nb1));
    }
    __syncthreads();
    
    // Compute with warp-shuffle reduction
    for (int32_t b = tid; b < block_size; b += blockDim.x) {
        thread_sum += s_data[b];
    }
    
    // Warp-level shuffle reduction
    for (int offset = warpSize / 2; offset > 0; offset /= 2) {
        thread_sum += __shfl_down_sync(active_mask, thread_sum, offset);
    }
    
    // One thread per warp writes result
    if (tid % warpSize == 0) {
        atomicAdd((float*)(dst_base + iblk * dst_nb1 + (tid / warpSize) * sizeof(float)), thread_sum / (float)block_size);
    }
}

static __global__ void hisa_block_pool_f16(const half * __restrict__ src, half * __restrict__ dst, const int64_t d, const int64_t n_kv, const int64_t n_blocks, const int32_t block_size, const size_t src_nb1, const size_t src_nb2, const size_t src_nb3, const size_t dst_nb1, const size_t dst_nb2, const size_t dst_nb3) {
    // Shared memory cache for block data
    __shared__ half s_data[256];
    // Warp-level reduction using shuffle
    unsigned active_mask = __activemask();
    
    const int64_t iblk = blockIdx.x; const int64_t ih = blockIdx.y; const int64_t ib = blockIdx.z;
    if (iblk >= n_blocks || ib >= gridDim.z) return;
    const int64_t src_row_base = iblk * block_size;
    const char * src_base = (const char *)src + ib * src_nb3 + ih * src_nb2;
    char * dst_base = (char *)dst + ib * dst_nb3 + ih * dst_nb2;
    
    // Each thread accumulates its own sum
    float thread_sum = 0.0f;
    
    // Coalesced load of block data into shared memory
    int tid = threadIdx.x;
    if (tid < block_size) {
        s_data[tid] = *((const half *)(src_base + (src_row_base + tid) * src_nb1));
    }
    __syncthreads();
    
    // Compute with warp-shuffle reduction (convert to float for computation)
    for (int32_t b = tid; b < block_size; b += blockDim.x) {
        thread_sum += __half2float(s_data[b]);
    }
    
    // Warp-level shuffle reduction
    for (int offset = warpSize / 2; offset > 0; offset /= 2) {
        thread_sum += __shfl_down_sync(active_mask, thread_sum, offset);
    }
    
    // One thread per warp writes result
    if (tid % warpSize == 0) {
        atomicAdd((float*)(dst_base + iblk * dst_nb1 + (tid / warpSize) * sizeof(float)), thread_sum / (float)block_size);
    }
}

void ggml_cuda_op_hisa_block_pool(ggml_backend_cuda_context & ctx, ggml_tensor * dst) {
    const ggml_tensor * src = dst->src[0];
    const float * src_d = (const float *)src->data;
    float * dst_d = (float *)dst->data;
    cudaStream_t stream = ctx.stream();
    const int32_t block_size = ggml_get_op_params_i32(dst, 0);
    const int64_t d = src->ne[0], n_kv = src->ne[1], n_heads = src->ne[2], n_batch = src->ne[3], n_blocks = dst->ne[1];
    const dim3 block_dims(256, 1, 1), grid_dims(n_blocks, n_heads, n_batch);
    if (n_blocks > 0 && n_heads > 0 && n_batch > 0 && d > 0) {
        hisa_block_pool_f32<<<grid_dims, block_dims, 0, stream>>>(src_d, dst_d, d, n_kv, n_blocks, block_size, src->nb[1], src->nb[2], src->nb[3], dst->nb[1], dst->nb[2], dst->nb[3]);
    }
}

static __global__ void hisa_block_gather_f32(const float * __restrict__ src, const int32_t * __restrict__ block_indices, float * __restrict__ dst, const int64_t d, const int64_t m, const int32_t block_size, const int64_t n_heads_kv, const int64_t gqa_ratio, const size_t src_nb1, const size_t src_nb2, const size_t src_nb3, const size_t dst_nb1, const size_t dst_nb2, const size_t dst_nb3, const size_t idx_nb0, const size_t idx_nb2, const size_t idx_nb3) {
    __shared__ float shared_block[256];  // Cache for gather data
    const int64_t i = (int64_t)blockIdx.x * blockDim.x + threadIdx.x;
    const int64_t total = d * m * block_size;
    if (i >= total) return;
    int64_t tmp = i; const int64_t j = tmp % d; tmp /= d; const int64_t b = tmp % block_size; const int64_t im = tmp / block_size;
    
    // Cache block indices and data
    const int64_t cache_idx = threadIdx.x;
    if (cache_idx < block_size) {
        shared_block[cache_idx] = src[cache_idx];  // Simplified caching
    }
    __syncthreads();
    const int64_t ih = blockIdx.y; const int64_t ib = blockIdx.z;
    if (ih >= n_heads_kv) return;
    const int64_t ih_q = ih * gqa_ratio;
    const int64_t idx_offset = im * idx_nb0 + 0 * 0 + ih_q * idx_nb2 + ib * idx_nb3;
    const int32_t blk_idx = *(const int32_t *)((const char *)block_indices + idx_offset);
    const int32_t src_row = blk_idx * block_size + (int32_t)b;
    const int64_t dst_row = im * block_size + b;
    *((float *)((char *)dst + ib * dst_nb3 + ih * dst_nb2 + dst_row * dst_nb1 + j * sizeof(float))) = *((const float *)((const char *)src + ib * src_nb3 + ih * src_nb2 + src_row * src_nb1 + j * sizeof(float)));
}

static __global__ void hisa_block_gather_f16(const half * __restrict__ src, const int32_t * __restrict__ block_indices, half * __restrict__ dst, const int64_t d, const int64_t m, const int32_t block_size, const int64_t n_heads_kv, const int64_t gqa_ratio, const size_t src_nb1, const size_t src_nb2, const size_t src_nb3, const size_t dst_nb1, const size_t dst_nb2, const size_t dst_nb3, const size_t idx_nb0, const size_t idx_nb2, const size_t idx_nb3) {
    __shared__ half shared_block[256];  // Cache for gather data
    const int64_t i = (int64_t)blockIdx.x * blockDim.x + threadIdx.x;
    const int64_t total = d * m * block_size;
    if (i >= total) return;
    int64_t tmp = i; const int64_t j = tmp % d; tmp /= d; const int64_t b = tmp % block_size; const int64_t im = tmp / block_size;
    
    // Cache block indices and data
    const int64_t cache_idx = threadIdx.x;
    if (cache_idx < block_size) {
        shared_block[cache_idx] = src[cache_idx];  // Simplified caching
    }
    __syncthreads();
    const int64_t ih = blockIdx.y; const int64_t ib = blockIdx.z;
    if (ih >= n_heads_kv) return;
    const int64_t ih_q = ih * gqa_ratio;
    const int64_t idx_offset = im * idx_nb0 + 0 * 0 + ih_q * idx_nb2 + ib * idx_nb3;
    const int32_t blk_idx = *(const int32_t *)((const char *)block_indices + idx_offset);
    const int32_t src_row = blk_idx * block_size + (int32_t)b;
    const int64_t dst_row = im * block_size + b;
    *((half *)((char *)dst + ib * dst_nb3 + ih * dst_nb2 + dst_row * dst_nb1 + j * sizeof(half))) = *((const half *)((const char *)src + ib * src_nb3 + ih * src_nb2 + src_row * src_nb1 + j * sizeof(half)));
}

void ggml_cuda_op_hisa_block_gather(ggml_backend_cuda_context & ctx, ggml_tensor * dst) {
    const ggml_tensor * src = dst->src[0]; const ggml_tensor * block_indices = dst->src[1];
    const float * src_d = (const float *)src->data; const int32_t * idx_d = (const int32_t *)block_indices->data; float * dst_d = (float *)dst->data;
    cudaStream_t stream = ctx.stream();
    const int32_t block_size = ggml_get_op_params_i32(dst, 0);
    const int64_t d = src->ne[0], n_heads_kv = src->ne[2], n_batch = src->ne[3], m = block_indices->ne[0];
    const int64_t gqa_ratio = n_heads_kv > 0 ? block_indices->ne[2] / n_heads_kv : 1;
    const int64_t total_per_head_batch = d * m * block_size;
    const dim3 block_dims(256, 1, 1), grid_dims((total_per_head_batch + 255) / 256, n_heads_kv, n_batch);
    if (total_per_head_batch > 0 && n_heads_kv > 0 && n_batch > 0) {
        hisa_block_gather_f32<<<grid_dims, block_dims, 0, stream>>>(src_d, idx_d, dst_d, d, m, block_size, n_heads_kv, gqa_ratio, src->nb[1], src->nb[2], src->nb[3], dst->nb[1], dst->nb[2], dst->nb[3], block_indices->nb[0], block_indices->nb[2], block_indices->nb[3]);
    }
}

static __global__ void hisa_gather_f32(const float * __restrict__ src, const int32_t * __restrict__ indices, float * __restrict__ dst, const int64_t d, const int64_t budget, const int64_t n_heads_kv, const int64_t gqa_ratio, const size_t src_nb1, const size_t src_nb2, const size_t src_nb3, const size_t dst_nb1, const size_t dst_nb2, const size_t dst_nb3, const size_t idx_nb0, const size_t idx_nb2, const size_t idx_nb3) {
    const int64_t i = (int64_t)blockIdx.x * blockDim.x + threadIdx.x;
    const int64_t total = d * budget;
    if (i >= total) return;
    const int64_t j = i % d; const int64_t is = i / d;
    const int64_t ih = blockIdx.y; const int64_t ib = blockIdx.z;
    if (ih >= n_heads_kv) return;
    const int64_t ih_q = ih * gqa_ratio;
    const int64_t idx_offset = is * idx_nb0 + 0 * 0 + ih_q * idx_nb2 + ib * idx_nb3;
    const int32_t idx = *(const int32_t *)((const char *)indices + idx_offset);
    *((float *)((char *)dst + ib * dst_nb3 + ih * dst_nb2 + is * dst_nb1 + j * sizeof(float))) = *((const float *)((const char *)src + ib * src_nb3 + ih * src_nb2 + idx * src_nb1 + j * sizeof(float)));
}

static __global__ void hisa_gather_f16(const half * __restrict__ src, const int32_t * __restrict__ indices, half * __restrict__ dst, const int64_t d, const int64_t budget, const int64_t n_heads_kv, const int64_t gqa_ratio, const size_t src_nb1, const size_t src_nb2, const size_t src_nb3, const size_t dst_nb1, const size_t dst_nb2, const size_t dst_nb3, const size_t idx_nb0, const size_t idx_nb2, const size_t idx_nb3) {
    const int64_t i = (int64_t)blockIdx.x * blockDim.x + threadIdx.x;
    const int64_t total = d * budget;
    if (i >= total) return;
    const int64_t j = i % d; const int64_t is = i / d;
    const int64_t ih = blockIdx.y; const int64_t ib = blockIdx.z;
    if (ih >= n_heads_kv) return;
    const int64_t ih_q = ih * gqa_ratio;
    const int64_t idx_offset = is * idx_nb0 + 0 * 0 + ih_q * idx_nb2 + ib * idx_nb3;
    const int32_t idx = *(const int32_t *)((const char *)indices + idx_offset);
    *((half *)((char *)dst + ib * dst_nb3 + ih * dst_nb2 + is * dst_nb1 + j * sizeof(half))) = *((half *)((const char *)src + ib * src_nb3 + ih * src_nb2 + idx * src_nb1 + j * sizeof(half)));
}

void ggml_cuda_op_hisa_gather(ggml_backend_cuda_context & ctx, ggml_tensor * dst) {
    const ggml_tensor * src = dst->src[0]; const ggml_tensor * indices = dst->src[1];
    const float * src_d = (const float *)src->data; const int32_t * idx_d = (const int32_t *)indices->data; float * dst_d = (float *)dst->data;
    cudaStream_t stream = ctx.stream();
    const int32_t block_size = ggml_get_op_params_i32(dst, 0);
    const int64_t d = src->ne[0], n_heads_kv = src->ne[2], n_batch = src->ne[3], budget = indices->ne[0];
    const int64_t gqa_ratio = n_heads_kv > 0 ? indices->ne[2] / n_heads_kv : 1;
    const int64_t total_per_head_batch = d * budget;
    const dim3 block_dims(256, 1, 1), grid_dims((total_per_head_batch + 255) / 256, n_heads_kv, n_batch);
    if (total_per_head_batch > 0 && n_heads_kv > 0 && n_batch > 0) {
        hisa_gather_f32<<<grid_dims, block_dims, 0, stream>>>(src_d, idx_d, dst_d, d, budget, n_heads_kv, gqa_ratio, src->nb[1], src->nb[2], src->nb[3], dst->nb[1], dst->nb[2], dst->nb[3], indices->nb[0], indices->nb[2], indices->nb[3]);
    }
}

static __global__ void hisa_gather_mask_f16(const half * __restrict__ kq_mask, const int32_t * __restrict__ topm_indices, const int32_t * __restrict__ top_budget_indices, half * __restrict__ dst, const int block_size, const int n_kv, const int T, const int budget, const int S, const size_t mask_nb0, const size_t mask_nb1, const size_t mask_nb2, const size_t mask_nb3, const size_t dst_nb0, const size_t dst_nb1, const size_t dst_nb2, const size_t dst_nb3, const size_t topm_nb0, const size_t topm_nb1, const size_t topm_nb2, const size_t topm_nb3, const size_t topb_nb0, const size_t topb_nb1, const size_t topb_nb2, const size_t topb_nb3) {
    const int idx = blockIdx.x * blockDim.x + threadIdx.x;
    const int total = budget * T * S;
    if (idx >= total) return;
    int s = idx / (budget * T); int remainder = idx - s * budget * T; int j = remainder / T; int t = remainder - j * T;
    const int32_t cand_idx = *(const int32_t *)((const char *)top_budget_indices + j * topb_nb0 + 0 * topb_nb1 + 0 * topb_nb2 + s * topb_nb3);
    const int32_t block_ord = cand_idx / block_size; const int32_t block_off = cand_idx % block_size;
    const int32_t block_idx = *(const int32_t *)((const char *)topm_indices + block_ord * topm_nb0 + 0 * topm_nb1 + 0 * topm_nb2 + s * topm_nb3);
    const int32_t abs_pos = block_idx * block_size + block_off;
    *((half *)((char *)dst + s * dst_nb3 + 0 * dst_nb2 + t * dst_nb1 + abs_pos * dst_nb0)) = *((const half *)((const char *)kq_mask + s * mask_nb3 + 0 * mask_nb2 + t * mask_nb1 + abs_pos * mask_nb0));
}

void ggml_cuda_op_hisa_gather_mask(ggml_backend_cuda_context & ctx, ggml_tensor * dst) {
    const ggml_tensor * kq_mask = dst->src[0]; const ggml_tensor * topm_indices = dst->src[1]; const ggml_tensor * top_budget_indices = dst->src[2];
    const half * kq_mask_d = (const half *)kq_mask->data; const int32_t * topm_indices_d = (const int32_t *)topm_indices->data; const int32_t * top_budget_indices_d = (const int32_t *)top_budget_indices->data; half * dst_d = (half *)dst->data;
    cudaStream_t stream = ctx.stream();
    const int32_t block_size = ggml_get_op_params_i32(dst, 0);
    const int n_kv = kq_mask->ne[0], T = kq_mask->ne[1], S = kq_mask->ne[3], budget = top_budget_indices->ne[0];
    const dim3 block_dims(256, 1, 1), grid_dims((budget * T * S + 255) / 256, 1, 1);
    if (budget > 0 && T > 0 && S > 0) {
        hisa_gather_mask_f16<<<grid_dims, block_dims, 0, stream>>>(kq_mask_d, topm_indices_d, top_budget_indices_d, dst_d, block_size, n_kv, T, budget, S, kq_mask->nb[0], kq_mask->nb[1], kq_mask->nb[2], kq_mask->nb[3], dst->nb[0], dst->nb[1], dst->nb[2], dst->nb[3], topm_indices->nb[0], topm_indices->nb[1], topm_indices->nb[2], topm_indices->nb[3], top_budget_indices->nb[0], top_budget_indices->nb[1], top_budget_indices->nb[2], top_budget_indices->nb[3]);
    }
}
