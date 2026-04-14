// CPU implementation of HISA operations for llama.cpp

#include "ggml.h"
#include "ggml-common.h"  // For uint16_t (uint16_t)
#include "ggml-cpu/ops.h"
#include "ggml-cpu/vec.h"
#include <cstring>
#include <algorithm>
#include <cstdint>

// Helper to get tensor data pointers
static float* get_data_f32(ggml_tensor * t) {
    return (float *)t->data;
}

static const float* get_data_f32_const(const ggml_tensor * t) {
    return (const float *)t->data;
}

static const uint16_t* get_data_hf16_const(const ggml_tensor * t) {
    return (const uint16_t *)t->data;
}

static uint16_t* get_data_hf16(ggml_tensor * t) {
    return (uint16_t *)t->data;
}

// Helper to compute output dimensions
static void get_output_dims(const ggml_tensor * dst,
                           int64_t & ne0, int64_t & ne1,
                           int64_t & ne2, int64_t & ne3,
                           size_t & nb0, size_t & nb1,
                           size_t & nb2, size_t & nb3) {
    ne0 = dst->ne[0];
    ne1 = dst->ne[1];
    ne2 = dst->ne[2];
    ne3 = dst->ne[3];
    nb0 = dst->nb[0];
    nb1 = dst->nb[1];
    nb2 = dst->nb[2];
    nb3 = dst->nb[3];
}

// ============================================================ //
// HISA Block Pool - CPU Implementation                         //
// ============================================================ //

void ggml_compute_forward_hisa_block_pool(const ggml_compute_params * params, ggml_tensor * dst) {
    if (params->ith != 0) return;
    
    const ggml_tensor * src = dst->src[0];
    if (!src) return;
    
    const int32_t block_size = ggml_get_op_params_i32(dst, 0);
    if (block_size <= 0) return;
    
    GGML_TENSOR_LOCALS(int64_t, ne0, src, ne)
    GGML_TENSOR_LOCALS(size_t,  nb0, src, nb)
    GGML_TENSOR_LOCALS(int64_t, ne,  dst,  ne)
    GGML_TENSOR_LOCALS(size_t,  nb,  dst,  nb)
    
    const int64_t d = ne0;
    const int64_t n_kv = ne1;
    const int64_t n_heads = ne2;
    const int64_t n_batch = ne3;
    const int64_t n_blocks = dst->ne[1];
    
    const float * src_data = get_data_f32_const(src);
    float * dst_data = get_data_f32(dst);
    
    (void)d; (void)n_kv; (void)n_heads; (void)n_batch; (void)n_blocks;
    
    // Simple CPU implementation
    #pragma omp parallel for collapse(2) schedule(dynamic)
    for (int64_t iblk = 0; iblk < n_blocks; iblk++) {
        for (int64_t ih = 0; ih < n_heads; ih++) {
            const int64_t src_row_base = iblk * block_size;
            const char * src_base = (const char *)src_data + ih * src->nb[2];
            char * dst_base = (char *)dst_data + ih * dst->nb[2];
            
            for (int64_t i = 0; i < block_size; i++) {
                float sum = 0.0f;
                for (int64_t j = 0; j < d; j++) {
                    sum += ((const float *)src_base)[(src_row_base + i) * d + j];
                }
                ((float *)dst_base)[i * d] = sum / (float)block_size;
            }
        }
    }
}

// ============================================================ //
// HISA Gather - CPU Implementation                            //
// ============================================================ //

void ggml_compute_forward_hisa_gather(const ggml_compute_params * params, ggml_tensor * dst) {
    if (params->ith != 0) return;
    
    const ggml_tensor * src = dst->src[0];
    const ggml_tensor * indices = dst->src[1];
    if (!src || !indices) return;
    
    const int32_t block_size = ggml_get_op_params_i32(dst, 0);
    if (block_size <= 0) return;
    
    GGML_TENSOR_LOCALS(int64_t, ne0, src, ne)
    GGML_TENSOR_LOCALS(size_t,  nb0, src, nb)
    GGML_TENSOR_LOCALS(int64_t, ne,  dst,  ne)
    GGML_TENSOR_LOCALS(size_t,  nb,  dst,  nb)
    
    const int64_t d = ne0;
    const int64_t n_heads = ne1;
    const int64_t n_batch = ne3;
    const int64_t n_tokens = indices->ne[0];
    
    const float * src_data = get_data_f32_const(src);
    const int32_t * idx_data = (const int32_t *)indices->data;
    float * dst_data = get_data_f32(dst);
    
    (void)d; (void)n_heads; (void)n_batch; (void)n_tokens;
    (void)block_size;
    
    // Simple CPU implementation
    #pragma omp parallel for collapse(2) schedule(dynamic)
    for (int64_t ib = 0; ib < n_batch; ib++) {
        for (int64_t ih = 0; ih < n_heads; ih++) {
            for (int64_t i = 0; i < n_tokens; i++) {
                const int32_t idx = idx_data[i];
                if (idx < 0 || idx >= (int32_t)(ne0 / block_size)) {
                    for (int64_t j = 0; j < d; j++) {
                        ((float *)((char *)dst_data + ib * dst->nb[3] + 
                                  ih * dst->nb[2]))[i * d + j] = 0.0f;
                    }
                    continue;
                }
                
                const int64_t src_row = idx * block_size;
                for (int64_t j = 0; j < d; j++) {
                    ((float *)((char *)dst_data + ib * dst->nb[3] + 
                              ih * dst->nb[2]))[i * d + j] = 
                        ((const float *)((const char *)src_data + ib * src->nb[3] + 
                                        ih * src->nb[2]))[src_row * d + j];
                }
            }
        }
    }
}

// ============================================================ //
// HISA Block Gather - CPU Implementation                       //
// ============================================================ //

void ggml_compute_forward_hisa_block_gather(const ggml_compute_params * params, ggml_tensor * dst) {
    if (params->ith != 0) return;
    
    const ggml_tensor * src = dst->src[0];
    const ggml_tensor * block_indices = dst->src[1];
    if (!src || !block_indices) return;
    
    const int32_t block_size = ggml_get_op_params_i32(dst, 0);
    if (block_size <= 0) return;
    
    GGML_TENSOR_LOCALS(int64_t, ne0, src, ne)
    GGML_TENSOR_LOCALS(size_t,  nb0, src, nb)
    GGML_TENSOR_LOCALS(int64_t, ne,  dst,  ne)
    GGML_TENSOR_LOCALS(size_t,  nb,  dst,  nb)
    
    const int64_t d = ne0;
    const int64_t m = block_indices->ne[0];
    const int64_t n_batch = ne3;
    
    const float * src_data = get_data_f32_const(src);
    const int32_t * idx_data = (const int32_t *)block_indices->data;
    float * dst_data = get_data_f32(dst);
    
    (void)d; (void)m; (void)n_batch;
    (void)block_size;
    
    #pragma omp parallel for schedule(dynamic)
    for (int64_t ib = 0; ib < n_batch; ib++) {
        for (int64_t i = 0; i < m; i++) {
            const int32_t block_idx = idx_data[i];
            const int64_t src_row = block_idx * block_size;
            
            for (int64_t j = 0; j < d; j++) {
                ((float *)((char *)dst_data + ib * dst->nb[3]))[i * d + j] = 
                    ((const float *)((const char *)src_data + ib * src->nb[3]))
                    [src_row * d + j];
            }
        }
    }
}

// ============================================================ //
// HISA Gather Mask - CPU Implementation                        //
// ============================================================ //

void ggml_compute_forward_hisa_gather_mask(const ggml_compute_params * params, ggml_tensor * dst) {
    if (params->ith != 0) return;
    
    const ggml_tensor * kq_mask = dst->src[0];
    const ggml_tensor * topm_indices = dst->src[1];
    const ggml_tensor * top_budget_indices = dst->src[2];
    
    if (!kq_mask || !topm_indices || !top_budget_indices) return;
    
    const int32_t block_size = ggml_get_op_params_i32(dst, 0);
    if (block_size <= 0) return;
    
    GGML_TENSOR_LOCALS(int64_t, ne0, kq_mask, ne)
    GGML_TENSOR_LOCALS(size_t,  nb0, kq_mask, nb)
    GGML_TENSOR_LOCALS(int64_t, ne,  dst,  ne)
    GGML_TENSOR_LOCALS(size_t,  nb,  dst,  nb)
    
    const int n_kv = ne0;
    const int T = ne1;
    const int S = ne3;
    const int budget = top_budget_indices->ne[0];
    
    const uint16_t * kq_mask_data = get_data_hf16_const(kq_mask);
    const int32_t * topm_data = (const int32_t *)topm_indices->data;
    const int32_t * topb_data = (const int32_t *)top_budget_indices->data;
    uint16_t * dst_data = get_data_hf16(dst);
    
    (void)n_kv; (void)T; (void)S; (void)budget;
    (void)kq_mask_data; (void)topm_data; (void)topb_data; (void)dst_data;
    (void)block_size;
    
    // Simple CPU implementation
    const int total = budget * T * S;
    
    #pragma omp parallel for schedule(static)
    for (int idx = 0; idx < total; idx++) {
        const int s = idx / (budget * T);
        int rem = idx - s * budget * T;
        const int j = rem / T;
        const int t = rem - j * T;
        
        const int32_t cand_idx = topb_data[j];
        const int32_t block_ord = cand_idx / block_size;
        const int32_t block_off = cand_idx % block_size;
        const int32_t block_idx = topm_data[block_ord];
        const int32_t abs_pos = block_idx * block_size + block_off;
        
        ((uint16_t *)((char *)dst_data + s * dst->nb[3] + 
                       t * dst->nb[1] + abs_pos * dst->nb[0]))[0] = 
            kq_mask_data[s * kq_mask->nb[3] + 
                        t * kq_mask->nb[1] + abs_pos * kq_mask->nb[0]];
    }
}
