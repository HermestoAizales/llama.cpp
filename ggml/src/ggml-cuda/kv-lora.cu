#include "kv-lora.cuh"

// KV-LoRA Project: Extract low-rank factors from KV cache
void ggml_cuda_op_kv_lora_project(
    const ggml_tensor* dst, const ggml_tensor* src0,
    const ggml_tensor* src1, ggml_tensor* dst2,
    const char* src0_host, const char* src1_host,
    const char* dst_host, const char* dst2_host,
    cudaStream_t stream) {
    
    const int64_t n_embd = src0->ne[0];
    const int64_t n_tokens = src0->ne[1];
    const int rank = dst->op_params[0];  // rank stored in op_params
    
    // For project, we use the existing mul_mat path
    // This is a placeholder - real implementation would do QR/SVD
    ggml_cuda_mul_mat(n_embd, n_tokens, rank,
                     src0->data, dst->data,
                     src0->type, GGML_TYPE_F32,
                     stream);
}

// KV-LoRA Reconstruct: Rebuild KV cache from low-rank factors
void ggml_cuda_op_kv_lora_reconstruct(
    const ggml_tensor* dst, const ggml_tensor* src0,
    const ggml_tensor* src1, ggml_tensor* dst2,
    const char* src0_host, const char* src1_host,
    const char* dst_host, const char* dst2_host,
    cudaStream_t stream) {
    
    const int64_t n_embd = dst->ne[0];
    const int64_t n_tokens = dst->ne[1];
    const int rank = src0->ne[1];
    
    // Reconstruction: kv = lora_b @ lora_a^T
    // Use standard matrix multiplication
    ggml_cuda_mul_mat(rank, n_tokens, n_embd,
                     src1->data, dst->data,
                     GGML_TYPE_F32, GGML_TYPE_F32,
                     stream);
}