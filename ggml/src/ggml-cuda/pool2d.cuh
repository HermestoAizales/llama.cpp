#include "common.cuh"

#define CUDA_POOL2D_BLOCK_SIZE 256

void ggml_cuda_op_pool2d(ggml_backend_cuda_context & ctx, ggml_tensor * dst);
void ggml_cuda_op_hisa_block_pool(ggml_backend_cuda_context & ctx, ggml_tensor * dst);
void ggml_cuda_op_hisa_gather(ggml_backend_cuda_context & ctx, ggml_tensor * dst);
void ggml_cuda_op_hisa_block_gather(ggml_backend_cuda_context & ctx, ggml_tensor * dst);
void ggml_cuda_op_hisa_gather_mask(ggml_backend_cuda_context & ctx, ggml_tensor * dst);
