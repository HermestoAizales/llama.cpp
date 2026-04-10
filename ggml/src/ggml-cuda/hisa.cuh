#pragma once

#include "common.cuh"

void ggml_cuda_op_hisa_block_pool(ggml_backend_cuda_context & ctx, ggml_tensor * dst);
void ggml_cuda_op_hisa_block_gather(ggml_backend_cuda_context & ctx, ggml_tensor * dst);
void ggml_cuda_op_hisa_gather(ggml_backend_cuda_context & ctx, ggml_tensor * dst);
