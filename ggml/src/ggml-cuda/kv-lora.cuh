#pragma once

#include "ggml-cuda/common.cuh"
#include <cuda_runtime.h>
#include <cuda_fp16.h>

// KV-LoRA Compression Kernels for GPU VRAM savings
// Project and reconstruct KV cache via low-rank factorization

// kv_lora_project_kernel: Extract low-rank factors from KV cache
// Input: kv [n_embd, n_tokens] (FP16/F32) - uses simplified column sampling
// Output: lora_b [rank, n_tokens] (FP32)
template<typename T>
__global__ void kv_lora_project_kernel(
    const T* __restrict__ kv,      // [n_embd, n_tokens]
    float* __restrict__ lora_b   , // [rank, n_tokens] - output
    int n_embd, int n_tokens, int rank) {

    const int tid = blockIdx.x * blockDim.x + threadIdx.x;
    const int n_total = n_tokens * rank;

    if (tid >= n_total) return;

    const int token = tid / rank;
    const int r = tid % rank;

    // Simple projection: sample columns evenly
    float sum = 0.0f;
    for (int i = r; i < n_embd; i += rank) {
        float val = kv[token * n_embd + i];
        sum += val * val;
    }
    lora_b[token * rank + r] = sqrtf(fabsf(sum / n_embd));
}

// kv_lora_reconstruct_kernel: Reconstruct KV from low-rank factors
// Input: lora_a [n_embd, rank], lora_b [rank, n_tokens]
// Output: kv_recon [n_embd, n_tokens]
// kv = lora_b @ lora_a^T (matrix multiplication)
template<typename T>
__global__ void kv_lora_reconstruct_kernel(
    const float* __restrict__ lora_a,  // [n_embd, rank]
    const float* __restrict__ lora_b,  // [rank, n_tokens]
    T* __restrict__ kv_recon,          // [n_embd, n_tokens]
    int n_embd, int n_tokens, int rank) {

    const int tid = blockIdx.x * blockDim.x + threadIdx.x;
    const int n_total = n_embd * n_tokens;

    if (tid >= n_total) return;

    const int emb = tid / n_tokens;
    const int token = tid % n_tokens;

    // Reconstruct: sum_r lora_a[emb,r] * lora_b[r,token]
    float val = 0.0f;
    for (int r = 0; r < rank; r++) {
        val += lora_a[emb * rank + r] * lora_b[r * n_tokens + token];
    }

    kv_recon[token * n_embd + emb] = (T)val;
}