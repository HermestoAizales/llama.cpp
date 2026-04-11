# Hierarchical Indexed Sparse Attention (HISA)

## Overview

HISA (Hierarchical Indexed Sparse Attention) is an optimization technique for transformer attention mechanisms that significantly reduces computational cost for long-context models by selectively attending to only the most relevant tokens. HISA employs a two-level selection strategy:

1. **Block-level filtering**: Coarse filtering of K/V tokens into blocks using mean pooling
2. **Token-level refinement**: Selecting top tokens within each block for final attention

## How It Works

### Block-Level Selection

For K/V tokens with length `n_kv`, HISA divides them into blocks of size `B` (typically 128):
```
n_blocks = n_kv / B
```

Each block is represented by the mean of its `B` token vectors. This reduces the attention cost from `O(n_kv²)` to `O(n_blocks × n_tokens)`.

### Token-Level Refinement

Within each selected block, individual tokens are scored and the top `budget` tokens are selected for final attention:

```
token_scores = Q @ K_cand^T  # Score each candidate token
top_budget_tokens = argsort(token_scores, k=budget)
```

### Final Attention

Only the top `budget` tokens attend to queries, reducing the attention cost to `O(budget × n_tokens)`.

## Parameters

| Parameter | Description | Default |
|-----------|-------------|---------|
| `--hisa` | Enable HISA (default: disabled) | `off` |
| `--hisa-block-size B` | Block size for coarse filtering | `128` |
| `--hisa-top-m M` | Number of top blocks to select | `auto (n_blocks/4)` |
| `--hisa-budget N` | Final token budget | `2048` |
| `--hisa-min-tokens N` | Activate HISA only when KV length exceeds this | `4096` |

## Usage

### With llama-batched-bench

```sh
# Basic usage with HISA enabled
llama-batched-bench -m model.gguf -p 'your prompt' --hisa --hisa-min-tokens 1 -n 100 -b 8 -t 4

# With custom parameters
llama-batched-bench -m model.gguf -p 'your prompt'   --hisa   --hisa-block-size 256   --hisa-top-m 16   --hisa-budget 1024   --hisa-min-tokens 1024   -n 50 -b 8 -t 4
```

### With llama-perplexity

```sh
# Measure perplexity with HISA
llama-perplexity -m model.gguf -p 'your text' --hisa --hisa-min-tokens 1
```

### Command Line Examples

```sh
# Small model (HISA disabled by default due to small context)
llama-batched-bench -m tinyllama.gguf -p 'test' -n 10

# Medium-large model (HISA enabled)
llama-batched-bench -m qwen2.5-0.5b.gguf -p 'your text here'   --hisa --hisa-min-tokens 100 -n 50

# Long-context model with aggressive HISA
llama-batched-bench -m gemma4-large.gguf -p 'long context text...'   --hisa --hisa-block-size 128 --hisa-top-m 32 --hisa-budget 2048   --hisa-min-tokens 8192 -n 100
```

## When HISA Is Activated

HISA is automatically enabled when:
1. Flash Attention is enabled (auto, on, or explicitly)
2. KV length > `hisa-min-tokens` (default: 4096)
3. KV length is divisible by `hisa-block-size` (default: 128)

If any condition is not met, HISA is automatically disabled for that layer with a debug log message.

## Performance Characteristics

### Time Complexity

| Stage | Normal Attention | HISA (with budget=N) |
|-------|------------------|---------------------|
| Block pooling | - | O(n_kv × B × d) |
| Block scoring | O(n_blocks × n_tokens × d) | - |
| Token selection | - | O(n_cand × n_tokens × d) |
| Final attention | O(n_kv × n_tokens × d) | O(N × n_tokens × d) |

### Space Complexity

| Stage | Normal Attention | HISA |
|-------|------------------|------|
| KV cache | O(n_kv × n_head_kv × d) | - |
| Block indices | - | O(m × n_tokens × n_head_q) |
| Gathered K/V | - | O(N × n_head_kv × d) |

Where:
- `n_kv` = total KV tokens
- `B` = block size
- `m` = number of selected blocks (default: n_blocks/4)
- `n_cand` = total candidate tokens (m × B)
- `N` = final budget
- `n_tokens` = number of query tokens
- `n_head_q` = number of query heads
- `n_head_kv` = number of key/value heads (GQA)
- `d` = embedding dimension

## Practical Considerations

### Benefits

- **Significant speedup for long contexts**: Reduces attention cost by 75-95% depending on budget
- **Lower memory usage**: Only stores budget tokens instead of full KV cache
- **Automatic activation**: Works automatically with Flash Attention
- **No model changes**: Works with existing GGUF models

### Trade-offs

- **Quality**: Slight reduction in attention quality due to coarse block selection
- **Budget requirement**: Requires sufficient tokens for effective filtering
- **Context length restrictions**: KV length must be divisible by block size

### Best Use Cases

- **Long-context models** (8k-32k context windows)
- **Generative inference** (large KV cache)
- **Batched processing** with many tokens
- **Resource-constrained environments**

### When To Disable HISA

- Short context lengths (n_kv < hisa-min-tokens)
- Small models (insignificant benefits)
- Tasks requiring exact attention patterns
- When KV length is not divisible by block size

## Implementation Details

### Op Types

HISA uses custom GGML operations:

| Op Type | Description |
|---------|-------------|
| `GGML_OP_HISA_BLOCK_POOL` | Mean-pool K rows into blocks |
| `GGML_OP_HISA_BLOCK_GATHER` | Gather full blocks by block index list |
| `GGML_OP_HISA_GATHER` | Gather rows by index list |
| `GGML_OP_HISA_GATHER_MASK` | Gather mask rows via two-level index mapping |

### CUDA Support

HISA CUDA kernels are implemented in `ggml/src/ggml-cuda/hisa.cu`:
- `ggml_cuda_op_hisa_block_pool`
- `ggml_cuda_op_hisa_block_gather`
- `ggml_cuda_op_hisa_gather`
- `ggml_cuda_op_hisa_gather_mask`

### CPU Support

HISA is supported on CPU using ggml-compatible implementations.

## Troubleshooting

### "KV length not divisible by HISA block size"

**Solution**: Either:
1. Reduce `--hisa-block-size` to divide evenly into n_kv
2. Disable HISA for this model (HISA will be auto-disabled with a log message)

### "HISA disabled for layer X"

**Explanation**: HISA was disabled for a specific layer because one of the activation conditions wasn't met.

**Check**:
- Is n_kv > hisa-min-tokens?
- Is n_kv % hisa-block-size == 0?
- Is flash_attn enabled?

### No performance improvement

**Explanation**: For small models or short contexts, HISA overhead may outweigh benefits.

**Solution**:
- Use `--hisa-min-tokens` higher to avoid unnecessary overhead
- Test with different block sizes (16, 32, 64, 128, 256)

## References

- Paper: [Hierarchical Sparse Transformer](https://arxiv.org/abs/1909.07758)
- Related: Flash Attention, Memory Efficient Attention
