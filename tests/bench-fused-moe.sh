#!/bin/bash
# Benchmark script for Fused MoE performance comparison
# Measures tokens/sec with --fused-moe=off vs --fused-moe=on
#
# Usage: ./tests/bench-fused-moe.sh <model_path>
# Example: ./tests/bench-fused-moe.sh /models/qwen36-35b-a3b.gguf

set -euo pipefail

MODEL="${1:?Usage: $0 <model_path>}"
PROMPT="The quick brown fox jumps over the lazy dog. This is a longer prompt to ensure meaningful benchmark results across multiple tokens."
N_TOKENS=128
LLAMA_BIN="./build/bin/llama-cli"

if [ ! -f "$LLAMA_BIN" ]; then
    echo "ERROR: $LLAMA_BIN not found. Build first with: cmake --build build"
    exit 1
fi

if [ ! -f "$MODEL" ]; then
    echo "ERROR: Model file not found: $MODEL"
    exit 1
fi

echo "=== Fused MoE Benchmark ==="
echo "Model: $MODEL"
echo "Tokens: $N_TOKENS"
echo ""

run_benchmark() {
    local label="$1"
    local extra_args="$2"
    echo "--- $label ---"

    # Run and capture timing from stderr
    local output
    output=$($LLAMA_BIN -m "$MODEL" -p "$PROMPT" -n "$N_TOKENS" $extra_args -ngl 99 --temp 0 --seed 42 2>&1 1>/dev/null)

    # Extract timing info
    local tokens_per_sec
    tokens_per_sec=$(echo "$output" | grep -oP 'tokens per second:\s+\K[0-9.]+' || echo "N/A")

    local prompt_tps
    prompt_tps=$(echo "$output" | grep -oP 'prompt tokens per second:\s+\K[0-9.]+' || echo "N/A")

    local eval_tps
    eval_tps=$(echo "$output" | grep -oP 'eval tokens per second:\s+\K[0-9.]+' || echo "N/A")

    echo "  Prompt TPS: $prompt_tps"
    echo "  Eval TPS:   $eval_tps"
    echo "  Total TPS:  $tokens_per_sec"
    echo ""

    echo "$tokens_per_sec"
}

# Baseline
BASELINE_TPS=$(run_benchmark "Baseline (fused-moe off)" "--fused-moe off")

# Fused MoE
FUSED_TPS=$(run_benchmark "Fused MoE (fused-moe on)" "--fused-moe on")

# Compare
echo "--- Results ---"
echo "Baseline TPS: $BASELINE_TPS"
echo "Fused MoE TPS: $FUSED_TPS"

if [ "$BASELINE_TPS" != "N/A" ] && [ "$FUSED_TPS" != "N/A" ]; then
    SPEEDUP=$(echo "scale=2; $FUSED_TPS / $BASELINE_TPS" | bc 2>/dev/null || echo "N/A")
    echo "Speedup: ${SPEEDUP}x"

    if (( $(echo "$SPEEDUP > 1.0" | bc -l 2>/dev/null || echo 0) )); then
        echo "✅ Fused MoE is faster"
    elif (( $(echo "$SPEEDUP < 1.0" | bc -l 2>/dev/null || echo 0) )); then
        echo "⚠️  Fused MoE is slower (may need tuning)"
    else
        echo "➡️  No significant difference"
    fi
else
    echo "Could not calculate speedup (missing timing data)"
fi
