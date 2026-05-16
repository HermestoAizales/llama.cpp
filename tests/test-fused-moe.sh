#!/bin/bash
# Test script for Fused MoE correctness validation
# Compares output with --fused-moe=off vs --fused-moe=on
#
# Usage: ./tests/test-fused-moe.sh <model_path>
# Example: ./tests/test-fused-moe.sh /models/qwen36-35b-a3b.gguf

set -euo pipefail

MODEL="${1:?Usage: $0 <model_path>}"
PROMPT="The capital of France is"
N_TOKENS=32
LLAMA_BIN="./build/bin/llama-cli"

if [ ! -f "$LLAMA_BIN" ]; then
    echo "ERROR: $LLAMA_BIN not found. Build first with: cmake --build build"
    exit 1
fi

if [ ! -f "$MODEL" ]; then
    echo "ERROR: Model file not found: $MODEL"
    exit 1
fi

echo "=== Fused MoE Correctness Test ==="
echo "Model: $MODEL"
echo "Prompt: $PROMPT"
echo "Tokens: $N_TOKENS"
echo ""

# Run without fused MoE (baseline)
echo "--- Running without fused MoE (baseline) ---"
OUTPUT_BASELINE=$(mktemp)
$LLAMA_BIN -m "$MODEL" -p "$PROMPT" -n "$N_TOKENS" --fused-moe off -ngl 99 --temp 0 --seed 42 2>/dev/null | tee "$OUTPUT_BASELINE"
echo ""

# Run with fused MoE
echo "--- Running with fused MoE ---"
OUTPUT_FUSED=$(mktemp)
$LLAMA_BIN -m "$MODEL" -p "$PROMPT" -n "$N_TOKENS" --fused-moe on -ngl 99 --temp 0 --seed 42 2>/dev/null | tee "$OUTPUT_FUSED"
echo ""

# Compare outputs
echo "--- Comparing outputs ---"
if diff -q "$OUTPUT_BASELINE" "$OUTPUT_FUSED" > /dev/null 2>&1; then
    echo "✅ PASS: Outputs are identical"
    RESULT=0
else
    echo "❌ FAIL: Outputs differ"
    echo ""
    echo "Diff:"
    diff "$OUTPUT_BASELINE" "$OUTPUT_FUSED" || true
    RESULT=1
fi

# Cleanup
rm -f "$OUTPUT_BASELINE" "$OUTPUT_FUSED"

exit $RESULT
