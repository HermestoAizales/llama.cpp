Problem description: ops.cpp HISA implementation integration issues

Current state:
- The upstream base file (from git show upstream/master:ggml/src/ggml-cpu/ops.cpp, 11214 lines) must be restored
- HISA functions (block_pool_32, block_pool_64, gather_common, gather, block_gather, gather_mask) need to be correctly integrated.


What needs to be done next (after restart):
1. Start from clean upstream base (done via git checkout upstream/master -- ggml/src/ggml-cpu/ops.cpp or cp /tmp/base_ops.cpp).
2. Carefully extract ONLY the C code (no patch metadata) for the 6 HISA functions.
(see feature/hisa-opti branch)
3. Verify with: make -C build-linux && nm -D build-linux/bin/libggml-cpu.so | grep hisa
4. Commit and push.

Notes:
- The HISA functions are: block_pool_32, block_pool_64, gather_common, gather, block_gather, gather_mask (4bit specialized).
- They depend on GGML_TENSOR_LOCALS macros — these are fine as long as the functions are placed in the same file scope.
- The upstream file uses `src->ne[3]`, `ne[1]`, `ne[3]` etc. — no standalone `ne[]` array, use tensor->ne[] or the locals created by GGML_TENSOR_LOCALS.