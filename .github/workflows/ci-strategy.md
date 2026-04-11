# CI/CD Strategy for llama.cpp Fork

To avoid wasteful full-system builds during iterative debugging, the CI has been shifted from "Push-Triggered" to "On-Demand".

## 1. Triggering Builds
Do NOT rely on automatic builds during push. Use the GitHub Actions UI:
- Go to **Actions** -> **Fork CI** -> **Run workflow**.
- Select only the target platform needed:
  - `build_mac`: Build for Mac mini M2 (Metal)
  - `build_win`: Build for Windows CUDA/CPU
  - `build_linux`: Build for Ubuntu ARM64

## 2. Build Targets
- **macOS**: Produces `llama-bin-macos-arm64` (Includes `llama-cli`, `llama-perplexity`).
- **Windows**: Produces `llama-bin-win-cpu-x64` and `ggml-cuda-win-x64-cuda13.1`.
- **Linux**: Produces `llama-bin-ubuntu-arm64`.

## 3. Full Release
Full release builds for all systems are only performed on `master` or specific release tags.

## 4. Workflow for Debugging
1. Local edit on Linux.
2. Push to feature branch (no build triggered).
3. Manually trigger `Fork CI` with `build_mac=true`.
4. Download artifact and test on Mac.
