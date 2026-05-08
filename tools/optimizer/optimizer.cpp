#include "optimizer.h"

#include "ggml.h"

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <algorithm>
#include <chrono>
#include <iomanip>
#include <iostream>
#include <numeric>
#include <string>
#include <vector>
#include <thread>

// ---------------------------------------------------------------------------
// helpers
// ---------------------------------------------------------------------------

static std::string trim(const std::string & s) {
    size_t a = s.find_first_not_of(" \t\r\n");
    if (a == std::string::npos) return "";
    size_t b = s.find_last_not_of(" \t\r\n");
    return s.substr(a, b - a + 1);
}

static std::string read_line(const std::string & prompt) {
    std::cout << prompt;
    std::string line;
    if (!std::getline(std::cin, line)) return "";
    return trim(line);
}

static int read_int(const std::string & prompt, int def) {
    std::string line = read_line(prompt);
    if (line.empty()) return def;
    try { return std::stoi(line); } catch (...) { return def; }
}

static bool read_bool(const std::string & prompt, bool def) {
    std::string line = read_line(prompt);
    if (line.empty()) return def;
    auto l = line;
    std::transform(l.begin(), l.end(), l.begin(), ::tolower);
    if (l == "y" || l == "yes" || l == "true" || l == "1") return true;
    if (l == "n" || l == "no" || l == "false" || l == "0") return false;
    return def;
}

static const char * cache_type_name(int t) {
    return ggml_type_name(static_cast<ggml_type>(t));
}

static std::string priority_str(optimizer_user_params::priority p) {
    switch (p) {
        case optimizer_user_params::priority::speed:    return "speed";
        case optimizer_user_params::priority::balanced: return "balanced";
        case optimizer_user_params::priority::quality:  return "quality";
    }
    return "unknown";
}

static const char * split_mode_name(int m) {
    switch (m) {
        case 0: return "none";
        case 1: return "layer";
        case 2: return "row";
        case 3: return "tensor";
    }
    return "unknown";
}

// ---------------------------------------------------------------------------
// optimizer implementation
// ---------------------------------------------------------------------------

optimizer::optimizer()  = default;
optimizer::~optimizer() = default;

// ---------------------------------------------------------------------------
// Interactive setup
// ---------------------------------------------------------------------------

bool optimizer::interactive_setup(optimizer_user_params & up) {
    std::cout << "\n";
    std::cout << "============================================================\n";
    std::cout << "  llama.cpp Runtime Optimizer\n";
    std::cout << "  Benchmark-driven parameter tuning for maximum TPS\n";
    std::cout << "============================================================\n";
    std::cout << "\n";

    // --- Model path ---
    while (up.model_path.empty()) {
        up.model_path = read_line("[?] Model path (.gguf): ");
        if (up.model_path.empty()) {
            std::cout << "  Please enter a valid model path.\n";
        }
    }

    // --- Detect model info ---
    {
        struct llama_model_params mparams = llama_model_default_params();
        mparams.use_mmap  = true;
        mparams.use_mlock = false;
        mparams.no_alloc  = true;

        llama_model * model = llama_model_load_from_file(up.model_path.c_str(), mparams);
        if (!model) {
            std::cerr << "  ERROR: Cannot load model metadata from '" << up.model_path << "'\n";
            return false;
        }

        m_n_layers       = llama_model_n_layer(model);
        m_n_ctx_train    = llama_model_n_ctx_train(model);
        m_n_gpu_devices  = 0;
        m_has_gpu        = false;

        int n_devices = llama_model_n_devices(model);
        for (int i = 0; i < n_devices; i++) {
            ggml_backend_dev_t dev = llama_model_get_device(model, i);
            if (dev) {
                auto type = ggml_backend_dev_type(dev);
                if (type == GGML_BACKEND_DEVICE_TYPE_GPU ||
                    type == GGML_BACKEND_DEVICE_TYPE_IGPU) {
                    m_has_gpu = true;
                    m_n_gpu_devices++;
                }
            }
        }

        m_is_moe = (llama_model_n_expert(model) > 0);
        m_n_cpu_threads = std::thread::hardware_concurrency();

        // Estimate model file size
        // GGUF header doesn't store this directly; use a rough heuristic
        m_model_size_bytes = llama_model_n_params(model) * sizeof(float); // rough

        llama_model_free(model);
    }

    std::cout << "\n  Detected:\n";
    std::cout << "    Layers:       " << m_n_layers << "\n";
    std::cout << "    Train ctx:    " << m_n_ctx_train << "\n";
    std::cout << "    GPU devices:  " << (m_has_gpu ? std::to_string(m_n_gpu_devices) + " (found)" : "none (CPU only)") << "\n";
    std::cout << "    CPU threads:  " << m_n_cpu_threads << "\n";
    if (m_is_moe) std::cout << "    Architecture: MoE\n";

    // --- Context size ---
    up.desired_ctx = read_int(
        "[?] Desired context size [" + std::to_string(m_n_ctx_train) + "]: ",
        m_n_ctx_train);
    if (up.desired_ctx <= 0) up.desired_ctx = m_n_ctx_train;
    // Round up to nearest 256 for CUDA alignment
    up.desired_ctx = ((up.desired_ctx + 255) / 256) * 256;

    // --- Parallel requests ---
    up.n_parallel = read_int("[?] Parallel requests (n-parallel) [1]: ", 1);
    if (up.n_parallel <= 0) up.n_parallel = 1;

    // --- Use-case ---
    std::cout << "\n  Use-case:\n";
    up.is_chat         = read_bool("    Chat (short prompts, many turns)? [y/N]: ", false);
    up.is_batch        = read_bool("    Batch processing (long prompts, throughput)? [y/N]: ", false);
    up.is_long_context = (up.desired_ctx > 8192);
    if (up.is_long_context) {
        std::cout << "    (auto-detected: long context mode, ctx > 8K)\n";
    }

    // --- Optimization priority ---
    std::cout << "\n  Optimization priority:\n";
    std::cout << "    speed    = maximum TPS, may reduce quality\n";
    std::cout << "    balanced = good quality, good speed (default)\n";
    std::cout << "    quality  = best possible quality, may be slower\n";
    while (true) {
        std::string p = read_line("    Priority (speed/balanced/quality) [balanced]: ");
        if (p.empty() || p == "balanced") { up.optimization_goal = optimizer_user_params::priority::balanced; break; }
        if (p == "speed")               { up.optimization_goal = optimizer_user_params::priority::speed;    break; }
        if (p == "quality")             { up.optimization_goal = optimizer_user_params::priority::quality;  break; }
        std::cout << "    Please enter speed, balanced, or quality.\n";
    }

    // --- GPU layer hint ---
    if (m_has_gpu) {
        std::cout << "\n  GPU offload:\n";
        gpu_layers_again:
        std::string hint = read_line(
            "    GPU layers — 'all', 'auto', or exact number [auto]: ");
        if (hint.empty() || hint == "auto") {
            up.n_gpu_layers_hint = -1;
        } else if (hint == "all") {
            up.n_gpu_layers_hint = -2; // means "all" in llama.cpp
        } else {
            try {
                int v = std::stoi(hint);
                if (v < 0 || v > m_n_layers + 1) {
                    std::cout << "    Please enter 0.." << (m_n_layers + 1) << ", 'auto', or 'all'.\n";
                    goto gpu_layers_again;
                }
                up.n_gpu_layers_hint = v;
            } catch (...) {
                std::cout << "    Invalid number.\n";
                goto gpu_layers_again;
            }
        }
    } else {
        up.n_gpu_layers_hint = 0;
    }

    // --- IO hints ---
    std::cout << "\n  Storage / IO:\n";
    up.model_on_ssd  = read_bool("    Model on SSD/NVMe? [Y/n]: ", true);
    up.model_on_nfs  = read_bool("    Model on network filesystem (NFS/SMB)? [y/N]: ", false);
    if (!up.model_on_nfs) {
        up.prefer_mmap = read_bool("    Use mmap (recommended for SSD)? [Y/n]: ", true);
    } else {
        std::cout << "    (NFS: recommending direct IO, no mmap)\n";
        up.prefer_mmap = false;
    }

    // --- Quality floor ---
    std::cout << "\n  Quality:\n";
    up.allow_quant_cache = read_bool("    Allow quantized KV cache (Q8_0) for speed? [Y/n]: ", true);
    std::string tps_str = read_line(
        "    Minimum acceptable generation TPS (0 = no minimum) [0]: ");
    if (!tps_str.empty()) {
        try { up.min_acceptable_tps = std::stof(tps_str); } catch (...) {}
    }

    // --- Summary ---
    std::cout << "\n  ============================================================\n";
    std::cout << "  Summary\n";
    std::cout << "  ============================================================\n";
    std::cout << "    Model:        " << up.model_path << "\n";
    std::cout << "    Ctx:          " << up.desired_ctx << "\n";
    std::cout << "    Parallel:     " << up.n_parallel << "\n";
    std::cout << "    Priority:     " << priority_str(up.optimization_goal) << "\n";
    std::cout << "    GPU layers:   " << (up.n_gpu_layers_hint == -1 ? "auto" : up.n_gpu_layers_hint == -2 ? "all" : std::to_string(up.n_gpu_layers_hint)) << "\n";
    std::cout << "    mmap:         " << (up.prefer_mmap ? "yes" : "no") << "\n";
    std::cout << "    quant cache:  " << (up.allow_quant_cache ? "yes" : "no") << "\n";
    std::cout << "    min TPS:      " << (up.min_acceptable_tps > 0 ? std::to_string((int)up.min_acceptable_tps) : "none") << "\n";
    std::cout << "\n";

    return read_bool("  Proceed with benchmarks? [Y/n]: ", true);
}

// ---------------------------------------------------------------------------
// Config generation — the intelligence lives here
// ---------------------------------------------------------------------------

std::vector<optimizer_config> optimizer::generate_configs(const optimizer_user_params & up) const {
    std::vector<optimizer_config> configs;

    // Baseline defaults
    optimizer_config base;
    base.n_gpu_layers     = up.n_gpu_layers_hint;
    base.split_mode       = 1;  // layer
    base.cache_type_k     = -1; // f16 default
    base.cache_type_v     = -1;
    base.n_batch          = 2048;
    base.n_ubatch         = 512;
    base.n_threads        = 0;  // auto
    base.n_threads_batch  = 0;
    base.n_parallel       = up.n_parallel;
    base.pipeline_partial = false;
    base.use_mmap         = up.prefer_mmap;
    base.use_direct_io    = !up.prefer_mmap && !up.model_on_ssd;
    base.use_mlock        = false;
    base.offload_kqv      = (up.n_gpu_layers_hint > 0 || up.n_gpu_layers_hint == -2);
    base.op_offload       = true;
    base.no_extra_bufts   = false;
    base.fit_target_mib   = 1024;  // 1 GiB default headroom

    // ============================================================
    // Phase 1: Sweep GPU layer counts (only if GPU available)
    // ============================================================
    std::vector<int> ngl_values;
    if (m_has_gpu && !up.force_cpu) {
        if (up.n_gpu_layers_hint >= 0) {
            ngl_values.push_back(up.n_gpu_layers_hint);
            // Also test all and 0 for comparison
            ngl_values.push_back(0);
            ngl_values.push_back(m_n_layers);
        } else {
            // Auto sweep: test 0, quarter, half, three-quarter, all
            ngl_values = {0,
                          m_n_layers / 4,
                          m_n_layers / 2,
                          (3 * m_n_layers) / 4,
                          m_n_layers + 1};  // +1 to include output layer
            // Deduplicate and clamp
            std::sort(ngl_values.begin(), ngl_values.end());
            ngl_values.erase(std::unique(ngl_values.begin(), ngl_values.end()), ngl_values.end();
            for (auto & v : ngl_values) v = std::min(v, m_n_layers);
        }
    } else {
        ngl_values = {0};
    }

    // ============================================================
    // Phase 2: Sweep KV cache types
    // ============================================================
    std::vector<std::pair<int,int>> cache_type_pairs;
    if (up.allow_quant_cache) {
        if (up.optimization_goal == optimizer_user_params::priority::speed) {
            cache_type_pairs = {{GGML_TYPE_Q8_0, GGML_TYPE_Q8_0},
                                {GGML_TYPE_F16,   GGML_TYPE_F16}};
        } else {
            cache_type_pairs = {{GGML_TYPE_F16,   GGML_TYPE_F16},
                                {GGML_TYPE_Q8_0,   GGML_TYPE_Q8_0}};
        }
    } else {
        cache_type_pairs = {{GGML_TYPE_F16, GGML_TYPE_F16}};
    }

    // ============================================================
    // Phase 3: Pipeline partial (only for partial GPU offload)
    // ============================================================
    std::vector<bool> pp_values = {false};
    if (m_has_gpu && !up.force_cpu) {
        // Only interesting if some layers are offloaded
        for (int ngl : ngl_values) {
            if (ngl > 0 && ngl < m_n_layers) {
                pp_values = {false, true};
                break;
            }
        }
    }

    // ============================================================
    // Phase 4: IO variants (only test for best config later, or a subset)
    // ============================================================
    struct io_variant {
        bool mmap;
        bool direct_io;
        const char * label;
    };
    std::vector<io_variant> io_variants;
    if (up.prefer_mmap) {
        io_variants = {{true, false, "mmap"}};
        if (!up.model_on_ssd) {
            io_variants.push_back({true, true, "mmap+direct"});
        }
    } else {
        io_variants = {{false, true,  "direct"}};
    }

    // ============================================================
    // Phase 5: Batch size variants (mainly for batch processing)
    // ============================================================
    std::vector<std::pair<int,int>> batch_pairs;
    if (up.is_batch) {
        batch_pairs = {{4096, 1024},
                       {2048, 512},
                       {8192, 2048}};
    } else {
        batch_pairs = {{2048, 512}};
    }

    // ============================================================
    // Build full cartesian product (but cap total configs)
    // ============================================================
    int total = (int)(ngl_values.size() * cache_type_pairs.size() * pp_values.size() *
                      io_variants.size() * batch_pairs.size());

    for (int ngl : ngl_values) {
        for (auto [ctk, ctv] : cache_type_pairs) {
            for (bool pp : pp_values) {
                for (auto & io : io_variants) {
                    for (auto [nb, nub] : batch_pairs) {
                        optimizer_config cfg = base;
                        cfg.n_gpu_layers     = ngl;
                        cfg.cache_type_k     = ctk;
                        cfg.cache_type_v     = ctv;
                        cfg.pipeline_partial = pp;
                        cfg.use_mmap         = io.mmap;
                        cfg.use_direct_io    = io.direct_io;
                        cfg.n_batch          = nb;
                        cfg.n_ubatch         = nub;
                        cfg.offload_kqv      = (ngl > 0);

                        // Build compact label
                        cfg.label = "ngl=" + std::to_string(ngl)
                                  + " k=" + cache_type_name(ctk)
                                  + " v=" + cache_type_name(ctv);
                        if (pp)          cfg.label += " pp=yes";
                        cfg.label += " io=" + std::string(io.label);
                        if (nb != 2048)  cfg.label += " b=" + std::to_string(nb);

                        configs.push_back(cfg);
                    }
                }
            }
        }
    }

    // Store total for progress display
    m_benchmark_tokens = up.is_batch ? 20 : 30;

    return configs;
}

// ---------------------------------------------------------------------------
// Single benchmark
// ---------------------------------------------------------------------------

// Minimal logger for benchmark runs
static void bench_logger(ggml_log_level level, const char * text, void *) {
    if (level >= GGML_LOG_LEVEL_WARN) {
        fprintf(stderr, "%s", text);
    }
}

optimizer_result optimizer::benchmark_single(const optimizer_config & cfg,
                                            const optimizer_user_params & up) const {
    optimizer_result result;
    result.config   = cfg;
    result.success  = false;
    result.gen_tps  = 0.0f;
    result.prompt_tps = 0.0f;

    auto prev_logger = ggml_log_set(bench_logger, nullptr);

    // --- Build model params ---
    struct llama_model_params mparams = llama_model_default_params();
    mparams.n_gpu_layers  = cfg.n_gpu_layers;
    mparams.use_mmap      = cfg.use_mmap;
    mparams.use_direct_io = cfg.use_direct_io;
    mparams.use_mlock     = cfg.use_mlock;
    mparams.split_mode    = static_cast<llama_split_mode>(cfg.split_mode < 0 ? 1 : cfg.split_mode);

    if (cfg.no_extra_bufts) {
        mparams.tensor_buft_overrides = nullptr; // minimal
    }

    llama_model * model = llama_model_load_from_file(up.model_path.c_str(), mparams);
    if (!model) {
        result.error = "model load failed (OOM or file error)";
        ggml_log_set(prev_logger, nullptr);
        return result;
    }

    // --- Build context params ---
    struct llama_context_params cparams = llama_context_default_params();
    cparams.n_ctx             = up.desired_ctx;
    cparams.n_batch           = cfg.n_batch;
    cparams.n_ubatch          = cfg.n_ubatch;
    cparams.n_seq_max         = cfg.n_parallel;
    cparams.type_k            = static_cast<ggml_type>(cfg.cache_type_k < 0 ? GGML_TYPE_F16 : cfg.cache_type_k);
    cparams.type_v            = static_cast<ggml_type>(cfg.cache_type_v < 0 ? GGML_TYPE_F16 : cfg.cache_type_v);
    cparams.pipeline_partial  = cfg.pipeline_partial;
    cparams.offload_kqv       = cfg.offload_kqv;
    cparams.op_offload        = cfg.op_offload;
    cparams.pipeline_parallel = false; // managed by scheduler internally

    llama_context * ctx = llama_init_from_model(model, cparams);
    if (!ctx) {
        result.error = "context creation failed (OOM)";
        llama_model_free(model);
        ggml_log_set(prev_logger, nullptr);
        return result;
    }

    // --- Warmup: feed a small prompt ---
    {
        llama_token bos = llama_model_bos_token(model);
        if (bos < 0) bos = 1;
        llama_token tokens[6] = {bos, 1, 2, 3, 4, 5};
        int n_warmup = std::min(6, m_warmup_tokens);

        for (int i = 0; i < n_warmup; i++) {
            llama_batch batch = llama_batch_get_one(&tokens[i], 1);
            if (llama_decode(ctx, batch) != 0) {
                // shrink warmup
                n_warmup = i;
                break;
            }
        }
    }

    // --- Prompt benchmark (optional, for batch-heavy workloads) ---
    if (up.is_batch && m_prompt_bench_tokens > n_warmup) {
        int remaining = m_prompt_bench_tokens - n_warmup;
        // Generate dummy tokens for prompt bench
        std::vector<llama_token> dummy(remaining, 1);
        auto t0 = std::chrono::steady_clock::now();

        // Feed in batches
        int offset = 0;
        while (offset < remaining) {
            int chunk = std::min(remaining - offset, (int)cparams.n_batch);
            llama_batch batch = llama_batch_get_one(&dummy[offset], chunk);
            if (llama_decode(ctx, batch) != 0) break;
            offset += chunk;
        }

        auto t1 = std::chrono::steady_clock::now();
        float ms = std::chrono::duration<float, std::milli>(t1 - t0).count();
        if (offset > 0 && ms > 0) {
            result.prompt_tps = (offset * 1000.0f) / ms;
        }
    }

    // --- Generation benchmark ---
    const int n_gen = m_benchmark_tokens;
    float total_ms = 0.0f;
    int   n_ok = 0;

    for (int i = 0; i < n_gen; i++) {
        auto t0 = std::chrono::steady_clock::now();

        llama_token token = 1; // dummy
        llama_batch batch = llama_batch_get_one(&token, 1);
        if (llama_decode(ctx, batch) != 0) break;

        auto t1 = std::chrono::steady_clock::now();
        total_ms += std::chrono::duration<float, std::milli>(t1 - t0).count();
        n_ok++;
    }

    if (n_ok > 0) {
        result.success = true;
        result.gen_tps = (n_ok * 1000.0f) / total_ms;
    } else {
        result.error = "all generation steps failed";
    }

    // --- Cleanup ---
    llama_free(ctx);
    llama_model_free(model);
    ggml_log_set(prev_logger, nullptr);

    return result;
}

// ---------------------------------------------------------------------------
// Run all benchmarks
// ---------------------------------------------------------------------------

std::vector<optimizer_result> optimizer::run_benchmarks(const optimizer_user_params & up) const {
    auto configs = generate_configs(up);
    std::vector<optimizer_result> results;

    std::cout << "\n  Running " << configs.size() << " configurations...\n";
    std::cout << "  (Each: load model + warmup + " << m_benchmark_tokens << " gen tokens)\n\n";

    int idx = 0;
    for (const auto & cfg : configs) {
        idx++;
        std::cout << "  [" << std::setw(2) << idx << "/" << configs.size() << "] "
                  << std::left << std::setw(52) << cfg.label << " ... " << std::flush;

        auto r = benchmark_single(cfg, up);
        results.push_back(r);

        if (r.success) {
            std::cout << std::fixed << std::setprecision(1) << std::setw(6) << r.gen_tps << " t/s gen";
            if (r.prompt_tps > 0.0f) {
                std::cout << "  " << std::setprecision(0) << r.prompt_tps << " t/s prompt";
            }
            std::cout << "\n";
        } else {
            std::cout << "FAIL  " << r.error << "\n";
        }
    }

    return results;
}

// ---------------------------------------------------------------------------
// Select best
// ---------------------------------------------------------------------------

optimizer_result optimizer::select_best(const std::vector<optimizer_result> & results,
                                       const optimizer_user_params & up) const {
    optimizer_result best;
    best.success = false;
    best.gen_tps = -1.0f;

    for (const auto & r : results) {
        if (!r.success) continue;

        // Quality: skip quantized cache
        if (up.optimization_goal == optimizer_user_params::priority::quality) {
            if (r.config.cache_type_k == GGML_TYPE_Q8_0 ||
                r.config.cache_type_v == GGML_TYPE_Q8_0) continue;
        }

        // Minimum TPS floor
        if (up.min_acceptable_tps > 0 && r.gen_tps < up.min_acceptable_tps) continue;

        // Score: for batch workloads, weight prompt speed too
        float score = r.gen_tps;
        if (up.is_batch && r.prompt_tps > 0.0f) {
            score = r.gen_tps * 0.4f + r.prompt_tps * 0.006f; // normalize prompt
        }

        if (score > best.gen_tps) {
            best = r;
        }
    }

    return best;
}

// ---------------------------------------------------------------------------
// Print comparison table
// ---------------------------------------------------------------------------

void optimizer::print_comparison_table(const std::vector<optimizer_result> & results) const {
    std::cout << "\n  ============================================================\n";
    std::cout << "  Benchmark Results\n";
    std::cout << "  ============================================================\n\n";

    std::cout << "  " << std::left
              << std::setw(52) << "Configuration"
              << std::right
              << std::setw(8)  << "Gen t/s"
              << std::setw(10) << "Prompt/s"
              << std::setw(6)  << "Stat"
              << "\n";
    std::cout << "  " << std::string(76, '-') << "\n";

    for (const auto & r : results) {
        std::cout << "  " << std::left << std::setw(52) << r.config.label;
        if (r.success) {
            std::cout << std::right << std::fixed << std::setprecision(1)
                      << std::setw(8) << r.gen_tps;
            if (r.prompt_tps > 0.0f) {
                std::cout << std::setprecision(0) << std::setw(10) << r.prompt_tps;
            } else {
                std::cout << std::setw(10) << "—";
            }
            std::cout << std::setw(6) << "OK";
        } else {
            std::cout << std::setw(8) << "—"
                      << std::setw(10) << "—"
                      << std::left << std::setw(6) << "FAIL";
        }
        std::cout << "\n";
    }
    std::cout << "\n";
}

// ---------------------------------------------------------------------------
// Print report — the main output, copy-paste ready
// ---------------------------------------------------------------------------

void optimizer::print_report(const optimizer_result & best,
                             const optimizer_user_params & up) const {
    if (!best.success) {
        std::cout << "  No successful benchmark results to report.\n";
        return;
    }

    const auto & c = best.config;

    std::cout << "\n";
    std::cout << "  ╔══════════════════════════════════════════════════════════╗\n";
    std::cout << "  ║           OPTIMAL RUNTIME PARAMETERS                    ║\n";
    std::cout << "  ╚══════════════════════════════════════════════════════════╝\n\n";

    std::cout << "  Measured:   " << std::fixed << std::setprecision(1) << best.gen_tps << " tokens/sec generation";
    if (best.prompt_tps > 0.0f) {
        std::cout << ", " << std::setprecision(0) << best.prompt_tps << " tokens/sec prompt";
    }
    std::cout << "\n\n";

    // --- llama-server command ---
    std::cout << "  ┌─────────────────────────────────────────────────────────┐\n";
    std::cout << "  │  llama-server                                           │\n";
    std::cout << "  └─────────────────────────────────────────────────────────┘\n\n";
    std::cout << "    llama-server \\\n";
    std::cout << "      --model \"" << up.model_path << "\" \\\n";
    std::cout << "      --ctx-size " << up.desired_ctx << " \\\n";
    if (c.n_gpu_layers >= 0)
        std::cout << "      --n-gpu-layers " << c.n_gpu_layers << " \\\n";
    else if (c.n_gpu_layers == -2)
        std::cout << "      --n-gpu-layers all \\\n";
    else
        std::cout << "      --n-gpu-layers auto \\\n";
    std::cout << "      --split-mode " << split_mode_name(c.split_mode) << " \\\n";
    if (c.cache_type_k >= 0)
        std::cout << "      --cache-type-k " << cache_type_name(c.cache_type_k) << " \\\n";
    if (c.cache_type_v >= 0)
        std::cout << "      --cache-type-v " << cache_type_name(c.cache_type_v) << " \\\n";
    std::cout << "      --batch-size " << c.n_batch << " \\\n";
    std::cout << "      --ubatch-size " << c.n_ubatch << " \\\n";
    if (c.n_parallel > 1)
        std::cout << "      --parallel " << c.n_parallel << " \\\n";
    if (c.pipeline_partial)
        std::cout << "      --pipeline-partial 1 \\\n";
    if (c.offload_kqv)
        std::cout << "      --offload-kqv \\\n";
    else
        std::cout << "      --no-kv-offload \\\n";
    if (!c.use_mmap)
        std::cout << "      --no-mmap \\\n";
    if (c.use_direct_io)
        std::cout << "      --direct-io \\\n";
    if (c.use_mlock)
        std::cout << "      --mlock \\\n";
    if (c.no_extra_bufts)
        std::cout << "      --no-repack \\\n";
    if (c.fit_target_mib != 1024)
        std::cout << "      --fit-target " << c.fit_target_mib << " \\\n";
    std::cout << "      --host 0.0.0.0 --port 8080\n\n";

    // --- llama-cli command ---
    std::cout << "  ┌─────────────────────────────────────────────────────────┐\n";
    std::cout << "  │  llama-cli                                              │\n";
    std::cout << "  └─────────────────────────────────────────────────────────┘\n\n";
    std::cout << "    llama-cli \\\n";
    std::cout << "      --model \"" << up.model_path << "\" \\\n";
    std::cout << "      --ctx-size " << up.desired_ctx << " \\\n";
    if (c.n_gpu_layers >= 0)
        std::cout << "      --n-gpu-layers " << c.n_gpu_layers << " \\\n";
    else if (c.n_gpu_layers == -2)
        std::cout << "      --n-gpu-layers all \\\n";
    else
        std::cout << "      --n-gpu-layers auto \\\n";
    if (c.cache_type_k >= 0)
        std::cout << "      --cache-type-k " << cache_type_name(c.cache_type_k) << " \\\n";
    if (c.cache_type_v >= 0)
        std::cout << "      --cache-type-v " << cache_type_name(c.cache_type_v) << " \\\n";
    std::cout << "      --batch-size " << c.n_batch << " \\\n";
    std::cout << "      --ubatch-size " << c.n_ubatch << " \\\n";
    if (c.pipeline_partial)
        std::cout << "      --pipeline-partial 1 \\\n";
    if (!c.offload_kqv)
        std::cout << "      --no-kv-offload \\\n";
    if (!c.use_mmap)
        std::cout << "      --no-mmap \\\n";
    if (c.use_direct_io)
        std::cout << "      --direct-io \\\n";
    std::cout << "      -p \"Your prompt here\"\n\n";

    // --- Flat flag list for copy-paste ---
    std::cout << "  ┌─────────────────────────────────────────────────────────┐\n";
    std::cout << "  │  Flat flag list (append to any llama.* command)         │\n";
    std::cout << "  └─────────────────────────────────────────────────────────┘\n\n";
    std::cout << "    --ctx-size " << up.desired_ctx;
    if (c.n_gpu_layers >= 0)
        std::cout << " --n-gpu-layers " << c.n_gpu_layers;
    else if (c.n_gpu_layers == -2)
        std::cout << " --n-gpu-layers all";
    else
        std::cout << " --n-gpu-layers auto";
    std::cout << " --split-mode " << split_mode_name(c.split_mode);
    if (c.cache_type_k >= 0)
        std::cout << " --cache-type-k " << cache_type_name(c.cache_type_k);
    if (c.cache_type_v >= 0)
        std::cout << " --cache-type-v " << cache_type_name(c.cache_type_v);
    std::cout << " --batch-size " << c.n_batch;
    std::cout << " --ubatch-size " << c.n_ubatch;
    if (c.n_parallel > 1)
        std::cout << " --parallel " << c.n_parallel;
    if (c.pipeline_partial)
        std::cout << " --pipeline-partial 1";
    if (!c.offload_kqv)
        std::cout << " --no-kv-offload";
    if (!c.use_mmap)
        std::cout << " --no-mmap";
    if (c.use_direct_io)
        std::cout << " --direct-io";
    if (c.use_mlock)
        std::cout << " --mlock";
    if (c.no_extra_bufts)
        std::cout << " --no-repack";
    if (c.fit_target_mib != 1024)
        std::cout << " --fit-target " << c.fit_target_mib;
    std::cout << "\n\n";

    // --- Notes ---
    std::cout << "  ┌─────────────────────────────────────────────────────────┐\n";
    std::cout << "  │  Notes                                                  │\n";
    std::cout << "  └─────────────────────────────────────────────────────────┘\n\n";
    if (c.n_gpu_layers == 0) {
        std::cout << "  • Running fully on CPU. Consider a GPU for better perf.\n";
    } else if (c.n_gpu_layers < m_n_layers) {
        std::cout << "  • Partial offload: " << c.n_gpu_layers << "/" << m_n_layers << " layers on GPU.\n";
        if (c.pipeline_partial) {
            std::cout << "  • Pipeline parallelism enabled — overlaps GPU with CPU.\n";
        }
    } else {
        std::cout << "  • Full GPU offload: all " << m_n_layers << " layers.\n";
    }
    if (c.cache_type_k == GGML_TYPE_Q8_0) {
        std::cout << "  • KV cache quantized to Q8_0 — saves ~50% VRAM.\n";
    }
    if (!c.use_mmap) {
        std::cout << "  • mmap disabled — uses direct file reads.\n";
    }
    if (up.is_long_context) {
        std::cout << "  • Long context: consider --cache-type-k q8_0 to reduce VRAM.\n";
    }
    if (up.is_moe && c.n_gpu_layers > 0 && c.n_gpu_layers < m_n_layers) {
        std::cout << "  • MoE model: expert weights on CPU may bottleneck. Try --n-gpu-layers all.\n";
    }
    std::cout << "\n";
}
