#include "optimizer.h"

#include "ggml.h"
#include "ggml-backend.h"
#include "llama.h"

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
    if (t < 0) return "default";
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
    return "auto";
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

    // --- Detect model + hardware ---
    {
        struct llama_model_params mparams = {};
        mparams.use_mmap  = true;
        mparams.use_mlock = false;
        mparams.no_alloc  = true;

        llama_model * model = llama_model_load_from_file(up.model_path.c_str(), mparams);
        if (!model) {
            std::cerr << "  ERROR: Cannot load model metadata from '" << up.model_path << "'\n";
            return false;
        }

        m_n_layers      = llama_model_n_layer(model);
        m_n_ctx_train   = llama_model_n_ctx_train(model);
        m_n_gpu_devices = 0;
        m_has_gpu       = false;

        // Detect GPUs via ggml backend device enumeration
        size_t n_devices = ggml_backend_dev_count();
        for (size_t i = 0; i < n_devices; i++) {
            ggml_backend_dev_t dev = ggml_backend_dev_get(i);
            if (dev) {
                auto type = ggml_backend_dev_type(dev);
                if (type == GGML_BACKEND_DEVICE_TYPE_GPU ||
                    type == GGML_BACKEND_DEVICE_TYPE_IGPU) {
                    m_has_gpu = true;
                    m_n_gpu_devices++;
                }
            }
        }

        m_is_moe        = (llama_model_n_expert(model) > 0);
        m_n_cpu_threads = std::thread::hardware_concurrency();
        m_model_size_bytes = llama_model_n_params(model) * sizeof(float);

        llama_model_free(model);

        // Heuristic: detect if model is on slow storage by checking if it's a symlink
        // or if the path contains nfs/cifs/remote
        up.model_on_nfs = (up.model_path.find("nfs") != std::string::npos ||
                           up.model_path.find("cifs") != std::string::npos ||
                           up.model_path.find("/mnt/") != std::string::npos);
    }

    std::cout << "\n  Detected:\n";
    std::cout << "    Layers:       " << m_n_layers << "\n";
    std::cout << "    Train ctx:    " << m_n_ctx_train << "\n";
    std::cout << "    GPU devices:  " << (m_has_gpu ? std::to_string(m_n_gpu_devices) + " (found)" : "none (CPU only)") << "\n";
    std::cout << "    CPU threads:  " << m_n_cpu_threads << "\n";
    if (m_is_moe) std::cout << "    Architecture: MoE (experts detected)\n";

    // ================================================================
    // 1. Context size
    // ================================================================
    up.desired_ctx = read_int(
        "\n[?] Desired context size [" + std::to_string(m_n_ctx_train) + "]: ",
        m_n_ctx_train);
    if (up.desired_ctx <= 0) up.desired_ctx = m_n_ctx_train;
    up.desired_ctx = ((up.desired_ctx + 255) / 256) * 256;  // align 256

    // ================================================================
    // 2. Parallel requests
    // ================================================================
    up.n_parallel = read_int("[?] Parallel requests (n-parallel) [1]: ", 1);
    if (up.n_parallel <= 0) up.n_parallel = 1;

    // ================================================================
    // 3. Use-case
    // ================================================================
    std::cout << "\n  Use-case profile:\n";
    up.is_chat     = read_bool("    Interactive chat (short prompts, many turns)? [y/N]: ", false);
    up.is_batch    = read_bool("    Batch processing (long prompts, throughput)? [y/N]: ", false);
    up.is_long_context = (up.desired_ctx > 8192);
    if (up.is_long_context) {
        std::cout << "    (auto-detected: long context mode, ctx=" << up.desired_ctx << ")\n";
    }

    // ================================================================
    // 4. Optimization priority
    // ================================================================
    std::cout << "\n  Optimization priority:\n";
    std::cout << "    speed    = max TPS, may reduce quality\n";
    std::cout << "    balanced = good quality + good speed\n";
    std::cout << "    quality  = best quality, may be slower\n";
    while (true) {
        std::string p = read_line("    Choice (speed/balanced/quality) [balanced]: ");
        if (p.empty() || p == "balanced") { up.optimization_goal = optimizer_user_params::priority::balanced; break; }
        if (p == "speed")               { up.optimization_goal = optimizer_user_params::priority::speed;    break; }
        if (p == "quality")             { up.optimization_goal = optimizer_user_params::priority::quality;  break; }
        std::cout << "    Please enter speed, balanced, or quality.\n";
    }

    // ================================================================
    // 5. GPU offload
    // ================================================================
    if (m_has_gpu) {
        std::cout << "\n  GPU offload:\n";
        std::string hint = read_line(
            "    GPU layers — 'all', 'auto', or exact number [auto]: ");
        if (hint.empty() || hint == "auto") {
            up.n_gpu_layers_hint = -1;
        } else if (hint == "all") {
            up.n_gpu_layers_hint = -2;
        } else {
            try {
                up.n_gpu_layers_hint = std::stoi(hint);
                up.n_gpu_layers_hint = std::min(up.n_gpu_layers_hint, m_n_layers);
            } catch (...) {
                up.n_gpu_layers_hint = -1;
            }
        }
    }

    // ================================================================
    // 6. Storage / IO
    // ================================================================
    std::cout << "\n  Storage / IO:\n";
    if (!up.model_on_nfs) {
        up.model_on_ssd  = read_bool("    Model on local SSD/NVMe? [Y/n]: ", true);
        up.model_on_nfs  = read_bool("    Model on network filesystem (NFS/SMB)? [y/N]: ", false);
    }
    up.prefer_mmap = !up.model_on_nfs;
    if (up.model_on_nfs) {
        std::cout << "    (NFS: recommending --no-mmap for stability)\n";
    }

    // ================================================================
    // 7. KV cache quality
    // ================================================================
    std::cout << "\n  KV cache:\n";
    up.allow_quant_cache = read_bool("    Allow quantized KV cache (Q8_0) to save VRAM? [Y/n]: ", true);

    // ================================================================
    // 8. MoE offload (only for MoE models)
    // ================================================================
    if (m_is_moe) {
        std::cout << "\n  MoE offload:\n";
        std::cout << "    This is a Mixture-of-Experts model.\n";
        std::cout << "    Expert weights are large — offloading some to CPU saves VRAM\n";
        std::cout << "    but may reduce speed due to PCIe transfers.\n";
        up.allow_moe_cpu = read_bool("    Allow CPU offload for MoE experts? [Y/n]: ", true);
        if (up.allow_moe_cpu) {
            std::cout << "    The optimizer will test different n-cpu-moe values.\n";
        }
    } else {
        up.allow_moe_cpu = false;
    }

    // ================================================================
    // 8b. Fused MoE kernel (only for MoE models with GPU)
    // ================================================================
    if (m_is_moe && m_has_gpu) {
        std::cout << "\n  Fused MoE kernel:\n";
        std::cout << "    The fused MoE kernel combines gate_up + SiLU + down\n";
        std::cout << "    in a single kernel launch with async weight prefetch.\n";
        std::cout << "    This can significantly reduce kernel launch overhead.\n";
        up.allow_fused_moe = read_bool("    Test fused MoE kernel? [Y/n]: ", true);
        if (up.allow_fused_moe) {
            up.fused_moe_streams = read_int("    Prefetch streams (1-8) [2]: ", 2);
            up.fused_moe_streams = std::max(1, std::min(8, up.fused_moe_streams));
            up.moe_max_vram_mb = read_int("    VRAM budget for expert cache in MB (0=auto) [0]: ", 0);
            std::cout << "    The optimizer will test different VRAM budgets.\n";
        }
    } else {
        up.allow_fused_moe = false;
    }

    // ================================================================
    // 9. Speculative decoding
    // ================================================================
    std::cout << "\n  Speculative decoding:\n";
    std::cout << "    Speculation can significantly improve generation speed.\n";
    up.allow_speculative = read_bool("    Test speculative decoding (ngram)? [Y/n]: ", true);
    if (up.allow_speculative) {
        up.spec_ngram_size = read_int("    N-gram size (2-5) [3]: ", 3);
        up.spec_ngram_size = std::max(2, std::min(5, up.spec_ngram_size));
    }

    // ================================================================
    // 10. Flash Attention
    // ================================================================
    std::cout << "\n  Attention:\n";
    up.flash_attn = read_bool("    Use Flash Attention (recommended)? [Y/n]: ", true);

    // ================================================================
    // 11. SWA (Sliding Window Attention)
    // ================================================================
    // Only relevant for models that support SWA
    up.swa_full = false;
    if (up.is_long_context) {
        up.swa_full = read_bool("    Use full SWA cache (for long context)? [y/N]: ", false);
    }

    // ================================================================
    // 12. NUMA
    // ================================================================
    up.use_numa = false;
    if (m_n_cpu_threads >= 8) {
        up.use_numa = read_bool("    Enable NUMA optimization (multi-socket systems)? [y/N]: ", false);
    }

    // ================================================================
    // 13. Context shift
    // ================================================================
    up.ctx_shift = false;
    if (up.is_chat && up.is_long_context) {
        up.ctx_shift = read_bool("    Enable ctx-shift (for infinite chat)? [Y/n]: ", true);
    }

    // ================================================================
    // 14. Continuous batching
    // ================================================================
    up.cont_batching = true;
    if (up.n_parallel > 1) {
        up.cont_batching = read_bool("    Enable continuous batching (for parallel requests)? [Y/n]: ", true);
    }

    // ================================================================
    // 15. Dry run / quality floor
    // ================================================================
    std::cout << "\n  Quality floor:\n";
    std::string tps_str = read_line(
        "    Minimum acceptable generation TPS (0 = none) [0]: ");
    if (!tps_str.empty()) {
        try { up.min_acceptable_tps = std::stof(tps_str); } catch (...) {}
    }

    // ================================================================
    // Summary
    // ================================================================
    std::cout << "\n  ============================================================\n";
    std::cout << "  Summary\n";
    std::cout << "  ============================================================\n";
    std::cout << "    Model:          " << up.model_path << "\n";
    std::cout << "    Ctx:            " << up.desired_ctx << "\n";
    std::cout << "    Parallel:       " << up.n_parallel << "\n";
    std::cout << "    Priority:       " << priority_str(up.optimization_goal) << "\n";
    std::cout << "    GPU layers:     " << (up.n_gpu_layers_hint == -1 ? "auto" : up.n_gpu_layers_hint == -2 ? "all" : std::to_string(up.n_gpu_layers_hint)) << "\n";
    std::cout << "    mmap:           " << (up.prefer_mmap ? "yes" : "no") << "\n";
    std::cout << "    quant cache:    " << (up.allow_quant_cache ? "yes" : "no") << "\n";
    std::cout << "    MoE CPU:        " << (up.allow_moe_cpu ? "yes" : "no") << "\n";
    std::cout << "    Fused MoE:      " << (up.allow_fused_moe ? "yes" : "no") << "\n";
    if (up.allow_fused_moe) {
        std::cout << "      streams:      " << up.fused_moe_streams << "\n";
        std::cout << "      vram budget:  " << (up.moe_max_vram_mb > 0 ? std::to_string(up.moe_max_vram_mb) + " MB" : "auto") << "\n";
    }
    std::cout << "    Speculative:    " << (up.allow_speculative ? "ngram-" + std::to_string(up.spec_ngram_size) : "no") << "\n";
    std::cout << "    Flash Attn:     " << (up.flash_attn ? "yes" : "no") << "\n";
    std::cout << "    SWA full:       " << (up.swa_full ? "yes" : "no") << "\n";
    std::cout << "    NUMA:           " << (up.use_numa ? "yes" : "no") << "\n";
    std::cout << "    ctx-shift:      " << (up.ctx_shift ? "yes" : "no") << "\n";
    std::cout << "    cont-batching:  " << (up.cont_batching ? "yes" : "no") << "\n";
    std::cout << "    min TPS:        " << (up.min_acceptable_tps > 0 ? std::to_string((int)up.min_acceptable_tps) : "none") << "\n";
    std::cout << "\n";

    return read_bool("  Proceed with benchmarks? [Y/n]: ", true);
}

// ---------------------------------------------------------------------------
// Config generation
// ---------------------------------------------------------------------------

std::vector<optimizer_config> optimizer::generate_configs(const optimizer_user_params & up) const {
    std::vector<optimizer_config> configs;

    // Baseline
    optimizer_config base;
    base.n_gpu_layers    = up.n_gpu_layers_hint;
    base.split_mode      = 1;   // layer
    base.cache_type_k    = -1;  // f16
    base.cache_type_v    = -1;
    base.n_batch         = 2048;
    base.n_ubatch        = 512;
    base.n_threads       = 0;
    base.n_threads_batch = 0;
    base.n_parallel      = up.n_parallel;
    base.pipeline_partial = false;
    base.use_mmap        = up.prefer_mmap;
    base.use_direct_io   = !up.prefer_mmap;
    base.use_mlock       = false;
    base.offload_kqv     = (up.n_gpu_layers_hint > 0 || up.n_gpu_layers_hint == -2);
    base.op_offload      = true;
    base.no_extra_bufts  = false;
    base.fit_target_mib  = 1024;
    base.n_cpu_moe       = -1;  // don't override
    base.spec_type       = "";
    base.spec_ngram_size = 0;

    // ================================================================
    // Phase 1: GPU layer sweep
    // ================================================================
    std::vector<int> ngl_values;
    if (m_has_gpu && up.n_gpu_layers_hint != 0) {
        if (up.n_gpu_layers_hint >= 0) {
            ngl_values = {up.n_gpu_layers_hint, 0, m_n_layers};
        } else {
            ngl_values = {0, m_n_layers / 4, m_n_layers / 2, (3 * m_n_layers) / 4, m_n_layers};
            std::sort(ngl_values.begin(), ngl_values.end());
            ngl_values.erase(std::unique(ngl_values.begin(), ngl_values.end()), ngl_values.end());
        }
    } else {
        ngl_values = {0};
    }

    // ================================================================
    // Phase 2: KV cache types
    // ================================================================
    std::vector<std::pair<int,int>> cache_pairs;
    if (up.allow_quant_cache) {
        if (up.optimization_goal == optimizer_user_params::priority::speed) {
            cache_pairs = {{GGML_TYPE_Q8_0, GGML_TYPE_Q8_0},
                           {GGML_TYPE_F16,   GGML_TYPE_F16}};
        } else {
            cache_pairs = {{GGML_TYPE_F16,   GGML_TYPE_F16},
                           {GGML_TYPE_Q8_0,   GGML_TYPE_Q8_0}};
        }
    } else {
        cache_pairs = {{GGML_TYPE_F16, GGML_TYPE_F16}};
    }

    // ================================================================
    // Phase 3: Pipeline partial
    // ================================================================
    std::vector<bool> pp_values = {false};
    if (m_has_gpu) {
        for (int ngl : ngl_values) {
            if (ngl > 0 && ngl < m_n_layers) {
                pp_values = {false, true};
                break;
            }
        }
    }

    // ================================================================
    // Phase 4: IO variants
    // ================================================================
    struct io_var { bool mmap; bool direct; const char * label; };
    std::vector<io_var> io_values;
    if (up.prefer_mmap) {
        io_values = {{true, false, "mmap"}};
    } else {
        io_values = {{false, true, "direct"}};
    }

    // ================================================================
    // Phase 5: Batch sizes
    // ================================================================
    std::vector<std::pair<int,int>> batch_pairs;
    if (up.is_batch) {
        batch_pairs = {{4096, 1024}, {2048, 512}, {8192, 2048}};
    } else {
        batch_pairs = {{2048, 512}};
    }

    // ================================================================
    // Phase 6: MoE CPU offload (only for MoE models)
    // ================================================================
    // n_cpu_moe: -1 = don't override, 0 = all experts on CPU, N = first N layers
    // Phase A uses coarse values; Phase B refines around the best with step=1.
    std::vector<int> moe_values;
    if (m_is_moe && up.allow_moe_cpu && m_has_gpu) {
        // Coarse sweep: all on GPU (0), quarter, half, three-quarter, all on CPU
        int q1 = m_n_layers / 4;
        int q3 = (3 * m_n_layers) / 4;
        moe_values = {-1, 0, q1, m_n_layers / 2, q3, m_n_layers};
        std::sort(moe_values.begin(), moe_values.end());
        moe_values.erase(std::unique(moe_values.begin(), moe_values.end()), moe_values.end());
    } else {
        moe_values = {-1};  // don't override
    }

    // ================================================================
    // Phase 7: Speculative decoding
    // ================================================================
    std::vector<std::string> spec_values;
    if (up.allow_speculative) {
        spec_values = {"", "ngram"};  // "" = no spec, "ngram" = n-gram speculation
    } else {
        spec_values = {""};
    }

    // ================================================================
    // Build cartesian product
    // To keep total configs manageable, we use a two-phase approach:
    // Phase A: Main sweep (ngl × cache × pp × io × batch × moe × spec)
    // Phase B: For the best Phase A result, test fine-grained variants
    //
    // Fused MoE is handled specially: we first find the best base config,
    // then test fused-moe variants on top of it (Phase C).
    // ================================================================

    for (int ngl : ngl_values) {
        for (auto [ctk, ctv] : cache_pairs) {
            for (bool pp : pp_values) {
                for (auto & io : io_values) {
                    for (auto [nb, nub] : batch_pairs) {
                        for (int moe : moe_values) {
                            for (auto & spec : spec_values) {
                                optimizer_config cfg = base;
                                cfg.n_gpu_layers     = ngl;
                                cfg.cache_type_k     = ctk;
                                cfg.cache_type_v     = ctv;
                                cfg.pipeline_partial = pp;
                                cfg.use_mmap         = io.mmap;
                                cfg.use_direct_io    = io.direct;
                                cfg.n_batch          = nb;
                                cfg.n_ubatch         = nub;
                                cfg.offload_kqv      = (ngl > 0);
                                cfg.n_cpu_moe        = moe;
                                cfg.spec_type        = spec;
                                cfg.spec_ngram_size  = (spec == "ngram") ? up.spec_ngram_size : 0;
                                // Fused MoE defaults (don't override in Phase A)
                                cfg.fused_moe        = -1;
                                cfg.moe_prefetch_streams = 0;
                                cfg.moe_max_vram_mb  = 0;

                                // Build compact label
                                cfg.label = "ngl=" + std::to_string(ngl)
                                          + " k=" + cache_type_name(ctk)
                                          + " v=" + cache_type_name(ctv);
                                if (pp)                          cfg.label += " pp=yes";
                                cfg.label += " io=" + std::string(io.label);
                                if (nb != 2048)                  cfg.label += " b=" + std::to_string(nb);
                                if (moe >= 0)                    cfg.label += " moe_cpu=" + std::to_string(moe);
                                if (spec == "ngram")             cfg.label += " spec=ngram" + std::to_string(up.spec_ngram_size);

                                configs.push_back(cfg);
                            }
                        }
                    }
                }
            }
        }
    }

    // Cap at 40 configs to keep runtime reasonable
    // If too many, prune: keep only most promising combinations
    if ((int)configs.size() > 40) {
        // Priority-based pruning: keep diverse ngl values, reduce other dims
        std::vector<optimizer_config> pruned;
        // Always keep ngl=0 and ngl=all as baselines
        for (auto & c : configs) {
            if (c.n_gpu_layers == 0 || c.n_gpu_layers == m_n_layers) {
                pruned.push_back(c);
            }
        }
        // Fill remaining slots with diverse configs
        for (auto & c : configs) {
            if ((int)pruned.size() >= 40) break;
            if (c.n_gpu_layers != 0 && c.n_gpu_layers != m_n_layers) {
                pruned.push_back(c);
            }
        }
        configs = pruned;
    }

    m_benchmark_tokens = up.is_batch ? 20 : 30;

    return configs;
}

// ---------------------------------------------------------------------------
// Single benchmark
// ---------------------------------------------------------------------------

static void bench_logger(ggml_log_level level, const char * text, void *) {
    if (level >= GGML_LOG_LEVEL_WARN) fprintf(stderr, "%s", text);
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
    struct llama_model_params mparams = {};
    mparams.n_gpu_layers  = cfg.n_gpu_layers;
    mparams.use_mmap      = cfg.use_mmap;
    mparams.use_direct_io = cfg.use_direct_io;
    mparams.use_mlock     = cfg.use_mlock;
    mparams.split_mode    = static_cast<llama_split_mode>(cfg.split_mode < 0 ? 1 : cfg.split_mode);

    // MoE CPU offload via tensor_buft_override
    std::vector<llama_model_tensor_buft_override> moe_overrides;
    if (cfg.n_cpu_moe >= 0) {
        // Build override: first cfg.n_cpu_moe layers' MoE weights → CPU
        // Pattern: blk.<il>.ffn_(up|down|gate_up|gate)_(ch|)exps
        // For simplicity, we use the global MoE CPU override if n_cpu_moe >= m_n_layers
        if (cfg.n_cpu_moe >= m_n_layers) {
            // All MoE on CPU — use the built-in override
            static const std::string pattern_moe = "blk\\.\\d+\\.ffn_(up|down|gate_up|gate)_(ch|)exps";
            moe_overrides.push_back({pattern_moe.c_str(), ggml_backend_cpu_buffer_type()});
            moe_overrides.push_back({nullptr, nullptr});
            mparams.tensor_buft_overrides = moe_overrides.data();
        }
        // For partial MoE offload (n_cpu_moe < m_n_layers but > 0),
        // we'd need per-layer overrides which is complex.
        // We test all-on-CPU vs all-on-GPU as the two extremes.
    }

    llama_model * model = llama_model_load_from_file(up.model_path.c_str(), mparams);
    if (!model) {
        result.error = "model load failed (OOM?)";
        ggml_log_set(prev_logger, nullptr);
        return result;
    }

    // --- Build context params ---
    struct llama_context_params cparams = {};
    cparams.n_ctx             = up.desired_ctx;
    cparams.n_batch           = cfg.n_batch;
    cparams.n_ubatch          = cfg.n_ubatch;
    cparams.n_seq_max         = cfg.n_parallel;
    cparams.type_k            = static_cast<ggml_type>(cfg.cache_type_k < 0 ? GGML_TYPE_F16 : cfg.cache_type_k);
    cparams.type_v            = static_cast<ggml_type>(cfg.cache_type_v < 0 ? GGML_TYPE_F16 : cfg.cache_type_v);
    cparams.offload_kqv       = cfg.offload_kqv;
    cparams.op_offload        = cfg.op_offload;
    cparams.swa_full          = up.swa_full;

    // Fused MoE parameters
    if (cfg.fused_moe >= 0) {
        cparams.fused_moe = (cfg.fused_moe == 1);
    }
    if (cfg.moe_prefetch_streams > 0) {
        cparams.moe_prefetch_streams = cfg.moe_prefetch_streams;
    }
    if (cfg.moe_max_vram_mb > 0) {
        cparams.moe_max_vram_mb = cfg.moe_max_vram_mb;
    }

    llama_context * ctx = llama_init_from_model(model, cparams);
    if (!ctx) {
        result.error = "context creation failed (OOM)";
        llama_model_free(model);
        ggml_log_set(prev_logger, nullptr);
        return result;
    }

    // --- Warmup ---
    {
        const llama_vocab * vocab = llama_model_get_vocab(model);
        llama_token bos = llama_vocab_bos(vocab);
        if (bos < 0) bos = 1;
        llama_token tokens[6] = {bos, 1, 2, 3, 4, 5};
        int n_w = std::min(6, m_warmup_tokens);
        for (int i = 0; i < n_w; i++) {
            llama_batch batch = llama_batch_get_one(&tokens[i], 1);
            if (llama_decode(ctx, batch) != 0) { n_w = i; break; }
        }
    }

    // --- Prompt benchmark (batch mode) ---
    if (up.is_batch && m_prompt_bench_tokens > 6) {
        int remaining = m_prompt_bench_tokens - 6;
        std::vector<llama_token> dummy(remaining, 1);
        auto t0 = std::chrono::steady_clock::now();
        int offset = 0;
        while (offset < remaining) {
            int chunk = std::min(remaining - offset, (int)cparams.n_batch);
            llama_batch batch = llama_batch_get_one(&dummy[offset], chunk);
            if (llama_decode(ctx, batch) != 0) break;
            offset += chunk;
        }
        auto t1 = std::chrono::steady_clock::now();
        float ms = std::chrono::duration<float, std::milli>(t1 - t0).count();
        if (offset > 0 && ms > 0) result.prompt_tps = (offset * 1000.0f) / ms;
    }

    // --- Generation benchmark ---
    const int n_gen = m_benchmark_tokens;
    float total_ms = 0.0f;
    int n_ok = 0;

    for (int i = 0; i < n_gen; i++) {
        auto t0 = std::chrono::steady_clock::now();
        llama_token token = 1;
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
        result.error = "all gen steps failed";
    }

    llama_free(ctx);
    llama_model_free(model);
    ggml_log_set(prev_logger, nullptr);
    return result;
}

// ---------------------------------------------------------------------------
// Run all benchmarks
// ---------------------------------------------------------------------------

std::vector<optimizer_result> optimizer::run_benchmarks(const optimizer_user_params & up) const {
    // ================================================================
    // Phase A: Coarse sweep
    // ================================================================
    auto configs_a = generate_configs(up);
    std::vector<optimizer_result> results;

    std::cout << "\n  Phase A: Coarse sweep (" << configs_a.size() << " configs)...\n";
    std::cout << "  (Each: load model + warmup + " << m_benchmark_tokens << " gen tokens)\n\n";

    int idx = 0;
    for (const auto & cfg : configs_a) {
        idx++;
        std::cout << "  [A" << std::setw(2) << idx << "/" << configs_a.size() << "] "
                  << std::left << std::setw(56) << cfg.label << " ... " << std::flush;

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

    // ================================================================
    // Phase B: Fine-grained MoE refinement
    // Find the best Phase A result that used n_cpu_moe, then test
    // moe_best ± range in step=1 to catch steep cliffs.
    // ================================================================
    if (m_is_moe && up.allow_moe_cpu && m_has_gpu) {
        // Find best Phase A result with n_cpu_moe set
        int best_moe_val = -1;
        float best_moe_tps = -1.0f;
        optimizer_config best_moe_cfg;

        for (const auto & r : results) {
            if (!r.success) continue;
            if (r.config.n_cpu_moe < 0) continue;  // skip don't-override
            if (r.gen_tps > best_moe_tps) {
                best_moe_tps = r.gen_tps;
                best_moe_val = r.config.n_cpu_moe;
                best_moe_cfg = r.config;
            }
        }

        if (best_moe_val >= 0) {
            int refine_range = 5;
            std::vector<int> refine_values;
            for (int delta = -refine_range; delta <= refine_range; ++delta) {
                int v = best_moe_val + delta;
                if (v < 0 || v > m_n_layers) continue;
                // Skip if already tested in Phase A
                bool already_tested = false;
                for (const auto & r : results) {
                    if (r.config.n_cpu_moe == v) { already_tested = true; break; }
                }
                if (!already_tested) {
                    refine_values.push_back(v);
                }
            }

            if (!refine_values.empty()) {
                std::cout << "\n  Phase B: MoE refinement around n_cpu_moe=" << best_moe_val
                          << " (" << refine_values.size() << " configs)...\n\n";

                // Use the best config as base, only vary n_cpu_moe
                for (int moe : refine_values) {
                    optimizer_config cfg = best_moe_cfg;
                    cfg.n_cpu_moe = moe;
                    cfg.label = "refine moe=" + std::to_string(moe);

                    std::cout << "  [B ] " << std::left << std::setw(56) << cfg.label << " ... " << std::flush;

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
            }
        }
    }

    // ================================================================
    // Phase C: Fused MoE variants on the overall best
    // ================================================================
    if (m_is_moe && up.allow_fused_moe && m_has_gpu) {
        // Find overall best result so far
        optimizer_result overall_best;
        overall_best.success = false;
        overall_best.gen_tps = -1.0f;
        for (const auto & r : results) {
            if (r.success && r.gen_tps > overall_best.gen_tps) {
                overall_best = r;
            }
        }

        if (overall_best.success) {
            std::vector<optimizer_config> fm_configs;
            int fm_streams = up.fused_moe_streams;
            std::vector<int> fm_vram_values = {0, 256, 512, 1024};
            std::sort(fm_vram_values.begin(), fm_vram_values.end());
            fm_vram_values.erase(std::unique(fm_vram_values.begin(), fm_vram_values.end()), fm_vram_values.end());

            for (int fm : {0, 1}) {
                for (int vram : fm_vram_values) {
                    optimizer_config cfg = overall_best.config;
                    cfg.fused_moe = fm;
                    cfg.moe_prefetch_streams = fm_streams;
                    cfg.moe_max_vram_mb = vram;
                    cfg.label = "fm=" + std::to_string(fm)
                              + " streams=" + std::to_string(fm_streams)
                              + " vram=" + std::to_string(vram) + "M";
                    fm_configs.push_back(cfg);
                }
            }

            std::cout << "\n  Phase C: Fused MoE variants (" << fm_configs.size() << " configs)...\n\n";

            for (const auto & cfg : fm_configs) {
                std::cout << "  [C ] " << std::left << std::setw(56) << cfg.label << " ... " << std::flush;

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
        if (up.optimization_goal == optimizer_user_params::priority::quality) {
            if (r.config.cache_type_k == GGML_TYPE_Q8_0 ||
                r.config.cache_type_v == GGML_TYPE_Q8_0) continue;
        }
        if (up.min_acceptable_tps > 0 && r.gen_tps < up.min_acceptable_tps) continue;

        float score = r.gen_tps;
        if (up.is_batch && r.prompt_tps > 0.0f) {
            score = r.gen_tps * 0.4f + r.prompt_tps * 0.006f;
        }
        if (score > best.gen_tps) best = r;
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
              << std::setw(56) << "Configuration"
              << std::right
              << std::setw(8)  << "Gen/s"
              << std::setw(10) << "Prompt/s"
              << std::setw(6)  << "Stat"
              << "\n";
    std::cout << "  " << std::string(80, '-') << "\n";
    for (const auto & r : results) {
        std::cout << "  " << std::left << std::setw(56) << r.config.label;
        if (r.success) {
            std::cout << std::right << std::fixed << std::setprecision(1) << std::setw(8) << r.gen_tps;
            if (r.prompt_tps > 0.0f)
                std::cout << std::setprecision(0) << std::setw(10) << r.prompt_tps;
            else
                std::cout << std::setw(10) << "—";
            std::cout << std::setw(6) << "OK";
        } else {
            std::cout << std::setw(8) << "—" << std::setw(10) << "—" << std::left << std::setw(6) << "FAIL";
        }
        std::cout << "\n";
    }
    std::cout << "\n";
}

// ---------------------------------------------------------------------------
// Print report
// ---------------------------------------------------------------------------

void optimizer::print_report(const optimizer_result & best,
                             const optimizer_user_params & up) const {
    if (!best.success) { std::cout << "  No successful results.\n"; return; }
    const auto & c = best.config;

    std::cout << "\n";
    std::cout << "  ╔══════════════════════════════════════════════════════════╗\n";
    std::cout << "  ║           OPTIMAL RUNTIME PARAMETERS                    ║\n";
    std::cout << "  ╚══════════════════════════════════════════════════════════╝\n\n";
    std::cout << "  Measured:   " << std::fixed << std::setprecision(1) << best.gen_tps << " tokens/sec generation";
    if (best.prompt_tps > 0.0f) std::cout << ", " << std::setprecision(0) << best.prompt_tps << " tokens/sec prompt";
    std::cout << "\n\n";

    // --- llama-server ---
    std::cout << "  ┌─────────────────────────────────────────────────────────┐\n";
    std::cout << "  │  llama-server                                           │\n";
    std::cout << "  └─────────────────────────────────────────────────────────┘\n\n";
    std::cout << "    llama-server \\\n";
    std::cout << "      --model \"" << up.model_path << "\" \\\n";
    std::cout << "      --ctx-size " << up.desired_ctx << " \\\n";
    if (c.n_gpu_layers >= 0)      std::cout << "      --n-gpu-layers " << c.n_gpu_layers << " \\\n";
    else if (c.n_gpu_layers == -2) std::cout << "      --n-gpu-layers all \\\n";
    else                           std::cout << "      --n-gpu-layers auto \\\n";
    std::cout << "      --split-mode " << split_mode_name(c.split_mode) << " \\\n";
    if (c.cache_type_k >= 0)      std::cout << "      --cache-type-k " << cache_type_name(c.cache_type_k) << " \\\n";
    if (c.cache_type_v >= 0)      std::cout << "      --cache-type-v " << cache_type_name(c.cache_type_v) << " \\\n";
    std::cout << "      --batch-size " << c.n_batch << " \\\n";
    std::cout << "      --ubatch-size " << c.n_ubatch << " \\\n";
    if (c.n_parallel > 1)         std::cout << "      --parallel " << c.n_parallel << " \\\n";
    if (c.pipeline_partial)        std::cout << "      --pipeline-partial 1 \\\n";
    if (c.offload_kqv)            std::cout << "      --offload-kqv \\\n";
    else                           std::cout << "      --no-kv-offload \\\n";
    if (!c.use_mmap)              std::cout << "      --no-mmap \\\n";
    if (c.use_direct_io)          std::cout << "      --direct-io \\\n";
    if (c.use_mlock)              std::cout << "      --mlock \\\n";
    if (c.n_cpu_moe >= m_n_layers) std::cout << "      --cpu-moe \\\n";
    if (c.fused_moe == 1)         std::cout << "      --fused-moe on \\\n";
    else if (c.fused_moe == 0)    std::cout << "      --fused-moe off \\\n";
    if (c.moe_prefetch_streams > 0) std::cout << "      --moe-prefetch-streams " << c.moe_prefetch_streams << " \\\n";
    if (c.moe_max_vram_mb > 0)    std::cout << "      --moe-max-vram " << c.moe_max_vram_mb << " \\\n";
    if (c.spec_type == "ngram")   std::cout << "      --spec-ngram " << c.spec_ngram_size << " \\\n";
    if (!up.flash_attn)           std::cout << "      --flash-attn off \\\n";
    if (up.swa_full)              std::cout << "      --swa-full \\\n";
    if (up.use_numa)              std::cout << "      --numa distribute \\\n";
    if (up.ctx_shift)             std::cout << "      --ctx-shift \\\n";
    if (!up.cont_batching)        std::cout << "      --no-cont-batching \\\n";
    if (c.fit_target_mib != 1024) std::cout << "      --fit-target " << c.fit_target_mib << " \\\n";
    std::cout << "      --host 0.0.0.0 --port 8080\n\n";

    // --- llama-cli ---
    std::cout << "  ┌─────────────────────────────────────────────────────────┐\n";
    std::cout << "  │  llama-cli                                              │\n";
    std::cout << "  └─────────────────────────────────────────────────────────┘\n\n";
    std::cout << "    llama-cli \\\n";
    std::cout << "      --model \"" << up.model_path << "\" \\\n";
    std::cout << "      --ctx-size " << up.desired_ctx << " \\\n";
    if (c.n_gpu_layers >= 0)      std::cout << "      --n-gpu-layers " << c.n_gpu_layers << " \\\n";
    else if (c.n_gpu_layers == -2) std::cout << "      --n-gpu-layers all \\\n";
    else                           std::cout << "      --n-gpu-layers auto \\\n";
    if (c.cache_type_k >= 0)      std::cout << "      --cache-type-k " << cache_type_name(c.cache_type_k) << " \\\n";
    if (c.cache_type_v >= 0)      std::cout << "      --cache-type-v " << cache_type_name(c.cache_type_v) << " \\\n";
    std::cout << "      --batch-size " << c.n_batch << " \\\n";
    std::cout << "      --ubatch-size " << c.n_ubatch << " \\\n";
    if (c.pipeline_partial)        std::cout << "      --pipeline-partial 1 \\\n";
    if (!c.offload_kqv)           std::cout << "      --no-kv-offload \\\n";
    if (!c.use_mmap)              std::cout << "      --no-mmap \\\n";
    if (c.use_direct_io)          std::cout << "      --direct-io \\\n";
    if (c.n_cpu_moe >= m_n_layers) std::cout << "      --cpu-moe \\\n";
    if (c.fused_moe == 1)         std::cout << "      --fused-moe on \\\n";
    else if (c.fused_moe == 0)    std::cout << "      --fused-moe off \\\n";
    if (c.moe_prefetch_streams > 0) std::cout << "      --moe-prefetch-streams " << c.moe_prefetch_streams << " \\\n";
    if (c.moe_max_vram_mb > 0)    std::cout << "      --moe-max-vram " << c.moe_max_vram_mb << " \\\n";
    if (c.spec_type == "ngram")   std::cout << "      --spec-ngram " << c.spec_ngram_size << " \\\n";
    if (!up.flash_attn)           std::cout << "      --flash-attn off \\\n";
    std::cout << "      -p \"Your prompt here\"\n\n";

    // --- Flat flags ---
    std::cout << "  ┌─────────────────────────────────────────────────────────┐\n";
    std::cout << "  │  Flat flag list (append to any llama.* command)         │\n";
    std::cout << "  └─────────────────────────────────────────────────────────┘\n\n";
    std::cout << "    --ctx-size " << up.desired_ctx;
    if (c.n_gpu_layers >= 0)      std::cout << " --n-gpu-layers " << c.n_gpu_layers;
    else if (c.n_gpu_layers == -2) std::cout << " --n-gpu-layers all";
    else                           std::cout << " --n-gpu-layers auto";
    std::cout << " --split-mode " << split_mode_name(c.split_mode);
    if (c.cache_type_k >= 0)      std::cout << " --cache-type-k " << cache_type_name(c.cache_type_k);
    if (c.cache_type_v >= 0)      std::cout << " --cache-type-v " << cache_type_name(c.cache_type_v);
    std::cout << " --batch-size " << c.n_batch << " --ubatch-size " << c.n_ubatch;
    if (c.n_parallel > 1)         std::cout << " --parallel " << c.n_parallel;
    if (c.pipeline_partial)        std::cout << " --pipeline-partial 1";
    if (!c.offload_kqv)           std::cout << " --no-kv-offload";
    if (!c.use_mmap)              std::cout << " --no-mmap";
    if (c.use_direct_io)          std::cout << " --direct-io";
    if (c.use_mlock)              std::cout << " --mlock";
    if (c.n_cpu_moe >= m_n_layers) std::cout << " --cpu-moe";
    if (c.fused_moe == 1)         std::cout << " --fused-moe on";
    else if (c.fused_moe == 0)    std::cout << " --fused-moe off";
    if (c.moe_prefetch_streams > 0) std::cout << " --moe-prefetch-streams " << c.moe_prefetch_streams;
    if (c.moe_max_vram_mb > 0)    std::cout << " --moe-max-vram " << c.moe_max_vram_mb;
    if (c.spec_type == "ngram")   std::cout << " --spec-ngram " << c.spec_ngram_size;
    if (!up.flash_attn)           std::cout << " --flash-attn off";
    if (up.swa_full)              std::cout << " --swa-full";
    if (up.use_numa)              std::cout << " --numa distribute";
    if (up.ctx_shift)             std::cout << " --ctx-shift";
    if (!up.cont_batching)        std::cout << " --no-cont-batching";
    if (c.fit_target_mib != 1024) std::cout << " --fit-target " << c.fit_target_mib;
    std::cout << "\n\n";

    // --- Notes ---
    std::cout << "  ┌─────────────────────────────────────────────────────────┐\n";
    std::cout << "  │  Notes                                                  │\n";
    std::cout << "  └─────────────────────────────────────────────────────────┘\n\n";
    if (c.n_gpu_layers == 0)
        std::cout << "  • Running fully on CPU. Consider a GPU for better perf.\n";
    else if (c.n_gpu_layers < m_n_layers)
        std::cout << "  • Partial offload: " << c.n_gpu_layers << "/" << m_n_layers << " layers on GPU.\n";
    else
        std::cout << "  • Full GPU offload: all " << m_n_layers << " layers.\n";
    if (c.pipeline_partial)
        std::cout << "  • Pipeline parallelism enabled — overlaps GPU with CPU compute.\n";
    if (c.cache_type_k == GGML_TYPE_Q8_0)
        std::cout << "  • KV cache quantized to Q8_0 — saves ~50% VRAM.\n";
    if (!c.use_mmap)
        std::cout << "  • mmap disabled — using direct file reads.\n";
    if (c.n_cpu_moe >= m_n_layers)
        std::cout << "  • All MoE experts on CPU — saves significant VRAM.\n";
    else if (c.n_cpu_moe > 0)
        std::cout << "  • First " << c.n_cpu_moe << " MoE layers on CPU.\n";
    if (c.fused_moe == 1)
        std::cout << "  • Fused MoE kernel enabled — combines gate_up + SiLU + down.\n";
    if (c.moe_max_vram_mb > 0)
        std::cout << "  • MoE expert cache VRAM budget: " << c.moe_max_vram_mb << " MB.\n";
    if (c.spec_type == "ngram")
        std::cout << "  • N-gram speculative decoding (size=" << c.spec_ngram_size << ") — may boost TPS.\n";
    if (up.is_long_context)
        std::cout << "  • Long context: consider --cache-type-k q8_0 to reduce VRAM.\n";
    if (m_is_moe && c.n_gpu_layers > 0 && c.n_gpu_layers < m_n_layers)
        std::cout << "  • MoE model: expert weights on CPU may bottleneck. Try --n-gpu-layers all.\n";
    if (up.use_numa)
        std::cout << "  • NUMA enabled — beneficial for multi-socket systems.\n";
    std::cout << "\n";
}
