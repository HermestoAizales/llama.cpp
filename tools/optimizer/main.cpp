#include "optimizer.h"
#include "preset.h"

#include "common/common.h"
#include "common/log.h"

#include "llama.h"

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <vector>
#include <iostream>

static void print_usage(const char * argv0) {
    fprintf(stderr, "Usage: %s --model <gguf_path> [options]\n", argv0);
    fprintf(stderr, "\n");
    fprintf(stderr, "Options:\n");
    fprintf(stderr, "  --model <path>        Path to GGUF model (required)\n");
    fprintf(stderr, "  --ctx-size <n>        Desired context size (default: model train ctx)\n");
    fprintf(stderr, "  --parallel <n>        Number of parallel requests (default: 1)\n");
    fprintf(stderr, "  --priority <p>        Optimization priority: speed|balanced|quality (default: balanced)\n");
    fprintf(stderr, "  --save-preset <path>  Also save best config as INI preset file (optional)\n");
    fprintf(stderr, "  --non-interactive     Skip prompts, use defaults\n");
    fprintf(stderr, "  --help                Show this help\n");
    fprintf(stderr, "\n");
    fprintf(stderr, "Output: prints a copy-paste-ready report of optimal runtime parameters.\n");
    fprintf(stderr, "\n");
    fprintf(stderr, "Example:\n");
    fprintf(stderr, "  %s --model model.gguf --priority speed\n", argv0);
}

int main(int argc, char ** argv) {
    std::string model_path;
    int ctx_size = 0;
    int n_parallel = 0;
    std::string priority_input;
    std::string save_preset_path;
    bool non_interactive = false;

    for (int i = 1; i < argc; i++) {
        std::string arg = argv[i];
        if (arg == "--model" && i + 1 < argc) {
            model_path = argv[++i];
        } else if (arg == "--ctx-size" && i + 1 < argc) {
            ctx_size = std::stoi(argv[++i]);
        } else if (arg == "--parallel" && i + 1 < argc) {
            n_parallel = std::stoi(argv[++i]);
        } else if (arg == "--priority" && i + 1 < argc) {
            priority_input = argv[++i];
        } else if (arg == "--save-preset" && i + 1 < argc) {
            save_preset_path = argv[++i];
        } else if (arg == "--non-interactive") {
            non_interactive = true;
        } else if (arg == "--help" || arg == "-h") {
            print_usage(argv[0]);
            return 0;
        } else {
            fprintf(stderr, "Unknown option: %s\n", arg.c_str());
            print_usage(argv[0]);
            return 1;
        }
    }

    if (model_path.empty()) {
        fprintf(stderr, "error: --model is required\n");
        print_usage(argv[0]);
        return 1;
    }

    // Suppress excessive llama.cpp logging during benchmarks
    llama_log_set([](ggml_log_level level, const char * text, void *) {
        if (level >= GGML_LOG_LEVEL_WARN) fprintf(stderr, "%s", text);
    }, nullptr);

    optimizer opt;
    optimizer_user_params up;

    if (non_interactive) {
        // Auto-detect model info
        struct llama_model_params mparams = llama_model_default_params();
        mparams.use_mmap  = false;  // no_alloc + mmap triggers assert on backends with buffer_from_host_ptr
        mparams.no_alloc = true;
        llama_model * model = llama_model_load_from_file(model_path.c_str(), mparams);
        if (!model) {
            fprintf(stderr, "error: cannot load model metadata\n");
            return 1;
        }
        int n_ctx_train = llama_model_n_ctx_train(model);
        llama_model_free(model);

        up.model_path      = model_path;
        up.desired_ctx     = (ctx_size > 0) ? ctx_size : n_ctx_train;
        up.n_parallel      = (n_parallel > 0) ? n_parallel : 1;
        up.optimization_goal = optimizer_user_params::priority::balanced;
        if (priority_input == "speed")    up.optimization_goal = optimizer_user_params::priority::speed;
        if (priority_input == "quality")  up.optimization_goal = optimizer_user_params::priority::quality;
        up.allow_quant_cache = true;
        up.prefer_mmap     = true;
        up.model_on_ssd    = true;
        up.allow_moe_cpu   = true;
        up.allow_speculative = true;
        up.spec_ngram_size = 3;
        up.flash_attn      = true;
        up.swa_full        = false;
        up.use_numa        = false;
        up.ctx_shift       = false;
        up.cont_batching   = true;
    } else {
        if (!opt.interactive_setup(up)) {
            std::cout << "  Aborted.\n";
            return 0;
        }
        // Override with CLI args if provided
        if (ctx_size > 0)    up.desired_ctx = ctx_size;
        if (n_parallel > 0)  up.n_parallel = n_parallel;
        if (priority_input == "speed")    up.optimization_goal = optimizer_user_params::priority::speed;
        else if (priority_input == "quality") up.optimization_goal = optimizer_user_params::priority::quality;
    }

    // Run benchmarks
    auto results = opt.run_benchmarks(up);

    // Print comparison table
    opt.print_comparison_table(results);

    // Select best
    auto best = opt.select_best(results, up);

    if (!best.success) {
        std::cout << "  ERROR: No configuration succeeded. Try reducing --ctx-size.\n";
        return 1;
    }

    // Print the main report (copy-paste ready)
    opt.print_report(best, up);

    // Optionally save preset
    if (!save_preset_path.empty()) {
        preset_params pp;
        pp.name             = "optimizer-" + std::to_string(up.desired_ctx);
        pp.n_ctx            = up.desired_ctx;
        pp.n_gpu_layers     = best.config.n_gpu_layers;
        pp.cache_type_k     = (best.config.cache_type_k >= 0) ? ggml_type_name(static_cast<ggml_type>(best.config.cache_type_k)) : "";
        pp.cache_type_v     = (best.config.cache_type_v >= 0) ? ggml_type_name(static_cast<ggml_type>(best.config.cache_type_v)) : "";
        pp.n_batch          = best.config.n_batch;
        pp.n_ubatch         = best.config.n_ubatch;
        pp.n_parallel       = up.n_parallel;
        pp.use_mmap_set     = true;
        pp.use_mmap         = best.config.use_mmap;
        pp.use_direct_io    = best.config.use_direct_io;
        pp.use_mlock        = best.config.use_mlock;
        pp.no_kv_offload    = !best.config.offload_kqv;
        pp.no_op_offload    = !best.config.op_offload;
        pp.no_extra_bufts   = best.config.no_extra_bufts;

        if (preset_save(save_preset_path, pp)) {
            std::cout << "  Preset saved to: " << save_preset_path << "\n\n";
        } else {
            std::cerr << "  WARNING: Failed to save preset to " << save_preset_path << "\n\n";
        }
    }

    return 0;
}
