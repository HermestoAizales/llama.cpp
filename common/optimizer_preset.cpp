#include "optimizer_preset.h"

#include "common.h"

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <sstream>

static std::string trim(const std::string & s) {
    size_t a = s.find_first_not_of(" \t\r\n");
    if (a == std::string::npos) return "";
    size_t b = s.find_last_not_of(" \t\r\n");
    return s.substr(a, b - a + 1);
}

static ggml_type cache_type_from_str(const std::string & s) {
    for (int t = GGML_TYPE_F16; t < GGML_TYPE_COUNT; ++t) {
        if (s == ggml_type_name(static_cast<ggml_type>(t)))
            return static_cast<ggml_type>(t);
    }
    return GGML_TYPE_COUNT;
}

static int to_int(const std::string & s, int def) {
    try { return std::stoi(s); } catch (...) { return def; }
}

static bool to_bool(const std::string & s) {
    auto l = s;
    std::transform(l.begin(), l.end(), l.begin(), ::tolower);
    return (l == "1" || l == "true" || l == "yes" || l == "on");
}

bool optimizer_preset_load(const std::string & path, optimizer_preset_params & out) {
    std::ifstream f(path);
    if (!f.is_open()) {
        fprintf(stderr, "preset: cannot open '%s'\n", path.c_str());
        return false;
    }

    std::string line, section;
    while (std::getline(f, line)) {
        line = trim(line);
        if (line.empty() || line[0] == ';' || line[0] == '#') continue;
        if (line.front() == '[' && line.back() == ']') {
            section = trim(line.substr(1, line.size() - 2));
            continue;
        }
        if (section != "preset") continue;

        auto eq = line.find('=');
        if (eq == std::string::npos) continue;

        std::string key   = trim(line.substr(0, eq));
        std::string value = trim(line.substr(eq + 1));

        if      (key == "name")              out.name             = value;
        else if (key == "n_ctx")             out.n_ctx            = to_int(value, 0);
        else if (key == "n_gpu_layers")      out.n_gpu_layers     = to_int(value, -999);
        else if (key == "split_mode")        out.split_mode       = to_int(value, -1);
        else if (key == "cache_type_k")      out.cache_type_k     = value;
        else if (key == "cache_type_v")      out.cache_type_v     = value;
        else if (key == "n_batch")           out.n_batch          = to_int(value, 0);
        else if (key == "n_ubatch")          out.n_ubatch         = to_int(value, 0);
        else if (key == "n_threads")         out.n_threads        = to_int(value, 0);
        else if (key == "n_threads_batch")   out.n_threads_batch  = to_int(value, 0);
        else if (key == "n_parallel")        out.n_parallel       = to_int(value, 0);
        else if (key == "mmap") {
            out.use_mmap_set = true;
            out.use_mmap     = to_bool(value);
        }
        else if (key == "direct_io")         out.use_direct_io    = to_bool(value);
        else if (key == "mlock")             out.use_mlock        = to_bool(value);
        else if (key == "no_kv_offload")     out.no_kv_offload    = to_bool(value);
        else if (key == "no_op_offload")     out.no_op_offload    = to_bool(value);
        else if (key == "no_repack")         out.no_extra_bufts   = to_bool(value);
        else if (key == "fit_target_mib")    out.fit_target_mib   = to_int(value, 0);
        else if (key == "n_cpu_moe")         out.n_cpu_moe        = to_int(value, -1);
        else if (key == "fused_moe")         out.fused_moe        = to_int(value, -1);
        else if (key == "moe_prefetch_streams") out.moe_prefetch_streams = to_int(value, 0);
        else if (key == "moe_max_vram_mb")   out.moe_max_vram_mb  = to_int(value, 0);
        else if (key == "spec_type")         out.spec_type        = value;
        else if (key == "spec_ngram_size")   out.spec_ngram_size  = to_int(value, 0);
        else if (key == "flash_attn") {      out.flash_attn_set = true; out.flash_attn = to_bool(value); }
        else if (key == "swa_full")           out.swa_full         = to_bool(value);
        else if (key == "numa")              out.use_numa         = to_bool(value);
        else if (key == "ctx_shift")         out.ctx_shift        = to_bool(value);
        else if (key == "cont_batching")     out.cont_batching    = to_bool(value);
    }
    return true;
}

bool optimizer_preset_save(const std::string & path, const optimizer_preset_params & in) {
    std::ofstream f(path);
    if (!f.is_open()) {
        fprintf(stderr, "preset: cannot write '%s'\n", path.c_str());
        return false;
    }

    f << "# llama.cpp optimizer preset\n";
    f << "# Usage: llama-server --model model.gguf --model-preset thisfile.ini\n\n";
    f << "[preset]\n";

    auto w_str = [&](const char * k, const std::string & v) { if (!v.empty()) f << k << " = " << v << "\n"; };
    auto w_int = [&](const char * k, int v, int sentinel) { if (v != sentinel) f << k << " = " << v << "\n"; };
    auto w_bool = [&](const char * k, bool v) { if (v) f << k << " = true\n"; };

    w_str("name", in.name);
    w_int("n_ctx", in.n_ctx, 0);
    w_int("n_gpu_layers", in.n_gpu_layers, -999);
    w_int("split_mode", in.split_mode, -1);
    w_str("cache_type_k", in.cache_type_k);
    w_str("cache_type_v", in.cache_type_v);
    w_int("n_batch", in.n_batch, 0);
    w_int("n_ubatch", in.n_ubatch, 0);
    w_int("n_threads", in.n_threads, 0);
    w_int("n_threads_batch", in.n_threads_batch, 0);
    w_int("n_parallel", in.n_parallel, 0);
    if (in.use_mmap_set) {
        f << "mmap = " << (in.use_mmap ? "true" : "false") << "\n";
    }
    w_bool("direct_io", in.use_direct_io);
    w_bool("mlock", in.use_mlock);
    w_bool("no_kv_offload", in.no_kv_offload);
    w_bool("no_op_offload", in.no_op_offload);
    w_bool("no_repack", in.no_extra_bufts);
    w_int("fit_target_mib", in.fit_target_mib, 0);
    w_int("n_cpu_moe", in.n_cpu_moe, -1);
    w_int("fused_moe", in.fused_moe, -1);
    w_int("moe_prefetch_streams", in.moe_prefetch_streams, 0);
    w_int("moe_max_vram_mb", in.moe_max_vram_mb, 0);

    return true;
}

void optimizer_preset_apply(const optimizer_preset_params & p, common_params & params) {
    if (p.n_ctx > 0)                    params.n_ctx           = p.n_ctx;
    if (p.n_gpu_layers != -999)         params.n_gpu_layers    = p.n_gpu_layers;
    if (p.split_mode >= 0)              params.split_mode      = static_cast<llama_split_mode>(p.split_mode);
    if (!p.cache_type_k.empty()) {
        auto t = cache_type_from_str(p.cache_type_k);
        if (t < GGML_TYPE_COUNT) params.cache_type_k = t;
    }
    if (!p.cache_type_v.empty()) {
        auto t = cache_type_from_str(p.cache_type_v);
        if (t < GGML_TYPE_COUNT) params.cache_type_v = t;
    }
    if (p.n_batch > 0)                  params.n_batch         = p.n_batch;
    if (p.n_ubatch > 0)                 params.n_ubatch        = p.n_ubatch;
    if (p.n_parallel > 0)               params.n_parallel      = p.n_parallel;
    if (p.use_mmap_set) {
        params.use_mmap                 = p.use_mmap;
        params.use_direct_io            = p.use_direct_io;
    } else {
        if (p.use_direct_io) params.use_direct_io = true;
    }
    if (p.use_mlock)                    params.use_mlock       = true;
    if (p.no_kv_offload)               params.no_kv_offload   = true;
    if (p.no_op_offload)               params.no_op_offload   = true;
    if (p.no_extra_bufts)              params.no_extra_bufts  = true;
    if (p.fit_target_mib > 0) {
        size_t bytes = static_cast<size_t>(p.fit_target_mib) * 1024 * 1024;
        std::fill(params.fit_params_target.begin(),
                  params.fit_params_target.end(), bytes);
    }
    if (p.fused_moe >= 0)               params.fused_moe       = (p.fused_moe == 1);
    if (p.moe_prefetch_streams > 0)     params.moe_prefetch_streams = p.moe_prefetch_streams;
    if (p.moe_max_vram_mb > 0)          params.moe_max_vram_mb = p.moe_max_vram_mb;
}
