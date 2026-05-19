#pragma once

#include <string>
#include <cstdint>

// Parsed optimizer preset parameters from an INI file.
// Fields at sentinel values mean "don't override".

struct optimizer_preset_params {
    std::string name;

    int32_t n_ctx             = 0;
    int32_t n_gpu_layers      = -999;
    int32_t split_mode        = -1;
    std::string cache_type_k;
    std::string cache_type_v;
    int32_t n_batch           = 0;
    int32_t n_ubatch          = 0;
    int32_t n_threads         = 0;
    int32_t n_threads_batch   = 0;
    int32_t n_parallel        = 0;
    bool    use_mmap_set      = false;
    bool    use_mmap          = true;
    bool    use_direct_io     = false;
    bool    use_mlock         = false;
    bool    no_kv_offload     = false;
    bool    no_op_offload     = false;
    bool    no_extra_bufts    = false;
    int32_t fit_target_mib    = 0;
    int32_t n_cpu_moe         = -1;
    int32_t fused_moe         = -1;
    int32_t moe_prefetch_streams = 0;
    int32_t moe_max_vram_mb   = 0;
    std::string spec_type;
    int32_t spec_ngram_size   = 0;
    bool    flash_attn_set    = false;
    bool    flash_attn        = true;
    bool    swa_full          = false;
    bool    use_numa          = false;
    bool    ctx_shift         = false;
    bool    cont_batching     = true;
};

bool optimizer_preset_load(const std::string & path, optimizer_preset_params & out);
bool optimizer_preset_save(const std::string & path, const optimizer_preset_params & in);
void optimizer_preset_apply(const optimizer_preset_params & preset, struct common_params & params);
