#pragma once

#include <string>
#include <cstdint>

// Parsed preset parameters from an INI file.
// Fields at sentinel values mean "don't override".

struct preset_params {
    std::string name;

    int32_t n_ctx             = 0;
    int32_t n_gpu_layers      = -999;   // -999 = don't override, -1 = auto, -2 = all
    int32_t split_mode        = -1;     // -1 = don't override
    std::string cache_type_k;
    std::string cache_type_v;
    int32_t n_batch           = 0;
    int32_t n_ubatch          = 0;
    int32_t n_threads         = 0;
    int32_t n_threads_batch   = 0;
    int32_t n_parallel        = 0;
    bool    pipeline_partial  = false;
    bool    use_mmap_set      = false;  // false = don't override
    bool    use_mmap          = true;
    bool    use_direct_io     = false;
    bool    use_mlock         = false;
    bool    no_kv_offload     = false;
    bool    no_op_offload     = false;
    bool    no_extra_bufts    = false;
    int32_t kv_cache_bounded  = 0;
    int32_t fit_target_mib    = 0;
    int32_t n_cpu_moe         = -1;
    std::string spec_type;
    int32_t spec_ngram_size    = 0;
    bool    flash_attn_set     = false;
    bool    flash_attn         = true;
    bool    swa_full           = false;
    bool    use_numa           = false;
    bool    ctx_shift          = false;
    bool    cont_batching      = true;
};

bool preset_load(const std::string & path, preset_params & out);
bool preset_save(const std::string & path, const preset_params & in);
void preset_apply(const preset_params & preset, struct common_params & params);
