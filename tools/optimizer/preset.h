#pragma once

#include "optimizer_preset.h"

// Alias for backward compatibility within the optimizer
using preset_params = optimizer_preset_params;

inline bool preset_load(const std::string & path, preset_params & out) {
    return optimizer_preset_load(path, out);
}

inline bool preset_save(const std::string & path, const preset_params & in) {
    return optimizer_preset_save(path, in);
}

inline void preset_apply(const preset_params & preset, common_params & params) {
    optimizer_preset_apply(preset, params);
}
