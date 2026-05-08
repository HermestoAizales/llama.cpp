#pragma once

// Optimizer preset: delegates to common/preset.h for the canonical implementation.
// This header provides a convenience wrapper so the optimizer tool does not need
// to know internal common types.

#include "common/preset.h"

// Alias for use in optimizer code — maps to the common types.
using preset_params = common_optimizer_preset_params;

inline bool preset_load(const std::string & path, preset_params & out) {
    return common_optimizer_preset_load(path, out);
}

inline bool preset_save(const std::string & path, const preset_params & in) {
    return common_optimizer_preset_save(path, in);
}

inline void preset_apply(const preset_params & preset, common_params & params) {
    common_optimizer_preset_apply(preset, params);
}
