#include "arg.h"
#include "preset.h"
#include "peg-parser.h"
#include "log.h"
#include "download.h"

#include <fstream>
#include <sstream>
#include <filesystem>

static std::string rm_leading_dashes(const std::string & str) {
    size_t pos = 0;
    while (pos < str.size() && str[pos] == '-') {
        ++pos;
    }
    return str.substr(pos);
}

// only allow a subset of args for remote presets for security reasons
// do not add more args unless absolutely necessary
// args that output to files are strictly prohibited
static std::set<std::string> get_remote_preset_whitelist(const std::map<std::string, common_arg> & key_to_opt) {
    static const std::set<std::string> allowed_options = {
        "model-url",
        "hf-repo",
        "hf-repo-draft",
        "hf-repo-v", // vocoder
        "hf-file-v", // vocoder
        "mmproj-url",
        "pooling",
        "jinja",
        "batch-size",
        "ubatch-size",
        "cache-reuse",
        "chat-template-kwargs",
        "mmap",
        // note: sampling params are automatically allowed by default
        // negated args will be added automatically if the positive arg is specified above
    };

    std::set<std::string> allowed_keys;

    for (const auto & it : key_to_opt) {
        const std::string & key = it.first;
        const common_arg & opt = it.second;
        if (allowed_options.find(key) != allowed_options.end() || opt.is_sampling) {
            allowed_keys.insert(key);
            // also add variant keys (args without leading dashes and env vars)
            for (const auto & arg : opt.get_args()) {
                allowed_keys.insert(rm_leading_dashes(arg));
            }
            for (const auto & env : opt.get_env()) {
                allowed_keys.insert(env);
            }
        }
    }

    return allowed_keys;
}

std::vector<std::string> common_preset::to_args(const std::string & bin_path) const {
    std::vector<std::string> args;

    if (!bin_path.empty()) {
        args.push_back(bin_path);
    }

    for (const auto & [opt, value] : options) {
        if (opt.is_preset_only) {
            continue; // skip preset-only options (they are not CLI args)
        }

        // use the last arg as the main arg (i.e. --long-form)
        args.push_back(opt.args.back());

        // handle value(s)
        if (opt.value_hint == nullptr && opt.value_hint_2 == nullptr) {
            // flag option, no value
            if (common_arg_utils::is_falsey(value)) {
                // use negative arg if available
                if (!opt.args_neg.empty()) {
                    args.back() = opt.args_neg.back();
                } else {
                    // otherwise, skip the flag
                    // TODO: maybe throw an error instead?
                    args.pop_back();
                }
            }
        }
        if (opt.value_hint != nullptr) {
            // single value
            args.push_back(value);
        }
        if (opt.value_hint != nullptr && opt.value_hint_2 != nullptr) {
            throw std::runtime_error(string_format(
                "common_preset::to_args(): option '%s' has two values, which is not supported yet",
                opt.args.back()
            ));
        }
    }

    return args;
}

std::string common_preset::to_ini() const {
    std::ostringstream ss;

    ss << "[" << name << "]\n";
    for (const auto & [opt, value] : options) {
        auto espaced_value = value;
        string_replace_all(espaced_value, "\n", "\\\n");
        ss << rm_leading_dashes(opt.args.back()) << " = ";
        ss << espaced_value << "\n";
    }
    ss << "\n";

    return ss.str();
}

void common_preset::set_option(const common_preset_context & ctx, const std::string & env, const std::string & value) {
    // try if option exists, update it
    for (auto & [opt, val] : options) {
        if (opt.env && env == opt.env) {
            val = value;
            return;
        }
    }
    // if option does not exist, we need to add it
    if (ctx.key_to_opt.find(env) == ctx.key_to_opt.end()) {
        throw std::runtime_error(string_format(
            "%s: option with env '%s' not found in ctx_params",
            __func__, env.c_str()
        ));
    }
    options[ctx.key_to_opt.at(env)] = value;
}

void common_preset::unset_option(const std::string & env) {
    for (auto it = options.begin(); it != options.end(); ) {
        const common_arg & opt = it->first;
        if (opt.env && env == opt.env) {
            it = options.erase(it);
            return;
        } else {
            ++it;
        }
    }
}

bool common_preset::get_option(const std::string & env, std::string & value) const {
    for (const auto & [opt, val] : options) {
        if (opt.env && env == opt.env) {
            value = val;
            return true;
        }
    }
    return false;
}

void common_preset::merge(const common_preset & other) {
    for (const auto & [opt, val] : other.options) {
        options[opt] = val; // overwrite existing options
    }
}

void common_preset::apply_to_params(common_params & params) const {
    for (const auto & [opt, val] : options) {
        // apply each option to params
        if (opt.handler_string) {
            opt.handler_string(params, val);
        } else if (opt.handler_int) {
            opt.handler_int(params, std::stoi(val));
        } else if (opt.handler_bool) {
            opt.handler_bool(params, common_arg_utils::is_truthy(val));
        } else if (opt.handler_str_str) {
            // not supported yet
            throw std::runtime_error(string_format(
                "%s: option with two values is not supported yet",
                __func__
            ));
        } else if (opt.handler_void) {
            opt.handler_void(params);
        } else {
            GGML_ABORT("unknown handler type");
        }
    }
}

static std::map<std::string, std::map<std::string, std::string>> parse_ini_from_file(const std::string & path) {
    std::map<std::string, std::map<std::string, std::string>> parsed;

    if (!std::filesystem::exists(path)) {
        throw std::runtime_error("preset file does not exist: " + path);
    }

    std::ifstream file(path);
    if (!file.good()) {
        throw std::runtime_error("failed to open server preset file: " + path);
    }

    std::string contents((std::istreambuf_iterator<char>(file)), std::istreambuf_iterator<char>());

    static const auto parser = build_peg_parser([](auto & p) {
        // newline ::= "\r\n" / "\n" / "\r"
        auto newline = p.rule("newline", p.literal("\r\n") | p.literal("\n") | p.literal("\r"));

        // ws ::= [ \t]*
        auto ws = p.rule("ws", p.chars("[ \t]", 0, -1));

        // comment ::= [;#] (!newline .)*
        auto comment = p.rule("comment", p.chars("[;#]", 1, 1) + p.zero_or_more(p.negate(newline) + p.any()));

        // eol ::= ws comment? (newline / EOF)
        auto eol = p.rule("eol", ws + p.optional(comment) + (newline | p.end()));

        // ident ::= [a-zA-Z_] [a-zA-Z0-9_.-]*
        auto ident = p.rule("ident", p.chars("[a-zA-Z_]", 1, 1) + p.chars("[a-zA-Z0-9_.-]", 0, -1));

        // value ::= (!eol-start .)*
        auto eol_start = p.rule("eol-start", ws + (p.chars("[;#]", 1, 1) | newline | p.end()));
        auto value = p.rule("value", p.zero_or_more(p.negate(eol_start) + p.any()));

        // header-line ::= "[" ws ident ws "]" eol
        auto header_line = p.rule("header-line", "[" + ws + p.tag("section-name", p.chars("[^]]")) + ws + "]" + eol);

        // kv-line ::= ident ws "=" ws value eol
        auto kv_line = p.rule("kv-line", p.tag("key", ident) + ws + "=" + ws + p.tag("value", value) + eol);

        // comment-line ::= ws comment (newline / EOF)
        auto comment_line = p.rule("comment-line", ws + comment + (newline | p.end()));

        // blank-line ::= ws (newline / EOF)
        auto blank_line = p.rule("blank-line", ws + (newline | p.end()));

        // line ::= header-line / kv-line / comment-line / blank-line
        auto line = p.rule("line", header_line | kv_line | comment_line | blank_line);

        // ini ::= line* EOF
        auto ini = p.rule("ini", p.zero_or_more(line) + p.end());

        return ini;
    });

    common_peg_parse_context ctx(contents);
    const auto result = parser.parse(ctx);
    if (!result.success()) {
        throw std::runtime_error("failed to parse server config file: " + path);
    }

    std::string current_section = COMMON_PRESET_DEFAULT_NAME;
    std::string current_key;

    ctx.ast.visit(result, [&](const auto & node) {
        if (node.tag == "section-name") {
            const std::string section = std::string(node.text);
            current_section = section;
            parsed[current_section] = {};
        } else if (node.tag == "key") {
            const std::string key = std::string(node.text);
            current_key = key;
        } else if (node.tag == "value" && !current_key.empty() && !current_section.empty()) {
            parsed[current_section][current_key] = std::string(node.text);
            current_key.clear();
        }
    });

    return parsed;
}

static std::map<std::string, common_arg> get_map_key_opt(common_params_context & ctx_params) {
    std::map<std::string, common_arg> mapping;
    for (const auto & opt : ctx_params.options) {
        for (const auto & env : opt.get_env()) {
            mapping[env] = opt;
        }
        for (const auto & arg : opt.get_args()) {
            mapping[rm_leading_dashes(arg)] = opt;
        }
    }
    return mapping;
}

static bool is_bool_arg(const common_arg & arg) {
    return !arg.args_neg.empty();
}

static std::string parse_bool_arg(const common_arg & arg, const std::string & key, const std::string & value) {
    // if this is a negated arg, we need to reverse the value
    for (const auto & neg_arg : arg.args_neg) {
        if (rm_leading_dashes(neg_arg) == key) {
            return common_arg_utils::is_truthy(value) ? "false" : "true";
        }
    }
    // otherwise, not negated
    return value;
}

common_preset_context::common_preset_context(llama_example ex, bool only_remote_allowed)
        : ctx_params(common_params_parser_init(default_params, ex)) {
    common_params_add_preset_options(ctx_params.options);
    key_to_opt = get_map_key_opt(ctx_params);

    // setup allowed keys if only_remote_allowed is true
    if (only_remote_allowed) {
        filter_allowed_keys = true;
        allowed_keys = get_remote_preset_whitelist(key_to_opt);
    }
}

common_presets common_preset_context::load_from_ini(const std::string & path, common_preset & global) const {
    common_presets out;
    auto ini_data = parse_ini_from_file(path);

    for (auto section : ini_data) {
        common_preset preset;
        if (section.first.empty()) {
            preset.name = COMMON_PRESET_DEFAULT_NAME;
        } else {
            preset.name = section.first;
        }
        LOG_DBG("loading preset: %s\n", preset.name.c_str());
        for (const auto & [key, value] : section.second) {
            if (key == "version") {
                // skip version key (reserved for future use)
                continue;
            }

            LOG_DBG("option: %s = %s\n", key.c_str(), value.c_str());
            if (filter_allowed_keys && allowed_keys.find(key) == allowed_keys.end()) {
                throw std::runtime_error(string_format(
                    "option '%s' is not allowed in remote presets",
                    key.c_str()
                ));
            }
            if (key_to_opt.find(key) != key_to_opt.end()) {
                const auto & opt = key_to_opt.at(key);
                if (is_bool_arg(opt)) {
                    preset.options[opt] = parse_bool_arg(opt, key, value);
                } else {
                    preset.options[opt] = value;
                }
                LOG_DBG("accepted option: %s = %s\n", key.c_str(), preset.options[opt].c_str());
            } else {
                throw std::runtime_error(string_format(
                    "option '%s' not recognized in preset '%s'",
                    key.c_str(), preset.name.c_str()
                ));
            }
        }

        if (preset.name == "*") {
            // handle global preset
            global = preset;
        } else {
            out[preset.name] = preset;
        }
    }

    return out;
}

common_presets common_preset_context::load_from_cache() const {
    common_presets out;

    auto cached_models = common_list_cached_models();
    for (const auto & model : cached_models) {
        common_preset preset;
        preset.name = model.to_string();
        preset.set_option(*this, "LLAMA_ARG_HF_REPO", model.to_string());
        out[preset.name] = preset;
    }

    return out;
}

struct local_model {
    std::string name;
    std::string path;
    std::string path_mmproj;
};

common_presets common_preset_context::load_from_models_dir(const std::string & models_dir) const {
    if (!std::filesystem::exists(models_dir) || !std::filesystem::is_directory(models_dir)) {
        throw std::runtime_error(string_format("error: '%s' does not exist or is not a directory\n", models_dir.c_str()));
    }

    std::vector<local_model> models;
    auto scan_subdir = [&models](const std::string & subdir_path, const std::string & name) {
        auto files = fs_list(subdir_path, false);
        common_file_info model_file;
        common_file_info first_shard_file;
        common_file_info mmproj_file;
        for (const auto & file : files) {
            if (string_ends_with(file.name, ".gguf")) {
                if (file.name.find("mmproj") != std::string::npos) {
                    mmproj_file = file;
                } else if (file.name.find("-00001-of-") != std::string::npos) {
                    first_shard_file = file;
                } else {
                    model_file = file;
                }
            }
        }
        // single file model
        local_model model{
            /* name        */ name,
            /* path        */ first_shard_file.path.empty() ? model_file.path : first_shard_file.path,
            /* path_mmproj */ mmproj_file.path // can be empty
        };
        if (!model.path.empty()) {
            models.push_back(model);
        }
    };

    auto files = fs_list(models_dir, true);
    for (const auto & file : files) {
        if (file.is_dir) {
            scan_subdir(file.path, file.name);
        } else if (string_ends_with(file.name, ".gguf")) {
            // single file model
            std::string name = file.name;
            string_replace_all(name, ".gguf", "");
            local_model model{
                /* name        */ name,
                /* path        */ file.path,
                /* path_mmproj */ ""
            };
            models.push_back(model);
        }
    }

    // convert local models to presets
    common_presets out;
    for (const auto & model : models) {
        common_preset preset;
        preset.name = model.name;
        preset.set_option(*this, "LLAMA_ARG_MODEL", model.path);
        if (!model.path_mmproj.empty()) {
            preset.set_option(*this, "LLAMA_ARG_MMPROJ", model.path_mmproj);
        }
        out[preset.name] = preset;
    }

    return out;
}

common_preset common_preset_context::load_from_args(int argc, char ** argv) const {
    common_preset preset;
    preset.name = COMMON_PRESET_DEFAULT_NAME;

    bool ok = common_params_to_map(argc, argv, ctx_params.ex, preset.options);
    if (!ok) {
        throw std::runtime_error("failed to parse CLI arguments into preset");
    }

    return preset;
}

common_presets common_preset_context::cascade(const common_presets & base, const common_presets & added) const {
    common_presets out = base; // copy
    for (const auto & [name, preset_added] : added) {
        if (out.find(name) != out.end()) {
            // if exists, merge
            common_preset & target = out[name];
            target.merge(preset_added);
        } else {
            // otherwise, add directly
            out[name] = preset_added;
        }
    }
    return out;
}

common_presets common_preset_context::cascade(const common_preset & base, const common_presets & presets) const {
    common_presets out;
    for (const auto & [name, preset] : presets) {
        common_preset tmp = base; // copy
        tmp.name = name;
        tmp.merge(preset);
        out[name] = std::move(tmp);
    }
    return out;
}

//
// Optimizer preset implementation
//

bool common_optimizer_preset_load(const std::string & path, common_optimizer_preset_params & out) {
    std::ifstream file(path);
    if (!file.is_open()) {
        LOG_WRN("%s: failed to open preset file '%s'\n", __func__, path.c_str());
        return false;
    }
    std::string line;
    std::string section;
    while (std::getline(file, line)) {
        size_t start = line.find_first_not_of(" \t\r\n");
        if (start == std::string::npos) continue;
        size_t end = line.find_last_not_of(" \t\r\n");
        line = line.substr(start, end - start + 1);
        if (line.empty() || line[0] == ';' || line[0] == '#') continue;
        if (line.front() == '[' && line.back() == ']') {
            section = line.substr(1, line.size() - 2);
            continue;
        }
        auto eq_pos = line.find('=');
        if (eq_pos == std::string::npos) continue;
        std::string key = line.substr(0, eq_pos);
        std::string val = line.substr(eq_pos + 1);
        end = key.find_last_not_of(" \t");
        if (end != std::string::npos) key = key.substr(0, end + 1);
        start = val.find_first_not_of(" \t");
        if (start != std::string::npos) val = val.substr(start);

        if (section == "preset") {
            if (key == "name") out.name = val;
        } else if (section == "context") {
            if (key == "n_ctx") out.n_ctx = std::stoi(val);
            else if (key == "cache_type_k") out.cache_type_k = val;
            else if (key == "cache_type_v") out.cache_type_v = val;
        } else if (section == "gpu") {
            if (key == "n_gpu_layers") out.n_gpu_layers = std::stoi(val);
            else if (key == "split_mode") out.split_mode = std::stoi(val);
            else if (key == "no_kv_offload") out.no_kv_offload = (val == "1" || val == "true");
            else if (key == "no_op_offload") out.no_op_offload = (val == "1" || val == "true");
            else if (key == "no_extra_bufts") out.no_extra_bufts = (val == "1" || val == "true");
            else if (key == "flash_attn") { out.flash_attn_set = (val == "1" || val == "true") ? 1 : -1; }
            else if (key == "swa_full") out.swa_full = (val == "1" || val == "true");
        } else if (section == "batch") {
            if (key == "n_batch") out.n_batch = std::stoi(val);
            else if (key == "n_ubatch") out.n_ubatch = std::stoi(val);
        } else if (section == "server") {
            if (key == "n_parallel") out.n_parallel = std::stoi(val);
        } else if (section == "memory") {
            if (key == "use_mmap") { out.use_mmap_set = true; out.use_mmap = (val == "1" || val == "true"); }
            else if (key == "use_direct_io") out.use_direct_io = (val == "1" || val == "true");
            else if (key == "use_mlock") out.use_mlock = (val == "1" || val == "true");
        } else if (section == "speculative") {
            if (key == "spec_type") out.spec_type = val;
        } else if (section == "advanced") {
            if (key == "numa") out.numa_str = val;
            else if (key == "ctx_shift") out.ctx_shift = (val == "1" || val == "true");
            else if (key == "cont_batching") out.cont_batching = (val == "1" || val == "true");
        }
    }
    LOG_INF("%s: loaded optimizer preset '%s' from '%s'\n", __func__, out.name.c_str(), path.c_str());
    return true;
}

bool common_optimizer_preset_save(const std::string & path, const common_optimizer_preset_params & in) {
    std::ofstream file(path);
    if (!file.is_open()) {
        LOG_WRN("%s: failed to open preset file '%s' for writing\n", __func__, path.c_str());
        return false;
    }
    file << "# llama-optimizer generated preset: " << in.name << "\n";
    file << "[preset]\n";
    file << "name=" << in.name << "\n";
    file << "\n[context]\n";
    if (in.n_ctx > 0) file << "n_ctx=" << in.n_ctx << "\n";
    if (!in.cache_type_k.empty()) file << "cache_type_k=" << in.cache_type_k << "\n";
    if (!in.cache_type_v.empty()) file << "cache_type_v=" << in.cache_type_v << "\n";
    file << "\n[gpu]\n";
    if (in.n_gpu_layers != -999) file << "n_gpu_layers=" << in.n_gpu_layers << "\n";
    if (in.split_mode >= 0) file << "split_mode=" << in.split_mode << "\n";
    file << "no_kv_offload=" << (in.no_kv_offload ? "1" : "0") << "\n";
    file << "no_op_offload=" << (in.no_op_offload ? "1" : "0") << "\n";
    file << "no_extra_bufts=" << (in.no_extra_bufts ? "1" : "0") << "\n";
    if (in.flash_attn_set != 0) file << "flash_attn=" << (in.flash_attn_set > 0 ? "1" : "0") << "\n";
    file << "swa_full=" << (in.swa_full ? "1" : "0") << "\n";
    file << "\n[batch]\n";
    if (in.n_batch > 0) file << "n_batch=" << in.n_batch << "\n";
    if (in.n_ubatch > 0) file << "n_ubatch=" << in.n_ubatch << "\n";
    file << "\n[server]\n";
    if (in.n_parallel > 0) file << "n_parallel=" << in.n_parallel << "\n";
    file << "\n[memory]\n";
    if (in.use_mmap_set) file << "use_mmap=" << (in.use_mmap ? "1" : "0") << "\n";
    file << "use_direct_io=" << (in.use_direct_io ? "1" : "0") << "\n";
    file << "use_mlock=" << (in.use_mlock ? "1" : "0") << "\n";
    file << "\n[speculative]\n";
    if (!in.spec_type.empty()) file << "spec_type=" << in.spec_type << "\n";
    file << "\n[advanced]\n";
    if (!in.numa_str.empty()) file << "numa=" << in.numa_str << "\n";
    file << "ctx_shift=" << (in.ctx_shift ? "1" : "0") << "\n";
    file << "cont_batching=" << (in.cont_batching ? "1" : "0") << "\n";
    LOG_INF("%s: saved optimizer preset '%s' to '%s'\n", __func__, in.name.c_str(), path.c_str());
    return true;
}

void common_optimizer_preset_apply(const common_optimizer_preset_params & preset, common_params & params) {
    // Only set fields that map directly to common_params members.
    // Complex types (cache_type_k/v as ggml_type, spec_type as enum,
    // numa as ggml_numa_strategy) are NOT set here — they are output
    // as CLI flags in the optimizer report instead.

    if (preset.n_ctx > 0)                 params.n_ctx           = preset.n_ctx;
    if (preset.n_gpu_layers != -999)      params.n_gpu_layers    = preset.n_gpu_layers;
    if (preset.split_mode >= 0)           params.split_mode      = (enum llama_split_mode)preset.split_mode;
    if (preset.n_batch > 0)               params.n_batch         = preset.n_batch;
    if (preset.n_ubatch > 0)              params.n_ubatch        = preset.n_ubatch;
    if (preset.n_parallel > 0)            params.n_parallel      = preset.n_parallel;
    if (preset.use_mmap_set)              params.use_mmap        = preset.use_mmap;
    if (preset.use_direct_io)             params.use_direct_io   = true;
    if (preset.use_mlock)                 params.use_mlock       = true;
    if (preset.no_kv_offload)             params.no_kv_offload   = true;
    if (preset.no_op_offload)             params.no_op_offload   = true;
    if (preset.no_extra_bufts)            params.no_extra_bufts  = true;
    if (preset.swa_full)                  params.swa_full        = true;
    if (preset.ctx_shift)                 params.ctx_shift       = true;
                                      params.cont_batching      = preset.cont_batching;
}
