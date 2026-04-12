File unchanged since last read. The content from the earlier read_file result in this conversation is still current — refer to that instead of re-reading.

ggml_tensor * llm_graph_context::build_hisa_sparse_attn(
        ggml_tensor * q,
        ggml_tensor * k,
        ggml_tensor * v,
        ggml_tensor * kq_mask,
        ggml_tensor * sinks,
        ggml_tensor * v_mla,
        float   kq_scale,
        int     il) const {
    const int64_t n_kv = k->ne[1];
    (void)sinks;
    const uint32_t B = cparams.hisa_block_size > 0 ? cparams.hisa_block_size : hparams.hisa_block_size;
    if (B == 0) return nullptr;
    uint32_t m_resolved = cparams.hisa_top_m > 0 ? cparams.hisa_top_m : (hparams.hisa_top_m > 0 ? hparams.hisa_top_m : 4u);
    uint32_t budget = cparams.hisa_budget > 0 ? cparams.hisa_budget : hparams.hisa_budget;
    const uint32_t n_blocks = (n_kv > 0 && n_kv % B == 0) ? (uint32_t)(n_kv / B) : 0;
    uint32_t m = (n_blocks > 0) ? std::min(m_resolved, n_blocks) : 0;
    const uint32_t n_cand = m * B;
    if (budget > 0 && budget > n_cand) budget = n_cand;
    if (n_kv == 0 || n_kv % B != 0 || n_blocks == 0) return nullptr;
    ggml_tensor * k_bp = ggml_cont(ctx0, k);
    ggml_tensor * v_bp = ggml_cont(ctx0, v);
    ggml_tensor * q_cont = ggml_cont(ctx0, q);
    ggml_tensor * q_scaled = ggml_scale(ctx0, q_cont, kq_scale);
    cb(q_scaled, "hisa_q_scaled", il);
    ggml_tensor * k_blocks = ggml_hisa_block_pool(ctx0, k_bp, B);
    cb(k_blocks, "hisa_k_blocks", il);
    ggml_tensor * block_scores = ggml_mul_mat(ctx0, k_blocks, q_scaled);
    cb(block_scores, "hisa_block_scores", il);
    ggml_tensor * topm_indices = ggml_top_k(ctx0, block_scores, m);
    cb(topm_indices, "hisa_topm_indices", il);
    ggml_tensor * k_cand = ggml_hisa_block_gather(ctx0, k_bp, topm_indices, B);
    cb(k_cand, "hisa_k_cand", il);
    ggml_tensor * v_cand = ggml_hisa_block_gather(ctx0, v_bp, topm_indices, B);
    cb(v_cand, "hisa_v_cand", il);
    ggml_tensor * k_final = nullptr, * v_final = nullptr, * top_budget_indices = nullptr;
    if (budget > 0 && budget < n_cand) {
        ggml_tensor * token_scores = ggml_mul_mat(ctx0, k_cand, q_scaled);
        cb(token_scores, "hisa_token_scores", il);
        top_budget_indices = ggml_top_k(ctx0, token_scores, budget);
        cb(top_budget_indices, "hisa_top_budget_indices", il);
        k_final = ggml_hisa_gather(ctx0, k_cand, top_budget_indices);
        cb(k_final, "hisa_k_final", il);
        v_final = ggml_hisa_gather(ctx0, v_cand, top_budget_indices);
        cb(v_final, "hisa_v_final", il);
    } else {
        k_final = k_cand; v_final = v_cand;
    }
    ggml_tensor * mask_hisa = nullptr;
    if (kq_mask != nullptr) {
        if (budget > 0 && budget < n_cand) {
            mask_hisa = ggml_hisa_gather_mask(ctx0, kq_mask, topm_indices, top_budget_indices, B);
        } else {
            mask_hisa = nullptr;
        }
        if (mask_hisa != nullptr) {
            mask_hisa = ggml_cont(ctx0, mask_hisa);
            mask_hisa = ggml_cast(ctx0, mask_hisa, GGML_TYPE_F16);
        }
        cb(mask_hisa, "hisa_mask_gathered", il);
    }
    ggml_tensor * cur = ggml_flash_attn_ext(ctx0, q, k_final, v_final, mask_hisa, kq_scale,
                                              hparams.f_max_alibi_bias,
                                              hparams.attn_soft_cap ? hparams.f_attn_logit_softcapping : 0.0f);
    cb(cur, LLAMA_TENSOR_NAME_FATTN "_hisa", il);
    ggml_flash_attn_ext_set_prec(cur, GGML_PREC_F32);
    if (v_mla) {
        cur = ggml_permute(ctx0, cur, 0, 2, 1, 3);
        cur = ggml_mul_mat(ctx0, v_mla, cur);
        cb(cur, "fattn_mla_hisa", il);
        cur = ggml_permute(ctx0, cur, 0, 2, 1, 3);
        cur = ggml_cont(ctx0, cur);
    }
    cur = ggml_reshape_2d(ctx0, cur, cur->ne[0]*cur->ne[1], cur->ne[2]*cur->ne[3]);
    return cur;
}
