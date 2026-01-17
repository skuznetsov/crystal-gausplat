// Persistent Threadgroups for Transformer Blocks
// Reduces kernel launch overhead by processing multiple layers in one dispatch
//
// Key optimizations:
// 1. Threadgroups stay resident, process all layers without relaunching
// 2. Intermediate activations stay in threadgroup memory
// 3. Only final output written to device memory
// 4. Fused operations: LN + Attention, LN + MLP

#include <metal_stdlib>
using namespace metal;

// ============================================================================
// Constants
// ============================================================================

constant uint TILE_SIZE = 16;
constant uint MAX_SEQ_LEN = 512;      // Max sequence length for threadgroup memory
constant uint MAX_EMBED_DIM = 1024;   // Max embedding dimension
constant uint MAX_HEAD_DIM = 128;     // Max head dimension

// ============================================================================
// Helper Functions
// ============================================================================

// GELU activation: 0.5 * x * (1 + tanh(sqrt(2/pi) * (x + 0.044715 * x^3)))
inline float gelu(float x) {
    const float sqrt_2_pi = 0.7978845608f;  // sqrt(2/pi)
    return 0.5f * x * (1.0f + tanh(sqrt_2_pi * (x + 0.044715f * x * x * x)));
}

// ============================================================================
// Fused Transformer Block Kernel (Full Block in One Pass)
// ============================================================================

// Process one transformer block: LN1 -> Attention -> Residual -> LN2 -> MLP -> Residual
// Each threadgroup processes one batch element through the entire block
//
// Weights layout in buffer (concatenated):
// - ln1_gamma: [embed_dim]
// - ln1_beta: [embed_dim]
// - wq: [embed_dim, embed_dim]
// - wk: [embed_dim, embed_dim]
// - wv: [embed_dim, embed_dim]
// - wo: [embed_dim, embed_dim]
// - ln2_gamma: [embed_dim]
// - ln2_beta: [embed_dim]
// - mlp_fc1_weight: [hidden_dim, embed_dim]
// - mlp_fc1_bias: [hidden_dim]
// - mlp_fc2_weight: [embed_dim, hidden_dim]
// - mlp_fc2_bias: [embed_dim]

kernel void transformer_block_fused(
    device const float* input [[buffer(0)]],        // [batch, seq_len, embed_dim]
    device float* output [[buffer(1)]],              // [batch, seq_len, embed_dim]
    device const float* weights [[buffer(2)]],       // All weights concatenated
    constant uint& batch [[buffer(3)]],
    constant uint& seq_len [[buffer(4)]],
    constant uint& embed_dim [[buffer(5)]],
    constant uint& num_heads [[buffer(6)]],
    constant uint& hidden_dim [[buffer(7)]],         // MLP hidden dim (usually 4 * embed_dim)
    constant float& eps [[buffer(8)]],
    threadgroup float* shared_x [[threadgroup(0)]],  // [seq_len, embed_dim] for activations
    threadgroup float* shared_tmp [[threadgroup(1)]], // Temp buffer for intermediate results
    uint3 gid [[threadgroup_position_in_grid]],
    uint lid [[thread_index_in_threadgroup]],
    uint3 tg_size [[threads_per_threadgroup]]
) {
    uint batch_idx = gid.x;
    if (batch_idx >= batch) return;

    uint head_dim = embed_dim / num_heads;

    // Weight offsets
    uint ln1_gamma_off = 0;
    uint ln1_beta_off = embed_dim;
    uint wq_off = 2 * embed_dim;
    uint wk_off = wq_off + embed_dim * embed_dim;
    uint wv_off = wk_off + embed_dim * embed_dim;
    uint wo_off = wv_off + embed_dim * embed_dim;
    uint ln2_gamma_off = wo_off + embed_dim * embed_dim;
    uint ln2_beta_off = ln2_gamma_off + embed_dim;
    uint fc1_w_off = ln2_beta_off + embed_dim;
    uint fc1_b_off = fc1_w_off + hidden_dim * embed_dim;
    uint fc2_w_off = fc1_b_off + hidden_dim;
    uint fc2_b_off = fc2_w_off + embed_dim * hidden_dim;

    // Base pointer for this batch element
    uint batch_offset = batch_idx * seq_len * embed_dim;

    // ========================================================================
    // Step 1: Load input to shared memory
    // ========================================================================
    for (uint i = lid; i < seq_len * embed_dim; i += tg_size.x) {
        shared_x[i] = input[batch_offset + i];
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    // ========================================================================
    // Step 2: LayerNorm 1 (in-place in shared_x)
    // ========================================================================
    // Each thread handles one position in sequence
    for (uint pos = lid; pos < seq_len; pos += tg_size.x) {
        // Compute mean
        float sum = 0.0f;
        for (uint e = 0; e < embed_dim; e++) {
            sum += shared_x[pos * embed_dim + e];
        }
        float mean = sum / float(embed_dim);

        // Compute variance
        float var_sum = 0.0f;
        for (uint e = 0; e < embed_dim; e++) {
            float diff = shared_x[pos * embed_dim + e] - mean;
            var_sum += diff * diff;
        }
        float inv_std = rsqrt(var_sum / float(embed_dim) + eps);

        // Normalize and apply gamma/beta
        for (uint e = 0; e < embed_dim; e++) {
            float x = shared_x[pos * embed_dim + e];
            float normed = (x - mean) * inv_std;
            shared_x[pos * embed_dim + e] = normed * weights[ln1_gamma_off + e] + weights[ln1_beta_off + e];
        }
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    // ========================================================================
    // Step 3: Self-Attention (QKV projection, attention, output projection)
    // ========================================================================
    // For simplicity, we compute attention sequentially per head
    // In production, this would be parallelized across threads

    // Allocate Q, K, V in shared_tmp: [seq_len, embed_dim] each
    threadgroup float* shared_q = shared_tmp;
    threadgroup float* shared_k = shared_tmp + seq_len * embed_dim;
    threadgroup float* shared_v = shared_tmp + 2 * seq_len * embed_dim;
    threadgroup float* shared_attn_out = shared_tmp + 3 * seq_len * embed_dim;

    // Q projection: Q = X @ Wq^T
    for (uint i = lid; i < seq_len * embed_dim; i += tg_size.x) {
        uint pos = i / embed_dim;
        uint out_e = i % embed_dim;

        float sum = 0.0f;
        for (uint k = 0; k < embed_dim; k++) {
            sum += shared_x[pos * embed_dim + k] * weights[wq_off + out_e * embed_dim + k];
        }
        shared_q[i] = sum;
    }

    // K projection: K = X @ Wk^T
    for (uint i = lid; i < seq_len * embed_dim; i += tg_size.x) {
        uint pos = i / embed_dim;
        uint out_e = i % embed_dim;

        float sum = 0.0f;
        for (uint k = 0; k < embed_dim; k++) {
            sum += shared_x[pos * embed_dim + k] * weights[wk_off + out_e * embed_dim + k];
        }
        shared_k[i] = sum;
    }

    // V projection: V = X @ Wv^T
    for (uint i = lid; i < seq_len * embed_dim; i += tg_size.x) {
        uint pos = i / embed_dim;
        uint out_e = i % embed_dim;

        float sum = 0.0f;
        for (uint k = 0; k < embed_dim; k++) {
            sum += shared_x[pos * embed_dim + k] * weights[wv_off + out_e * embed_dim + k];
        }
        shared_v[i] = sum;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    // Attention: for each head, compute softmax(Q @ K^T / sqrt(d)) @ V
    float scale = 1.0f / sqrt(float(head_dim));

    // Each thread handles one (position, head) pair
    for (uint i = lid; i < seq_len * num_heads; i += tg_size.x) {
        uint pos = i / num_heads;
        uint head = i % num_heads;
        uint head_start = head * head_dim;

        // Compute attention scores for this query
        float scores[MAX_SEQ_LEN];
        float max_score = -INFINITY;

        for (uint k_pos = 0; k_pos < seq_len; k_pos++) {
            float dot = 0.0f;
            for (uint d = 0; d < head_dim; d++) {
                dot += shared_q[pos * embed_dim + head_start + d] *
                       shared_k[k_pos * embed_dim + head_start + d];
            }
            scores[k_pos] = dot * scale;
            max_score = max(max_score, scores[k_pos]);
        }

        // Softmax
        float sum_exp = 0.0f;
        for (uint k_pos = 0; k_pos < seq_len; k_pos++) {
            scores[k_pos] = exp(scores[k_pos] - max_score);
            sum_exp += scores[k_pos];
        }
        for (uint k_pos = 0; k_pos < seq_len; k_pos++) {
            scores[k_pos] /= sum_exp;
        }

        // Weighted sum of values
        for (uint d = 0; d < head_dim; d++) {
            float sum = 0.0f;
            for (uint v_pos = 0; v_pos < seq_len; v_pos++) {
                sum += scores[v_pos] * shared_v[v_pos * embed_dim + head_start + d];
            }
            shared_attn_out[pos * embed_dim + head_start + d] = sum;
        }
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    // Output projection: out = attn_out @ Wo^T
    for (uint i = lid; i < seq_len * embed_dim; i += tg_size.x) {
        uint pos = i / embed_dim;
        uint out_e = i % embed_dim;

        float sum = 0.0f;
        for (uint k = 0; k < embed_dim; k++) {
            sum += shared_attn_out[pos * embed_dim + k] * weights[wo_off + out_e * embed_dim + k];
        }
        // Store back to shared_tmp (reusing attn_out space)
        shared_attn_out[i] = sum;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    // ========================================================================
    // Step 4: Residual connection 1 (add original input)
    // ========================================================================
    // Need to reload original input since we overwrote shared_x with LN output
    // Actually, let's modify: keep residual in a separate buffer
    // For now, add the attention output to the normalized input (which is a simplification)
    // In proper implementation, we'd keep the original input

    // For this implementation, we add attn_out to shared_x (LN1 output)
    // This is slightly different from standard transformer but demonstrates the concept
    for (uint i = lid; i < seq_len * embed_dim; i += tg_size.x) {
        shared_x[i] = input[batch_offset + i] + shared_attn_out[i];  // Residual with original input
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    // ========================================================================
    // Step 5: LayerNorm 2 (store in shared_tmp for MLP)
    // ========================================================================
    threadgroup float* ln2_out = shared_tmp;  // Reuse shared_tmp

    for (uint pos = lid; pos < seq_len; pos += tg_size.x) {
        float sum = 0.0f;
        for (uint e = 0; e < embed_dim; e++) {
            sum += shared_x[pos * embed_dim + e];
        }
        float mean = sum / float(embed_dim);

        float var_sum = 0.0f;
        for (uint e = 0; e < embed_dim; e++) {
            float diff = shared_x[pos * embed_dim + e] - mean;
            var_sum += diff * diff;
        }
        float inv_std = rsqrt(var_sum / float(embed_dim) + eps);

        for (uint e = 0; e < embed_dim; e++) {
            float x = shared_x[pos * embed_dim + e];
            float normed = (x - mean) * inv_std;
            ln2_out[pos * embed_dim + e] = normed * weights[ln2_gamma_off + e] + weights[ln2_beta_off + e];
        }
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    // ========================================================================
    // Step 6: MLP (FC1 + GELU + FC2)
    // ========================================================================
    threadgroup float* mlp_hidden = shared_tmp + seq_len * embed_dim;  // [seq_len, hidden_dim]

    // FC1: hidden = GELU(ln2_out @ W1^T + b1)
    for (uint i = lid; i < seq_len * hidden_dim; i += tg_size.x) {
        uint pos = i / hidden_dim;
        uint h = i % hidden_dim;

        float sum = weights[fc1_b_off + h];  // bias
        for (uint k = 0; k < embed_dim; k++) {
            sum += ln2_out[pos * embed_dim + k] * weights[fc1_w_off + h * embed_dim + k];
        }
        mlp_hidden[i] = gelu(sum);
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    // FC2: out = hidden @ W2^T + b2
    threadgroup float* mlp_out = shared_tmp;  // Reuse ln2_out space

    for (uint i = lid; i < seq_len * embed_dim; i += tg_size.x) {
        uint pos = i / embed_dim;
        uint out_e = i % embed_dim;

        float sum = weights[fc2_b_off + out_e];  // bias
        for (uint k = 0; k < hidden_dim; k++) {
            sum += mlp_hidden[pos * hidden_dim + k] * weights[fc2_w_off + out_e * hidden_dim + k];
        }
        mlp_out[i] = sum;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    // ========================================================================
    // Step 7: Residual connection 2 and write output
    // ========================================================================
    for (uint i = lid; i < seq_len * embed_dim; i += tg_size.x) {
        output[batch_offset + i] = shared_x[i] + mlp_out[i];
    }
}

// ============================================================================
// Multi-Block Persistent Transformer Kernel
// ============================================================================

// Process multiple transformer blocks without returning to CPU
// Uses ping-pong buffers to chain blocks together
kernel void persistent_transformer_multi_block(
    device const float* input [[buffer(0)]],         // [batch, seq_len, embed_dim]
    device float* output [[buffer(1)]],               // [batch, seq_len, embed_dim]
    device const float* all_weights [[buffer(2)]],    // All blocks' weights concatenated
    constant uint& batch [[buffer(3)]],
    constant uint& seq_len [[buffer(4)]],
    constant uint& embed_dim [[buffer(5)]],
    constant uint& num_heads [[buffer(6)]],
    constant uint& hidden_dim [[buffer(7)]],
    constant uint& num_blocks [[buffer(8)]],          // Number of transformer blocks
    constant float& eps [[buffer(9)]],
    device float* temp_buffer [[buffer(10)]],         // [batch, seq_len, embed_dim] for ping-pong
    threadgroup float* shared [[threadgroup(0)]],     // Large shared memory pool
    uint3 gid [[threadgroup_position_in_grid]],
    uint lid [[thread_index_in_threadgroup]],
    uint3 tg_size [[threads_per_threadgroup]]
) {
    uint batch_idx = gid.x;
    if (batch_idx >= batch) return;

    uint head_dim = embed_dim / num_heads;

    // Calculate weight size per block
    // ln1_gamma + ln1_beta + wq + wk + wv + wo + ln2_gamma + ln2_beta + fc1_w + fc1_b + fc2_w + fc2_b
    uint weights_per_block = 2 * embed_dim +                    // ln1 gamma/beta
                             4 * embed_dim * embed_dim +        // wq, wk, wv, wo
                             2 * embed_dim +                    // ln2 gamma/beta
                             hidden_dim * embed_dim + hidden_dim +  // fc1 weight/bias
                             embed_dim * hidden_dim + embed_dim;    // fc2 weight/bias

    uint batch_offset = batch_idx * seq_len * embed_dim;

    // Shared memory layout:
    // [0, seq_len * embed_dim): activation buffer A
    // [seq_len * embed_dim, 2 * seq_len * embed_dim): activation buffer B
    // [2 * seq_len * embed_dim, ...): working space for attention/MLP

    threadgroup float* buf_a = shared;
    threadgroup float* buf_b = shared + seq_len * embed_dim;
    threadgroup float* workspace = shared + 2 * seq_len * embed_dim;

    // Load input to buffer A
    for (uint i = lid; i < seq_len * embed_dim; i += tg_size.x) {
        buf_a[i] = input[batch_offset + i];
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    // Process each block
    for (uint block = 0; block < num_blocks; block++) {
        device const float* block_weights = all_weights + block * weights_per_block;

        // Determine input/output buffers (ping-pong)
        threadgroup float* current_in = (block % 2 == 0) ? buf_a : buf_b;
        threadgroup float* current_out = (block % 2 == 0) ? buf_b : buf_a;

        // Weight offsets within this block
        uint ln1_gamma_off = 0;
        uint ln1_beta_off = embed_dim;
        uint wq_off = 2 * embed_dim;
        uint wk_off = wq_off + embed_dim * embed_dim;
        uint wv_off = wk_off + embed_dim * embed_dim;
        uint wo_off = wv_off + embed_dim * embed_dim;
        uint ln2_gamma_off = wo_off + embed_dim * embed_dim;
        uint ln2_beta_off = ln2_gamma_off + embed_dim;
        uint fc1_w_off = ln2_beta_off + embed_dim;
        uint fc1_b_off = fc1_w_off + hidden_dim * embed_dim;
        uint fc2_w_off = fc1_b_off + hidden_dim;
        uint fc2_b_off = fc2_w_off + embed_dim * hidden_dim;

        // Store residual (copy current_in to workspace temporarily)
        threadgroup float* residual = workspace;
        for (uint i = lid; i < seq_len * embed_dim; i += tg_size.x) {
            residual[i] = current_in[i];
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);

        // ====== LayerNorm 1 ======
        threadgroup float* ln1_out = workspace + seq_len * embed_dim;
        for (uint pos = lid; pos < seq_len; pos += tg_size.x) {
            float sum = 0.0f;
            for (uint e = 0; e < embed_dim; e++) {
                sum += current_in[pos * embed_dim + e];
            }
            float mean = sum / float(embed_dim);

            float var_sum = 0.0f;
            for (uint e = 0; e < embed_dim; e++) {
                float diff = current_in[pos * embed_dim + e] - mean;
                var_sum += diff * diff;
            }
            float inv_std = rsqrt(var_sum / float(embed_dim) + eps);

            for (uint e = 0; e < embed_dim; e++) {
                float x = current_in[pos * embed_dim + e];
                float normed = (x - mean) * inv_std;
                ln1_out[pos * embed_dim + e] = normed * block_weights[ln1_gamma_off + e] +
                                                block_weights[ln1_beta_off + e];
            }
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);

        // ====== QKV Projections ======
        threadgroup float* q = workspace + 2 * seq_len * embed_dim;
        threadgroup float* k = q + seq_len * embed_dim;
        threadgroup float* v = k + seq_len * embed_dim;

        // Q projection
        for (uint i = lid; i < seq_len * embed_dim; i += tg_size.x) {
            uint pos = i / embed_dim;
            uint out_e = i % embed_dim;
            float sum = 0.0f;
            for (uint kk = 0; kk < embed_dim; kk++) {
                sum += ln1_out[pos * embed_dim + kk] * block_weights[wq_off + out_e * embed_dim + kk];
            }
            q[i] = sum;
        }

        // K projection
        for (uint i = lid; i < seq_len * embed_dim; i += tg_size.x) {
            uint pos = i / embed_dim;
            uint out_e = i % embed_dim;
            float sum = 0.0f;
            for (uint kk = 0; kk < embed_dim; kk++) {
                sum += ln1_out[pos * embed_dim + kk] * block_weights[wk_off + out_e * embed_dim + kk];
            }
            k[i] = sum;
        }

        // V projection
        for (uint i = lid; i < seq_len * embed_dim; i += tg_size.x) {
            uint pos = i / embed_dim;
            uint out_e = i % embed_dim;
            float sum = 0.0f;
            for (uint kk = 0; kk < embed_dim; kk++) {
                sum += ln1_out[pos * embed_dim + kk] * block_weights[wv_off + out_e * embed_dim + kk];
            }
            v[i] = sum;
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);

        // ====== Self-Attention ======
        threadgroup float* attn_out = ln1_out;  // Reuse ln1_out space
        float scale = 1.0f / sqrt(float(head_dim));

        for (uint i = lid; i < seq_len * num_heads; i += tg_size.x) {
            uint pos = i / num_heads;
            uint head = i % num_heads;
            uint head_start = head * head_dim;

            float scores[MAX_SEQ_LEN];
            float max_score = -INFINITY;

            for (uint k_pos = 0; k_pos < seq_len; k_pos++) {
                float dot = 0.0f;
                for (uint d = 0; d < head_dim; d++) {
                    dot += q[pos * embed_dim + head_start + d] *
                           k[k_pos * embed_dim + head_start + d];
                }
                scores[k_pos] = dot * scale;
                max_score = max(max_score, scores[k_pos]);
            }

            float sum_exp = 0.0f;
            for (uint k_pos = 0; k_pos < seq_len; k_pos++) {
                scores[k_pos] = exp(scores[k_pos] - max_score);
                sum_exp += scores[k_pos];
            }
            for (uint k_pos = 0; k_pos < seq_len; k_pos++) {
                scores[k_pos] /= sum_exp;
            }

            for (uint d = 0; d < head_dim; d++) {
                float sum = 0.0f;
                for (uint v_pos = 0; v_pos < seq_len; v_pos++) {
                    sum += scores[v_pos] * v[v_pos * embed_dim + head_start + d];
                }
                attn_out[pos * embed_dim + head_start + d] = sum;
            }
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);

        // Output projection
        threadgroup float* proj_out = q;  // Reuse q space
        for (uint i = lid; i < seq_len * embed_dim; i += tg_size.x) {
            uint pos = i / embed_dim;
            uint out_e = i % embed_dim;
            float sum = 0.0f;
            for (uint kk = 0; kk < embed_dim; kk++) {
                sum += attn_out[pos * embed_dim + kk] * block_weights[wo_off + out_e * embed_dim + kk];
            }
            proj_out[i] = sum;
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);

        // ====== Residual 1 ======
        for (uint i = lid; i < seq_len * embed_dim; i += tg_size.x) {
            proj_out[i] = residual[i] + proj_out[i];
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);

        // Store for residual 2
        for (uint i = lid; i < seq_len * embed_dim; i += tg_size.x) {
            residual[i] = proj_out[i];
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);

        // ====== LayerNorm 2 ======
        threadgroup float* ln2_out = attn_out;  // Reuse
        for (uint pos = lid; pos < seq_len; pos += tg_size.x) {
            float sum = 0.0f;
            for (uint e = 0; e < embed_dim; e++) {
                sum += proj_out[pos * embed_dim + e];
            }
            float mean = sum / float(embed_dim);

            float var_sum = 0.0f;
            for (uint e = 0; e < embed_dim; e++) {
                float diff = proj_out[pos * embed_dim + e] - mean;
                var_sum += diff * diff;
            }
            float inv_std = rsqrt(var_sum / float(embed_dim) + eps);

            for (uint e = 0; e < embed_dim; e++) {
                float x = proj_out[pos * embed_dim + e];
                float normed = (x - mean) * inv_std;
                ln2_out[pos * embed_dim + e] = normed * block_weights[ln2_gamma_off + e] +
                                                block_weights[ln2_beta_off + e];
            }
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);

        // ====== MLP FC1 + GELU ======
        threadgroup float* mlp_hidden = k;  // Reuse k space (need hidden_dim size, assuming hidden_dim <= embed_dim * 2)
        for (uint i = lid; i < seq_len * hidden_dim; i += tg_size.x) {
            uint pos = i / hidden_dim;
            uint h = i % hidden_dim;
            float sum = block_weights[fc1_b_off + h];
            for (uint kk = 0; kk < embed_dim; kk++) {
                sum += ln2_out[pos * embed_dim + kk] * block_weights[fc1_w_off + h * embed_dim + kk];
            }
            mlp_hidden[i] = gelu(sum);
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);

        // ====== MLP FC2 ======
        for (uint i = lid; i < seq_len * embed_dim; i += tg_size.x) {
            uint pos = i / embed_dim;
            uint out_e = i % embed_dim;
            float sum = block_weights[fc2_b_off + out_e];
            for (uint kk = 0; kk < hidden_dim; kk++) {
                sum += mlp_hidden[pos * hidden_dim + kk] * block_weights[fc2_w_off + out_e * hidden_dim + kk];
            }
            // Residual 2 and write to output buffer
            current_out[i] = residual[i] + sum;
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }

    // Write final output (from last buffer used)
    threadgroup float* final_buf = (num_blocks % 2 == 0) ? buf_a : buf_b;
    for (uint i = lid; i < seq_len * embed_dim; i += tg_size.x) {
        output[batch_offset + i] = final_buf[i];
    }
}

// ============================================================================
// Lightweight version for smaller models (less shared memory)
// ============================================================================

kernel void transformer_block_light(
    device const float* input [[buffer(0)]],
    device float* output [[buffer(1)]],
    device const float* weights [[buffer(2)]],
    constant uint& batch [[buffer(3)]],
    constant uint& seq_len [[buffer(4)]],
    constant uint& embed_dim [[buffer(5)]],
    constant uint& num_heads [[buffer(6)]],
    constant uint& hidden_dim [[buffer(7)]],
    constant float& eps [[buffer(8)]],
    uint3 gid [[threadgroup_position_in_grid]],
    uint lid [[thread_index_in_threadgroup]],
    uint3 tg_size [[threads_per_threadgroup]]
) {
    // Simplified version that uses device memory for intermediates
    // Good for when threadgroup memory is limited

    uint batch_idx = gid.x;
    uint pos = gid.y;  // Each threadgroup handles one position

    if (batch_idx >= batch || pos >= seq_len) return;

    uint head_dim = embed_dim / num_heads;
    uint batch_offset = batch_idx * seq_len * embed_dim + pos * embed_dim;

    // Weight offsets
    uint ln1_gamma_off = 0;
    uint ln1_beta_off = embed_dim;

    // Thread-local storage for this position's embedding
    float x[MAX_EMBED_DIM];

    // Load and compute LayerNorm for this position
    float sum = 0.0f;
    for (uint e = lid; e < embed_dim; e += tg_size.x) {
        x[e] = input[batch_offset - pos * embed_dim + pos * embed_dim + e];
        sum += x[e];
    }

    // Reduce sum across threads (simplified - in production use simd)
    // For now, single-threaded LayerNorm per position
    if (lid == 0) {
        sum = 0.0f;
        for (uint e = 0; e < embed_dim; e++) {
            x[e] = input[batch_offset - pos * embed_dim + pos * embed_dim + e];
            sum += x[e];
        }
        float mean = sum / float(embed_dim);

        float var_sum = 0.0f;
        for (uint e = 0; e < embed_dim; e++) {
            float diff = x[e] - mean;
            var_sum += diff * diff;
        }
        float inv_std = rsqrt(var_sum / float(embed_dim) + eps);

        for (uint e = 0; e < embed_dim; e++) {
            float normed = (x[e] - mean) * inv_std;
            output[batch_offset - pos * embed_dim + pos * embed_dim + e] =
                normed * weights[ln1_gamma_off + e] + weights[ln1_beta_off + e];
        }
    }
}
