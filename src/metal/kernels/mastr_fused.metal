//
// MASt3R Fused Metal Kernels
// Optimized for Apple Silicon (M2 Max)
//

#include <metal_stdlib>
using namespace metal;

// Constants
constant int TILE_SIZE = 16;
constant float GELU_COEF = 0.044715f;
constant float SQRT_2_PI = 0.7978845608f;

// ============================================================================
// Fused LayerNorm + QKV Projection
// Combines: normalize -> qkv_proj for attention
// Input: x [batch, seq, embed_dim]
// Output: qkv [batch, seq, 3 * embed_dim]
// ============================================================================
kernel void fused_layernorm_qkv(
    device const float* x [[buffer(0)]],
    device const float* ln_weight [[buffer(1)]],
    device const float* ln_bias [[buffer(2)]],
    device const float* qkv_weight [[buffer(3)]],
    device const float* qkv_bias [[buffer(4)]],
    device float* out [[buffer(5)]],
    constant int& batch [[buffer(6)]],
    constant int& seq_len [[buffer(7)]],
    constant int& embed_dim [[buffer(8)]],
    constant float& eps [[buffer(9)]],
    uint3 gid [[thread_position_in_grid]],
    uint3 tid [[thread_position_in_threadgroup]],
    uint3 tgsize [[threads_per_threadgroup]]
) {
    int b = gid.z;
    int s = gid.y;
    int d = gid.x;

    if (b >= batch || s >= seq_len || d >= 3 * embed_dim) return;

    int base_idx = b * seq_len * embed_dim + s * embed_dim;

    // Compute mean (parallel reduction would be faster for large embed_dim)
    float mean = 0.0f;
    for (int i = 0; i < embed_dim; i++) {
        mean += x[base_idx + i];
    }
    mean /= float(embed_dim);

    // Compute variance
    float var = 0.0f;
    for (int i = 0; i < embed_dim; i++) {
        float diff = x[base_idx + i] - mean;
        var += diff * diff;
    }
    var /= float(embed_dim);

    float inv_std = rsqrt(var + eps);

    // Compute QKV output
    float sum = 0.0f;
    int out_idx = d;  // Which output dimension (0 to 3*embed_dim-1)

    for (int i = 0; i < embed_dim; i++) {
        // Normalized input
        float x_norm = (x[base_idx + i] - mean) * inv_std;
        x_norm = x_norm * ln_weight[i] + ln_bias[i];

        // QKV projection: out[d] = sum_i(x_norm[i] * qkv_weight[d, i])
        sum += x_norm * qkv_weight[out_idx * embed_dim + i];
    }

    // Add bias
    sum += qkv_bias[out_idx];

    int out_base = b * seq_len * 3 * embed_dim + s * 3 * embed_dim;
    out[out_base + d] = sum;
}

// ============================================================================
// Fused Attention Score + Softmax
// Computes: scores = softmax(Q @ K^T / sqrt(d))
// With optional RoPE rotation
// ============================================================================
kernel void fused_attention_scores(
    device const float* q [[buffer(0)]],        // [batch, heads, seq, head_dim]
    device const float* k [[buffer(1)]],        // [batch, heads, seq, head_dim]
    device float* scores [[buffer(2)]],         // [batch, heads, seq, seq]
    device const float* rope_cos [[buffer(3)]], // [seq, head_dim/2] or nullptr
    device const float* rope_sin [[buffer(4)]], // [seq, head_dim/2] or nullptr
    constant int& batch [[buffer(5)]],
    constant int& num_heads [[buffer(6)]],
    constant int& seq_len [[buffer(7)]],
    constant int& head_dim [[buffer(8)]],
    constant int& use_rope [[buffer(9)]],
    uint3 gid [[thread_position_in_grid]]
) {
    int b = gid.z / num_heads;
    int h = gid.z % num_heads;
    int i = gid.y;  // Query position
    int j = gid.x;  // Key position

    if (b >= batch || i >= seq_len || j >= seq_len) return;

    float scale = rsqrt(float(head_dim));

    // Base indices
    int q_base = b * num_heads * seq_len * head_dim + h * seq_len * head_dim + i * head_dim;
    int k_base = b * num_heads * seq_len * head_dim + h * seq_len * head_dim + j * head_dim;

    // Dot product with optional RoPE
    float dot = 0.0f;

    if (use_rope) {
        int half_dim = head_dim / 2;

        for (int d = 0; d < half_dim; d++) {
            // Apply RoPE rotation
            float cos_i = rope_cos[i * half_dim + d];
            float sin_i = rope_sin[i * half_dim + d];
            float cos_j = rope_cos[j * half_dim + d];
            float sin_j = rope_sin[j * half_dim + d];

            float q0 = q[q_base + d];
            float q1 = q[q_base + d + half_dim];
            float k0 = k[k_base + d];
            float k1 = k[k_base + d + half_dim];

            // Rotated q
            float q_rot0 = q0 * cos_i - q1 * sin_i;
            float q_rot1 = q0 * sin_i + q1 * cos_i;

            // Rotated k
            float k_rot0 = k0 * cos_j - k1 * sin_j;
            float k_rot1 = k0 * sin_j + k1 * cos_j;

            dot += q_rot0 * k_rot0 + q_rot1 * k_rot1;
        }
    } else {
        for (int d = 0; d < head_dim; d++) {
            dot += q[q_base + d] * k[k_base + d];
        }
    }

    int out_idx = b * num_heads * seq_len * seq_len + h * seq_len * seq_len + i * seq_len + j;
    scores[out_idx] = dot * scale;
}

// Softmax kernel (row-wise)
kernel void softmax_rows(
    device float* scores [[buffer(0)]],  // [batch * heads, seq, seq]
    constant int& rows [[buffer(1)]],     // batch * heads * seq
    constant int& cols [[buffer(2)]],     // seq
    uint gid [[thread_position_in_grid]]
) {
    if (gid >= uint(rows)) return;

    int base = gid * cols;

    // Find max for numerical stability
    float max_val = scores[base];
    for (int i = 1; i < cols; i++) {
        max_val = max(max_val, scores[base + i]);
    }

    // Exp and sum
    float sum = 0.0f;
    for (int i = 0; i < cols; i++) {
        scores[base + i] = exp(scores[base + i] - max_val);
        sum += scores[base + i];
    }

    // Normalize
    float inv_sum = 1.0f / sum;
    for (int i = 0; i < cols; i++) {
        scores[base + i] *= inv_sum;
    }
}

// ============================================================================
// Fused MLP: GELU(x @ W1 + b1) @ W2 + b2
// ============================================================================
kernel void fused_mlp(
    device const float* x [[buffer(0)]],
    device const float* w1 [[buffer(1)]],
    device const float* b1 [[buffer(2)]],
    device const float* w2 [[buffer(3)]],
    device const float* b2 [[buffer(4)]],
    device float* out [[buffer(5)]],
    constant int& batch_seq [[buffer(6)]],  // batch * seq
    constant int& in_dim [[buffer(7)]],
    constant int& hidden_dim [[buffer(8)]],
    constant int& out_dim [[buffer(9)]],
    uint2 gid [[thread_position_in_grid]],
    uint2 tid [[thread_position_in_threadgroup]],
    threadgroup float* shared [[threadgroup(0)]]
) {
    int row = gid.y;
    int col = gid.x;

    if (row >= batch_seq || col >= out_dim) return;

    int x_base = row * in_dim;

    // First layer: hidden = GELU(x @ W1 + b1)
    // We compute this on-the-fly for each output
    float sum = 0.0f;

    for (int h = 0; h < hidden_dim; h++) {
        // Compute hidden[h] = GELU(sum_i(x[i] * w1[h, i]) + b1[h])
        float hidden_val = 0.0f;
        for (int i = 0; i < in_dim; i++) {
            hidden_val += x[x_base + i] * w1[h * in_dim + i];
        }
        hidden_val += b1[h];

        // GELU activation
        float gelu = 0.5f * hidden_val * (1.0f + tanh(SQRT_2_PI * (hidden_val + GELU_COEF * hidden_val * hidden_val * hidden_val)));

        // Accumulate for second layer
        sum += gelu * w2[col * hidden_dim + h];
    }

    sum += b2[col];
    out[row * out_dim + col] = sum;
}

// ============================================================================
// Fused Cross-Attention for Decoder
// Q from decoder, K/V from encoder
// ============================================================================
kernel void fused_cross_attention(
    device const float* q [[buffer(0)]],        // [batch, seq_q, embed_dim]
    device const float* k [[buffer(1)]],        // [batch, seq_kv, embed_dim]
    device const float* v [[buffer(2)]],        // [batch, seq_kv, embed_dim]
    device const float* wq [[buffer(3)]],       // [embed_dim, embed_dim]
    device const float* wk [[buffer(4)]],       // [embed_dim, embed_dim]
    device const float* wv [[buffer(5)]],       // [embed_dim, embed_dim]
    device const float* wo [[buffer(6)]],       // [embed_dim, embed_dim]
    device float* out [[buffer(7)]],            // [batch, seq_q, embed_dim]
    constant int& batch [[buffer(8)]],
    constant int& seq_q [[buffer(9)]],
    constant int& seq_kv [[buffer(10)]],
    constant int& embed_dim [[buffer(11)]],
    constant int& num_heads [[buffer(12)]],
    uint3 gid [[thread_position_in_grid]]
) {
    int b = gid.z;
    int i = gid.y;  // Query position
    int d = gid.x;  // Output dimension

    if (b >= batch || i >= seq_q || d >= embed_dim) return;

    int head_dim = embed_dim / num_heads;
    int head = d / head_dim;
    int d_in_head = d % head_dim;

    float scale = rsqrt(float(head_dim));

    // Project query
    int q_base = b * seq_q * embed_dim + i * embed_dim;
    float q_proj = 0.0f;
    for (int j = 0; j < embed_dim; j++) {
        q_proj += q[q_base + j] * wq[d * embed_dim + j];
    }

    // Compute attention for this head position
    float weighted_sum = 0.0f;
    float attn_sum = 0.0f;
    float max_score = -INFINITY;

    // First pass: compute max for softmax stability
    for (int kv_pos = 0; kv_pos < seq_kv; kv_pos++) {
        int k_base = b * seq_kv * embed_dim + kv_pos * embed_dim;
        float score = 0.0f;

        for (int j = 0; j < head_dim; j++) {
            int kd = head * head_dim + j;
            float k_proj = 0.0f;
            for (int m = 0; m < embed_dim; m++) {
                k_proj += k[k_base + m] * wk[kd * embed_dim + m];
            }
            if (j == d_in_head) {
                score += q_proj * k_proj;
            }
        }
        score *= scale;
        max_score = max(max_score, score);
    }

    // Second pass: compute softmax and weighted sum
    for (int kv_pos = 0; kv_pos < seq_kv; kv_pos++) {
        int k_base = b * seq_kv * embed_dim + kv_pos * embed_dim;
        int v_base = b * seq_kv * embed_dim + kv_pos * embed_dim;

        float score = 0.0f;
        for (int j = 0; j < head_dim; j++) {
            int kd = head * head_dim + j;
            float k_proj = 0.0f;
            for (int m = 0; m < embed_dim; m++) {
                k_proj += k[k_base + m] * wk[kd * embed_dim + m];
            }
            if (j == d_in_head) {
                score += q_proj * k_proj;
            }
        }
        score *= scale;

        float attn = exp(score - max_score);
        attn_sum += attn;

        // V projection for this position
        float v_proj = 0.0f;
        for (int m = 0; m < embed_dim; m++) {
            v_proj += v[v_base + m] * wv[d * embed_dim + m];
        }

        weighted_sum += attn * v_proj;
    }

    // Normalize
    weighted_sum /= attn_sum;

    // Output projection
    float out_val = 0.0f;
    // Note: This is simplified - full impl would accumulate across heads
    out_val = weighted_sum;  // Apply wo in separate kernel for efficiency

    int out_idx = b * seq_q * embed_dim + i * embed_dim + d;
    out[out_idx] = out_val;
}

// ============================================================================
// RoPE (Rotary Position Embedding) Application
// ============================================================================
kernel void apply_rope_2d(
    device float* x [[buffer(0)]],              // [batch, seq, embed_dim] - modified in place
    device const float* freqs [[buffer(1)]],    // [max_seq, head_dim/2]
    constant int& batch [[buffer(2)]],
    constant int& height [[buffer(3)]],
    constant int& width [[buffer(4)]],
    constant int& embed_dim [[buffer(5)]],
    constant int& num_heads [[buffer(6)]],
    uint3 gid [[thread_position_in_grid]]
) {
    int b = gid.z;
    int pos = gid.y;
    int d = gid.x;

    int seq_len = height * width;
    if (b >= batch || pos >= seq_len || d >= embed_dim) return;

    int y = pos / width;
    int xi = pos % width;

    int head_dim = embed_dim / num_heads;
    int half_head_dim = head_dim / 2;

    int head = d / head_dim;
    int d_in_head = d % head_dim;

    int base_idx = b * seq_len * embed_dim + pos * embed_dim;

    if (d_in_head < half_head_dim) {
        // First half: rotate based on y coordinate
        float theta = float(y) * freqs[d_in_head];
        float cos_t = cos(theta);
        float sin_t = sin(theta);

        int idx0 = base_idx + d;
        int idx1 = base_idx + d + half_head_dim;

        float x0 = x[idx0];
        float x1 = x[idx1];

        x[idx0] = x0 * cos_t - x1 * sin_t;
        x[idx1] = x0 * sin_t + x1 * cos_t;
    }
    // Second half is handled by the first half's pair
}

// ============================================================================
// Fused Residual + LayerNorm
// out = LayerNorm(x + residual)
// ============================================================================
kernel void fused_residual_layernorm(
    device const float* x [[buffer(0)]],
    device const float* residual [[buffer(1)]],
    device const float* weight [[buffer(2)]],
    device const float* bias [[buffer(3)]],
    device float* out [[buffer(4)]],
    constant int& batch_seq [[buffer(5)]],
    constant int& dim [[buffer(6)]],
    constant float& eps [[buffer(7)]],
    uint2 gid [[thread_position_in_grid]]
) {
    int row = gid.y;
    int col = gid.x;

    if (row >= batch_seq || col >= dim) return;

    int base = row * dim;

    // Compute mean
    float mean = 0.0f;
    for (int i = 0; i < dim; i++) {
        mean += x[base + i] + residual[base + i];
    }
    mean /= float(dim);

    // Compute variance
    float var = 0.0f;
    for (int i = 0; i < dim; i++) {
        float val = x[base + i] + residual[base + i] - mean;
        var += val * val;
    }
    var /= float(dim);

    float inv_std = rsqrt(var + eps);

    // Normalize and output
    float val = x[base + col] + residual[base + col];
    val = (val - mean) * inv_std;
    out[base + col] = val * weight[col] + bias[col];
}
