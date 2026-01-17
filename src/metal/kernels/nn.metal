// Neural Network compute kernels for GPU acceleration
// Linear, LayerNorm, fused operations

#include <metal_stdlib>
using namespace metal;

// ============================================================================
// Constants
// ============================================================================

constant uint TILE_SIZE = 16;
constant float GELU_COEFF = 0.7978845608f;  // sqrt(2/pi)
constant float GELU_COEFF2 = 0.044715f;

// ============================================================================
// Linear Layer Forward
// ============================================================================

// Linear forward (tiled): out = input @ weight^T + bias
// input: [batch, in_features]
// weight: [out_features, in_features] (stored as [out, in], will be transposed)
// bias: [out_features]
// output: [batch, out_features]
kernel void linear_forward_tiled(
    device const float* input [[buffer(0)]],
    device const float* weight [[buffer(1)]],
    device const float* bias [[buffer(2)]],
    device float* output [[buffer(3)]],
    constant uint& batch [[buffer(4)]],
    constant uint& in_features [[buffer(5)]],
    constant uint& out_features [[buffer(6)]],
    constant uint& use_bias [[buffer(7)]],
    threadgroup float* tileA [[threadgroup(0)]],
    threadgroup float* tileB [[threadgroup(1)]],
    uint2 tid [[thread_position_in_threadgroup]],
    uint2 gid [[thread_position_in_grid]]
) {
    // out[b, o] = sum_i(input[b, i] * weight[o, i]) + bias[o]
    // This is: input[batch, in] @ weight^T[in, out] = output[batch, out]

    uint row = gid.y;  // batch index
    uint col = gid.x;  // output feature index

    float sum = 0.0f;
    uint numTiles = (in_features + TILE_SIZE - 1) / TILE_SIZE;

    for (uint t = 0; t < numTiles; t++) {
        // Load input tile: input[row, t*TILE_SIZE + tid.x]
        uint in_col = t * TILE_SIZE + tid.x;
        if (row < batch && in_col < in_features) {
            tileA[tid.y * TILE_SIZE + tid.x] = input[row * in_features + in_col];
        } else {
            tileA[tid.y * TILE_SIZE + tid.x] = 0.0f;
        }

        // Load weight tile (transposed): weight[col, t*TILE_SIZE + tid.y]
        uint w_row = t * TILE_SIZE + tid.y;
        if (col < out_features && w_row < in_features) {
            tileB[tid.y * TILE_SIZE + tid.x] = weight[col * in_features + w_row];
        } else {
            tileB[tid.y * TILE_SIZE + tid.x] = 0.0f;
        }

        threadgroup_barrier(mem_flags::mem_threadgroup);

        // Compute partial dot product
        for (uint k = 0; k < TILE_SIZE; k++) {
            sum += tileA[tid.y * TILE_SIZE + k] * tileB[k * TILE_SIZE + tid.x];
        }

        threadgroup_barrier(mem_flags::mem_threadgroup);
    }

    if (row < batch && col < out_features) {
        float result = sum;
        if (use_bias != 0) {
            result += bias[col];
        }
        output[row * out_features + col] = result;
    }
}

// Linear forward without tiling (for small matrices)
kernel void linear_forward_simple(
    device const float* input [[buffer(0)]],
    device const float* weight [[buffer(1)]],
    device const float* bias [[buffer(2)]],
    device float* output [[buffer(3)]],
    constant uint& batch [[buffer(4)]],
    constant uint& in_features [[buffer(5)]],
    constant uint& out_features [[buffer(6)]],
    constant uint& use_bias [[buffer(7)]],
    uint2 gid [[thread_position_in_grid]]
) {
    uint row = gid.y;  // batch index
    uint col = gid.x;  // output feature index

    if (row >= batch || col >= out_features) return;

    float sum = 0.0f;
    for (uint i = 0; i < in_features; i++) {
        sum += input[row * in_features + i] * weight[col * in_features + i];
    }

    if (use_bias != 0) {
        sum += bias[col];
    }

    output[row * out_features + col] = sum;
}

// ============================================================================
// Fused Linear + GELU
// ============================================================================

// GELU activation: 0.5 * x * (1 + tanh(sqrt(2/pi) * (x + 0.044715 * x^3)))
inline float gelu(float x) {
    float x3 = x * x * x;
    float inner = GELU_COEFF * (x + GELU_COEFF2 * x3);
    return 0.5f * x * (1.0f + tanh(inner));
}

// Fused linear + GELU: out = GELU(input @ weight^T + bias)
kernel void linear_gelu_forward(
    device const float* input [[buffer(0)]],
    device const float* weight [[buffer(1)]],
    device const float* bias [[buffer(2)]],
    device float* output [[buffer(3)]],
    constant uint& batch [[buffer(4)]],
    constant uint& in_features [[buffer(5)]],
    constant uint& out_features [[buffer(6)]],
    constant uint& use_bias [[buffer(7)]],
    threadgroup float* tileA [[threadgroup(0)]],
    threadgroup float* tileB [[threadgroup(1)]],
    uint2 tid [[thread_position_in_threadgroup]],
    uint2 gid [[thread_position_in_grid]]
) {
    uint row = gid.y;
    uint col = gid.x;

    float sum = 0.0f;
    uint numTiles = (in_features + TILE_SIZE - 1) / TILE_SIZE;

    for (uint t = 0; t < numTiles; t++) {
        uint in_col = t * TILE_SIZE + tid.x;
        if (row < batch && in_col < in_features) {
            tileA[tid.y * TILE_SIZE + tid.x] = input[row * in_features + in_col];
        } else {
            tileA[tid.y * TILE_SIZE + tid.x] = 0.0f;
        }

        uint w_row = t * TILE_SIZE + tid.y;
        if (col < out_features && w_row < in_features) {
            tileB[tid.y * TILE_SIZE + tid.x] = weight[col * in_features + w_row];
        } else {
            tileB[tid.y * TILE_SIZE + tid.x] = 0.0f;
        }

        threadgroup_barrier(mem_flags::mem_threadgroup);

        for (uint k = 0; k < TILE_SIZE; k++) {
            sum += tileA[tid.y * TILE_SIZE + k] * tileB[k * TILE_SIZE + tid.x];
        }

        threadgroup_barrier(mem_flags::mem_threadgroup);
    }

    if (row < batch && col < out_features) {
        if (use_bias != 0) {
            sum += bias[col];
        }
        output[row * out_features + col] = gelu(sum);
    }
}

// ============================================================================
// Fused Linear + ReLU
// ============================================================================

kernel void linear_relu_forward(
    device const float* input [[buffer(0)]],
    device const float* weight [[buffer(1)]],
    device const float* bias [[buffer(2)]],
    device float* output [[buffer(3)]],
    constant uint& batch [[buffer(4)]],
    constant uint& in_features [[buffer(5)]],
    constant uint& out_features [[buffer(6)]],
    constant uint& use_bias [[buffer(7)]],
    threadgroup float* tileA [[threadgroup(0)]],
    threadgroup float* tileB [[threadgroup(1)]],
    uint2 tid [[thread_position_in_threadgroup]],
    uint2 gid [[thread_position_in_grid]]
) {
    uint row = gid.y;
    uint col = gid.x;

    float sum = 0.0f;
    uint numTiles = (in_features + TILE_SIZE - 1) / TILE_SIZE;

    for (uint t = 0; t < numTiles; t++) {
        uint in_col = t * TILE_SIZE + tid.x;
        if (row < batch && in_col < in_features) {
            tileA[tid.y * TILE_SIZE + tid.x] = input[row * in_features + in_col];
        } else {
            tileA[tid.y * TILE_SIZE + tid.x] = 0.0f;
        }

        uint w_row = t * TILE_SIZE + tid.y;
        if (col < out_features && w_row < in_features) {
            tileB[tid.y * TILE_SIZE + tid.x] = weight[col * in_features + w_row];
        } else {
            tileB[tid.y * TILE_SIZE + tid.x] = 0.0f;
        }

        threadgroup_barrier(mem_flags::mem_threadgroup);

        for (uint k = 0; k < TILE_SIZE; k++) {
            sum += tileA[tid.y * TILE_SIZE + k] * tileB[k * TILE_SIZE + tid.x];
        }

        threadgroup_barrier(mem_flags::mem_threadgroup);
    }

    if (row < batch && col < out_features) {
        if (use_bias != 0) {
            sum += bias[col];
        }
        output[row * out_features + col] = max(0.0f, sum);
    }
}

// ============================================================================
// LayerNorm Forward
// ============================================================================

// Layer normalization with mean/var computation per sample
// input: [batch, features]
// gamma, beta: [features]
// output: [batch, features]
kernel void layernorm_forward(
    device const float* input [[buffer(0)]],
    device const float* gamma [[buffer(1)]],
    device const float* beta [[buffer(2)]],
    device float* output [[buffer(3)]],
    constant uint& batch [[buffer(4)]],
    constant uint& features [[buffer(5)]],
    constant float& eps [[buffer(6)]],
    uint batch_idx [[thread_position_in_grid]]
) {
    if (batch_idx >= batch) return;

    uint offset = batch_idx * features;

    // Compute mean
    float mean = 0.0f;
    for (uint f = 0; f < features; f++) {
        mean += input[offset + f];
    }
    mean /= float(features);

    // Compute variance
    float var = 0.0f;
    for (uint f = 0; f < features; f++) {
        float diff = input[offset + f] - mean;
        var += diff * diff;
    }
    var /= float(features);

    // Normalize and apply affine
    float inv_std = rsqrt(var + eps);
    for (uint f = 0; f < features; f++) {
        float normalized = (input[offset + f] - mean) * inv_std;
        output[offset + f] = normalized * gamma[f] + beta[f];
    }
}

// Parallel LayerNorm using threadgroup reduction for large feature dimensions
kernel void layernorm_parallel(
    device const float* input [[buffer(0)]],
    device const float* gamma [[buffer(1)]],
    device const float* beta [[buffer(2)]],
    device float* output [[buffer(3)]],
    constant uint& batch [[buffer(4)]],
    constant uint& features [[buffer(5)]],
    constant float& eps [[buffer(6)]],
    threadgroup float* shared_sum [[threadgroup(0)]],
    threadgroup float* shared_sq_sum [[threadgroup(1)]],
    uint batch_idx [[threadgroup_position_in_grid]],
    uint tid [[thread_index_in_threadgroup]],
    uint tg_size [[threads_per_threadgroup]]
) {
    if (batch_idx >= batch) return;

    uint offset = batch_idx * features;

    // Each thread computes partial sum over its assigned features
    float local_sum = 0.0f;
    for (uint f = tid; f < features; f += tg_size) {
        local_sum += input[offset + f];
    }
    shared_sum[tid] = local_sum;

    threadgroup_barrier(mem_flags::mem_threadgroup);

    // Parallel reduction for mean
    for (uint stride = tg_size / 2; stride > 0; stride /= 2) {
        if (tid < stride) {
            shared_sum[tid] += shared_sum[tid + stride];
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }

    float mean = shared_sum[0] / float(features);

    // Compute variance
    float local_sq_sum = 0.0f;
    for (uint f = tid; f < features; f += tg_size) {
        float diff = input[offset + f] - mean;
        local_sq_sum += diff * diff;
    }
    shared_sq_sum[tid] = local_sq_sum;

    threadgroup_barrier(mem_flags::mem_threadgroup);

    // Parallel reduction for variance
    for (uint stride = tg_size / 2; stride > 0; stride /= 2) {
        if (tid < stride) {
            shared_sq_sum[tid] += shared_sq_sum[tid + stride];
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }

    float var = shared_sq_sum[0] / float(features);
    float inv_std = rsqrt(var + eps);

    // Normalize and apply affine (each thread handles multiple features)
    for (uint f = tid; f < features; f += tg_size) {
        float normalized = (input[offset + f] - mean) * inv_std;
        output[offset + f] = normalized * gamma[f] + beta[f];
    }
}

// ============================================================================
// RMSNorm Forward
// ============================================================================

// RMS normalization: out = x / sqrt(mean(x^2) + eps) * gamma
kernel void rmsnorm_forward(
    device const float* input [[buffer(0)]],
    device const float* gamma [[buffer(1)]],
    device float* output [[buffer(2)]],
    constant uint& batch [[buffer(3)]],
    constant uint& features [[buffer(4)]],
    constant float& eps [[buffer(5)]],
    uint batch_idx [[thread_position_in_grid]]
) {
    if (batch_idx >= batch) return;

    uint offset = batch_idx * features;

    // Compute mean of squares
    float ms = 0.0f;
    for (uint f = 0; f < features; f++) {
        float x = input[offset + f];
        ms += x * x;
    }
    ms /= float(features);

    // Normalize and scale
    float inv_rms = rsqrt(ms + eps);
    for (uint f = 0; f < features; f++) {
        output[offset + f] = input[offset + f] * inv_rms * gamma[f];
    }
}

// ============================================================================
// Fused LayerNorm + Linear
// ============================================================================

// Fused: LayerNorm(input) @ weight^T + bias
// Saves one global memory read/write cycle
kernel void layernorm_linear_forward(
    device const float* input [[buffer(0)]],
    device const float* ln_gamma [[buffer(1)]],
    device const float* ln_beta [[buffer(2)]],
    device const float* weight [[buffer(3)]],
    device const float* bias [[buffer(4)]],
    device float* output [[buffer(5)]],
    constant uint& batch [[buffer(6)]],
    constant uint& in_features [[buffer(7)]],
    constant uint& out_features [[buffer(8)]],
    constant float& eps [[buffer(9)]],
    constant uint& use_bias [[buffer(10)]],
    uint2 gid [[thread_position_in_grid]]
) {
    // Simple version: each thread computes one output element
    uint row = gid.y;  // batch index
    uint col = gid.x;  // output feature index

    if (row >= batch || col >= out_features) return;

    uint offset = row * in_features;

    // Compute LayerNorm statistics (each thread does this for its batch element)
    float mean = 0.0f;
    for (uint f = 0; f < in_features; f++) {
        mean += input[offset + f];
    }
    mean /= float(in_features);

    float var = 0.0f;
    for (uint f = 0; f < in_features; f++) {
        float diff = input[offset + f] - mean;
        var += diff * diff;
    }
    var /= float(in_features);
    float inv_std = rsqrt(var + eps);

    // Compute linear projection with on-the-fly normalization
    float sum = 0.0f;
    for (uint i = 0; i < in_features; i++) {
        float normalized_val = (input[offset + i] - mean) * inv_std * ln_gamma[i] + ln_beta[i];
        sum += normalized_val * weight[col * in_features + i];
    }

    if (use_bias != 0) {
        sum += bias[col];
    }

    output[row * out_features + col] = sum;
}

// ============================================================================
// Linear Backward
// ============================================================================

// Linear backward for input gradient: grad_input = grad_output @ weight
kernel void linear_backward_input(
    device const float* grad_output [[buffer(0)]],
    device const float* weight [[buffer(1)]],
    device float* grad_input [[buffer(2)]],
    constant uint& batch [[buffer(3)]],
    constant uint& in_features [[buffer(4)]],
    constant uint& out_features [[buffer(5)]],
    threadgroup float* tileA [[threadgroup(0)]],
    threadgroup float* tileB [[threadgroup(1)]],
    uint2 tid [[thread_position_in_threadgroup]],
    uint2 gid [[thread_position_in_grid]]
) {
    // grad_input[b, i] = sum_o(grad_output[b, o] * weight[o, i])
    uint row = gid.y;  // batch index
    uint col = gid.x;  // in_features index

    float sum = 0.0f;
    uint numTiles = (out_features + TILE_SIZE - 1) / TILE_SIZE;

    for (uint t = 0; t < numTiles; t++) {
        // Load grad_output tile
        uint out_col = t * TILE_SIZE + tid.x;
        if (row < batch && out_col < out_features) {
            tileA[tid.y * TILE_SIZE + tid.x] = grad_output[row * out_features + out_col];
        } else {
            tileA[tid.y * TILE_SIZE + tid.x] = 0.0f;
        }

        // Load weight tile (not transposed this time)
        uint w_row = t * TILE_SIZE + tid.y;
        if (w_row < out_features && col < in_features) {
            tileB[tid.y * TILE_SIZE + tid.x] = weight[w_row * in_features + col];
        } else {
            tileB[tid.y * TILE_SIZE + tid.x] = 0.0f;
        }

        threadgroup_barrier(mem_flags::mem_threadgroup);

        for (uint k = 0; k < TILE_SIZE; k++) {
            sum += tileA[tid.y * TILE_SIZE + k] * tileB[k * TILE_SIZE + tid.x];
        }

        threadgroup_barrier(mem_flags::mem_threadgroup);
    }

    if (row < batch && col < in_features) {
        grad_input[row * in_features + col] = sum;
    }
}

// Linear backward for weight gradient: grad_weight = grad_output^T @ input
kernel void linear_backward_weight(
    device const float* grad_output [[buffer(0)]],
    device const float* input [[buffer(1)]],
    device float* grad_weight [[buffer(2)]],
    constant uint& batch [[buffer(3)]],
    constant uint& in_features [[buffer(4)]],
    constant uint& out_features [[buffer(5)]],
    threadgroup float* tileA [[threadgroup(0)]],
    threadgroup float* tileB [[threadgroup(1)]],
    uint2 tid [[thread_position_in_threadgroup]],
    uint2 gid [[thread_position_in_grid]]
) {
    // grad_weight[o, i] = sum_b(grad_output[b, o] * input[b, i])
    uint row = gid.y;  // out_features index
    uint col = gid.x;  // in_features index

    float sum = 0.0f;
    uint numTiles = (batch + TILE_SIZE - 1) / TILE_SIZE;

    for (uint t = 0; t < numTiles; t++) {
        // Load grad_output^T tile: grad_output[t*TILE_SIZE + tid.y, row]
        uint b_idx = t * TILE_SIZE + tid.y;
        if (b_idx < batch && row < out_features) {
            tileA[tid.y * TILE_SIZE + tid.x] = grad_output[b_idx * out_features + row];
        } else {
            tileA[tid.y * TILE_SIZE + tid.x] = 0.0f;
        }

        // Load input tile
        uint b_idx2 = t * TILE_SIZE + tid.y;
        if (b_idx2 < batch && col < in_features) {
            tileB[tid.y * TILE_SIZE + tid.x] = input[b_idx2 * in_features + col];
        } else {
            tileB[tid.y * TILE_SIZE + tid.x] = 0.0f;
        }

        threadgroup_barrier(mem_flags::mem_threadgroup);

        for (uint k = 0; k < TILE_SIZE; k++) {
            sum += tileA[k * TILE_SIZE + tid.y] * tileB[k * TILE_SIZE + tid.x];
        }

        threadgroup_barrier(mem_flags::mem_threadgroup);
    }

    if (row < out_features && col < in_features) {
        grad_weight[row * in_features + col] = sum;
    }
}

// Linear backward for bias gradient: grad_bias = sum(grad_output, dim=0)
kernel void linear_backward_bias(
    device const float* grad_output [[buffer(0)]],
    device float* grad_bias [[buffer(1)]],
    constant uint& batch [[buffer(2)]],
    constant uint& out_features [[buffer(3)]],
    threadgroup float* shared [[threadgroup(0)]],
    uint out_idx [[threadgroup_position_in_grid]],
    uint tid [[thread_index_in_threadgroup]],
    uint tg_size [[threads_per_threadgroup]]
) {
    if (out_idx >= out_features) return;

    // Each thread sums over part of the batch
    float local_sum = 0.0f;
    for (uint b = tid; b < batch; b += tg_size) {
        local_sum += grad_output[b * out_features + out_idx];
    }
    shared[tid] = local_sum;

    threadgroup_barrier(mem_flags::mem_threadgroup);

    // Parallel reduction
    for (uint stride = tg_size / 2; stride > 0; stride /= 2) {
        if (tid < stride) {
            shared[tid] += shared[tid + stride];
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }

    if (tid == 0) {
        grad_bias[out_idx] = shared[0];
    }
}

// ============================================================================
// Softmax (for attention)
// ============================================================================

// Online softmax per row with numerical stability
kernel void softmax_forward(
    device const float* input [[buffer(0)]],
    device float* output [[buffer(1)]],
    constant uint& rows [[buffer(2)]],
    constant uint& cols [[buffer(3)]],
    uint row [[thread_position_in_grid]]
) {
    if (row >= rows) return;

    uint offset = row * cols;

    // Find max for numerical stability
    float max_val = input[offset];
    for (uint c = 1; c < cols; c++) {
        max_val = max(max_val, input[offset + c]);
    }

    // Compute exp and sum
    float sum = 0.0f;
    for (uint c = 0; c < cols; c++) {
        float e = exp(input[offset + c] - max_val);
        output[offset + c] = e;
        sum += e;
    }

    // Normalize
    float inv_sum = 1.0f / sum;
    for (uint c = 0; c < cols; c++) {
        output[offset + c] *= inv_sum;
    }
}

// ============================================================================
// Batched Matrix Multiply (for attention QK^T and attn @ V)
// ============================================================================

// Batched matmul: C[b] = A[b] @ B[b]
// A: [batch, M, K], B: [batch, K, N], C: [batch, M, N]
kernel void batched_matmul(
    device const float* A [[buffer(0)]],
    device const float* B [[buffer(1)]],
    device float* C [[buffer(2)]],
    constant uint& batch [[buffer(3)]],
    constant uint& M [[buffer(4)]],
    constant uint& N [[buffer(5)]],
    constant uint& K [[buffer(6)]],
    threadgroup float* tileA [[threadgroup(0)]],
    threadgroup float* tileB [[threadgroup(1)]],
    uint3 tid [[thread_position_in_threadgroup]],
    uint3 gid [[thread_position_in_grid]]
) {
    uint b = gid.z;  // batch index
    uint row = gid.y;  // M index
    uint col = gid.x;  // N index

    if (b >= batch) return;

    uint A_offset = b * M * K;
    uint B_offset = b * K * N;
    uint C_offset = b * M * N;

    float sum = 0.0f;
    uint numTiles = (K + TILE_SIZE - 1) / TILE_SIZE;

    for (uint t = 0; t < numTiles; t++) {
        uint aCol = t * TILE_SIZE + tid.x;
        if (row < M && aCol < K) {
            tileA[tid.y * TILE_SIZE + tid.x] = A[A_offset + row * K + aCol];
        } else {
            tileA[tid.y * TILE_SIZE + tid.x] = 0.0f;
        }

        uint bRow = t * TILE_SIZE + tid.y;
        if (bRow < K && col < N) {
            tileB[tid.y * TILE_SIZE + tid.x] = B[B_offset + bRow * N + col];
        } else {
            tileB[tid.y * TILE_SIZE + tid.x] = 0.0f;
        }

        threadgroup_barrier(mem_flags::mem_threadgroup);

        for (uint k = 0; k < TILE_SIZE; k++) {
            sum += tileA[tid.y * TILE_SIZE + k] * tileB[k * TILE_SIZE + tid.x];
        }

        threadgroup_barrier(mem_flags::mem_threadgroup);
    }

    if (row < M && col < N) {
        C[C_offset + row * N + col] = sum;
    }
}

// Batched matmul with transpose: C[b] = A[b] @ B[b]^T
kernel void batched_matmul_tn(
    device const float* A [[buffer(0)]],
    device const float* B [[buffer(1)]],
    device float* C [[buffer(2)]],
    constant uint& batch [[buffer(3)]],
    constant uint& M [[buffer(4)]],
    constant uint& N [[buffer(5)]],
    constant uint& K [[buffer(6)]],
    constant float& scale [[buffer(7)]],
    uint3 gid [[thread_position_in_grid]]
) {
    // Simple implementation without tiling for clarity
    // A: [batch, M, K], B: [batch, N, K], C: [batch, M, N]
    // C = A @ B^T (each row of B is treated as a column)

    uint b = gid.z;
    uint row = gid.y;
    uint col = gid.x;

    if (b >= batch || row >= M || col >= N) return;

    uint A_offset = b * M * K;
    uint B_offset = b * N * K;
    uint C_offset = b * M * N;

    float sum = 0.0f;
    for (uint k = 0; k < K; k++) {
        sum += A[A_offset + row * K + k] * B[B_offset + col * K + k];
    }

    C[C_offset + row * N + col] = sum * scale;
}

// ============================================================================
// Fused Scaled Dot-Product Attention
// ============================================================================

// Fused attention: output = softmax(Q @ K^T / sqrt(d)) @ V
// Q, K, V: [batch * num_heads, seq_len, head_dim]
// Output: [batch * num_heads, seq_len, head_dim]
// This kernel fuses: matmul -> scale -> softmax -> matmul
kernel void fused_attention(
    device const float* Q [[buffer(0)]],
    device const float* K [[buffer(1)]],
    device const float* V [[buffer(2)]],
    device float* output [[buffer(3)]],
    constant uint& batch_heads [[buffer(4)]],  // batch * num_heads
    constant uint& seq_len [[buffer(5)]],
    constant uint& head_dim [[buffer(6)]],
    constant float& scale [[buffer(7)]],
    uint2 gid [[thread_position_in_grid]]  // (head_dim, batch_heads * seq_len)
) {
    uint bh_seq = gid.y;  // combined batch_heads and seq index
    uint d = gid.x;       // head_dim index

    if (d >= head_dim || bh_seq >= batch_heads * seq_len) return;

    uint bh = bh_seq / seq_len;  // batch_head index
    uint i = bh_seq % seq_len;   // query position

    uint Q_offset = bh * seq_len * head_dim;
    uint K_offset = bh * seq_len * head_dim;
    uint V_offset = bh * seq_len * head_dim;

    // Step 1: Compute attention scores for this query position
    // scores[j] = Q[i] @ K[j]^T * scale
    // Using online softmax to avoid storing all scores

    float max_score = -INFINITY;
    float sum_exp = 0.0f;
    float output_acc = 0.0f;

    // First pass: find max (for numerical stability)
    for (uint j = 0; j < seq_len; j++) {
        float score = 0.0f;
        for (uint k = 0; k < head_dim; k++) {
            score += Q[Q_offset + i * head_dim + k] * K[K_offset + j * head_dim + k];
        }
        score *= scale;
        max_score = max(max_score, score);
    }

    // Second pass: compute softmax and weighted sum of V
    for (uint j = 0; j < seq_len; j++) {
        // Recompute score (trade compute for memory)
        float score = 0.0f;
        for (uint k = 0; k < head_dim; k++) {
            score += Q[Q_offset + i * head_dim + k] * K[K_offset + j * head_dim + k];
        }
        score *= scale;

        float attn_weight = exp(score - max_score);
        sum_exp += attn_weight;

        // Accumulate weighted V[j][d]
        output_acc += attn_weight * V[V_offset + j * head_dim + d];
    }

    // Normalize
    output[bh * seq_len * head_dim + i * head_dim + d] = output_acc / sum_exp;
}

// Optimized fused attention using tiled approach for larger sequences
// One threadgroup handles one query position across all d dimensions
kernel void fused_attention_tiled(
    device const float* Q [[buffer(0)]],
    device const float* K [[buffer(1)]],
    device const float* V [[buffer(2)]],
    device float* output [[buffer(3)]],
    constant uint& batch_heads [[buffer(4)]],
    constant uint& seq_len [[buffer(5)]],
    constant uint& head_dim [[buffer(6)]],
    constant float& scale [[buffer(7)]],
    threadgroup float* shared_scores [[threadgroup(0)]],  // [seq_len] for attention scores
    threadgroup float* shared_v [[threadgroup(1)]],       // [head_dim] for partial V accumulation
    uint bh_i [[threadgroup_position_in_grid]],  // batch_head * seq_len + query_i
    uint tid [[thread_index_in_threadgroup]],
    uint tg_size [[threads_per_threadgroup]]
) {
    if (bh_i >= batch_heads * seq_len) return;

    uint bh = bh_i / seq_len;
    uint i = bh_i % seq_len;

    uint Q_offset = bh * seq_len * head_dim;
    uint K_offset = bh * seq_len * head_dim;
    uint V_offset = bh * seq_len * head_dim;

    // Step 1: Compute attention scores (parallelized over key positions)
    float local_max = -INFINITY;
    for (uint j = tid; j < seq_len; j += tg_size) {
        float score = 0.0f;
        for (uint k = 0; k < head_dim; k++) {
            score += Q[Q_offset + i * head_dim + k] * K[K_offset + j * head_dim + k];
        }
        score *= scale;
        shared_scores[j] = score;
        local_max = max(local_max, score);
    }

    // Reduce to find global max
    shared_v[tid] = local_max;
    threadgroup_barrier(mem_flags::mem_threadgroup);

    for (uint stride = tg_size / 2; stride > 0; stride /= 2) {
        if (tid < stride && tid + stride < tg_size) {
            shared_v[tid] = max(shared_v[tid], shared_v[tid + stride]);
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    float max_score = shared_v[0];
    threadgroup_barrier(mem_flags::mem_threadgroup);

    // Step 2: Compute softmax (exp and sum)
    float local_sum = 0.0f;
    for (uint j = tid; j < seq_len; j += tg_size) {
        float attn = exp(shared_scores[j] - max_score);
        shared_scores[j] = attn;  // Store attention weights
        local_sum += attn;
    }

    // Reduce sum
    shared_v[tid] = local_sum;
    threadgroup_barrier(mem_flags::mem_threadgroup);

    for (uint stride = tg_size / 2; stride > 0; stride /= 2) {
        if (tid < stride && tid + stride < tg_size) {
            shared_v[tid] += shared_v[tid + stride];
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    float sum_exp = shared_v[0];
    threadgroup_barrier(mem_flags::mem_threadgroup);

    // Normalize attention weights
    for (uint j = tid; j < seq_len; j += tg_size) {
        shared_scores[j] /= sum_exp;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    // Step 3: Compute output = attention @ V (parallelized over head_dim)
    for (uint d = tid; d < head_dim; d += tg_size) {
        float out_val = 0.0f;
        for (uint j = 0; j < seq_len; j++) {
            out_val += shared_scores[j] * V[V_offset + j * head_dim + d];
        }
        output[bh * seq_len * head_dim + i * head_dim + d] = out_val;
    }
}

// Flash attention style - memory efficient, single pass with online softmax
kernel void flash_attention(
    device const float* Q [[buffer(0)]],
    device const float* K [[buffer(1)]],
    device const float* V [[buffer(2)]],
    device float* output [[buffer(3)]],
    constant uint& batch_heads [[buffer(4)]],
    constant uint& seq_len [[buffer(5)]],
    constant uint& head_dim [[buffer(6)]],
    constant float& scale [[buffer(7)]],
    uint2 gid [[thread_position_in_grid]]  // (d, bh * seq_len)
) {
    uint bh_seq = gid.y;
    uint d = gid.x;

    if (d >= head_dim || bh_seq >= batch_heads * seq_len) return;

    uint bh = bh_seq / seq_len;
    uint i = bh_seq % seq_len;

    uint Q_offset = bh * seq_len * head_dim;
    uint K_offset = bh * seq_len * head_dim;
    uint V_offset = bh * seq_len * head_dim;

    // Online softmax with running statistics
    float max_score = -INFINITY;
    float sum_exp = 0.0f;
    float output_acc = 0.0f;

    for (uint j = 0; j < seq_len; j++) {
        // Compute score
        float score = 0.0f;
        for (uint k = 0; k < head_dim; k++) {
            score += Q[Q_offset + i * head_dim + k] * K[K_offset + j * head_dim + k];
        }
        score *= scale;

        // Online softmax update
        float new_max = max(max_score, score);
        float correction = exp(max_score - new_max);

        // Update running sum and output
        sum_exp = sum_exp * correction + exp(score - new_max);
        output_acc = output_acc * correction + exp(score - new_max) * V[V_offset + j * head_dim + d];

        max_score = new_max;
    }

    output[bh * seq_len * head_dim + i * head_dim + d] = output_acc / sum_exp;
}

// ============================================================================
// Fused LayerNorm + Linear
// ============================================================================

// Fused: output = Linear(LayerNorm(input))
// Saves one global memory round-trip
kernel void fused_layernorm_linear(
    device const float* input [[buffer(0)]],
    device const float* ln_gamma [[buffer(1)]],
    device const float* ln_beta [[buffer(2)]],
    device const float* weight [[buffer(3)]],
    device const float* bias [[buffer(4)]],
    device float* output [[buffer(5)]],
    constant uint& batch [[buffer(6)]],
    constant uint& in_features [[buffer(7)]],
    constant uint& out_features [[buffer(8)]],
    constant float& eps [[buffer(9)]],
    constant uint& use_bias [[buffer(10)]],
    uint2 gid [[thread_position_in_grid]]  // (out_features, batch)
) {
    uint row = gid.y;  // batch index
    uint col = gid.x;  // output feature index

    if (row >= batch || col >= out_features) return;

    uint in_offset = row * in_features;

    // Step 1: Compute LayerNorm statistics for this row
    float mean = 0.0f;
    for (uint f = 0; f < in_features; f++) {
        mean += input[in_offset + f];
    }
    mean /= float(in_features);

    float var = 0.0f;
    for (uint f = 0; f < in_features; f++) {
        float diff = input[in_offset + f] - mean;
        var += diff * diff;
    }
    var /= float(in_features);
    float inv_std = rsqrt(var + eps);

    // Step 2: Compute Linear with on-the-fly normalized input
    float sum = 0.0f;
    for (uint i = 0; i < in_features; i++) {
        float normalized = (input[in_offset + i] - mean) * inv_std;
        float ln_out = normalized * ln_gamma[i] + ln_beta[i];
        sum += ln_out * weight[col * in_features + i];
    }

    if (use_bias != 0) {
        sum += bias[col];
    }

    output[row * out_features + col] = sum;
}

// Fused LayerNorm + Linear + GELU
kernel void fused_layernorm_linear_gelu(
    device const float* input [[buffer(0)]],
    device const float* ln_gamma [[buffer(1)]],
    device const float* ln_beta [[buffer(2)]],
    device const float* weight [[buffer(3)]],
    device const float* bias [[buffer(4)]],
    device float* output [[buffer(5)]],
    constant uint& batch [[buffer(6)]],
    constant uint& in_features [[buffer(7)]],
    constant uint& out_features [[buffer(8)]],
    constant float& eps [[buffer(9)]],
    constant uint& use_bias [[buffer(10)]],
    uint2 gid [[thread_position_in_grid]]
) {
    uint row = gid.y;
    uint col = gid.x;

    if (row >= batch || col >= out_features) return;

    uint in_offset = row * in_features;

    // LayerNorm
    float mean = 0.0f;
    for (uint f = 0; f < in_features; f++) {
        mean += input[in_offset + f];
    }
    mean /= float(in_features);

    float var = 0.0f;
    for (uint f = 0; f < in_features; f++) {
        float diff = input[in_offset + f] - mean;
        var += diff * diff;
    }
    var /= float(in_features);
    float inv_std = rsqrt(var + eps);

    // Linear
    float sum = 0.0f;
    for (uint i = 0; i < in_features; i++) {
        float normalized = (input[in_offset + i] - mean) * inv_std;
        float ln_out = normalized * ln_gamma[i] + ln_beta[i];
        sum += ln_out * weight[col * in_features + i];
    }

    if (use_bias != 0) {
        sum += bias[col];
    }

    // GELU
    float x = sum;
    float x3 = x * x * x;
    float inner = GELU_COEFF * (x + GELU_COEFF2 * x3);
    output[row * out_features + col] = 0.5f * x * (1.0f + tanh(inner));
}

// ============================================================================
// Reshape utilities for attention
// ============================================================================

// Reshape [batch, seq, embed] -> [batch * heads, seq, head_dim]
kernel void reshape_for_heads(
    device const float* input [[buffer(0)]],
    device float* output [[buffer(1)]],
    constant uint& batch [[buffer(2)]],
    constant uint& seq_len [[buffer(3)]],
    constant uint& num_heads [[buffer(4)]],
    constant uint& head_dim [[buffer(5)]],
    uint3 gid [[thread_position_in_grid]]  // (head_dim, seq_len, batch * num_heads)
) {
    uint d = gid.x;
    uint s = gid.y;
    uint bh = gid.z;

    uint embed_dim = num_heads * head_dim;

    if (d >= head_dim || s >= seq_len || bh >= batch * num_heads) return;

    uint b = bh / num_heads;
    uint h = bh % num_heads;

    // Input index: [b, s, h * head_dim + d]
    uint in_idx = b * seq_len * embed_dim + s * embed_dim + h * head_dim + d;
    // Output index: [b * heads + h, s, d]
    uint out_idx = bh * seq_len * head_dim + s * head_dim + d;

    output[out_idx] = input[in_idx];
}

// Reshape [batch * heads, seq, head_dim] -> [batch, seq, embed]
kernel void reshape_from_heads(
    device const float* input [[buffer(0)]],
    device float* output [[buffer(1)]],
    constant uint& batch [[buffer(2)]],
    constant uint& seq_len [[buffer(3)]],
    constant uint& num_heads [[buffer(4)]],
    constant uint& head_dim [[buffer(5)]],
    uint3 gid [[thread_position_in_grid]]  // (embed_dim, seq_len, batch)
) {
    uint e = gid.x;  // embed_dim index
    uint s = gid.y;
    uint b = gid.z;

    uint embed_dim = num_heads * head_dim;

    if (e >= embed_dim || s >= seq_len || b >= batch) return;

    uint h = e / head_dim;
    uint d = e % head_dim;

    // Input index: [b * heads + h, s, d]
    uint in_idx = (b * num_heads + h) * seq_len * head_dim + s * head_dim + d;
    // Output index: [b, s, e]
    uint out_idx = b * seq_len * embed_dim + s * embed_dim + e;

    output[out_idx] = input[in_idx];
}
