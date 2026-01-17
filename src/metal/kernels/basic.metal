// Basic compute kernels for tensor operations
// Elementwise ops, reductions, matrix multiply

#include <metal_stdlib>
using namespace metal;

// ============================================================================
// Elementwise Operations
// ============================================================================

// Add two tensors: out = a + b
kernel void add(
    device const float* a [[buffer(0)]],
    device const float* b [[buffer(1)]],
    device float* out [[buffer(2)]],
    constant uint& count [[buffer(3)]],
    uint id [[thread_position_in_grid]]
) {
    if (id >= count) return;
    out[id] = a[id] + b[id];
}

// Add scalar: out = a + scalar
kernel void add_scalar(
    device const float* a [[buffer(0)]],
    device float* out [[buffer(1)]],
    constant float& scalar [[buffer(2)]],
    constant uint& count [[buffer(3)]],
    uint id [[thread_position_in_grid]]
) {
    if (id >= count) return;
    out[id] = a[id] + scalar;
}

// Subtract: out = a - b
kernel void sub(
    device const float* a [[buffer(0)]],
    device const float* b [[buffer(1)]],
    device float* out [[buffer(2)]],
    constant uint& count [[buffer(3)]],
    uint id [[thread_position_in_grid]]
) {
    if (id >= count) return;
    out[id] = a[id] - b[id];
}

// Multiply elementwise: out = a * b
kernel void mul(
    device const float* a [[buffer(0)]],
    device const float* b [[buffer(1)]],
    device float* out [[buffer(2)]],
    constant uint& count [[buffer(3)]],
    uint id [[thread_position_in_grid]]
) {
    if (id >= count) return;
    out[id] = a[id] * b[id];
}

// Multiply by scalar: out = a * scalar
kernel void mul_scalar(
    device const float* a [[buffer(0)]],
    device float* out [[buffer(1)]],
    constant float& scalar [[buffer(2)]],
    constant uint& count [[buffer(3)]],
    uint id [[thread_position_in_grid]]
) {
    if (id >= count) return;
    out[id] = a[id] * scalar;
}

// Divide: out = a / b
kernel void div_tensors(
    device const float* a [[buffer(0)]],
    device const float* b [[buffer(1)]],
    device float* out [[buffer(2)]],
    constant uint& count [[buffer(3)]],
    uint id [[thread_position_in_grid]]
) {
    if (id >= count) return;
    out[id] = a[id] / b[id];
}

// Negative: out = -a
kernel void neg(
    device const float* a [[buffer(0)]],
    device float* out [[buffer(1)]],
    constant uint& count [[buffer(2)]],
    uint id [[thread_position_in_grid]]
) {
    if (id >= count) return;
    out[id] = -a[id];
}

// Absolute value: out = |a|
kernel void abs_val(
    device const float* a [[buffer(0)]],
    device float* out [[buffer(1)]],
    constant uint& count [[buffer(2)]],
    uint id [[thread_position_in_grid]]
) {
    if (id >= count) return;
    out[id] = abs(a[id]);
}

// Square: out = a * a
kernel void square(
    device const float* a [[buffer(0)]],
    device float* out [[buffer(1)]],
    constant uint& count [[buffer(2)]],
    uint id [[thread_position_in_grid]]
) {
    if (id >= count) return;
    float val = a[id];
    out[id] = val * val;
}

// Square root: out = sqrt(a)
kernel void sqrt_val(
    device const float* a [[buffer(0)]],
    device float* out [[buffer(1)]],
    constant uint& count [[buffer(2)]],
    uint id [[thread_position_in_grid]]
) {
    if (id >= count) return;
    out[id] = sqrt(a[id]);
}

// Exponential: out = exp(a)
kernel void exp_val(
    device const float* a [[buffer(0)]],
    device float* out [[buffer(1)]],
    constant uint& count [[buffer(2)]],
    uint id [[thread_position_in_grid]]
) {
    if (id >= count) return;
    out[id] = exp(a[id]);
}

// Logarithm: out = log(a)
kernel void log_val(
    device const float* a [[buffer(0)]],
    device float* out [[buffer(1)]],
    constant uint& count [[buffer(2)]],
    uint id [[thread_position_in_grid]]
) {
    if (id >= count) return;
    out[id] = log(a[id]);
}

// Power: out = pow(a, p)
kernel void pow_val(
    device const float* a [[buffer(0)]],
    device float* out [[buffer(1)]],
    constant float& p [[buffer(2)]],
    constant uint& count [[buffer(3)]],
    uint id [[thread_position_in_grid]]
) {
    if (id >= count) return;
    out[id] = pow(a[id], p);
}

// Fill with constant: out = value
kernel void fill(
    device float* out [[buffer(0)]],
    constant float& value [[buffer(1)]],
    constant uint& count [[buffer(2)]],
    uint id [[thread_position_in_grid]]
) {
    if (id >= count) return;
    out[id] = value;
}

// Copy: out = a
kernel void copy(
    device const float* a [[buffer(0)]],
    device float* out [[buffer(1)]],
    constant uint& count [[buffer(2)]],
    uint id [[thread_position_in_grid]]
) {
    if (id >= count) return;
    out[id] = a[id];
}

// ============================================================================
// Activation Functions
// ============================================================================

// ReLU: out = max(0, a)
kernel void relu(
    device const float* a [[buffer(0)]],
    device float* out [[buffer(1)]],
    constant uint& count [[buffer(2)]],
    uint id [[thread_position_in_grid]]
) {
    if (id >= count) return;
    out[id] = max(0.0f, a[id]);
}

// ReLU backward: grad_in = grad_out * (a > 0)
kernel void relu_backward(
    device const float* a [[buffer(0)]],
    device const float* grad_out [[buffer(1)]],
    device float* grad_in [[buffer(2)]],
    constant uint& count [[buffer(3)]],
    uint id [[thread_position_in_grid]]
) {
    if (id >= count) return;
    grad_in[id] = a[id] > 0.0f ? grad_out[id] : 0.0f;
}

// Sigmoid: out = 1 / (1 + exp(-a))
kernel void sigmoid(
    device const float* a [[buffer(0)]],
    device float* out [[buffer(1)]],
    constant uint& count [[buffer(2)]],
    uint id [[thread_position_in_grid]]
) {
    if (id >= count) return;
    out[id] = 1.0f / (1.0f + exp(-a[id]));
}

// Sigmoid backward: grad_in = grad_out * out * (1 - out)
kernel void sigmoid_backward(
    device const float* out [[buffer(0)]],  // sigmoid output
    device const float* grad_out [[buffer(1)]],
    device float* grad_in [[buffer(2)]],
    constant uint& count [[buffer(3)]],
    uint id [[thread_position_in_grid]]
) {
    if (id >= count) return;
    float s = out[id];
    grad_in[id] = grad_out[id] * s * (1.0f - s);
}

// GELU: out = 0.5 * a * (1 + tanh(sqrt(2/pi) * (a + 0.044715 * a^3)))
kernel void gelu(
    device const float* a [[buffer(0)]],
    device float* out [[buffer(1)]],
    constant uint& count [[buffer(2)]],
    uint id [[thread_position_in_grid]]
) {
    if (id >= count) return;
    float x = a[id];
    float x3 = x * x * x;
    float inner = 0.7978845608f * (x + 0.044715f * x3);  // sqrt(2/pi) ≈ 0.7978845608
    out[id] = 0.5f * x * (1.0f + tanh(inner));
}

// GELU backward (approximate)
kernel void gelu_backward(
    device const float* a [[buffer(0)]],
    device const float* grad_out [[buffer(1)]],
    device float* grad_in [[buffer(2)]],
    constant uint& count [[buffer(3)]],
    uint id [[thread_position_in_grid]]
) {
    if (id >= count) return;
    float x = a[id];
    float x2 = x * x;
    float x3 = x2 * x;
    float inner = 0.7978845608f * (x + 0.044715f * x3);
    float tanh_inner = tanh(inner);
    float sech2 = 1.0f - tanh_inner * tanh_inner;
    float d_inner = 0.7978845608f * (1.0f + 0.134145f * x2);  // 3 * 0.044715 = 0.134145
    grad_in[id] = grad_out[id] * (0.5f * (1.0f + tanh_inner) + 0.5f * x * sech2 * d_inner);
}

// SiLU (Swish): out = a * sigmoid(a)
kernel void silu(
    device const float* a [[buffer(0)]],
    device float* out [[buffer(1)]],
    constant uint& count [[buffer(2)]],
    uint id [[thread_position_in_grid]]
) {
    if (id >= count) return;
    float x = a[id];
    float s = 1.0f / (1.0f + exp(-x));
    out[id] = x * s;
}

// Tanh: out = tanh(a)
kernel void tanh_val(
    device const float* a [[buffer(0)]],
    device float* out [[buffer(1)]],
    constant uint& count [[buffer(2)]],
    uint id [[thread_position_in_grid]]
) {
    if (id >= count) return;
    out[id] = tanh(a[id]);
}

// ============================================================================
// Reduction Operations (Single threadgroup for now, can be extended)
// ============================================================================

// Sum reduction
kernel void reduce_sum(
    device const float* input [[buffer(0)]],
    device float* output [[buffer(1)]],
    constant uint& count [[buffer(2)]],
    threadgroup float* shared [[threadgroup(0)]],
    uint tid [[thread_index_in_threadgroup]],
    uint tg_size [[threads_per_threadgroup]],
    uint gid [[threadgroup_position_in_grid]]
) {
    // Each thread loads and sums multiple elements
    float sum = 0.0f;
    for (uint i = tid; i < count; i += tg_size) {
        sum += input[i];
    }
    shared[tid] = sum;

    threadgroup_barrier(mem_flags::mem_threadgroup);

    // Parallel reduction in threadgroup
    for (uint stride = tg_size / 2; stride > 0; stride /= 2) {
        if (tid < stride) {
            shared[tid] += shared[tid + stride];
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }

    if (tid == 0) {
        output[gid] = shared[0];
    }
}

// Max reduction
kernel void reduce_max(
    device const float* input [[buffer(0)]],
    device float* output [[buffer(1)]],
    constant uint& count [[buffer(2)]],
    threadgroup float* shared [[threadgroup(0)]],
    uint tid [[thread_index_in_threadgroup]],
    uint tg_size [[threads_per_threadgroup]],
    uint gid [[threadgroup_position_in_grid]]
) {
    float max_val = -INFINITY;
    for (uint i = tid; i < count; i += tg_size) {
        max_val = max(max_val, input[i]);
    }
    shared[tid] = max_val;

    threadgroup_barrier(mem_flags::mem_threadgroup);

    for (uint stride = tg_size / 2; stride > 0; stride /= 2) {
        if (tid < stride) {
            shared[tid] = max(shared[tid], shared[tid + stride]);
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }

    if (tid == 0) {
        output[gid] = shared[0];
    }
}

// Min reduction
kernel void reduce_min(
    device const float* input [[buffer(0)]],
    device float* output [[buffer(1)]],
    constant uint& count [[buffer(2)]],
    threadgroup float* shared [[threadgroup(0)]],
    uint tid [[thread_index_in_threadgroup]],
    uint tg_size [[threads_per_threadgroup]],
    uint gid [[threadgroup_position_in_grid]]
) {
    float min_val = INFINITY;
    for (uint i = tid; i < count; i += tg_size) {
        min_val = min(min_val, input[i]);
    }
    shared[tid] = min_val;

    threadgroup_barrier(mem_flags::mem_threadgroup);

    for (uint stride = tg_size / 2; stride > 0; stride /= 2) {
        if (tid < stride) {
            shared[tid] = min(shared[tid], shared[tid + stride]);
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }

    if (tid == 0) {
        output[gid] = shared[0];
    }
}

// ============================================================================
// Matrix Multiplication
// ============================================================================

// Simple matmul: C[M,N] = A[M,K] @ B[K,N]
// Uses tiled approach with shared memory
constant uint TILE_SIZE = 16;

kernel void matmul(
    device const float* A [[buffer(0)]],
    device const float* B [[buffer(1)]],
    device float* C [[buffer(2)]],
    constant uint& M [[buffer(3)]],
    constant uint& N [[buffer(4)]],
    constant uint& K [[buffer(5)]],
    threadgroup float* tileA [[threadgroup(0)]],
    threadgroup float* tileB [[threadgroup(1)]],
    uint2 tid [[thread_position_in_threadgroup]],
    uint2 gid [[thread_position_in_grid]]
) {
    uint row = gid.y;
    uint col = gid.x;

    float sum = 0.0f;

    uint numTiles = (K + TILE_SIZE - 1) / TILE_SIZE;

    for (uint t = 0; t < numTiles; t++) {
        // Load A tile
        uint aRow = row;
        uint aCol = t * TILE_SIZE + tid.x;
        if (aRow < M && aCol < K) {
            tileA[tid.y * TILE_SIZE + tid.x] = A[aRow * K + aCol];
        } else {
            tileA[tid.y * TILE_SIZE + tid.x] = 0.0f;
        }

        // Load B tile
        uint bRow = t * TILE_SIZE + tid.y;
        uint bCol = col;
        if (bRow < K && bCol < N) {
            tileB[tid.y * TILE_SIZE + tid.x] = B[bRow * N + bCol];
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

    if (row < M && col < N) {
        C[row * N + col] = sum;
    }
}

// Matmul with alpha/beta: C = alpha * A @ B + beta * C
kernel void matmul_ab(
    device const float* A [[buffer(0)]],
    device const float* B [[buffer(1)]],
    device float* C [[buffer(2)]],
    constant uint& M [[buffer(3)]],
    constant uint& N [[buffer(4)]],
    constant uint& K [[buffer(5)]],
    constant float& alpha [[buffer(6)]],
    constant float& beta [[buffer(7)]],
    threadgroup float* tileA [[threadgroup(0)]],
    threadgroup float* tileB [[threadgroup(1)]],
    uint2 tid [[thread_position_in_threadgroup]],
    uint2 gid [[thread_position_in_grid]]
) {
    uint row = gid.y;
    uint col = gid.x;

    float sum = 0.0f;
    uint numTiles = (K + TILE_SIZE - 1) / TILE_SIZE;

    for (uint t = 0; t < numTiles; t++) {
        uint aRow = row;
        uint aCol = t * TILE_SIZE + tid.x;
        if (aRow < M && aCol < K) {
            tileA[tid.y * TILE_SIZE + tid.x] = A[aRow * K + aCol];
        } else {
            tileA[tid.y * TILE_SIZE + tid.x] = 0.0f;
        }

        uint bRow = t * TILE_SIZE + tid.y;
        uint bCol = col;
        if (bRow < K && bCol < N) {
            tileB[tid.y * TILE_SIZE + tid.x] = B[bRow * N + bCol];
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
        float c_val = C[row * N + col];
        C[row * N + col] = alpha * sum + beta * c_val;
    }
}

// ============================================================================
// Softmax
// ============================================================================

// Softmax along last dimension (row-wise for 2D)
// Two-pass: find max, then compute exp and sum
kernel void softmax_row(
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
// Layer Normalization
// ============================================================================

// LayerNorm: out = (x - mean) / sqrt(var + eps) * gamma + beta
kernel void layernorm(
    device const float* input [[buffer(0)]],
    device const float* gamma [[buffer(1)]],
    device const float* beta [[buffer(2)]],
    device float* output [[buffer(3)]],
    constant uint& batch_size [[buffer(4)]],
    constant uint& features [[buffer(5)]],
    constant float& eps [[buffer(6)]],
    uint batch_idx [[thread_position_in_grid]]
) {
    if (batch_idx >= batch_size) return;

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

// ============================================================================
// Utility Operations
// ============================================================================

// Clamp values to range
kernel void clamp_val(
    device const float* input [[buffer(0)]],
    device float* output [[buffer(1)]],
    constant float& min_val [[buffer(2)]],
    constant float& max_val [[buffer(3)]],
    constant uint& count [[buffer(4)]],
    uint id [[thread_position_in_grid]]
) {
    if (id >= count) return;
    output[id] = clamp(input[id], min_val, max_val);
}

// Where: out = condition ? a : b
kernel void where_val(
    device const float* condition [[buffer(0)]],
    device const float* a [[buffer(1)]],
    device const float* b [[buffer(2)]],
    device float* output [[buffer(3)]],
    constant uint& count [[buffer(4)]],
    uint id [[thread_position_in_grid]]
) {
    if (id >= count) return;
    output[id] = condition[id] > 0.5f ? a[id] : b[id];
}

// Transpose 2D: out[j,i] = in[i,j]
kernel void transpose_2d(
    device const float* input [[buffer(0)]],
    device float* output [[buffer(1)]],
    constant uint& rows [[buffer(2)]],
    constant uint& cols [[buffer(3)]],
    uint2 gid [[thread_position_in_grid]]
) {
    if (gid.x >= cols || gid.y >= rows) return;
    output[gid.x * rows + gid.y] = input[gid.y * cols + gid.x];
}
