// GPU Radix Sort for Gaussian Splatting tile binning
// Sorts (tile_id << 32 | depth) keys with associated gaussian IDs
// Uses parallel prefix sum for efficient scatter

#include <metal_stdlib>
using namespace metal;

// ============================================================================
// Constants
// ============================================================================

constant uint RADIX_BITS = 4;              // Bits per pass
constant uint NUM_BUCKETS = 16;            // 2^RADIX_BITS
constant uint BLOCK_SIZE = 256;            // Threads per block

// ============================================================================
// Local Histogram - Count digits per block
// ============================================================================

// Count occurrences of each digit (0-15) in block
kernel void radix_count_local(
    device const uint64_t* keys [[buffer(0)]],
    device uint* local_counts [[buffer(1)]],     // [num_blocks * 16]
    constant uint& n [[buffer(2)]],
    constant uint& bit_offset [[buffer(3)]],     // Which 4 bits to examine
    threadgroup uint* shared_counts [[threadgroup(0)]], // [16]
    uint gid [[thread_position_in_grid]],
    uint lid [[thread_index_in_threadgroup]],
    uint bid [[threadgroup_position_in_grid]],
    uint block_size [[threads_per_threadgroup]]
) {
    // Initialize shared counts
    if (lid < NUM_BUCKETS) {
        shared_counts[lid] = 0;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    // Count this thread's element
    if (gid < n) {
        uint64_t key = keys[gid];
        uint digit = (key >> bit_offset) & 0xF;
        atomic_fetch_add_explicit((threadgroup atomic_uint*)&shared_counts[digit], 1, memory_order_relaxed);
    }

    threadgroup_barrier(mem_flags::mem_threadgroup);

    // Write block counts to global memory
    if (lid < NUM_BUCKETS) {
        local_counts[bid * NUM_BUCKETS + lid] = shared_counts[lid];
    }
}

// ============================================================================
// Global Prefix Sum of Histograms
// ============================================================================

// Prefix sum over all block histograms for each digit
// Result: global_offsets[digit] = sum of counts for digit across all preceding blocks
kernel void radix_prefix_sum_histograms(
    device uint* local_counts [[buffer(0)]],     // [num_blocks * 16] in/out
    device uint* block_sums [[buffer(1)]],       // [16] total per digit
    constant uint& num_blocks [[buffer(2)]],
    threadgroup uint* shared [[threadgroup(0)]], // [2 * BLOCK_SIZE]
    uint gid [[thread_position_in_grid]],
    uint lid [[thread_index_in_threadgroup]],
    uint bid [[threadgroup_position_in_grid]]
) {
    // This kernel processes one digit (bid = digit index 0-15)
    // Scans local_counts for that digit across all blocks

    uint digit = bid;

    // Load counts for this digit from all blocks
    uint val = 0;
    if (gid < num_blocks) {
        val = local_counts[gid * NUM_BUCKETS + digit];
    }

    // Blelloch scan in shared memory
    uint ai = lid;
    uint bi = lid + BLOCK_SIZE;

    shared[ai] = (gid < num_blocks) ? val : 0;
    shared[bi] = (gid + BLOCK_SIZE < num_blocks) ? local_counts[(gid + BLOCK_SIZE) * NUM_BUCKETS + digit] : 0;

    uint offset = 1;

    // Up-sweep
    for (uint d = BLOCK_SIZE; d > 0; d >>= 1) {
        threadgroup_barrier(mem_flags::mem_threadgroup);
        if (lid < d) {
            uint ai_idx = offset * (2 * lid + 1) - 1;
            uint bi_idx = offset * (2 * lid + 2) - 1;
            shared[bi_idx] += shared[ai_idx];
        }
        offset *= 2;
    }

    // Store total and clear
    if (lid == 0) {
        block_sums[digit] = shared[2 * BLOCK_SIZE - 1];
        shared[2 * BLOCK_SIZE - 1] = 0;
    }

    // Down-sweep
    for (uint d = 1; d < 2 * BLOCK_SIZE; d *= 2) {
        offset >>= 1;
        threadgroup_barrier(mem_flags::mem_threadgroup);
        if (lid < d) {
            uint ai_idx = offset * (2 * lid + 1) - 1;
            uint bi_idx = offset * (2 * lid + 2) - 1;
            uint temp = shared[ai_idx];
            shared[ai_idx] = shared[bi_idx];
            shared[bi_idx] += temp;
        }
    }

    threadgroup_barrier(mem_flags::mem_threadgroup);

    // Write exclusive prefix sums back
    if (gid < num_blocks) {
        local_counts[gid * NUM_BUCKETS + digit] = shared[ai];
    }
    if (gid + BLOCK_SIZE < num_blocks) {
        local_counts[(gid + BLOCK_SIZE) * NUM_BUCKETS + digit] = shared[bi];
    }
}

// Add digit base offsets to all histogram entries
kernel void radix_add_digit_offsets(
    device uint* local_counts [[buffer(0)]],     // [num_blocks * 16]
    device const uint* digit_offsets [[buffer(1)]], // [16] scanned block_sums
    constant uint& num_blocks [[buffer(2)]],
    uint gid [[thread_position_in_grid]]
) {
    uint block_idx = gid / NUM_BUCKETS;
    uint digit = gid % NUM_BUCKETS;

    if (block_idx < num_blocks) {
        local_counts[gid] += digit_offsets[digit];
    }
}

// Simple prefix sum for digit offsets (16 elements, single thread)
kernel void radix_scan_digit_sums(
    device uint* block_sums [[buffer(0)]],       // [16] in: totals, out: offsets
    uint tid [[thread_position_in_grid]]
) {
    if (tid != 0) return;

    uint sum = 0;
    for (uint i = 0; i < NUM_BUCKETS; i++) {
        uint val = block_sums[i];
        block_sums[i] = sum;
        sum += val;
    }
}

// ============================================================================
// Scatter Elements
// ============================================================================

// Scatter keys and values to sorted positions
kernel void radix_scatter(
    device const uint64_t* keys_in [[buffer(0)]],
    device const uint* values_in [[buffer(1)]],
    device uint64_t* keys_out [[buffer(2)]],
    device uint* values_out [[buffer(3)]],
    device uint* local_counts [[buffer(4)]],     // [num_blocks * 16] - prefix sums
    constant uint& n [[buffer(5)]],
    constant uint& bit_offset [[buffer(6)]],
    threadgroup uint* shared_offsets [[threadgroup(0)]], // [16]
    threadgroup uint* shared_counts [[threadgroup(1)]],  // [16]
    uint gid [[thread_position_in_grid]],
    uint lid [[thread_index_in_threadgroup]],
    uint bid [[threadgroup_position_in_grid]],
    uint block_size [[threads_per_threadgroup]]
) {
    // Load block's starting offsets for each digit
    if (lid < NUM_BUCKETS) {
        shared_offsets[lid] = local_counts[bid * NUM_BUCKETS + lid];
        shared_counts[lid] = 0;  // Local counter for this block
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    if (gid < n) {
        uint64_t key = keys_in[gid];
        uint value = values_in[gid];
        uint digit = (key >> bit_offset) & 0xF;

        // Get local position within digit bucket for this block
        uint local_pos = atomic_fetch_add_explicit((threadgroup atomic_uint*)&shared_counts[digit], 1, memory_order_relaxed);

        // Global position = block's offset for this digit + local position
        uint global_pos = shared_offsets[digit] + local_pos;

        keys_out[global_pos] = key;
        values_out[global_pos] = value;
    }
}

// ============================================================================
// Single-Pass Radix Sort for Small Arrays (fits in threadgroup)
// ============================================================================

// Sort small arrays entirely within one threadgroup
kernel void radix_sort_local(
    device uint64_t* keys [[buffer(0)]],
    device uint* values [[buffer(1)]],
    constant uint& n [[buffer(2)]],
    threadgroup uint64_t* shared_keys [[threadgroup(0)]],  // [BLOCK_SIZE * 2]
    threadgroup uint* shared_values [[threadgroup(1)]],    // [BLOCK_SIZE * 2]
    threadgroup uint* shared_counts [[threadgroup(2)]],    // [16]
    threadgroup uint* shared_offsets [[threadgroup(3)]],   // [16]
    uint gid [[thread_position_in_grid]],
    uint lid [[thread_index_in_threadgroup]],
    uint block_size [[threads_per_threadgroup]]
) {
    // Load elements into shared memory
    uint idx = lid;
    if (idx < n) {
        shared_keys[idx] = keys[idx];
        shared_values[idx] = values[idx];
    } else {
        shared_keys[idx] = 0xFFFFFFFFFFFFFFFFULL;  // Sentinel (max uint64)
        shared_values[idx] = 0;
    }

    threadgroup_barrier(mem_flags::mem_threadgroup);

    // Sort all 64 bits, 4 bits at a time
    for (uint bit_offset = 0; bit_offset < 64; bit_offset += RADIX_BITS) {
        // Count digits
        if (lid < NUM_BUCKETS) {
            shared_counts[lid] = 0;
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);

        if (idx < n) {
            uint digit = (shared_keys[idx] >> bit_offset) & 0xF;
            atomic_fetch_add_explicit((threadgroup atomic_uint*)&shared_counts[digit], 1, memory_order_relaxed);
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);

        // Prefix sum of counts
        if (lid == 0) {
            uint sum = 0;
            for (uint i = 0; i < NUM_BUCKETS; i++) {
                uint val = shared_counts[i];
                shared_offsets[i] = sum;
                sum += val;
            }
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);

        // Scatter to temp positions
        if (idx < n) {
            uint64_t key = shared_keys[idx];
            uint value = shared_values[idx];
            uint digit = (key >> bit_offset) & 0xF;
            uint pos = atomic_fetch_add_explicit((threadgroup atomic_uint*)&shared_offsets[digit], 1, memory_order_relaxed);

            shared_keys[block_size + pos] = key;
            shared_values[block_size + pos] = value;
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);

        // Copy back
        if (idx < n) {
            shared_keys[idx] = shared_keys[block_size + idx];
            shared_values[idx] = shared_values[block_size + idx];
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }

    // Write results back
    if (idx < n) {
        keys[idx] = shared_keys[idx];
        values[idx] = shared_values[idx];
    }
}

// ============================================================================
// Compute Tile Ranges After Sorting
// ============================================================================

// Find start/end indices for each tile in sorted array
kernel void compute_tile_ranges(
    device const uint64_t* sorted_keys [[buffer(0)]],
    device uint* tile_ranges [[buffer(1)]],      // [num_tiles * 2] - start, end pairs
    constant uint& n [[buffer(2)]],
    constant uint& num_tiles [[buffer(3)]],
    uint gid [[thread_position_in_grid]]
) {
    if (gid >= n) return;

    // Get tile IDs for current and previous elements
    uint64_t key = sorted_keys[gid];
    uint tile_id = uint(key >> 32);

    uint prev_tile = (gid > 0) ? uint(sorted_keys[gid - 1] >> 32) : UINT_MAX;
    uint next_tile = (gid < n - 1) ? uint(sorted_keys[gid + 1] >> 32) : UINT_MAX;

    // Mark tile start
    if (tile_id != prev_tile && tile_id < num_tiles) {
        tile_ranges[tile_id * 2] = gid;  // start
    }

    // Mark tile end
    if (tile_id != next_tile && tile_id < num_tiles) {
        tile_ranges[tile_id * 2 + 1] = gid + 1;  // end (exclusive)
    }
}

// Initialize tile ranges to empty
kernel void init_tile_ranges(
    device uint* tile_ranges [[buffer(0)]],
    constant uint& num_tiles [[buffer(1)]],
    uint gid [[thread_position_in_grid]]
) {
    if (gid >= num_tiles) return;
    tile_ranges[gid * 2] = 0;      // start
    tile_ranges[gid * 2 + 1] = 0;  // end
}
