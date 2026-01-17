// Gaussian Splatting compute kernels
// Covariance computation, projection, tile binning, rasterization

#include <metal_stdlib>
using namespace metal;

// ============================================================================
// Constants and Structures
// ============================================================================

constant float SH_C0 = 0.28209479177387814f;  // 1 / (2 * sqrt(pi))
constant float SH_C1 = 0.4886025119029199f;   // sqrt(3) / (2 * sqrt(pi))
constant float SH_C2[] = {
    1.0925484305920792f,   // sqrt(15) / (2 * sqrt(pi))
    -1.0925484305920792f,
    0.31539156525252005f,  // sqrt(5) / (4 * sqrt(pi))
    -1.0925484305920792f,
    0.5462742152960396f    // sqrt(15) / (4 * sqrt(pi))
};
constant float SH_C3[] = {
    -0.5900435899266435f,
    2.890611442640554f,
    -0.4570457994644658f,
    0.3731763325901154f,
    -0.4570457994644658f,
    1.445305721320277f,
    -0.5900435899266435f
};

// Tile size for rasterization
constant uint TILE_SIZE = 16;

// Gaussian parameters structure (for kernel input)
struct GaussianParams {
    uint count;
    uint sh_degree;
    float scale_modifier;
};

// Camera parameters
struct CameraParams {
    float4x4 view_matrix;      // world to camera
    float4x4 proj_matrix;      // full projection
    float fx, fy, cx, cy;
    int width, height;
    float tan_fov_x, tan_fov_y;
    float3 camera_position;
};

// Per-gaussian projected data
struct ProjectedGaussian {
    float2 mean2d;           // Screen position
    float3 cov2d;            // 2D covariance (upper triangle: xx, xy, yy)
    float depth;             // Depth for sorting
    float radius;            // Bounding radius in pixels
    uint tiles_touched;      // Number of tiles overlapped
    bool valid;              // Passed frustum culling
};

// ============================================================================
// Quaternion to Rotation Matrix
// ============================================================================

float3x3 quat_to_matrix(float4 q) {
    // q = (w, x, y, z)
    float w = q.x, x = q.y, y = q.z, z = q.w;

    float3x3 R;
    R[0][0] = 1.0f - 2.0f * (y*y + z*z);
    R[0][1] = 2.0f * (x*y - z*w);
    R[0][2] = 2.0f * (x*z + y*w);

    R[1][0] = 2.0f * (x*y + z*w);
    R[1][1] = 1.0f - 2.0f * (x*x + z*z);
    R[1][2] = 2.0f * (y*z - x*w);

    R[2][0] = 2.0f * (x*z - y*w);
    R[2][1] = 2.0f * (y*z + x*w);
    R[2][2] = 1.0f - 2.0f * (x*x + y*y);

    return R;
}

// ============================================================================
// Compute 3D Covariance from Scale and Rotation
// ============================================================================

// Input: scale [N,3] (actual scale, not log), rotation [N,4] (quaternion)
// Output: cov3d [N,6] (upper triangular of symmetric 3x3)
kernel void compute_cov3d(
    device const float* scale [[buffer(0)]],
    device const float* rotation [[buffer(1)]],
    device float* cov3d [[buffer(2)]],
    constant uint& count [[buffer(3)]],
    uint id [[thread_position_in_grid]]
) {
    if (id >= count) return;

    // Load scale (already exp'd in Crystal)
    float3 s = float3(scale[id * 3], scale[id * 3 + 1], scale[id * 3 + 2]);

    // Load quaternion
    float4 q = float4(rotation[id * 4], rotation[id * 4 + 1],
                      rotation[id * 4 + 2], rotation[id * 4 + 3]);

    // Rotation matrix
    float3x3 R = quat_to_matrix(q);

    // Scale matrix S = diag(s)
    // Covariance Σ = R * S * S^T * R^T = R * diag(s²) * R^T
    float3 s2 = s * s;

    // Compute R * diag(s²) * R^T
    // This is sum over k of: R[:,k] * s²[k] * R[:,k]^T
    float3x3 cov;
    for (int i = 0; i < 3; i++) {
        for (int j = 0; j < 3; j++) {
            cov[i][j] = R[i][0] * s2.x * R[j][0] +
                        R[i][1] * s2.y * R[j][1] +
                        R[i][2] * s2.z * R[j][2];
        }
    }

    // Store upper triangular: [xx, xy, xz, yy, yz, zz]
    cov3d[id * 6 + 0] = cov[0][0];
    cov3d[id * 6 + 1] = cov[0][1];
    cov3d[id * 6 + 2] = cov[0][2];
    cov3d[id * 6 + 3] = cov[1][1];
    cov3d[id * 6 + 4] = cov[1][2];
    cov3d[id * 6 + 5] = cov[2][2];
}

// ============================================================================
// Project Gaussians to 2D
// ============================================================================

// Compute 2D projection and covariance for each Gaussian
kernel void project_gaussians(
    device const float* position [[buffer(0)]],      // [N, 3]
    device const float* cov3d [[buffer(1)]],         // [N, 6]
    device float* mean2d [[buffer(2)]],              // [N, 2] output
    device float* cov2d [[buffer(3)]],               // [N, 3] output (xx, xy, yy)
    device float* depth [[buffer(4)]],               // [N] output
    device float* radius [[buffer(5)]],              // [N] output
    device uint* tiles_touched [[buffer(6)]],        // [N] output
    constant CameraParams& camera [[buffer(7)]],
    constant uint& count [[buffer(8)]],
    uint id [[thread_position_in_grid]]
) {
    if (id >= count) return;

    // Load position
    float3 pos = float3(position[id * 3], position[id * 3 + 1], position[id * 3 + 2]);

    // Transform to camera space
    float4 pos_h = float4(pos, 1.0f);
    float4 pos_cam = camera.view_matrix * pos_h;

    // Frustum culling: behind camera or too far
    if (pos_cam.z <= 0.01f || pos_cam.z > 1000.0f) {
        mean2d[id * 2] = -1000.0f;
        mean2d[id * 2 + 1] = -1000.0f;
        depth[id] = 0.0f;
        radius[id] = 0.0f;
        tiles_touched[id] = 0;
        return;
    }

    // Project to screen
    float x_ndc = pos_cam.x / pos_cam.z;
    float y_ndc = pos_cam.y / pos_cam.z;

    // Frustum culling: outside FOV
    if (abs(x_ndc) > camera.tan_fov_x * 1.3f || abs(y_ndc) > camera.tan_fov_y * 1.3f) {
        mean2d[id * 2] = -1000.0f;
        mean2d[id * 2 + 1] = -1000.0f;
        depth[id] = 0.0f;
        radius[id] = 0.0f;
        tiles_touched[id] = 0;
        return;
    }

    // Screen coordinates
    float px = camera.fx * x_ndc + camera.cx;
    float py = camera.fy * y_ndc + camera.cy;

    mean2d[id * 2] = px;
    mean2d[id * 2 + 1] = py;
    depth[id] = pos_cam.z;

    // Compute Jacobian of projection
    float z2 = pos_cam.z * pos_cam.z;
    float J00 = camera.fx / pos_cam.z;
    float J02 = -camera.fx * pos_cam.x / z2;
    float J11 = camera.fy / pos_cam.z;
    float J12 = -camera.fy * pos_cam.y / z2;

    // Load 3D covariance
    float cov3d_xx = cov3d[id * 6 + 0];
    float cov3d_xy = cov3d[id * 6 + 1];
    float cov3d_xz = cov3d[id * 6 + 2];
    float cov3d_yy = cov3d[id * 6 + 3];
    float cov3d_yz = cov3d[id * 6 + 4];
    float cov3d_zz = cov3d[id * 6 + 5];

    // Transform 3D covariance to camera space
    // cov_cam = V * cov3d * V^T where V is view rotation
    float3x3 V;
    V[0] = camera.view_matrix[0].xyz;
    V[1] = camera.view_matrix[1].xyz;
    V[2] = camera.view_matrix[2].xyz;

    float3x3 cov3d_mat;
    cov3d_mat[0] = float3(cov3d_xx, cov3d_xy, cov3d_xz);
    cov3d_mat[1] = float3(cov3d_xy, cov3d_yy, cov3d_yz);
    cov3d_mat[2] = float3(cov3d_xz, cov3d_yz, cov3d_zz);

    float3x3 cov_cam = V * cov3d_mat * transpose(V);

    // Project to 2D: cov2d = J * cov_cam[:2,:2] * J^T + low-pass filter
    // Simplified: take top-left 2x2 and apply Jacobian
    float cov2d_xx = J00 * J00 * cov_cam[0][0] + 2.0f * J00 * J02 * cov_cam[0][2] + J02 * J02 * cov_cam[2][2];
    float cov2d_xy = J00 * J11 * cov_cam[0][1] + J00 * J12 * cov_cam[0][2] + J02 * J11 * cov_cam[2][1] + J02 * J12 * cov_cam[2][2];
    float cov2d_yy = J11 * J11 * cov_cam[1][1] + 2.0f * J11 * J12 * cov_cam[1][2] + J12 * J12 * cov_cam[2][2];

    // Add low-pass filter for anti-aliasing (0.3 pixel blur)
    cov2d_xx += 0.3f;
    cov2d_yy += 0.3f;

    cov2d[id * 3] = cov2d_xx;
    cov2d[id * 3 + 1] = cov2d_xy;
    cov2d[id * 3 + 2] = cov2d_yy;

    // Compute bounding radius (3 sigma)
    float det = cov2d_xx * cov2d_yy - cov2d_xy * cov2d_xy;
    float trace = cov2d_xx + cov2d_yy;
    float lambda_max = 0.5f * (trace + sqrt(max(0.0f, trace * trace - 4.0f * det)));
    float r = ceil(3.0f * sqrt(lambda_max));
    radius[id] = r;

    // Count tiles touched
    int tile_min_x = max(0, int(px - r) / int(TILE_SIZE));
    int tile_max_x = min(int(camera.width / TILE_SIZE) - 1, int(px + r) / int(TILE_SIZE));
    int tile_min_y = max(0, int(py - r) / int(TILE_SIZE));
    int tile_max_y = min(int(camera.height / TILE_SIZE) - 1, int(py + r) / int(TILE_SIZE));

    uint num_tiles = (tile_max_x - tile_min_x + 1) * (tile_max_y - tile_min_y + 1);
    tiles_touched[id] = max(0u, num_tiles);
}

// ============================================================================
// Spherical Harmonics Evaluation
// ============================================================================

// Evaluate SH for view direction, output RGB color
kernel void eval_sh(
    device const float* sh_coeffs [[buffer(0)]],     // [N, 16, 3]
    device const float* position [[buffer(1)]],      // [N, 3]
    device float* colors [[buffer(2)]],              // [N, 3] output RGB
    constant float3& camera_pos [[buffer(3)]],
    constant uint& count [[buffer(4)]],
    constant uint& sh_degree [[buffer(5)]],
    uint id [[thread_position_in_grid]]
) {
    if (id >= count) return;

    // View direction (from gaussian to camera)
    float3 pos = float3(position[id * 3], position[id * 3 + 1], position[id * 3 + 2]);
    float3 dir = normalize(camera_pos - pos);

    float x = dir.x;
    float y = dir.y;
    float z = dir.z;

    float3 result = float3(0.0f);

    // DC term (degree 0)
    uint base = id * 16 * 3;
    result.x += SH_C0 * sh_coeffs[base + 0];
    result.y += SH_C0 * sh_coeffs[base + 1];
    result.z += SH_C0 * sh_coeffs[base + 2];

    if (sh_degree >= 1) {
        // Degree 1
        result.x += SH_C1 * (-y * sh_coeffs[base + 3] + z * sh_coeffs[base + 6] - x * sh_coeffs[base + 9]);
        result.y += SH_C1 * (-y * sh_coeffs[base + 4] + z * sh_coeffs[base + 7] - x * sh_coeffs[base + 10]);
        result.z += SH_C1 * (-y * sh_coeffs[base + 5] + z * sh_coeffs[base + 8] - x * sh_coeffs[base + 11]);
    }

    if (sh_degree >= 2) {
        // Degree 2
        float xx = x * x, yy = y * y, zz = z * z;
        float xy = x * y, yz = y * z, xz = x * z;

        uint offset = base + 12;  // Start of degree 2 coefficients
        result.x += SH_C2[0] * xy * sh_coeffs[offset + 0] +
                    SH_C2[1] * yz * sh_coeffs[offset + 3] +
                    SH_C2[2] * (2.0f * zz - xx - yy) * sh_coeffs[offset + 6] +
                    SH_C2[3] * xz * sh_coeffs[offset + 9] +
                    SH_C2[4] * (xx - yy) * sh_coeffs[offset + 12];
        // ... similar for y, z components (abbreviated)
    }

    // Clamp to valid range
    colors[id * 3 + 0] = max(0.0f, result.x + 0.5f);  // SH gives values around 0, shift to positive
    colors[id * 3 + 1] = max(0.0f, result.y + 0.5f);
    colors[id * 3 + 2] = max(0.0f, result.z + 0.5f);
}

// ============================================================================
// Tile Binning (Create Sorted Keys)
// ============================================================================

// Create (tile_id, depth) key pairs for sorting
kernel void create_tile_keys(
    device const float* mean2d [[buffer(0)]],
    device const float* depth [[buffer(1)]],
    device const float* radius [[buffer(2)]],
    device uint64_t* keys [[buffer(3)]],           // Output: (tile_id << 32) | depth_bits
    device uint* gaussian_ids [[buffer(4)]],       // Output: gaussian index for each key
    device atomic_uint* key_counter [[buffer(5)]], // Atomic counter for keys
    constant uint& count [[buffer(6)]],
    constant uint& tiles_x [[buffer(7)]],
    constant uint& tiles_y [[buffer(8)]],
    uint id [[thread_position_in_grid]]
) {
    if (id >= count) return;

    float px = mean2d[id * 2];
    float py = mean2d[id * 2 + 1];
    float r = radius[id];
    float d = depth[id];

    // Skip if culled
    if (r <= 0.0f || d <= 0.0f) return;

    // Compute tile range
    int tile_min_x = max(0, int(px - r) / int(TILE_SIZE));
    int tile_max_x = min(int(tiles_x) - 1, int(px + r) / int(TILE_SIZE));
    int tile_min_y = max(0, int(py - r) / int(TILE_SIZE));
    int tile_max_y = min(int(tiles_y) - 1, int(py + r) / int(TILE_SIZE));

    // Convert depth to sortable uint32
    uint depth_bits = as_type<uint>(d);

    // Add key for each overlapped tile
    for (int ty = tile_min_y; ty <= tile_max_y; ty++) {
        for (int tx = tile_min_x; tx <= tile_max_x; tx++) {
            uint tile_id = ty * tiles_x + tx;
            uint key_idx = atomic_fetch_add_explicit(key_counter, 1, memory_order_relaxed);

            // Key: high 32 bits = tile_id, low 32 bits = depth
            keys[key_idx] = (uint64_t(tile_id) << 32) | uint64_t(depth_bits);
            gaussian_ids[key_idx] = id;
        }
    }
}

// ============================================================================
// Rasterize Forward Pass
// ============================================================================

// Render one tile
kernel void rasterize_tile(
    device const uint64_t* sorted_keys [[buffer(0)]],
    device const uint* sorted_gaussian_ids [[buffer(1)]],
    device const uint* tile_ranges [[buffer(2)]],    // [tiles_x * tiles_y, 2] start/end
    device const float* mean2d [[buffer(3)]],
    device const float* cov2d [[buffer(4)]],         // [N, 3] inverse covariance
    device const float* opacity [[buffer(5)]],       // [N] actual opacity
    device const float* colors [[buffer(6)]],        // [N, 3] SH-evaluated color
    device float* out_image [[buffer(7)]],           // [H, W, 3]
    device uint* n_contrib [[buffer(8)]],            // [H, W] for backward
    device float* final_T [[buffer(9)]],             // [H, W] for backward
    constant uint& tiles_x [[buffer(10)]],
    constant uint& width [[buffer(11)]],
    constant uint& height [[buffer(12)]],
    constant float3& background [[buffer(13)]],
    uint2 tile_id [[threadgroup_position_in_grid]],
    uint2 local_id [[thread_position_in_threadgroup]]
) {
    uint tile_idx = tile_id.y * tiles_x + tile_id.x;
    uint pixel_x = tile_id.x * TILE_SIZE + local_id.x;
    uint pixel_y = tile_id.y * TILE_SIZE + local_id.y;

    if (pixel_x >= width || pixel_y >= height) return;

    float2 pixel_pos = float2(pixel_x + 0.5f, pixel_y + 0.5f);

    // Get range of gaussians for this tile
    uint start = tile_ranges[tile_idx * 2];
    uint end = tile_ranges[tile_idx * 2 + 1];

    // Initialize accumulation
    float3 color = float3(0.0f);
    float T = 1.0f;  // Transmittance
    uint contributor_count = 0;

    // Iterate over gaussians in front-to-back order
    for (uint i = start; i < end && T > 0.001f; i++) {
        uint gid = sorted_gaussian_ids[i];

        float2 mean = float2(mean2d[gid * 2], mean2d[gid * 2 + 1]);
        float2 d = pixel_pos - mean;

        // Load inverse covariance (precomputed)
        float cov_xx = cov2d[gid * 3];
        float cov_xy = cov2d[gid * 3 + 1];
        float cov_yy = cov2d[gid * 3 + 2];

        // Compute determinant and inverse
        float det = cov_xx * cov_yy - cov_xy * cov_xy;
        if (det <= 0.0f) continue;

        float inv_det = 1.0f / det;
        float inv_xx = cov_yy * inv_det;
        float inv_xy = -cov_xy * inv_det;
        float inv_yy = cov_xx * inv_det;

        // Gaussian exponent
        float power = -0.5f * (d.x * d.x * inv_xx + 2.0f * d.x * d.y * inv_xy + d.y * d.y * inv_yy);

        if (power > 0.0f) continue;  // Outside Gaussian support

        float alpha = min(0.99f, opacity[gid] * exp(power));
        if (alpha < 1.0f / 255.0f) continue;

        // Accumulate color
        float3 c = float3(colors[gid * 3], colors[gid * 3 + 1], colors[gid * 3 + 2]);
        color += T * alpha * c;
        T *= (1.0f - alpha);
        contributor_count++;
    }

    // Add background
    color += T * background;

    // Write output
    uint pixel_idx = pixel_y * width + pixel_x;
    out_image[pixel_idx * 3 + 0] = color.x;
    out_image[pixel_idx * 3 + 1] = color.y;
    out_image[pixel_idx * 3 + 2] = color.z;
    n_contrib[pixel_idx] = contributor_count;
    final_T[pixel_idx] = T;
}

// ============================================================================
// Rasterize Backward Pass
// ============================================================================

// Compute gradients w.r.t. gaussian parameters
kernel void rasterize_backward(
    device const float* dL_dout_image [[buffer(0)]],  // [H, W, 3] gradient of loss
    device const uint64_t* sorted_keys [[buffer(1)]],
    device const uint* sorted_gaussian_ids [[buffer(2)]],
    device const uint* tile_ranges [[buffer(3)]],
    device const float* mean2d [[buffer(4)]],
    device const float* cov2d [[buffer(5)]],
    device const float* opacity [[buffer(6)]],
    device const float* colors [[buffer(7)]],
    device const uint* n_contrib [[buffer(8)]],
    device const float* final_T [[buffer(9)]],
    device float* dL_dmean2d [[buffer(10)]],          // [N, 2] output
    device float* dL_dcov2d [[buffer(11)]],           // [N, 3] output
    device float* dL_dopacity [[buffer(12)]],         // [N] output
    device float* dL_dcolors [[buffer(13)]],          // [N, 3] output
    constant uint& tiles_x [[buffer(14)]],
    constant uint& width [[buffer(15)]],
    constant uint& height [[buffer(16)]],
    constant float3& background [[buffer(17)]],
    uint2 tile_id [[threadgroup_position_in_grid]],
    uint2 local_id [[thread_position_in_threadgroup]]
) {
    uint tile_idx = tile_id.y * tiles_x + tile_id.x;
    uint pixel_x = tile_id.x * TILE_SIZE + local_id.x;
    uint pixel_y = tile_id.y * TILE_SIZE + local_id.y;

    if (pixel_x >= width || pixel_y >= height) return;

    uint pixel_idx = pixel_y * width + pixel_x;
    float2 pixel_pos = float2(pixel_x + 0.5f, pixel_y + 0.5f);

    // Load gradient from loss
    float3 dL_dpixel = float3(
        dL_dout_image[pixel_idx * 3 + 0],
        dL_dout_image[pixel_idx * 3 + 1],
        dL_dout_image[pixel_idx * 3 + 2]
    );

    // Get range and stored values
    uint start = tile_ranges[tile_idx * 2];
    uint end = tile_ranges[tile_idx * 2 + 1];
    float T_final = final_T[pixel_idx];

    // Backward: accumulate gradients in reverse order
    float T = T_final;
    float3 accum_color = background * T_final;

    // Need to iterate forward to reconstruct intermediate T values
    // Then iterate backward for gradients
    // (Simplified version - full impl would store per-pixel contributor list)

    for (int i = (int)end - 1; i >= (int)start; i--) {
        uint gid = sorted_gaussian_ids[i];

        float2 mean = float2(mean2d[gid * 2], mean2d[gid * 2 + 1]);
        float2 d = pixel_pos - mean;

        // Same Gaussian evaluation as forward
        float cov_xx = cov2d[gid * 3];
        float cov_xy = cov2d[gid * 3 + 1];
        float cov_yy = cov2d[gid * 3 + 2];

        float det = cov_xx * cov_yy - cov_xy * cov_xy;
        if (det <= 0.0f) continue;

        float inv_det = 1.0f / det;
        float inv_xx = cov_yy * inv_det;
        float inv_xy = -cov_xy * inv_det;
        float inv_yy = cov_xx * inv_det;

        float power = -0.5f * (d.x * d.x * inv_xx + 2.0f * d.x * d.y * inv_xy + d.y * d.y * inv_yy);
        if (power > 0.0f) continue;

        float G = exp(power);
        float alpha = min(0.99f, opacity[gid] * G);
        if (alpha < 1.0f / 255.0f) continue;

        float3 c = float3(colors[gid * 3], colors[gid * 3 + 1], colors[gid * 3 + 2]);

        // Recover T before this gaussian
        float T_before = T / (1.0f - alpha);

        // Gradients
        // dL/d(color) = T_before * alpha * dL_dpixel
        float3 dL_dc = T_before * alpha * dL_dpixel;

        // dL/d(alpha) = T_before * (c - accum_color / T) * dL_dpixel
        float dL_dalpha = dot(T_before * (c - accum_color / T), dL_dpixel);

        // dL/d(opacity) = dL_dalpha * G
        // dL/d(G) = dL_dalpha * opacity
        float dL_dG = dL_dalpha * opacity[gid];
        float dL_dop = dL_dalpha * G;

        // dL/d(power) = dL_dG * G
        float dL_dpower = dL_dG * G;

        // dL/d(mean2d) via dL/d(d) and dL/d(power)
        // d(power)/d(d) = -(inv_xx * d.x + inv_xy * d.y, inv_xy * d.x + inv_yy * d.y)
        float2 dL_dd = -dL_dpower * float2(
            inv_xx * d.x + inv_xy * d.y,
            inv_xy * d.x + inv_yy * d.y
        );

        // Atomic add to gradients (thread safety)
        atomic_fetch_add_explicit((device atomic_float*)&dL_dmean2d[gid * 2], -dL_dd.x, memory_order_relaxed);
        atomic_fetch_add_explicit((device atomic_float*)&dL_dmean2d[gid * 2 + 1], -dL_dd.y, memory_order_relaxed);
        atomic_fetch_add_explicit((device atomic_float*)&dL_dopacity[gid], dL_dop, memory_order_relaxed);
        atomic_fetch_add_explicit((device atomic_float*)&dL_dcolors[gid * 3], dL_dc.x, memory_order_relaxed);
        atomic_fetch_add_explicit((device atomic_float*)&dL_dcolors[gid * 3 + 1], dL_dc.y, memory_order_relaxed);
        atomic_fetch_add_explicit((device atomic_float*)&dL_dcolors[gid * 3 + 2], dL_dc.z, memory_order_relaxed);

        // Update for next iteration
        T = T_before;
        accum_color = (accum_color - T_before * alpha * c);
    }
}
