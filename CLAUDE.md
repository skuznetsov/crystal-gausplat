# CLAUDE.md — Crystal Gaussian Splatting + MASt3R

## Project Overview

**Goal:** 3D reconstruction from photos using MASt3R (dense stereo) + Gaussian Splatting, implemented in Crystal with Metal compute shaders for M2 Max.

**Target use case:** Scanning jewelry wax prototypes for casting — small objects (~5-30mm) requiring fine detail preservation.

**Platform:** macOS, Apple Silicon M2 Max 64GB unified memory

---

## Existing Infrastructure (from previous projects)

The following components already exist from weather prediction and protein folding projects:

### Crystal ML Framework
- [x] Tensor type with Metal buffer backend
- [x] Autograd with backward pass support
- [x] Custom backward for non-standard ops
- [x] AdamW optimizer
- [x] Metal FFI bindings
- [x] Compute shader dispatch infrastructure

### Available Metal Kernels (verify paths)
- [x] Matrix multiplication (matmul)
- [x] Attention mechanism (softmax, QKV projection)
- [x] Layer normalization
- [x] Elementwise ops (add, mul, gelu, silu)
- [x] Reduction ops (sum, mean, max)
- [x] Convolutions (likely from weather)
- [x] Scatter/gather operations (likely from protein)

### To Verify at Project Start
```crystal
# Check these exist and note their signatures:
# - src/metal/kernels/*.metal
# - src/tensor/autograd.cr
# - src/optim/adamw.cr
# - src/nn/attention.cr (if exists)
```

---

## Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                        Pipeline Overview                         │
├─────────────────────────────────────────────────────────────────┤
│                                                                  │
│  Photos (N images)                                               │
│       │                                                          │
│       ▼                                                          │
│  ┌─────────────┐                                                │
│  │   MASt3R    │  ViT encoder + decoder                         │
│  │  Inference  │  Output: dense point cloud + confidence        │
│  └──────┬──────┘                                                │
│         │                                                        │
│         ▼                                                        │
│  ┌─────────────┐                                                │
│  │  Gaussian   │  Points → 3D Gaussians                         │
│  │    Init     │  (position, covariance, opacity, SH)           │
│  └──────┬──────┘                                                │
│         │                                                        │
│         ▼                                                        │
│  ┌─────────────────────────────────────────┐                    │
│  │         Optimization Loop               │                    │
│  │  ┌─────────────────────────────────┐   │                    │
│  │  │  Rasterize (tile-based)         │   │                    │
│  │  │  → rendered image               │   │                    │
│  │  └───────────────┬─────────────────┘   │                    │
│  │                  │                      │                    │
│  │                  ▼                      │                    │
│  │  ┌─────────────────────────────────┐   │                    │
│  │  │  Loss (L1 + SSIM)               │   │                    │
│  │  └───────────────┬─────────────────┘   │                    │
│  │                  │                      │                    │
│  │                  ▼                      │                    │
│  │  ┌─────────────────────────────────┐   │                    │
│  │  │  Backward (rasterize_backward)  │   │                    │
│  │  └───────────────┬─────────────────┘   │                    │
│  │                  │                      │                    │
│  │                  ▼                      │                    │
│  │  ┌─────────────────────────────────┐   │                    │
│  │  │  AdamW update                   │   │                    │
│  │  └─────────────────────────────────┘   │                    │
│  └─────────────────────────────────────────┘                    │
│         │                                                        │
│         ▼                                                        │
│  ┌─────────────┐                                                │
│  │   Export    │  Marching cubes → STL/OBJ mesh                 │
│  └─────────────┘                                                │
│                                                                  │
└─────────────────────────────────────────────────────────────────┘
```

---

## Module Structure

```
src/
├── gaussian_splatting/
│   ├── gaussian.cr           # Gaussian3D struct
│   ├── camera.cr             # Camera model, projection
│   ├── rasterizer.cr         # Forward pass orchestration
│   ├── backward.cr           # Backward pass orchestration
│   ├── optimizer.cr          # Densification, pruning logic
│   └── scene.cr              # Scene management, I/O
│
├── mastr/
│   ├── model.cr              # MASt3R architecture
│   ├── weights.cr            # Weight loading (ONNX/safetensors)
│   ├── inference.cr          # Forward pass
│   └── point_cloud.cr        # Output processing
│
├── metal/kernels/
│   ├── gaussian_project.metal      # 3D → 2D projection
│   ├── tile_binning.metal          # Frustum cull + tile assign
│   ├── radix_sort.metal            # Per-tile depth sort
│   ├── rasterize_forward.metal     # Alpha compositing
│   ├── rasterize_backward.metal    # Gradient computation
│   ├── sh_eval.metal               # Spherical harmonics
│   └── covariance.metal            # Quaternion → cov matrix
│
├── export/
│   ├── marching_cubes.cr     # Mesh extraction
│   └── stl_writer.cr         # STL format output
│
└── utils/
    ├── image_io.cr           # PNG/JPEG loading
    ├── colmap_loader.cr      # Optional: COLMAP format
    └── metrics.cr            # PSNR, SSIM
```

---

## Implementation Phases

### Phase 1: Gaussian Core (Week 1)

**Goal:** Forward rasterizer that renders gaussians to image

#### 1.1 Data Structures

```crystal
# src/gaussian_splatting/gaussian.cr

struct Gaussian3D
  # Position (3)
  property position : Tensor  # [N, 3]
  
  # Covariance via scale + rotation (7)
  property scale : Tensor     # [N, 3] log-scale
  property rotation : Tensor  # [N, 4] quaternion (wxyz)
  
  # Appearance
  property opacity : Tensor   # [N, 1] logit
  property sh_coeffs : Tensor # [N, K, 3] spherical harmonics, K=16 for degree 3
  
  # All tensors require_grad for training
end

struct Camera
  property width : Int32
  property height : Int32
  property fx : Float32
  property fy : Float32
  property cx : Float32
  property cy : Float32
  property world_to_camera : Tensor  # [4, 4]
  property camera_to_world : Tensor  # [4, 4]
end
```

#### 1.2 Metal Kernels — Specifications

**Kernel: `compute_cov_3d`**
```metal
// Input:
//   scale: [N, 3] — log scale
//   rotation: [N, 4] — quaternion (w, x, y, z)
// Output:
//   cov3d: [N, 6] — upper triangular of 3x3 symmetric matrix

// Math:
// S = diag(exp(scale))
// R = quat_to_matrix(rotation)
// Σ = R @ S @ S^T @ R^T

kernel void compute_cov_3d(
    device const float* scale [[buffer(0)]],
    device const float* rotation [[buffer(1)]],
    device float* cov3d [[buffer(2)]],
    uint id [[thread_position_in_grid]]
);
```

**Kernel: `project_gaussians`**
```metal
// Input:
//   position: [N, 3]
//   cov3d: [N, 6]
//   viewmatrix: [4, 4]
//   projmatrix: [4, 4]  // full projection
//   focal_x, focal_y: float
//   tan_fovx, tan_fovy: float
//   image_width, image_height: int
// Output:
//   mean2d: [N, 2]
//   cov2d: [N, 3] — upper triangular 2x2
//   depth: [N]
//   radius: [N] — bounding circle in pixels
//   tiles_touched: [N] — count
//   clamped: [N] — bool, for gradient masking

// Math:
// p_cam = viewmatrix @ [position, 1]
// p_ndc = projmatrix @ p_cam
// J = jacobian of projection at p_cam
// cov2d = J @ cov3d @ J^T

kernel void project_gaussians(...);
```

**Kernel: `tile_binning`**
```metal
// Assigns each gaussian to tiles it overlaps
// Tile size: 16x16 pixels

// Input:
//   mean2d: [N, 2]
//   radius: [N]
//   depth: [N]
//   tiles_x, tiles_y: int
// Output:
//   keys: [M] — tile_id << 32 | depth_bits
//   values: [M] — gaussian index
//   tile_ranges: [tiles_x * tiles_y, 2] — start/end in sorted array
```

**Kernel: `rasterize_forward`**
```metal
// Per-tile rendering with alpha compositing

// Input:
//   sorted_keys, sorted_values: [M]
//   tile_ranges: [T, 2]
//   mean2d: [N, 2]
//   cov2d_inv: [N, 3]  // precomputed inverse
//   opacity: [N]
//   sh_colors: [N, 3]  // precomputed for this view
//   background: [3]
//   image_width, image_height: int
// Output:
//   out_image: [H, W, 3]
//   n_contrib: [H, W] — number of gaussians per pixel (for backward)
//   final_T: [H, W] — final transmittance (for backward)

kernel void rasterize_forward(...);
```

**Kernel: `rasterize_backward`**
```metal
// Compute gradients w.r.t. gaussian parameters

// Input:
//   dL_dout_image: [H, W, 3]
//   ... all forward inputs ...
//   ... n_contrib, final_T from forward ...
// Output:
//   dL_dmean2d: [N, 2]
//   dL_dcov2d: [N, 3]
//   dL_dopacity: [N]
//   dL_dsh_colors: [N, 3]

// Note: This is the most complex kernel.
// Requires storing per-pixel gaussian contributions or recomputing.
// Strategy: store auxiliary buffer during forward, or two-pass backward.

kernel void rasterize_backward(...);
```

#### 1.3 Tests for Phase 1

```crystal
# spec/gaussian_splatting/rasterizer_spec.cr

describe "GaussianRasterizer" do
  it "renders single gaussian as blob" do
    # Single gaussian at center
    # Expected: bright blob in center, falloff
  end
  
  it "handles depth ordering correctly" do
    # Two overlapping gaussians at different depths
    # Front one should occlude back one
  end
  
  it "gradient check: numerical vs analytical" do
    # Finite difference check for backward pass
    # This is critical for correctness
  end
end
```

---

### Phase 2: MASt3R Integration (Week 2)

**Goal:** Load MASt3R weights, run inference, get dense point cloud

#### 2.1 Weight Loading Strategy

**Option A: ONNX → Custom loader**
```bash
# Export from Python
python -c "
from mast3r.model import MASt3R
model = MASt3R.from_pretrained('naver/MASt3R_ViTLarge_BaseDecoder_512_catmlpdpt_metric')
torch.onnx.export(model, dummy_input, 'mast3r.onnx', opset_version=17)
"
```

**Option B: Safetensors direct load**
```crystal
# src/mastr/weights.cr

class MASt3RWeights
  def self.load(path : String) : Hash(String, Tensor)
    # Parse safetensors format
    # Map weight names to our layer names
  end
end
```

#### 2.2 Architecture Implementation

MASt3R is ViT encoder + cross-attention decoder. Core components:
- ViT blocks (already have attention)
- DPT-style decoder
- Regression heads for depth/confidence

```crystal
# src/mastr/model.cr

class MASt3R
  @encoder : ViTEncoder
  @decoder : DPTDecoder
  @depth_head : Linear
  @conf_head : Linear
  
  def forward(img1 : Tensor, img2 : Tensor) : {Tensor, Tensor, Tensor}
    # Returns: points3d, confidence, descriptors
  end
end
```

#### 2.3 Tests for Phase 2

```crystal
describe "MASt3R" do
  it "loads weights without error" do
    model = MASt3R.load("weights/mast3r.safetensors")
    model.should_not be_nil
  end
  
  it "produces reasonable depth for test pair" do
    # Use two known images
    # Check depth is in expected range
  end
end
```

---

### Phase 3: Training Loop (Week 3)

**Goal:** Optimize gaussians to match input images

#### 3.1 Loss Functions

```crystal
# src/gaussian_splatting/loss.cr

module Loss
  def self.l1(pred : Tensor, target : Tensor) : Tensor
    (pred - target).abs.mean
  end
  
  def self.ssim(pred : Tensor, target : Tensor, window_size = 11) : Tensor
    # Structural similarity
    # Requires gaussian blur kernel
  end
  
  def self.combined(pred : Tensor, target : Tensor, lambda_ssim = 0.2) : Tensor
    (1 - lambda_ssim) * l1(pred, target) + lambda_ssim * (1 - ssim(pred, target))
  end
end
```

#### 3.2 Adaptive Density Control

```crystal
# src/gaussian_splatting/optimizer.cr

class GaussianOptimizer
  # Clone gaussians with high gradient
  def densify_and_clone(gaussians : Gaussian3D, grad_threshold : Float32)
  end
  
  # Split large gaussians
  def densify_and_split(gaussians : Gaussian3D, scale_threshold : Float32)
  end
  
  # Remove transparent gaussians  
  def prune(gaussians : Gaussian3D, opacity_threshold : Float32)
  end
  
  # Reset opacity periodically
  def reset_opacity(gaussians : Gaussian3D)
  end
end
```

#### 3.3 Training Script

```crystal
# src/train.cr

def train(images : Array(Tensor), cameras : Array(Camera), iterations = 30000)
  # Initialize from MASt3R points
  points = MASt3R.infer(images[0], images[1])
  gaussians = Gaussian3D.from_points(points)
  
  optimizer = AdamW.new(gaussians.parameters, lr: 0.001)
  
  iterations.times do |i|
    camera = cameras.sample
    target = images[camera.index]
    
    rendered = Rasterizer.forward(gaussians, camera)
    loss = Loss.combined(rendered, target)
    
    loss.backward
    optimizer.step
    optimizer.zero_grad
    
    # Adaptive control every 100 iterations
    if i % 100 == 0 && i < 15000
      gaussians = densify_and_prune(gaussians)
    end
    
    # Log progress
    if i % 500 == 0
      puts "Iteration #{i}: loss=#{loss.item}, n_gaussians=#{gaussians.size}"
    end
  end
  
  gaussians
end
```

---

### Phase 4: Export & Polish (Week 4)

**Goal:** Mesh extraction, STL export for casting

#### 4.1 Marching Cubes

```crystal
# src/export/marching_cubes.cr

class MarchingCubes
  def self.extract(
    gaussians : Gaussian3D,
    resolution : Int32 = 256,
    threshold : Float32 = 0.5
  ) : Mesh
    # Sample density field on grid
    # Run marching cubes
    # Return vertices + faces
  end
end
```

#### 4.2 STL Export

```crystal
# src/export/stl_writer.cr

class STLWriter
  def self.write(mesh : Mesh, path : String, binary = true)
    # Binary STL format for casting software
  end
end
```

---

## Key Technical Decisions

### Memory Layout

All tensors use **row-major layout** matching Metal's default.
For gaussians: **Structure of Arrays (SoA)** not AoS for GPU efficiency.

### Precision

- Forward pass: **float32** (Metal doesn't benefit from fp16 on M2 for compute)
- Intermediate buffers: **float32**
- Gradients: **float32**

### Tile Size

**16x16 pixels** per tile (standard, good occupancy on Apple GPU)

### Spherical Harmonics

**Degree 3** (16 coefficients per color channel) — good quality/speed tradeoff

---

## Potential Blockers & Mitigations

| Risk | Mitigation |
|------|------------|
| MASt3R too slow on M2 | Use smaller ViT variant, or CoreML conversion |
| Numerical instability in backward | Add epsilon guards, gradient clipping |
| Wax texture too uniform | Add photometric augmentation, multi-light capture |
| Memory pressure on large scenes | Tile-based processing, streaming |

---

## Testing Strategy

1. **Unit tests** for each kernel (numerical gradient check)
2. **Integration test** with synthetic scene (known ground truth)
3. **Real test** with photographed object (visual inspection)
4. **Comparison** with Python original (PSNR/SSIM should match ±1%)

---

## References

- [3D Gaussian Splatting Paper](https://repo-sam.inria.fr/fungraph/3d-gaussian-splatting/)
- [Original CUDA Implementation](https://github.com/graphdeco-inria/gaussian-splatting)
- [MASt3R Paper](https://arxiv.org/abs/2406.09756)
- [MASt3R Code](https://github.com/naver/mast3r)

---

## Quick Start Commands

```bash
# Build
crystal build src/main.cr -o gsplat --release

# Test
crystal spec

# Train on images
./gsplat train --images ./photos/*.jpg --output scene.gs

# Export mesh
./gsplat export --input scene.gs --output model.stl --resolution 512
```

---

## Session Notes

*Add notes here during Claude Code sessions*

- [ ] Verify existing kernel inventory
- [ ] Confirm tensor autograd interface
- [ ] Test Metal dispatch latency
