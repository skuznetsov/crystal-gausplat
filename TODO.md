# 3D Scanner - Gaussian Splatting + MASt3R

## Project Status: PHASE 12 - GPU OPTIMIZATION 🚀

**Goal:** 3D reconstruction from photos using MASt3R (dense stereo) + Gaussian Splatting
**Platform:** Crystal + Metal compute shaders for M2 Max
**Tests:** 22 tests passing (including GPU benchmarks for Linear, LayerNorm, Attention, Marching Cubes, Radix Sort, Persistent Transformer)

---

## Completed Phases

### Phase 1: Core Infrastructure ✅
- [x] `src/core/buffer.cr` - MetalBuffer with RAII, BufferPool
- [x] `src/core/shape.cr` - Shape, Strides, ShapeOps
- [x] `src/core/tensor.cr` - Tensor class (shape, stride, buffer, DType)

### Phase 2: Metal Backend ✅
- [x] `src/metal/device.cr` - Metal device wrapper, command queue, pipeline cache
- [x] `src/metal/dispatch.cr` - Kernel launch wrapper, ComputeEncoder
- [x] `src/metal/bridge.mm` - Objective-C++ FFI implementation
- [x] `src/metal/kernels/basic.metal` - elementwise, reduction, matmul, activations
- [x] `src/metal/kernels/gaussian.metal` - covariance, projection, SH eval, rasterize fwd/bwd

### Phase 3: Autograd System ✅
- [x] `src/autograd/grad_fn.cr` - GradFn base class, common backward functions
- [x] `src/autograd/variable.cr` - Variable = Tensor + grad + grad_fn, backward()

### Phase 4: Optimizer ✅
- [x] `src/optim/adam.cr` - Adam/AdamW, SGD, learning rate schedulers

### Phase 5: Gaussian Splatting Core ✅
- [x] `src/gaussian_splatting/gaussian.cr` - Gaussian3D params, clone/concat/remove
- [x] `src/gaussian_splatting/camera.cr` - Camera intrinsics/extrinsics, COLMAP loader
- [x] `src/gaussian_splatting/rasterizer.cr` - Forward/backward orchestration
- [x] `src/gaussian_splatting/rasterizer_context.cr` - State storage for backward

### Phase 6: Training ✅
- [x] `src/ops/loss.cr` - L1, MSE, SSIM, combined loss, PSNR
- [x] `src/gaussian_splatting/trainer.cr` - Training loop with densification/pruning

### Phase 7: Entry Point ✅
- [x] `src/main.cr` - CLI with train/render/export/test commands
- [x] `shard.yml` - Crystal project config
- [x] `Makefile` - Build automation

---

## Additional Phases (Completed)

### Phase 8: Neural Network Layers (for MASt3R) ✅
- [x] `src/nn/linear.cr` - Linear layer (supports arbitrary batch dims)
- [x] `src/nn/layernorm.cr` - LayerNorm, RMSNorm modules
- [x] `src/nn/attention.cr` - Multi-head attention, Cross-attention
- [x] `src/nn/vit.cr` - PatchEmbedding, MLP, TransformerEncoderBlock, ViTEncoder

### Phase 9: MASt3R Integration ✅
- [x] `src/mastr/weights.cr` - Safetensors loader (FP16/BF16/FP32)
- [x] `src/mastr/encoder.cr` - ViT encoder with cross-attention
- [x] `src/mastr/decoder.cr` - DPT decoder with reassemble/fusion
- [x] `src/mastr/model.cr` - Full MASt3R model

### Phase 10: Export ✅
- [x] `src/export/marching_cubes.cr` - Mesh extraction (with lookup tables)
- [x] `src/export/stl.cr` - STL/OBJ/PLY format writers

### Phase 11: Polish ✅
- [x] Image loading (PNG/JPEG) - macOS ImageIO framework
- [x] COLMAP integration for camera poses (text + binary format)
- [x] Geometry utilities (Vec3, Kabsch, ICP, SVD, Distance Geometry)
- [x] End-to-end testing (16 tests)

---

## Phase 12: GPU Optimization ✅ (completed 2026-01-16)

### Quadrumvirate Analysis (2026-01-16)

**[CASSANDRA] Bottlenecks по приоритету:**
```
Current: Image → [CPU] MASt3R → [CPU] Gaussians → [GPU] Rasterize → [CPU] Mesh
                  ↑ SLOW         ↑ SLOW                              ↑ SLOW
```
1. NN на CPU (Linear, Attention, LayerNorm) — ~90% времени MASt3R
2. Marching Cubes на CPU — O(n³) grid sampling
3. Много мелких kernel launches — dispatch overhead
4. CPU↔GPU sync points

**[DAEDALUS] Упущенные возможности:**
- Metal matmul kernel существует (basic.metal) но НЕ используется в NN layers!
- Нет batch dispatching
- Нет kernel fusion (каждая операция = отдельный kernel)

**[MAIEUTIC] Bedrocks:**
- M2 Max: 38 GPU cores vs 12 CPU cores → GPU выигрывает на параллельных ops
- Unified memory (MTLResourceStorageModeShared) → zero-copy возможен
- Threadgroup memory ~100x быстрее device memory
- Kernel launch overhead ~5-10μs на Metal

**[ADVERSARY] Риски:**
- Малые батчи (batch<16): GPU overhead > benefit → нужен fallback
- FP16 vs FP32: точность может пострадать → нужны tolerance tests

### План оптимизации

#### Level 1: Quick Wins ✅ (completed 2026-01-16)
- [x] `src/metal/kernels/nn.metal` — GPU kernels для Linear, LayerNorm
- [x] `src/nn/gpu_ops.cr` — Crystal wrapper для GPU NN операций
- [x] Обновить `linear.cr` — использовать GPU matmul
- [x] Обновить `layernorm.cr` — использовать GPU kernel
- [x] Benchmark: GPU Linear ~1.2ms/iter (256x512→256), LayerNorm ~1.3ms/iter (256x256)

#### Level 2: Kernel Fusion ✅ (completed 2026-01-16)
- [x] Fused Linear+GELU kernel (implemented in nn.metal)
- [x] Fused Attention: `flash_attention` kernel (online softmax, Q×K^T → softmax → ×V)
- [x] Fused LayerNorm+Linear: `fused_layernorm_linear`, `fused_layernorm_linear_gelu`
- [x] GPU reshape utilities for attention: `reshape_for_heads`, `reshape_from_heads`
- [x] Benchmark: GPU Attention ~1.6ms/iter (2 batch, 4 heads, seq=32)

#### Level 3: Advanced ✅ (completed 2026-01-16)
- [x] GPU Marching Cubes (parallel prefix sum in marching_cubes.metal)
- [x] GPU MC Crystal wrapper (gpu_marching_cubes.cr)
- [x] Benchmark: GPU MC ~12ms vs CPU ~34ms at res=64 (2.7x speedup)
- [x] GPU radix sort for tile binning (radix_sort.metal + gpu_radix_sort.cr)
- [x] Benchmark: GPU radix sort ~15ms for 64K keys
- [x] Persistent threadgroups for Transformer (persistent_transformer.metal + persistent_transformer.cr)
- [x] Benchmark: ~35ms/block for batch=2, seq=16, dim=64; kernel overhead reduced from ~0.8ms to ~0ms per 12 blocks

### Target Architecture
```
Image → [GPU] MASt3R → [GPU] Gaussians → [GPU] Rasterize → [GPU] Mesh
         ↑ FAST          ↑ FAST            ↑ FAST           ↑ FAST

Fused kernels:
┌─────────────────────────────────────────────────────────┐
│ linear_gelu_fused    │ attention_fused │ layernorm_linear│
│ (weight×input+bias   │ (QK^T→softmax   │ (norm→project   │
│  →GELU in one pass)  │  →×V one pass)  │  one pass)      │
└─────────────────────────────────────────────────────────┘
```

---

## File Inventory (39 files created)

| File | Lines | Description |
|------|-------|-------------|
| `src/core/buffer.cr` | ~220 | MetalBuffer RAII, BufferPool, raw bytes I/O |
| `src/core/shape.cr` | ~200 | Shape, Strides, broadcasting |
| `src/core/tensor.cr` | ~350 | Tensor class, device transfer |
| `src/metal/device.cr` | ~250 | Metal device, pipeline cache |
| `src/metal/dispatch.cr` | ~280 | ComputeEncoder, kernel launch, threadgroup memory |
| `src/metal/bridge.mm` | ~470 | Objective-C++ FFI |
| `src/metal/gpu_radix_sort.cr` | ~240 | GPU Radix Sort for tile binning |
| `src/metal/kernels/basic.metal` | ~500 | Elementwise, matmul, activations |
| `src/metal/kernels/gaussian.metal` | ~450 | Covariance, projection, rasterize |
| `src/metal/kernels/nn.metal` | ~1090 | Linear, LayerNorm, Attention, fused ops (GPU) |
| `src/metal/kernels/marching_cubes.metal` | ~600 | GPU Marching Cubes, prefix sum |
| `src/metal/kernels/radix_sort.metal` | ~325 | GPU Radix Sort kernels |
| `src/metal/kernels/persistent_transformer.metal` | ~630 | Persistent threadgroups transformer |
| `src/metal/persistent_transformer.cr` | ~180 | Persistent transformer Crystal wrapper |
| `src/autograd/grad_fn.cr` | ~250 | Backward functions |
| `src/autograd/variable.cr` | ~350 | Autograd Variable |
| `src/optim/adam.cr` | ~300 | Adam/AdamW, SGD, schedulers |
| `src/ops/loss.cr` | ~250 | L1, MSE, SSIM, PSNR |
| `src/gaussian_splatting/gaussian.cr` | ~250 | Gaussian3D struct |
| `src/gaussian_splatting/camera.cr` | ~300 | Camera model, COLMAP |
| `src/gaussian_splatting/rasterizer.cr` | ~400 | Rasterizer forward/backward |
| `src/gaussian_splatting/rasterizer_context.cr` | ~150 | Context for backward |
| `src/gaussian_splatting/trainer.cr` | ~350 | Training loop |
| `src/nn/gpu_ops.cr` | ~475 | GPU dispatch wrapper for NN ops (fused attention, LN+Linear) |
| `src/nn/linear.cr` | ~300 | Linear layer, GPU/CPU matmul |
| `src/nn/layernorm.cr` | ~330 | LayerNorm, RMSNorm with GPU |
| `src/nn/attention.cr` | ~340 | MultiHeadAttention, CrossAttention (GPU fused) |
| `src/nn/vit.cr` | ~300 | PatchEmbedding, MLP, ViTEncoder |
| `src/mastr/weights.cr` | ~240 | Safetensors loader (FP16/BF16/FP32) |
| `src/mastr/encoder.cr` | ~300 | MASt3R encoder with cross-attention |
| `src/mastr/decoder.cr` | ~350 | DPT decoder with reassemble/fusion |
| `src/mastr/model.cr` | ~200 | Full MASt3R model |
| `src/export/marching_cubes.cr` | ~610 | Marching cubes mesh extraction |
| `src/export/gpu_marching_cubes.cr` | ~410 | GPU Marching Cubes wrapper |
| `src/export/stl.cr` | ~300 | STL/OBJ/PLY format writers |
| `src/utils/image_io.cr` | ~300 | PNG/JPEG loading via ImageIO |
| `src/utils/colmap_loader.cr` | ~350 | COLMAP format parser |
| `src/utils/geometry.cr` | ~450 | Vec3, Kabsch, ICP, SVD, Distance Geometry |
| `src/main.cr` | ~405 | CLI entry point with export command |

**Total: ~11,890 lines of Crystal + ~3,835 lines of Metal shaders**

---

## Build & Run

```bash
# Build (release)
make release

# or simply
make

# Run tests
make test

# Train (once image loading is implemented)
./gsplat train --images ./photos --output ./scene --iterations 30000

# Export mesh
./gsplat export --scene ./scene --output model.stl
```

---

## Key Technical Notes

- All tensors use **row-major layout** (Metal default)
- **float32** precision throughout
- **Tile size 16x16** for rasterizer
- **SH degree 3** (16 coefficients per color channel)
- FFI functions use **gs_** prefix
- Unified memory via `MTLResourceStorageModeShared`

---

## Architecture Diagram

```
┌──────────────┐     ┌──────────────┐
│   Images     │     │   Cameras    │
└──────┬───────┘     └──────┬───────┘
       │                    │
       ▼                    ▼
┌──────────────────────────────────┐
│           MASt3R                 │  ✅ GPU
│  ViT Encoder → DPT Decoder       │
└──────────────┬───────────────────┘
               │
               ▼ Point Cloud
┌──────────────────────────────────┐
│      Gaussian3D.from_points()    │
│  position, scale, rotation,      │
│  opacity, SH coefficients        │
└──────────────┬───────────────────┘
               │
    ┌──────────┴──────────┐
    │   Training Loop     │
    │  ┌───────────────┐  │
    │  │   Rasterizer  │──┼──▶ Rendered Image
    │  │   (Metal GPU) │  │
    │  └───────────────┘  │
    │         │           │
    │         ▼           │
    │  ┌───────────────┐  │
    │  │  Loss (L1+SSIM)│◀─┼── Target Image
    │  └───────────────┘  │
    │         │           │
    │         ▼           │
    │  ┌───────────────┐  │
    │  │   Backward    │  │
    │  └───────────────┘  │
    │         │           │
    │         ▼           │
    │  ┌───────────────┐  │
    │  │  Adam Update  │  │
    │  └───────────────┘  │
    │         │           │
    │         ▼           │
    │  Densify & Prune    │
    └─────────────────────┘
               │
               ▼
┌──────────────────────────────────┐
│     Marching Cubes → STL         │  ✅ GPU
└──────────────────────────────────┘
```
