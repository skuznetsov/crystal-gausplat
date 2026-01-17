# 3D Scanner - Gaussian Splatting + MASt3R

3D reconstruction from photos or video using MASt3R (dense stereo) + Gaussian Splatting, implemented in Crystal with Metal compute shaders for Apple Silicon.

## Features

- **Video Input**: Direct scanning from MOV/MP4/MKV with smart frame selection
- **MASt3R Integration**: ViT encoder + DPT decoder for dense stereo matching
- **Gaussian Splatting**: Differentiable rendering with tile-based rasterization
- **Full GPU Pipeline**: All operations accelerated on Metal (M2 Max optimized)
- **Mesh Export**: Marching Cubes extraction to STL/OBJ/PLY formats

## Requirements

- macOS with Apple Silicon (M1/M2/M3)
- Crystal 1.19+
- Xcode Command Line Tools (for Metal compiler)
- FFmpeg (`brew install ffmpeg`) - for video input

## Build

```bash
# Build release version
make

# Build debug version
make debug

# Run tests (22 tests)
make test
# or
./gsplat test
```

## Usage

```bash
# Scan object from video → 3D mesh (simplest workflow)
./gsplat scan --video object.MOV --output model.stl

# With options
./gsplat scan --video object.MOV --output model.stl --frames 50 --quality thorough

# Train Gaussian Splatting from images
./gsplat train --images ./photos --output ./scene --iterations 30000

# Render a view
./gsplat render --scene ./scene --output render.png

# Export mesh
./gsplat export --scene ./scene --output model.stl --resolution 256
```

### Video Scan Options

| Option | Description | Default |
|--------|-------------|---------|
| `--video` | Input video file (MOV, MP4, MKV, AVI, WebM) | required |
| `--output` | Output mesh path | output.stl |
| `--frames` | Maximum frames to extract | 50 |
| `--quality` | Frame selection: fast, normal, thorough | normal |
| `--resolution` | Mesh resolution | 256 |

## Architecture

```
Video / Images
       │
       ▼
┌──────────────────────────────────┐
│      Frame Extraction            │  FFmpeg + Smart Selection
│  (blur detect, motion filter)    │
└──────────────┬───────────────────┘
               │
               ▼
┌──────────────────────────────────┐
│           MASt3R                 │  GPU Accelerated
│  ViT Encoder → DPT Decoder       │
└──────────────┬───────────────────┘
               │
               ▼ Point Cloud
┌──────────────────────────────────┐
│      Gaussian3D Initialization   │
└──────────────┬───────────────────┘
               │
    ┌──────────┴──────────┐
    │   Training Loop     │
    │  Rasterize → Loss   │
    │  → Backward → Adam  │
    │  → Densify/Prune    │
    └──────────┬──────────┘
               │
               ▼
┌──────────────────────────────────┐
│     Marching Cubes → STL         │  GPU Accelerated
└──────────────────────────────────┘
```

## GPU Optimizations

| Component | Optimization | Speedup |
|-----------|-------------|---------|
| Linear/LayerNorm | Fused GPU kernels | ~10x vs CPU |
| Attention | Flash Attention (online softmax) | ~5x vs naive |
| Marching Cubes | Parallel prefix sum | 3x vs CPU |
| Tile Binning | GPU Radix Sort | Integrated |
| Transformer | Persistent Threadgroups | Reduced overhead |

## Project Structure

```
src/
├── core/           # Tensor, Buffer, Shape
├── metal/          # Metal backend, GPU kernels
│   └── kernels/    # .metal shader files
├── autograd/       # Automatic differentiation
├── optim/          # Adam, SGD optimizers
├── nn/             # Neural network layers
├── gaussian_splatting/  # Core GS implementation
├── mastr/          # MASt3R model
├── video/          # FFmpeg video reader, frame selection
├── export/         # Mesh extraction, file formats
└── utils/          # Image I/O, geometry, COLMAP
```

## Technical Notes

- All tensors use **row-major layout** (Metal default)
- **float32** precision throughout
- **Tile size 16x16** for rasterizer
- **SH degree 3** (16 coefficients per color channel)
- Unified memory via `MTLResourceStorageModeShared` (zero-copy)

## Stats

- **43 source files** (Crystal + C + Metal)
- **~13,500 lines** of Crystal
- **~3,835 lines** of Metal shaders
- **87 specs** passing

## License

MIT
