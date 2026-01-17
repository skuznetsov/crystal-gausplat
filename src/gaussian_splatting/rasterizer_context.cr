# Rasterizer Context: stores state between forward and backward passes
# Required because backward needs intermediate values from forward

require "../core/tensor"
require "../core/buffer"

module GS
  module GaussianSplatting
    # Context holding all intermediate data needed for backward pass
    class RasterizerContext
      # Camera info
      property width : Int32
      property height : Int32
      property tiles_x : Int32
      property tiles_y : Int32

      # Per-gaussian projected data
      property mean2d : Tensor           # [N, 2] screen positions
      property cov2d : Tensor            # [N, 3] 2D covariance (xx, xy, yy)
      property depths : Tensor           # [N] depth values
      property radii : Tensor            # [N] bounding radii
      property cov3d : Tensor            # [N, 6] 3D covariance (upper tri)
      property colors : Tensor           # [N, 3] SH-evaluated colors

      # Tile binning data
      property tile_keys : MetalBuffer?        # Sorted (tile_id, depth) keys
      property gaussian_ids : MetalBuffer?     # Gaussian index per key
      property tile_ranges : MetalBuffer?      # [tiles, 2] start/end per tile
      property num_rendered : Int32            # Total key count

      # Per-pixel data for backward
      property n_contrib : Tensor        # [H, W] number of contributors
      property final_transmittance : Tensor  # [H, W] final T value

      # Output image
      property rendered_image : Tensor   # [H, W, 3]

      # Background color
      property background : {Float32, Float32, Float32}

      def initialize(
        @width : Int32,
        @height : Int32,
        gaussian_count : Int32,
        device : Tensor::Device = Tensor::Device::GPU
      )
        @tiles_x = (@width + 15) // 16
        @tiles_y = (@height + 15) // 16

        # Pre-allocate tensors
        @mean2d = Tensor.zeros(gaussian_count, 2, device: device)
        @cov2d = Tensor.zeros(gaussian_count, 3, device: device)
        @depths = Tensor.zeros(gaussian_count, device: device)
        @radii = Tensor.zeros(gaussian_count, device: device)
        @cov3d = Tensor.zeros(gaussian_count, 6, device: device)
        @colors = Tensor.zeros(gaussian_count, 3, device: device)

        @n_contrib = Tensor.zeros(@height, @width, device: device)
        @final_transmittance = Tensor.zeros(@height, @width, device: device)
        @rendered_image = Tensor.zeros(@height, @width, 3, device: device)

        @tile_keys = nil
        @gaussian_ids = nil
        @tile_ranges = nil
        @num_rendered = 0

        @background = {0.0_f32, 0.0_f32, 0.0_f32}
      end

      # Resize context for different gaussian count
      def resize_gaussians(new_count : Int32) : Nil
        device = @mean2d.device

        @mean2d = Tensor.zeros(new_count, 2, device: device)
        @cov2d = Tensor.zeros(new_count, 3, device: device)
        @depths = Tensor.zeros(new_count, device: device)
        @radii = Tensor.zeros(new_count, device: device)
        @cov3d = Tensor.zeros(new_count, 6, device: device)
        @colors = Tensor.zeros(new_count, 3, device: device)
      end

      # Resize for different image dimensions
      def resize_image(new_width : Int32, new_height : Int32) : Nil
        return if new_width == @width && new_height == @height

        @width = new_width
        @height = new_height
        @tiles_x = (@width + 15) // 16
        @tiles_y = (@height + 15) // 16

        device = @rendered_image.device
        @n_contrib = Tensor.zeros(@height, @width, device: device)
        @final_transmittance = Tensor.zeros(@height, @width, device: device)
        @rendered_image = Tensor.zeros(@height, @width, 3, device: device)
      end

      # Allocate tile buffers based on expected key count
      def allocate_tile_buffers(max_keys : Int32) : Nil
        # Each key is 8 bytes (uint64), each id is 4 bytes (uint32)
        @tile_keys = GS.buffer_pool.acquire(max_keys.to_i64 * 8)
        @gaussian_ids = GS.buffer_pool.acquire(max_keys.to_i64 * 4)
        @tile_ranges = GS.buffer_pool.acquire(@tiles_x.to_i64 * @tiles_y * 8)  # 2 uints per tile
      end

      # Release tile buffers back to pool
      def release_tile_buffers : Nil
        if buf = @tile_keys
          GS.buffer_pool.release(buf)
          @tile_keys = nil
        end
        if buf = @gaussian_ids
          GS.buffer_pool.release(buf)
          @gaussian_ids = nil
        end
        if buf = @tile_ranges
          GS.buffer_pool.release(buf)
          @tile_ranges = nil
        end
      end

      # Check if context is valid for rendering
      def valid? : Bool
        @mean2d.numel > 0 && @rendered_image.numel > 0
      end

      # Get number of tiles
      def num_tiles : Int32
        @tiles_x * @tiles_y
      end
    end

    # Gradient outputs from backward pass
    struct RasterizerGradients
      property dL_dmean2d : Tensor      # [N, 2]
      property dL_dcov2d : Tensor       # [N, 3]
      property dL_dopacity : Tensor     # [N]
      property dL_dcolors : Tensor      # [N, 3]
      property dL_dposition : Tensor    # [N, 3] (after chain rule through projection)
      property dL_dcov3d : Tensor       # [N, 6]
      property dL_dsh : Tensor          # [N, 16, 3]

      def initialize(gaussian_count : Int32, device : Tensor::Device = Tensor::Device::GPU)
        @dL_dmean2d = Tensor.zeros(gaussian_count, 2, device: device)
        @dL_dcov2d = Tensor.zeros(gaussian_count, 3, device: device)
        @dL_dopacity = Tensor.zeros(gaussian_count, device: device)
        @dL_dcolors = Tensor.zeros(gaussian_count, 3, device: device)
        @dL_dposition = Tensor.zeros(gaussian_count, 3, device: device)
        @dL_dcov3d = Tensor.zeros(gaussian_count, 6, device: device)
        @dL_dsh = Tensor.zeros(gaussian_count, 16, 3, device: device)
      end

      # Zero all gradients
      def zero! : Nil
        @dL_dmean2d.fill!(0.0_f32)
        @dL_dcov2d.fill!(0.0_f32)
        @dL_dopacity.fill!(0.0_f32)
        @dL_dcolors.fill!(0.0_f32)
        @dL_dposition.fill!(0.0_f32)
        @dL_dcov3d.fill!(0.0_f32)
        @dL_dsh.fill!(0.0_f32)
      end
    end
  end
end
