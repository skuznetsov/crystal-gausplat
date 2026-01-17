# Gaussian Splatting Rasterizer
# Orchestrates forward and backward passes using Metal kernels

require "../core/tensor"
require "../metal/device"
require "../metal/dispatch"
require "../metal/gpu_radix_sort"
require "./gaussian"
require "./camera"
require "./rasterizer_context"

module GS
  module GaussianSplatting
    # Main rasterizer class
    class Rasterizer
      # Compiled pipelines (lazy initialized)
      @@compute_cov3d_pipeline : Metal::ComputePipeline?
      @@project_pipeline : Metal::ComputePipeline?
      @@eval_sh_pipeline : Metal::ComputePipeline?
      @@create_keys_pipeline : Metal::ComputePipeline?
      @@rasterize_pipeline : Metal::ComputePipeline?
      @@rasterize_backward_pipeline : Metal::ComputePipeline?

      # Kernel source (embedded for JIT compilation)
      KERNEL_SOURCE = {{ read_file("#{__DIR__}/../metal/kernels/gaussian.metal") }}

      # Initialize pipelines
      def self.init_pipelines : Bool
        return true if @@compute_cov3d_pipeline

        return false unless Metal::Device.available?

        begin
          @@compute_cov3d_pipeline = Metal::ComputePipeline.new("compute_cov3d", KERNEL_SOURCE, "compute_cov3d")
          @@project_pipeline = Metal::ComputePipeline.new("project_gaussians", KERNEL_SOURCE, "project_gaussians")
          @@eval_sh_pipeline = Metal::ComputePipeline.new("eval_sh", KERNEL_SOURCE, "eval_sh")
          @@create_keys_pipeline = Metal::ComputePipeline.new("create_tile_keys", KERNEL_SOURCE, "create_tile_keys")
          @@rasterize_pipeline = Metal::ComputePipeline.new("rasterize_tile", KERNEL_SOURCE, "rasterize_tile")
          @@rasterize_backward_pipeline = Metal::ComputePipeline.new("rasterize_backward", KERNEL_SOURCE, "rasterize_backward")
          true
        rescue ex
          puts "Failed to compile Gaussian kernels: #{ex.message}"
          false
        end
      end

      # Forward pass: render Gaussians to image
      def self.forward(
        gaussians : Gaussian3D,
        camera : Camera,
        context : RasterizerContext? = nil,
        background : {Float32, Float32, Float32} = {0.0_f32, 0.0_f32, 0.0_f32}
      ) : {Tensor, RasterizerContext}
        init_pipelines || raise "Metal not available"

        # Create or reuse context
        ctx = context || RasterizerContext.new(camera.width, camera.height, gaussians.count)
        ctx.resize_image(camera.width, camera.height)
        ctx.resize_gaussians(gaussians.count) if ctx.mean2d.shape[0] != gaussians.count
        ctx.background = background

        # Step 1: Compute 3D covariance from scale and rotation
        compute_covariance!(gaussians, ctx)

        # Step 2: Project Gaussians to 2D
        project_gaussians!(gaussians, camera, ctx)

        # Step 3: Evaluate spherical harmonics for view-dependent color
        eval_spherical_harmonics!(gaussians, camera, ctx)

        # Step 4: Tile binning and sorting
        tile_bin_and_sort!(gaussians, ctx)

        # Step 5: Rasterize tiles
        rasterize_tiles!(gaussians, ctx)

        {ctx.rendered_image, ctx}
      end

      # Backward pass: compute gradients
      def self.backward(
        dL_dimage : Tensor,
        gaussians : Gaussian3D,
        camera : Camera,
        context : RasterizerContext
      ) : RasterizerGradients
        init_pipelines || raise "Metal not available"

        grads = RasterizerGradients.new(gaussians.count, dL_dimage.device)

        # Step 1: Rasterize backward - get gradients w.r.t. 2D params
        rasterize_backward!(dL_dimage, context, grads)

        # Step 2: Project backward - chain rule to 3D params
        project_backward!(gaussians, camera, context, grads)

        # Step 3: SH backward - gradients to SH coefficients
        sh_backward!(gaussians, camera, context, grads)

        # Step 4: Covariance backward - gradients to scale and rotation
        covariance_backward!(gaussians, context, grads)

        grads
      end

      # ========================================================================
      # Forward Pass Steps
      # ========================================================================

      private def self.compute_covariance!(gaussians : Gaussian3D, ctx : RasterizerContext) : Nil
        # Get actual scale (exp of log_scale)
        scale = gaussians.scale
        rotation = gaussians.rotation.data

        # Normalize quaternions
        gaussians.normalize_rotations!

        pipeline = @@compute_cov3d_pipeline.not_nil!
        count = gaussians.count.to_u32

        Metal::Dispatch.execute(pipeline) do |encoder|
          encoder.set_tensor(scale, 0)
          encoder.set_tensor(rotation, 1)
          encoder.set_tensor(ctx.cov3d, 2)
          encoder.set_value(count, 3)
          encoder.dispatch_1d(gaussians.count)
        end
      end

      private def self.project_gaussians!(gaussians : Gaussian3D, camera : Camera, ctx : RasterizerContext) : Nil
        pipeline = @@project_pipeline.not_nil!

        # Prepare camera parameters
        camera_params = build_camera_params(camera)

        Metal::Dispatch.execute(pipeline) do |encoder|
          encoder.set_tensor(gaussians.position.data, 0)
          encoder.set_tensor(ctx.cov3d, 1)
          encoder.set_tensor(ctx.mean2d, 2)
          encoder.set_tensor(ctx.cov2d, 3)
          encoder.set_tensor(ctx.depths, 4)
          encoder.set_tensor(ctx.radii, 5)

          # tiles_touched as uint buffer
          tiles_touched = Tensor.zeros(gaussians.count, device: Tensor::Device::GPU)
          encoder.set_tensor(tiles_touched, 6)

          # Camera params as bytes
          encoder.set_bytes(pointerof(camera_params).as(Pointer(Void)), sizeof(CameraParamsGPU), 7)

          encoder.set_value(gaussians.count.to_u32, 8)
          encoder.dispatch_1d(gaussians.count)
        end
      end

      private def self.eval_spherical_harmonics!(gaussians : Gaussian3D, camera : Camera, ctx : RasterizerContext) : Nil
        pipeline = @@eval_sh_pipeline.not_nil!

        cam_pos = camera.position
        cam_pos_arr = StaticArray[cam_pos[0], cam_pos[1], cam_pos[2]]

        Metal::Dispatch.execute(pipeline) do |encoder|
          encoder.set_tensor(gaussians.sh_coeffs.data, 0)
          encoder.set_tensor(gaussians.position.data, 1)
          encoder.set_tensor(ctx.colors, 2)
          encoder.set_bytes(cam_pos_arr.to_unsafe.as(Pointer(Void)), 12, 3)
          encoder.set_value(gaussians.count.to_u32, 4)
          encoder.set_value(SH_DEGREE.to_u32, 5)
          encoder.dispatch_1d(gaussians.count)
        end
      end

      private def self.tile_bin_and_sort!(gaussians : Gaussian3D, ctx : RasterizerContext) : Nil
        # Estimate max keys (sum of tiles_touched)
        # For simplicity, use upper bound: count * average_tiles
        max_keys = gaussians.count * 16  # Assume average 16 tiles per gaussian

        ctx.allocate_tile_buffers(max_keys)

        # Create keys using GPU kernel
        pipeline = @@create_keys_pipeline.not_nil!

        # Atomic counter for keys
        key_counter = MetalBuffer.new(4)  # Single uint32
        key_counter.write([0_u32].map(&.to_f32))  # Zero it

        Metal::Dispatch.execute(pipeline) do |encoder|
          encoder.set_tensor(ctx.mean2d, 0)
          encoder.set_tensor(ctx.depths, 1)
          encoder.set_tensor(ctx.radii, 2)
          encoder.set_buffer(ctx.tile_keys.not_nil!, 3)
          encoder.set_buffer(ctx.gaussian_ids.not_nil!, 4)
          encoder.set_buffer(key_counter, 5)
          encoder.set_value(gaussians.count.to_u32, 6)
          encoder.set_value(ctx.tiles_x.to_u32, 7)
          encoder.set_value(ctx.tiles_y.to_u32, 8)
          encoder.dispatch_1d(gaussians.count)
        end

        # Read key count
        count_data = key_counter.read(1)
        ctx.num_rendered = count_data[0].to_i32

        # Sort keys by tile_id then depth
        if Metal::GPURadixSort.available?
          sort_tile_keys_gpu!(ctx)
        else
          sort_tile_keys_cpu!(ctx)
          compute_tile_ranges_cpu!(ctx)
        end
      end

      private def self.sort_tile_keys_gpu!(ctx : RasterizerContext) : Nil
        return if ctx.num_rendered == 0

        keys_buffer = ctx.tile_keys.not_nil!
        ids_buffer = ctx.gaussian_ids.not_nil!
        ranges_buffer = ctx.tile_ranges.not_nil!
        num_tiles = ctx.tiles_x * ctx.tiles_y

        # GPU radix sort
        Metal::GPURadixSort.sort!(keys_buffer, ids_buffer, ctx.num_rendered)

        # Compute tile ranges on GPU
        Metal::GPURadixSort.compute_tile_ranges!(keys_buffer, ranges_buffer, ctx.num_rendered, num_tiles)
      end

      private def self.sort_tile_keys_cpu!(ctx : RasterizerContext) : Nil
        return if ctx.num_rendered == 0

        # Read keys and ids from GPU
        keys_buffer = ctx.tile_keys.not_nil!
        ids_buffer = ctx.gaussian_ids.not_nil!

        # Read as raw bytes then interpret
        keys_bytes = keys_buffer.read(ctx.num_rendered * 2)  # 2 floats = 8 bytes = 1 uint64
        ids_data = ids_buffer.read(ctx.num_rendered)

        # Reinterpret float pairs as uint64 keys
        keys = Array(UInt64).new(ctx.num_rendered)
        ctx.num_rendered.times do |i|
          low = keys_bytes[i * 2].unsafe_as(UInt32)
          high = keys_bytes[i * 2 + 1].unsafe_as(UInt32)
          keys << (high.to_u64 << 32) | low.to_u64
        end

        ids = ids_data.map(&.to_u32)

        # Sort by key (tile_id in high bits, depth in low bits)
        indices = (0...ctx.num_rendered).to_a
        indices.sort! { |a, b| keys[a] <=> keys[b] }

        # Reorder
        sorted_keys = indices.map { |i| keys[i] }
        sorted_ids = indices.map { |i| ids[i] }

        # Write back
        sorted_keys_floats = sorted_keys.flat_map do |k|
          [(k & 0xFFFFFFFF).to_f32, (k >> 32).to_f32]
        end
        sorted_ids_floats = sorted_ids.map(&.to_f32)

        keys_buffer.write(sorted_keys_floats)
        ids_buffer.write(sorted_ids_floats)
      end

      private def self.compute_tile_ranges_cpu!(ctx : RasterizerContext) : Nil
        # Initialize ranges: [start, end) for each tile
        num_tiles = ctx.tiles_x * ctx.tiles_y
        ranges = Array(UInt32).new(num_tiles * 2, 0_u32)

        return if ctx.num_rendered == 0

        # Read sorted keys to find tile boundaries
        keys_buffer = ctx.tile_keys.not_nil!
        keys_bytes = keys_buffer.read(ctx.num_rendered * 2)

        current_tile = -1
        ctx.num_rendered.times do |i|
          high = keys_bytes[i * 2 + 1].unsafe_as(UInt32)
          tile_id = high.to_i32

          if tile_id != current_tile
            # End previous tile
            if current_tile >= 0 && current_tile < num_tiles
              ranges[current_tile * 2 + 1] = i.to_u32
            end
            # Start new tile
            if tile_id >= 0 && tile_id < num_tiles
              ranges[tile_id * 2] = i.to_u32
              current_tile = tile_id
            end
          end
        end

        # End last tile
        if current_tile >= 0 && current_tile < num_tiles
          ranges[current_tile * 2 + 1] = ctx.num_rendered.to_u32
        end

        # Write to GPU
        ctx.tile_ranges.not_nil!.write(ranges.map(&.to_f32))
      end

      private def self.rasterize_tiles!(gaussians : Gaussian3D, ctx : RasterizerContext) : Nil
        return if ctx.num_rendered == 0

        pipeline = @@rasterize_pipeline.not_nil!

        # Get actual opacity (sigmoid of logit)
        opacity = gaussians.opacity

        bg = StaticArray[ctx.background[0], ctx.background[1], ctx.background[2]]

        Metal::Dispatch.execute(pipeline) do |encoder|
          encoder.set_buffer(ctx.tile_keys.not_nil!, 0)
          encoder.set_buffer(ctx.gaussian_ids.not_nil!, 1)
          encoder.set_buffer(ctx.tile_ranges.not_nil!, 2)
          encoder.set_tensor(ctx.mean2d, 3)
          encoder.set_tensor(ctx.cov2d, 4)
          encoder.set_tensor(opacity, 5)
          encoder.set_tensor(ctx.colors, 6)
          encoder.set_tensor(ctx.rendered_image, 7)
          encoder.set_tensor(ctx.n_contrib, 8)
          encoder.set_tensor(ctx.final_transmittance, 9)
          encoder.set_value(ctx.tiles_x.to_u32, 10)
          encoder.set_value(ctx.width.to_u32, 11)
          encoder.set_value(ctx.height.to_u32, 12)
          encoder.set_bytes(bg.to_unsafe.as(Pointer(Void)), 12, 13)

          # Dispatch per tile with 16x16 threads per tile
          encoder.dispatch({ctx.tiles_x, ctx.tiles_y, 1}, {16, 16, 1})
        end
      end

      # ========================================================================
      # Backward Pass Steps
      # ========================================================================

      private def self.rasterize_backward!(dL_dimage : Tensor, ctx : RasterizerContext, grads : RasterizerGradients) : Nil
        return if ctx.num_rendered == 0

        pipeline = @@rasterize_backward_pipeline.not_nil!

        # Get actual opacity
        opacity_data = ctx.mean2d  # Placeholder - should store opacity in context
        # TODO: Store opacity in context during forward

        bg = StaticArray[ctx.background[0], ctx.background[1], ctx.background[2]]

        Metal::Dispatch.execute(pipeline) do |encoder|
          encoder.set_tensor(dL_dimage, 0)
          encoder.set_buffer(ctx.tile_keys.not_nil!, 1)
          encoder.set_buffer(ctx.gaussian_ids.not_nil!, 2)
          encoder.set_buffer(ctx.tile_ranges.not_nil!, 3)
          encoder.set_tensor(ctx.mean2d, 4)
          encoder.set_tensor(ctx.cov2d, 5)
          encoder.set_tensor(opacity_data, 6)  # Should be actual opacity
          encoder.set_tensor(ctx.colors, 7)
          encoder.set_tensor(ctx.n_contrib, 8)
          encoder.set_tensor(ctx.final_transmittance, 9)
          encoder.set_tensor(grads.dL_dmean2d, 10)
          encoder.set_tensor(grads.dL_dcov2d, 11)
          encoder.set_tensor(grads.dL_dopacity, 12)
          encoder.set_tensor(grads.dL_dcolors, 13)
          encoder.set_value(ctx.tiles_x.to_u32, 14)
          encoder.set_value(ctx.width.to_u32, 15)
          encoder.set_value(ctx.height.to_u32, 16)
          encoder.set_bytes(bg.to_unsafe.as(Pointer(Void)), 12, 17)

          encoder.dispatch({ctx.tiles_x, ctx.tiles_y, 1}, {16, 16, 1})
        end
      end

      private def self.project_backward!(gaussians : Gaussian3D, camera : Camera, ctx : RasterizerContext, grads : RasterizerGradients) : Nil
        # Chain rule: dL/d(position) = dL/d(mean2d) * d(mean2d)/d(position)
        # This requires the Jacobian of projection

        # CPU implementation for now
        # TODO: GPU kernel

        dL_dmean2d_cpu = grads.dL_dmean2d.to_cpu
        positions_cpu = gaussians.position.data.to_cpu
        grads.dL_dposition.to_cpu!

        dm2d = dL_dmean2d_cpu.cpu_data.not_nil!
        pos = positions_cpu.cpu_data.not_nil!
        dpos = grads.dL_dposition.cpu_data.not_nil!

        fx = camera.intrinsics.fx
        fy = camera.intrinsics.fy

        view_mat = camera.world_to_camera.to_cpu.cpu_data.not_nil!

        gaussians.count.times do |i|
          # Get position in camera space
          x = pos[i * 3]
          y = pos[i * 3 + 1]
          z = pos[i * 3 + 2]

          cx = view_mat[0]*x + view_mat[1]*y + view_mat[2]*z + view_mat[3]
          cy = view_mat[4]*x + view_mat[5]*y + view_mat[6]*z + view_mat[7]
          cz = view_mat[8]*x + view_mat[9]*y + view_mat[10]*z + view_mat[11]

          next if cz <= 0.01_f32

          z2 = cz * cz

          # Jacobian of projection w.r.t. camera-space position
          # d(px)/d(cx) = fx/cz, d(px)/d(cz) = -fx*cx/cz²
          # d(py)/d(cy) = fy/cz, d(py)/d(cz) = -fy*cy/cz²

          # Chain with view matrix to get d/d(world position)
          dL_dpx = dm2d[i * 2]
          dL_dpy = dm2d[i * 2 + 1]

          # d(camera)/d(world) = view_matrix rotation part
          3.times do |j|
            dL_dcx = dL_dpx * fx / cz
            dL_dcy = dL_dpy * fy / cz
            dL_dcz = dL_dpx * (-fx * cx / z2) + dL_dpy * (-fy * cy / z2)

            dpos[i * 3 + j] = dL_dcx * view_mat[j] +
                              dL_dcy * view_mat[4 + j] +
                              dL_dcz * view_mat[8 + j]
          end
        end

        grads.dL_dposition.to_gpu! if gaussians.position.data.on_gpu?
      end

      private def self.sh_backward!(gaussians : Gaussian3D, camera : Camera, ctx : RasterizerContext, grads : RasterizerGradients) : Nil
        # dL/d(sh_coeffs) from dL/d(colors)
        # Simplified: only DC term for now

        dL_dcolors_cpu = grads.dL_dcolors.to_cpu
        grads.dL_dsh.to_cpu!

        dc = dL_dcolors_cpu.cpu_data.not_nil!
        dsh = grads.dL_dsh.cpu_data.not_nil!

        # DC coefficient gradient: dL/d(sh[0]) = dL/d(color) * SH_C0
        sh_c0 = 0.28209479177387814_f32

        gaussians.count.times do |i|
          3.times do |c|
            dsh[i * 16 * 3 + c] = dc[i * 3 + c] / sh_c0
          end
        end

        grads.dL_dsh.to_gpu! if gaussians.sh_coeffs.data.on_gpu?
      end

      private def self.covariance_backward!(gaussians : Gaussian3D, ctx : RasterizerContext, grads : RasterizerGradients) : Nil
        # dL/d(scale), dL/d(rotation) from dL/d(cov3d)
        # Complex Jacobian - simplified CPU implementation

        # TODO: Full implementation with quaternion derivatives
        # For now, approximate with numerical gradients or skip
      end

      # ========================================================================
      # Helper Structures
      # ========================================================================

      # Camera parameters packed for GPU
      @[Packed]
      struct CameraParamsGPU
        property view_matrix : StaticArray(Float32, 16)
        property proj_matrix : StaticArray(Float32, 16)
        property fx : Float32
        property fy : Float32
        property cx : Float32
        property cy : Float32
        property width : Int32
        property height : Int32
        property tan_fov_x : Float32
        property tan_fov_y : Float32
        property cam_pos_x : Float32
        property cam_pos_y : Float32
        property cam_pos_z : Float32
        property _padding : Float32

        def initialize
          @view_matrix = StaticArray(Float32, 16).new(0.0_f32)
          @proj_matrix = StaticArray(Float32, 16).new(0.0_f32)
          @fx = 0.0_f32
          @fy = 0.0_f32
          @cx = 0.0_f32
          @cy = 0.0_f32
          @width = 0
          @height = 0
          @tan_fov_x = 0.0_f32
          @tan_fov_y = 0.0_f32
          @cam_pos_x = 0.0_f32
          @cam_pos_y = 0.0_f32
          @cam_pos_z = 0.0_f32
          @_padding = 0.0_f32
        end
      end

      private def self.build_camera_params(camera : Camera) : CameraParamsGPU
        params = CameraParamsGPU.new

        # Copy view matrix
        view_cpu = camera.world_to_camera.to_cpu
        view_data = view_cpu.cpu_data.not_nil!
        16.times { |i| params.view_matrix[i] = view_data[i] }

        # Copy projection matrix
        proj = camera.full_projection_matrix
        proj_cpu = proj.to_cpu
        proj_data = proj_cpu.cpu_data.not_nil!
        16.times { |i| params.proj_matrix[i] = proj_data[i] }

        params.fx = camera.intrinsics.fx
        params.fy = camera.intrinsics.fy
        params.cx = camera.intrinsics.cx
        params.cy = camera.intrinsics.cy
        params.width = camera.width
        params.height = camera.height
        params.tan_fov_x = camera.intrinsics.tan_half_fov_x
        params.tan_fov_y = camera.intrinsics.tan_half_fov_y

        pos = camera.position
        params.cam_pos_x = pos[0]
        params.cam_pos_y = pos[1]
        params.cam_pos_z = pos[2]

        params
      end
    end
  end
end
