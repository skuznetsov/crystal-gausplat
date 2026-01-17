# GPU-accelerated Marching Cubes via Metal compute shaders
# Uses parallel prefix sum for efficient mesh extraction

require "../core/buffer"
require "../core/tensor"
require "../metal/device"
require "../metal/dispatch"

module GS
  module Export
    # GPU Marching Cubes operations module
    module GPUMarchingCubes
      extend self

      # Load marching cubes kernel source at compile time
      MC_KERNEL_SOURCE = {{ read_file("#{__DIR__}/../metal/kernels/marching_cubes.metal") }}

      @@pipelines = Hash(String, Metal::ComputePipeline).new
      @@initialized = false

      # Lazy init
      def ensure_initialized
        return if @@initialized
        return unless Metal::Device.available?
        @@initialized = true
      end

      # Get or create pipeline
      def get_pipeline(name : String) : Metal::ComputePipeline
        @@pipelines[name] ||= Metal::ComputePipeline.new(name, MC_KERNEL_SOURCE, name)
      end

      # Check if GPU MC is available
      def available? : Bool
        Metal::Device.available?
      end

      # Full GPU marching cubes pipeline
      # Returns: Mesh
      def extract(
        points : Tensor,
        resolution : Int32 = 128,
        sigma : Float32 = 0.02_f32,
        isovalue : Float32 = 0.5_f32,
        bounds_min : {Float32, Float32, Float32}? = nil,
        bounds_max : {Float32, Float32, Float32}? = nil
      ) : Mesh
        raise "GPU not available" unless available?
        ensure_initialized

        # Ensure points are on GPU
        points_gpu = points.on_gpu? ? points : points.to_gpu
        n_points = points_gpu.shape[0]

        # Compute bounds
        origin, cell_size, extent = compute_bounds(points, bounds_min, bounds_max, sigma, resolution)

        # Step 1: Sample density field
        grid_size = resolution ** 3
        grid_buffer = MetalBuffer.new(grid_size.to_i64 * sizeof(Float32))

        sample_density(
          points_gpu.buffer.not_nil!,
          grid_buffer,
          n_points.to_u32,
          resolution.to_u32,
          origin,
          cell_size,
          sigma
        )

        # Step 2: Classify cubes
        cubes_per_dim = resolution - 1
        num_cubes = cubes_per_dim ** 3

        cube_index_buffer = MetalBuffer.new(num_cubes.to_i64)  # uchar
        vertex_count_buffer = MetalBuffer.new(num_cubes.to_i64)  # uchar
        tri_count_buffer = MetalBuffer.new(num_cubes.to_i64)  # uchar

        classify_cubes(
          grid_buffer,
          cube_index_buffer,
          vertex_count_buffer,
          tri_count_buffer,
          resolution.to_u32,
          isovalue
        )

        # Step 3: Prefix sum to compute offsets
        vertex_offset_buffer = MetalBuffer.new(num_cubes.to_i64 * sizeof(UInt32))
        tri_offset_buffer = MetalBuffer.new(num_cubes.to_i64 * sizeof(UInt32))

        total_vertices = prefix_sum(vertex_count_buffer, vertex_offset_buffer, num_cubes)
        total_triangles = prefix_sum(tri_count_buffer, tri_offset_buffer, num_cubes)

        mesh = Mesh.new
        return mesh if total_vertices == 0 || total_triangles == 0

        # Step 4: Generate vertices
        vertices_buffer = MetalBuffer.new(total_vertices.to_i64 * 3 * sizeof(Float32))

        generate_vertices(
          grid_buffer,
          cube_index_buffer,
          vertex_offset_buffer,
          vertices_buffer,
          resolution.to_u32,
          isovalue,
          origin,
          cell_size,
          cubes_per_dim.to_u32
        )

        # Step 5: Generate triangles
        triangles_buffer = MetalBuffer.new(total_triangles.to_i64 * 3 * sizeof(UInt32))

        generate_triangles(
          cube_index_buffer,
          vertex_offset_buffer,
          tri_offset_buffer,
          triangles_buffer,
          resolution.to_u32,
          cubes_per_dim.to_u32
        )

        # Step 6: Compute normals
        normals_buffer = MetalBuffer.new(total_vertices.to_i64 * 3 * sizeof(Float32))
        # Zero-initialize normals
        zeros = Array(Float32).new(total_vertices * 3, 0.0_f32)
        normals_buffer.write(zeros)

        compute_normals(
          vertices_buffer,
          triangles_buffer,
          normals_buffer,
          total_triangles.to_u32,
          total_vertices.to_u32
        )

        # Read results back to CPU and build mesh
        vertices = vertices_buffer.read(total_vertices * 3)
        normals = normals_buffer.read(total_vertices * 3)

        # Read triangles (stored as UInt32, need to convert to Int32)
        triangles_u32 = Array(UInt32).new(total_triangles * 3, 0_u32)
        MetalFFI.gs_buffer_read(triangles_buffer.handle, triangles_u32.to_unsafe.as(Pointer(Void)), (total_triangles * 3 * sizeof(UInt32)).to_i64)

        # Build mesh from raw arrays
        mesh = Mesh.new
        total_vertices.times do |i|
          mesh.add_vertex(vertices[i * 3], vertices[i * 3 + 1], vertices[i * 3 + 2])
        end

        # Set normals directly
        total_vertices.times do |i|
          mesh.normals[i * 3] = normals[i * 3]
          mesh.normals[i * 3 + 1] = normals[i * 3 + 1]
          mesh.normals[i * 3 + 2] = normals[i * 3 + 2]
        end

        # Add triangles
        total_triangles.times do |i|
          v0 = triangles_u32[i * 3].to_i32
          v1 = triangles_u32[i * 3 + 1].to_i32
          v2 = triangles_u32[i * 3 + 2].to_i32
          mesh.add_triangle(v0, v1, v2)
        end

        mesh
      end

      # Compute bounding box and grid parameters
      private def compute_bounds(
        points : Tensor,
        bounds_min : {Float32, Float32, Float32}?,
        bounds_max : {Float32, Float32, Float32}?,
        sigma : Float32,
        resolution : Int32
      ) : Tuple(Tuple(Float32, Float32, Float32), Float32, Float32)
        points_cpu = points.on_cpu? ? points : points.to_cpu
        points_d = points_cpu.cpu_data.not_nil!
        n_points = points_cpu.shape[0]

        min_x, min_y, min_z = Float32::MAX, Float32::MAX, Float32::MAX
        max_x, max_y, max_z = Float32::MIN, Float32::MIN, Float32::MIN

        n_points.times do |i|
          x, y, z = points_d[i * 3], points_d[i * 3 + 1], points_d[i * 3 + 2]
          min_x = x if x < min_x
          min_y = y if y < min_y
          min_z = z if z < min_z
          max_x = x if x > max_x
          max_y = y if y > max_y
          max_z = z if z > max_z
        end

        pad = sigma * 3
        if bmin = bounds_min
          min_x, min_y, min_z = bmin
        else
          min_x -= pad
          min_y -= pad
          min_z -= pad
        end

        if bmax = bounds_max
          max_x, max_y, max_z = bmax
        else
          max_x += pad
          max_y += pad
          max_z += pad
        end

        origin = {min_x, min_y, min_z}
        extent = {max_x - min_x, max_y - min_y, max_z - min_z}.max
        cell_size = extent / (resolution - 1).to_f32

        {origin, cell_size, extent}
      end

      # GPU kernel dispatch: sample density field
      private def sample_density(
        points : MetalBuffer,
        grid : MetalBuffer,
        num_points : UInt32,
        resolution : UInt32,
        origin : {Float32, Float32, Float32},
        cell_size : Float32,
        sigma : Float32
      )
        pipeline = get_pipeline("mc_sample_density")

        Metal::Dispatch.execute(pipeline) do |encoder|
          encoder.set_buffer(points, 0)
          encoder.set_buffer(grid, 1)
          encoder.set_value(num_points, 2)
          encoder.set_value(resolution, 3)
          encoder.set_value(StaticArray[origin[0], origin[1], origin[2]], 4)
          encoder.set_value(cell_size, 5)
          encoder.set_value(sigma, 6)

          res = resolution.to_i32
          encoder.dispatch_3d(res, res, res, {8, 8, 8})
        end
      end

      # GPU kernel dispatch: classify cubes
      private def classify_cubes(
        grid : MetalBuffer,
        cube_index : MetalBuffer,
        vertex_count : MetalBuffer,
        tri_count : MetalBuffer,
        resolution : UInt32,
        isovalue : Float32
      )
        pipeline = get_pipeline("mc_classify_cubes")

        Metal::Dispatch.execute(pipeline) do |encoder|
          encoder.set_buffer(grid, 0)
          encoder.set_buffer(cube_index, 1)
          encoder.set_buffer(vertex_count, 2)
          encoder.set_buffer(tri_count, 3)
          encoder.set_value(resolution, 4)
          encoder.set_value(isovalue, 5)

          cubes_per_dim = (resolution - 1).to_i32
          encoder.dispatch_3d(cubes_per_dim, cubes_per_dim, cubes_per_dim, {8, 8, 8})
        end
      end

      # Parallel prefix sum (Blelloch scan)
      # Returns total sum
      private def prefix_sum(input : MetalBuffer, output : MetalBuffer, n : Int32) : Int32
        block_size = 256
        num_blocks = (n + block_size * 2 - 1) // (block_size * 2)

        # Allocate block sums buffer
        block_sums_buffer = MetalBuffer.new(num_blocks.to_i64 * sizeof(UInt32))

        # First pass: local prefix sum within blocks
        pipeline_local = get_pipeline("mc_prefix_sum_local")

        Metal::Dispatch.execute(pipeline_local) do |encoder|
          encoder.set_buffer(input, 0)
          encoder.set_buffer(output, 1)
          encoder.set_buffer(block_sums_buffer, 2)
          encoder.set_value(n.to_u32, 3)
          encoder.set_threadgroup_memory((block_size * 2 * sizeof(UInt32)).to_i32, 0)
          encoder.dispatch_1d(num_blocks * block_size, block_size)
        end

        # If we have multiple blocks, need to scan block sums and add
        if num_blocks > 1
          # Scan block sums on CPU (for simplicity, could be GPU for large n)
          block_sums = Array(UInt32).new(num_blocks, 0_u32)
          MetalFFI.gs_buffer_read(block_sums_buffer.handle, block_sums.to_unsafe.as(Pointer(Void)), (num_blocks * sizeof(UInt32)).to_i64)

          # Exclusive scan of block sums
          block_offsets = Array(UInt32).new(num_blocks, 0_u32)
          running_sum = 0_u32
          num_blocks.times do |i|
            block_offsets[i] = running_sum
            running_sum += block_sums[i]
          end

          # Write back block offsets
          block_offsets_buffer = MetalBuffer.new(num_blocks.to_i64 * sizeof(UInt32))
          MetalFFI.gs_buffer_write(block_offsets_buffer.handle, block_offsets.to_unsafe.as(Pointer(Void)), (num_blocks * sizeof(UInt32)).to_i64)

          # Add block offsets to each element
          pipeline_add = get_pipeline("mc_prefix_sum_add_block")

          Metal::Dispatch.execute(pipeline_add) do |encoder|
            encoder.set_buffer(output, 0)
            encoder.set_buffer(block_offsets_buffer, 1)
            encoder.set_value(n.to_u32, 2)
            encoder.dispatch_1d(n, block_size)
          end

          return running_sum.to_i32
        else
          # Single block - read total from block_sums
          block_sums = Array(UInt32).new(1, 0_u32)
          MetalFFI.gs_buffer_read(block_sums_buffer.handle, block_sums.to_unsafe.as(Pointer(Void)), sizeof(UInt32).to_i64)
          return block_sums[0].to_i32
        end
      end

      # GPU kernel dispatch: generate vertices
      private def generate_vertices(
        grid : MetalBuffer,
        cube_index : MetalBuffer,
        vertex_offset : MetalBuffer,
        vertices : MetalBuffer,
        resolution : UInt32,
        isovalue : Float32,
        origin : {Float32, Float32, Float32},
        cell_size : Float32,
        cubes_per_dim : UInt32
      )
        pipeline = get_pipeline("mc_generate_vertices")

        Metal::Dispatch.execute(pipeline) do |encoder|
          encoder.set_buffer(grid, 0)
          encoder.set_buffer(cube_index, 1)
          encoder.set_buffer(vertex_offset, 2)
          encoder.set_buffer(vertices, 3)
          encoder.set_value(resolution, 4)
          encoder.set_value(isovalue, 5)
          encoder.set_value(StaticArray[origin[0], origin[1], origin[2]], 6)
          encoder.set_value(cell_size, 7)

          cpd = cubes_per_dim.to_i32
          encoder.dispatch_3d(cpd, cpd, cpd, {8, 8, 8})
        end
      end

      # GPU kernel dispatch: generate triangles
      private def generate_triangles(
        cube_index : MetalBuffer,
        vertex_offset : MetalBuffer,
        tri_offset : MetalBuffer,
        triangles : MetalBuffer,
        resolution : UInt32,
        cubes_per_dim : UInt32
      )
        pipeline = get_pipeline("mc_generate_triangles")

        Metal::Dispatch.execute(pipeline) do |encoder|
          encoder.set_buffer(cube_index, 0)
          encoder.set_buffer(vertex_offset, 1)
          encoder.set_buffer(tri_offset, 2)
          encoder.set_buffer(triangles, 3)
          encoder.set_value(resolution, 4)

          cpd = cubes_per_dim.to_i32
          encoder.dispatch_3d(cpd, cpd, cpd, {8, 8, 8})
        end
      end

      # GPU kernel dispatch: compute normals
      private def compute_normals(
        vertices : MetalBuffer,
        triangles : MetalBuffer,
        normals : MetalBuffer,
        num_triangles : UInt32,
        num_vertices : UInt32
      )
        # Phase 1: Accumulate face normals to vertices (atomic)
        pipeline_accum = get_pipeline("mc_compute_normals_accumulate")

        Metal::Dispatch.execute(pipeline_accum) do |encoder|
          encoder.set_buffer(vertices, 0)
          encoder.set_buffer(triangles, 1)
          encoder.set_buffer(normals, 2)
          encoder.set_value(num_triangles, 3)
          encoder.dispatch_1d(num_triangles.to_i32, 256)
        end

        # Phase 2: Normalize
        pipeline_norm = get_pipeline("mc_normalize_normals")

        Metal::Dispatch.execute(pipeline_norm) do |encoder|
          encoder.set_buffer(normals, 0)
          encoder.set_value(num_vertices, 1)
          encoder.dispatch_1d(num_vertices.to_i32, 256)
        end
      end
    end
  end
end
