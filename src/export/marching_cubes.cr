# Marching Cubes algorithm for mesh extraction
# Converts Gaussian point cloud to watertight mesh

require "../core/tensor"
require "./gpu_marching_cubes"

module GS
  module Export
    # Triangle mesh representation
    struct Mesh
      getter vertices : Array(Float32)   # [N * 3] flat array of xyz
      getter normals : Array(Float32)    # [N * 3] flat array of normals
      getter triangles : Array(Int32)    # [M * 3] triangle indices

      def initialize
        @vertices = Array(Float32).new
        @normals = Array(Float32).new
        @triangles = Array(Int32).new
      end

      def add_vertex(x : Float32, y : Float32, z : Float32) : Int32
        idx = @vertices.size // 3
        @vertices << x << y << z
        @normals << 0.0_f32 << 0.0_f32 << 0.0_f32  # Will be computed later
        idx
      end

      def add_triangle(v0 : Int32, v1 : Int32, v2 : Int32)
        @triangles << v0 << v1 << v2
      end

      def vertex_count : Int32
        @vertices.size // 3
      end

      def triangle_count : Int32
        @triangles.size // 3
      end

      # Compute normals from triangles
      def compute_normals!
        # Reset normals
        @normals.fill(0.0_f32)

        # Accumulate face normals to vertices
        (@triangles.size // 3).times do |i|
          i0 = @triangles[i * 3]
          i1 = @triangles[i * 3 + 1]
          i2 = @triangles[i * 3 + 2]

          # Get vertices
          v0x, v0y, v0z = @vertices[i0 * 3], @vertices[i0 * 3 + 1], @vertices[i0 * 3 + 2]
          v1x, v1y, v1z = @vertices[i1 * 3], @vertices[i1 * 3 + 1], @vertices[i1 * 3 + 2]
          v2x, v2y, v2z = @vertices[i2 * 3], @vertices[i2 * 3 + 1], @vertices[i2 * 3 + 2]

          # Edge vectors
          e1x, e1y, e1z = v1x - v0x, v1y - v0y, v1z - v0z
          e2x, e2y, e2z = v2x - v0x, v2y - v0y, v2z - v0z

          # Cross product
          nx = e1y * e2z - e1z * e2y
          ny = e1z * e2x - e1x * e2z
          nz = e1x * e2y - e1y * e2x

          # Add to each vertex
          [i0, i1, i2].each do |vi|
            @normals[vi * 3] += nx
            @normals[vi * 3 + 1] += ny
            @normals[vi * 3 + 2] += nz
          end
        end

        # Normalize
        (@normals.size // 3).times do |i|
          nx, ny, nz = @normals[i * 3], @normals[i * 3 + 1], @normals[i * 3 + 2]
          len = Math.sqrt(nx * nx + ny * ny + nz * nz).to_f32
          if len > 1e-8_f32
            @normals[i * 3] = nx / len
            @normals[i * 3 + 1] = ny / len
            @normals[i * 3 + 2] = nz / len
          end
        end
      end
    end

    # Marching cubes implementation
    class MarchingCubes
      # Grid dimensions
      getter resolution : Int32
      getter grid : Array(Float32)
      getter origin : {Float32, Float32, Float32}
      getter cell_size : Float32

      def initialize(@resolution : Int32 = 128)
        @grid = Array(Float32).new(@resolution ** 3, 0.0_f32)
        @origin = {0.0_f32, 0.0_f32, 0.0_f32}
        @cell_size = 1.0_f32
      end

      # Sample density field from Gaussians
      # gaussians: point cloud with positions [N, 3]
      # bounds: min/max coordinates
      def sample_from_points(
        points : Tensor,
        sigma : Float32 = 0.02_f32,
        bounds_min : {Float32, Float32, Float32}? = nil,
        bounds_max : {Float32, Float32, Float32}? = nil
      )
        points_cpu = points.on_cpu? ? points : points.to_cpu
        points_d = points_cpu.cpu_data.not_nil!
        n_points = points_cpu.shape[0]

        # Compute bounds if not provided
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

        # Add padding
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

        @origin = {min_x, min_y, min_z}
        extent = {max_x - min_x, max_y - min_y, max_z - min_z}.max
        @cell_size = extent / (@resolution - 1).to_f32

        # Reset grid
        @grid.fill(0.0_f32)

        # For each point, add Gaussian contribution to nearby cells
        inv_2sigma2 = 1.0_f32 / (2.0_f32 * sigma * sigma)
        radius_cells = (3.0_f32 * sigma / @cell_size).ceil.to_i32

        n_points.times do |pi|
          px, py, pz = points_d[pi * 3], points_d[pi * 3 + 1], points_d[pi * 3 + 2]

          # Grid coordinates
          gx = ((px - min_x) / @cell_size).to_i32
          gy = ((py - min_y) / @cell_size).to_i32
          gz = ((pz - min_z) / @cell_size).to_i32

          # Sample neighborhood
          (-radius_cells..radius_cells).each do |di|
            (-radius_cells..radius_cells).each do |dj|
              (-radius_cells..radius_cells).each do |dk|
                ix = gx + di
                iy = gy + dj
                iz = gz + dk

                next if ix < 0 || ix >= @resolution
                next if iy < 0 || iy >= @resolution
                next if iz < 0 || iz >= @resolution

                # World position of cell center
                cx = min_x + (ix + 0.5_f32) * @cell_size
                cy = min_y + (iy + 0.5_f32) * @cell_size
                cz = min_z + (iz + 0.5_f32) * @cell_size

                # Distance squared
                dx, dy, dz = cx - px, cy - py, cz - pz
                dist2 = dx * dx + dy * dy + dz * dz

                # Gaussian contribution
                contribution = Math.exp(-dist2 * inv_2sigma2).to_f32
                @grid[ix + iy * @resolution + iz * @resolution * @resolution] += contribution
              end
            end
          end
        end
      end

      # Extract mesh using GPU if available, otherwise CPU
      def self.extract_from_points(
        points : Tensor,
        resolution : Int32 = 128,
        sigma : Float32 = 0.02_f32,
        isovalue : Float32 = 0.5_f32,
        use_gpu : Bool = true
      ) : Mesh
        if use_gpu && GPUMarchingCubes.available?
          GPUMarchingCubes.extract(points, resolution, sigma, isovalue)
        else
          mc = new(resolution)
          mc.sample_from_points(points, sigma)
          mc.extract(isovalue)
        end
      end

      # Extract mesh at given isovalue
      def extract(isovalue : Float32 = 0.5_f32) : Mesh
        mesh = Mesh.new

        # Edge vertex cache to avoid duplicates
        # Key: edge index (cube_idx * 12 + edge_idx)
        edge_vertices = Hash(Int64, Int32).new

        # Process each cube
        (@resolution - 1).times do |iz|
          (@resolution - 1).times do |iy|
            (@resolution - 1).times do |ix|
              process_cube(ix, iy, iz, isovalue, mesh, edge_vertices)
            end
          end
        end

        mesh.compute_normals!
        mesh
      end

      private def process_cube(
        ix : Int32, iy : Int32, iz : Int32,
        isovalue : Float32,
        mesh : Mesh,
        edge_vertices : Hash(Int64, Int32)
      )
        # Get cube corner values
        corners = StaticArray(Float32, 8).new(0.0_f32)
        CORNER_OFFSETS.each_with_index do |off, i|
          cx = ix + off[0]
          cy = iy + off[1]
          cz = iz + off[2]
          corners[i] = @grid[cx + cy * @resolution + cz * @resolution * @resolution]
        end

        # Compute cube index (which corners are inside)
        cube_index = 0
        8.times { |i| cube_index |= (1 << i) if corners[i] < isovalue }

        # Skip if entirely inside or outside
        return if cube_index == 0 || cube_index == 255

        # Get edges that intersect the isosurface
        edges = EDGE_TABLE[cube_index]
        return if edges == 0

        # Interpolate vertices on active edges
        vert_list = StaticArray(Int32, 12).new(-1)
        12.times do |e|
          next unless (edges & (1 << e)) != 0

          # Edge endpoints
          v0, v1 = EDGE_VERTICES[e]

          # Compute unique edge key
          cube_idx = ix.to_i64 + iy.to_i64 * @resolution + iz.to_i64 * @resolution * @resolution
          edge_key = cube_idx * 12 + e

          if cached = edge_vertices[edge_key]?
            vert_list[e] = cached
          else
            # Interpolate position
            val0, val1 = corners[v0], corners[v1]
            t = (isovalue - val0) / (val1 - val0 + 1e-10_f32)
            t = t.clamp(0.0_f32, 1.0_f32)

            off0, off1 = CORNER_OFFSETS[v0], CORNER_OFFSETS[v1]
            x = @origin[0] + (ix + off0[0] + t * (off1[0] - off0[0])) * @cell_size
            y = @origin[1] + (iy + off0[1] + t * (off1[1] - off0[1])) * @cell_size
            z = @origin[2] + (iz + off0[2] + t * (off1[2] - off0[2])) * @cell_size

            vidx = mesh.add_vertex(x, y, z)
            edge_vertices[edge_key] = vidx
            vert_list[e] = vidx
          end
        end

        # Create triangles
        tri_idx = 0
        while TRI_TABLE[cube_index][tri_idx] != -1
          v0 = vert_list[TRI_TABLE[cube_index][tri_idx]]
          v1 = vert_list[TRI_TABLE[cube_index][tri_idx + 1]]
          v2 = vert_list[TRI_TABLE[cube_index][tri_idx + 2]]
          mesh.add_triangle(v0, v1, v2) if v0 >= 0 && v1 >= 0 && v2 >= 0
          tri_idx += 3
        end
      end

      # Cube corner offsets (relative to cell origin)
      CORNER_OFFSETS = [
        {0, 0, 0}, {1, 0, 0}, {1, 1, 0}, {0, 1, 0},
        {0, 0, 1}, {1, 0, 1}, {1, 1, 1}, {0, 1, 1},
      ]

      # Edge vertices (indices into CORNER_OFFSETS)
      EDGE_VERTICES = [
        {0, 1}, {1, 2}, {2, 3}, {3, 0},  # Bottom face
        {4, 5}, {5, 6}, {6, 7}, {7, 4},  # Top face
        {0, 4}, {1, 5}, {2, 6}, {3, 7},  # Vertical edges
      ]

      # Edge table: which edges are intersected for each cube configuration
      # 256 entries, 12 bits each
      EDGE_TABLE = [
        0x000, 0x109, 0x203, 0x30a, 0x406, 0x50f, 0x605, 0x70c,
        0x80c, 0x905, 0xa0f, 0xb06, 0xc0a, 0xd03, 0xe09, 0xf00,
        0x190, 0x099, 0x393, 0x29a, 0x596, 0x49f, 0x795, 0x69c,
        0x99c, 0x895, 0xb9f, 0xa96, 0xd9a, 0xc93, 0xf99, 0xe90,
        0x230, 0x339, 0x033, 0x13a, 0x636, 0x73f, 0x435, 0x53c,
        0xa3c, 0xb35, 0x83f, 0x936, 0xe3a, 0xf33, 0xc39, 0xd30,
        0x3a0, 0x2a9, 0x1a3, 0x0aa, 0x7a6, 0x6af, 0x5a5, 0x4ac,
        0xbac, 0xaa5, 0x9af, 0x8a6, 0xfaa, 0xea3, 0xda9, 0xca0,
        0x460, 0x569, 0x663, 0x76a, 0x066, 0x16f, 0x265, 0x36c,
        0xc6c, 0xd65, 0xe6f, 0xf66, 0x86a, 0x963, 0xa69, 0xb60,
        0x5f0, 0x4f9, 0x7f3, 0x6fa, 0x1f6, 0x0ff, 0x3f5, 0x2fc,
        0xdfc, 0xcf5, 0xfff, 0xef6, 0x9fa, 0x8f3, 0xbf9, 0xaf0,
        0x650, 0x759, 0x453, 0x55a, 0x256, 0x35f, 0x055, 0x15c,
        0xe5c, 0xf55, 0xc5f, 0xd56, 0xa5a, 0xb53, 0x859, 0x950,
        0x7c0, 0x6c9, 0x5c3, 0x4ca, 0x3c6, 0x2cf, 0x1c5, 0x0cc,
        0xfcc, 0xec5, 0xdcf, 0xcc6, 0xbca, 0xac3, 0x9c9, 0x8c0,
        0x8c0, 0x9c9, 0xac3, 0xbca, 0xcc6, 0xdcf, 0xec5, 0xfcc,
        0x0cc, 0x1c5, 0x2cf, 0x3c6, 0x4ca, 0x5c3, 0x6c9, 0x7c0,
        0x950, 0x859, 0xb53, 0xa5a, 0xd56, 0xc5f, 0xf55, 0xe5c,
        0x15c, 0x055, 0x35f, 0x256, 0x55a, 0x453, 0x759, 0x650,
        0xaf0, 0xbf9, 0x8f3, 0x9fa, 0xef6, 0xfff, 0xcf5, 0xdfc,
        0x2fc, 0x3f5, 0x0ff, 0x1f6, 0x6fa, 0x7f3, 0x4f9, 0x5f0,
        0xb60, 0xa69, 0x963, 0x86a, 0xf66, 0xe6f, 0xd65, 0xc6c,
        0x36c, 0x265, 0x16f, 0x066, 0x76a, 0x663, 0x569, 0x460,
        0xca0, 0xda9, 0xea3, 0xfaa, 0x8a6, 0x9af, 0xaa5, 0xbac,
        0x4ac, 0x5a5, 0x6af, 0x7a6, 0x0aa, 0x1a3, 0x2a9, 0x3a0,
        0xd30, 0xc39, 0xf33, 0xe3a, 0x936, 0x83f, 0xb35, 0xa3c,
        0x53c, 0x435, 0x73f, 0x636, 0x13a, 0x033, 0x339, 0x230,
        0xe90, 0xf99, 0xc93, 0xd9a, 0xa96, 0xb9f, 0x895, 0x99c,
        0x69c, 0x795, 0x49f, 0x596, 0x29a, 0x393, 0x099, 0x190,
        0xf00, 0xe09, 0xd03, 0xc0a, 0xb06, 0xa0f, 0x905, 0x80c,
        0x70c, 0x605, 0x50f, 0x406, 0x30a, 0x203, 0x109, 0x000,
      ]

      # Triangle table: which triangles to generate for each configuration
      # -1 terminates the list
      TRI_TABLE = [
        StaticArray[-1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1],
        StaticArray[0, 8, 3, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1],
        StaticArray[0, 1, 9, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1],
        StaticArray[1, 8, 3, 9, 8, 1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1],
        StaticArray[1, 2, 10, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1],
        StaticArray[0, 8, 3, 1, 2, 10, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1],
        StaticArray[9, 2, 10, 0, 2, 9, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1],
        StaticArray[2, 8, 3, 2, 10, 8, 10, 9, 8, -1, -1, -1, -1, -1, -1, -1],
        StaticArray[3, 11, 2, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1],
        StaticArray[0, 11, 2, 8, 11, 0, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1],
        StaticArray[1, 9, 0, 2, 3, 11, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1],
        StaticArray[1, 11, 2, 1, 9, 11, 9, 8, 11, -1, -1, -1, -1, -1, -1, -1],
        StaticArray[3, 10, 1, 11, 10, 3, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1],
        StaticArray[0, 10, 1, 0, 8, 10, 8, 11, 10, -1, -1, -1, -1, -1, -1, -1],
        StaticArray[3, 9, 0, 3, 11, 9, 11, 10, 9, -1, -1, -1, -1, -1, -1, -1],
        StaticArray[9, 8, 10, 10, 8, 11, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1],
        StaticArray[4, 7, 8, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1],
        StaticArray[4, 3, 0, 7, 3, 4, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1],
        StaticArray[0, 1, 9, 8, 4, 7, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1],
        StaticArray[4, 1, 9, 4, 7, 1, 7, 3, 1, -1, -1, -1, -1, -1, -1, -1],
        StaticArray[1, 2, 10, 8, 4, 7, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1],
        StaticArray[3, 4, 7, 3, 0, 4, 1, 2, 10, -1, -1, -1, -1, -1, -1, -1],
        StaticArray[9, 2, 10, 9, 0, 2, 8, 4, 7, -1, -1, -1, -1, -1, -1, -1],
        StaticArray[2, 10, 9, 2, 9, 7, 2, 7, 3, 7, 9, 4, -1, -1, -1, -1],
        StaticArray[8, 4, 7, 3, 11, 2, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1],
        StaticArray[11, 4, 7, 11, 2, 4, 2, 0, 4, -1, -1, -1, -1, -1, -1, -1],
        StaticArray[9, 0, 1, 8, 4, 7, 2, 3, 11, -1, -1, -1, -1, -1, -1, -1],
        StaticArray[4, 7, 11, 9, 4, 11, 9, 11, 2, 9, 2, 1, -1, -1, -1, -1],
        StaticArray[3, 10, 1, 3, 11, 10, 7, 8, 4, -1, -1, -1, -1, -1, -1, -1],
        StaticArray[1, 11, 10, 1, 4, 11, 1, 0, 4, 7, 11, 4, -1, -1, -1, -1],
        StaticArray[4, 7, 8, 9, 0, 11, 9, 11, 10, 11, 0, 3, -1, -1, -1, -1],
        StaticArray[4, 7, 11, 4, 11, 9, 9, 11, 10, -1, -1, -1, -1, -1, -1, -1],
        StaticArray[9, 5, 4, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1],
        StaticArray[9, 5, 4, 0, 8, 3, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1],
        StaticArray[0, 5, 4, 1, 5, 0, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1],
        StaticArray[8, 5, 4, 8, 3, 5, 3, 1, 5, -1, -1, -1, -1, -1, -1, -1],
        StaticArray[1, 2, 10, 9, 5, 4, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1],
        StaticArray[3, 0, 8, 1, 2, 10, 4, 9, 5, -1, -1, -1, -1, -1, -1, -1],
        StaticArray[5, 2, 10, 5, 4, 2, 4, 0, 2, -1, -1, -1, -1, -1, -1, -1],
        StaticArray[2, 10, 5, 3, 2, 5, 3, 5, 4, 3, 4, 8, -1, -1, -1, -1],
        StaticArray[9, 5, 4, 2, 3, 11, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1],
        StaticArray[0, 11, 2, 0, 8, 11, 4, 9, 5, -1, -1, -1, -1, -1, -1, -1],
        StaticArray[0, 5, 4, 0, 1, 5, 2, 3, 11, -1, -1, -1, -1, -1, -1, -1],
        StaticArray[2, 1, 5, 2, 5, 8, 2, 8, 11, 4, 8, 5, -1, -1, -1, -1],
        StaticArray[10, 3, 11, 10, 1, 3, 9, 5, 4, -1, -1, -1, -1, -1, -1, -1],
        StaticArray[4, 9, 5, 0, 8, 1, 8, 10, 1, 8, 11, 10, -1, -1, -1, -1],
        StaticArray[5, 4, 0, 5, 0, 11, 5, 11, 10, 11, 0, 3, -1, -1, -1, -1],
        StaticArray[5, 4, 8, 5, 8, 10, 10, 8, 11, -1, -1, -1, -1, -1, -1, -1],
        StaticArray[9, 7, 8, 5, 7, 9, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1],
        StaticArray[9, 3, 0, 9, 5, 3, 5, 7, 3, -1, -1, -1, -1, -1, -1, -1],
        StaticArray[0, 7, 8, 0, 1, 7, 1, 5, 7, -1, -1, -1, -1, -1, -1, -1],
        StaticArray[1, 5, 3, 3, 5, 7, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1],
        StaticArray[9, 7, 8, 9, 5, 7, 10, 1, 2, -1, -1, -1, -1, -1, -1, -1],
        StaticArray[10, 1, 2, 9, 5, 0, 5, 3, 0, 5, 7, 3, -1, -1, -1, -1],
        StaticArray[8, 0, 2, 8, 2, 5, 8, 5, 7, 10, 5, 2, -1, -1, -1, -1],
        StaticArray[2, 10, 5, 2, 5, 3, 3, 5, 7, -1, -1, -1, -1, -1, -1, -1],
        StaticArray[7, 9, 5, 7, 8, 9, 3, 11, 2, -1, -1, -1, -1, -1, -1, -1],
        StaticArray[9, 5, 7, 9, 7, 2, 9, 2, 0, 2, 7, 11, -1, -1, -1, -1],
        StaticArray[2, 3, 11, 0, 1, 8, 1, 7, 8, 1, 5, 7, -1, -1, -1, -1],
        StaticArray[11, 2, 1, 11, 1, 7, 7, 1, 5, -1, -1, -1, -1, -1, -1, -1],
        StaticArray[9, 5, 8, 8, 5, 7, 10, 1, 3, 10, 3, 11, -1, -1, -1, -1],
        StaticArray[5, 7, 0, 5, 0, 9, 7, 11, 0, 1, 0, 10, 11, 10, 0, -1],
        StaticArray[11, 10, 0, 11, 0, 3, 10, 5, 0, 8, 0, 7, 5, 7, 0, -1],
        StaticArray[11, 10, 5, 7, 11, 5, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1],
        StaticArray[10, 6, 5, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1],
        StaticArray[0, 8, 3, 5, 10, 6, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1],
        StaticArray[9, 0, 1, 5, 10, 6, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1],
        StaticArray[1, 8, 3, 1, 9, 8, 5, 10, 6, -1, -1, -1, -1, -1, -1, -1],
        StaticArray[1, 6, 5, 2, 6, 1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1],
        StaticArray[1, 6, 5, 1, 2, 6, 3, 0, 8, -1, -1, -1, -1, -1, -1, -1],
        StaticArray[9, 6, 5, 9, 0, 6, 0, 2, 6, -1, -1, -1, -1, -1, -1, -1],
        StaticArray[5, 9, 8, 5, 8, 2, 5, 2, 6, 3, 2, 8, -1, -1, -1, -1],
        StaticArray[2, 3, 11, 10, 6, 5, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1],
        StaticArray[11, 0, 8, 11, 2, 0, 10, 6, 5, -1, -1, -1, -1, -1, -1, -1],
        StaticArray[0, 1, 9, 2, 3, 11, 5, 10, 6, -1, -1, -1, -1, -1, -1, -1],
        StaticArray[5, 10, 6, 1, 9, 2, 9, 11, 2, 9, 8, 11, -1, -1, -1, -1],
        StaticArray[6, 3, 11, 6, 5, 3, 5, 1, 3, -1, -1, -1, -1, -1, -1, -1],
        StaticArray[0, 8, 11, 0, 11, 5, 0, 5, 1, 5, 11, 6, -1, -1, -1, -1],
        StaticArray[3, 11, 6, 0, 3, 6, 0, 6, 5, 0, 5, 9, -1, -1, -1, -1],
        StaticArray[6, 5, 9, 6, 9, 11, 11, 9, 8, -1, -1, -1, -1, -1, -1, -1],
        StaticArray[5, 10, 6, 4, 7, 8, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1],
        StaticArray[4, 3, 0, 4, 7, 3, 6, 5, 10, -1, -1, -1, -1, -1, -1, -1],
        StaticArray[1, 9, 0, 5, 10, 6, 8, 4, 7, -1, -1, -1, -1, -1, -1, -1],
        StaticArray[10, 6, 5, 1, 9, 7, 1, 7, 3, 7, 9, 4, -1, -1, -1, -1],
        StaticArray[6, 1, 2, 6, 5, 1, 4, 7, 8, -1, -1, -1, -1, -1, -1, -1],
        StaticArray[1, 2, 5, 5, 2, 6, 3, 0, 4, 3, 4, 7, -1, -1, -1, -1],
        StaticArray[8, 4, 7, 9, 0, 5, 0, 6, 5, 0, 2, 6, -1, -1, -1, -1],
        StaticArray[7, 3, 9, 7, 9, 4, 3, 2, 9, 5, 9, 6, 2, 6, 9, -1],
        StaticArray[3, 11, 2, 7, 8, 4, 10, 6, 5, -1, -1, -1, -1, -1, -1, -1],
        StaticArray[5, 10, 6, 4, 7, 2, 4, 2, 0, 2, 7, 11, -1, -1, -1, -1],
        StaticArray[0, 1, 9, 4, 7, 8, 2, 3, 11, 5, 10, 6, -1, -1, -1, -1],
        StaticArray[9, 2, 1, 9, 11, 2, 9, 4, 11, 7, 11, 4, 5, 10, 6, -1],
        StaticArray[8, 4, 7, 3, 11, 5, 3, 5, 1, 5, 11, 6, -1, -1, -1, -1],
        StaticArray[5, 1, 11, 5, 11, 6, 1, 0, 11, 7, 11, 4, 0, 4, 11, -1],
        StaticArray[0, 5, 9, 0, 6, 5, 0, 3, 6, 11, 6, 3, 8, 4, 7, -1],
        StaticArray[6, 5, 9, 6, 9, 11, 4, 7, 9, 7, 11, 9, -1, -1, -1, -1],
        StaticArray[10, 4, 9, 6, 4, 10, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1],
        StaticArray[4, 10, 6, 4, 9, 10, 0, 8, 3, -1, -1, -1, -1, -1, -1, -1],
        StaticArray[10, 0, 1, 10, 6, 0, 6, 4, 0, -1, -1, -1, -1, -1, -1, -1],
        StaticArray[8, 3, 1, 8, 1, 6, 8, 6, 4, 6, 1, 10, -1, -1, -1, -1],
        StaticArray[1, 4, 9, 1, 2, 4, 2, 6, 4, -1, -1, -1, -1, -1, -1, -1],
        StaticArray[3, 0, 8, 1, 2, 9, 2, 4, 9, 2, 6, 4, -1, -1, -1, -1],
        StaticArray[0, 2, 4, 4, 2, 6, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1],
        StaticArray[8, 3, 2, 8, 2, 4, 4, 2, 6, -1, -1, -1, -1, -1, -1, -1],
        StaticArray[10, 4, 9, 10, 6, 4, 11, 2, 3, -1, -1, -1, -1, -1, -1, -1],
        StaticArray[0, 8, 2, 2, 8, 11, 4, 9, 10, 4, 10, 6, -1, -1, -1, -1],
        StaticArray[3, 11, 2, 0, 1, 6, 0, 6, 4, 6, 1, 10, -1, -1, -1, -1],
        StaticArray[6, 4, 1, 6, 1, 10, 4, 8, 1, 2, 1, 11, 8, 11, 1, -1],
        StaticArray[9, 6, 4, 9, 3, 6, 9, 1, 3, 11, 6, 3, -1, -1, -1, -1],
        StaticArray[8, 11, 1, 8, 1, 0, 11, 6, 1, 9, 1, 4, 6, 4, 1, -1],
        StaticArray[3, 11, 6, 3, 6, 0, 0, 6, 4, -1, -1, -1, -1, -1, -1, -1],
        StaticArray[6, 4, 8, 11, 6, 8, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1],
        StaticArray[7, 10, 6, 7, 8, 10, 8, 9, 10, -1, -1, -1, -1, -1, -1, -1],
        StaticArray[0, 7, 3, 0, 10, 7, 0, 9, 10, 6, 7, 10, -1, -1, -1, -1],
        StaticArray[10, 6, 7, 1, 10, 7, 1, 7, 8, 1, 8, 0, -1, -1, -1, -1],
        StaticArray[10, 6, 7, 10, 7, 1, 1, 7, 3, -1, -1, -1, -1, -1, -1, -1],
        StaticArray[1, 2, 6, 1, 6, 8, 1, 8, 9, 8, 6, 7, -1, -1, -1, -1],
        StaticArray[2, 6, 9, 2, 9, 1, 6, 7, 9, 0, 9, 3, 7, 3, 9, -1],
        StaticArray[7, 8, 0, 7, 0, 6, 6, 0, 2, -1, -1, -1, -1, -1, -1, -1],
        StaticArray[7, 3, 2, 6, 7, 2, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1],
        StaticArray[2, 3, 11, 10, 6, 8, 10, 8, 9, 8, 6, 7, -1, -1, -1, -1],
        StaticArray[2, 0, 7, 2, 7, 11, 0, 9, 7, 6, 7, 10, 9, 10, 7, -1],
        StaticArray[1, 8, 0, 1, 7, 8, 1, 10, 7, 6, 7, 10, 2, 3, 11, -1],
        StaticArray[11, 2, 1, 11, 1, 7, 10, 6, 1, 6, 7, 1, -1, -1, -1, -1],
        StaticArray[8, 9, 6, 8, 6, 7, 9, 1, 6, 11, 6, 3, 1, 3, 6, -1],
        StaticArray[0, 9, 1, 11, 6, 7, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1],
        StaticArray[7, 8, 0, 7, 0, 6, 3, 11, 0, 11, 6, 0, -1, -1, -1, -1],
        StaticArray[7, 11, 6, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1],
        StaticArray[7, 6, 11, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1],
        StaticArray[3, 0, 8, 11, 7, 6, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1],
        StaticArray[0, 1, 9, 11, 7, 6, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1],
        StaticArray[8, 1, 9, 8, 3, 1, 11, 7, 6, -1, -1, -1, -1, -1, -1, -1],
        StaticArray[10, 1, 2, 6, 11, 7, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1],
        StaticArray[1, 2, 10, 3, 0, 8, 6, 11, 7, -1, -1, -1, -1, -1, -1, -1],
        StaticArray[2, 9, 0, 2, 10, 9, 6, 11, 7, -1, -1, -1, -1, -1, -1, -1],
        StaticArray[6, 11, 7, 2, 10, 3, 10, 8, 3, 10, 9, 8, -1, -1, -1, -1],
        StaticArray[7, 2, 3, 6, 2, 7, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1],
        StaticArray[7, 0, 8, 7, 6, 0, 6, 2, 0, -1, -1, -1, -1, -1, -1, -1],
        StaticArray[2, 7, 6, 2, 3, 7, 0, 1, 9, -1, -1, -1, -1, -1, -1, -1],
        StaticArray[1, 6, 2, 1, 8, 6, 1, 9, 8, 8, 7, 6, -1, -1, -1, -1],
        StaticArray[10, 7, 6, 10, 1, 7, 1, 3, 7, -1, -1, -1, -1, -1, -1, -1],
        StaticArray[10, 7, 6, 1, 7, 10, 1, 8, 7, 1, 0, 8, -1, -1, -1, -1],
        StaticArray[0, 3, 7, 0, 7, 10, 0, 10, 9, 6, 10, 7, -1, -1, -1, -1],
        StaticArray[7, 6, 10, 7, 10, 8, 8, 10, 9, -1, -1, -1, -1, -1, -1, -1],
        StaticArray[6, 8, 4, 11, 8, 6, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1],
        StaticArray[3, 6, 11, 3, 0, 6, 0, 4, 6, -1, -1, -1, -1, -1, -1, -1],
        StaticArray[8, 6, 11, 8, 4, 6, 9, 0, 1, -1, -1, -1, -1, -1, -1, -1],
        StaticArray[9, 4, 6, 9, 6, 3, 9, 3, 1, 11, 3, 6, -1, -1, -1, -1],
        StaticArray[6, 8, 4, 6, 11, 8, 2, 10, 1, -1, -1, -1, -1, -1, -1, -1],
        StaticArray[1, 2, 10, 3, 0, 11, 0, 6, 11, 0, 4, 6, -1, -1, -1, -1],
        StaticArray[4, 11, 8, 4, 6, 11, 0, 2, 9, 2, 10, 9, -1, -1, -1, -1],
        StaticArray[10, 9, 3, 10, 3, 2, 9, 4, 3, 11, 3, 6, 4, 6, 3, -1],
        StaticArray[8, 2, 3, 8, 4, 2, 4, 6, 2, -1, -1, -1, -1, -1, -1, -1],
        StaticArray[0, 4, 2, 4, 6, 2, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1],
        StaticArray[1, 9, 0, 2, 3, 4, 2, 4, 6, 4, 3, 8, -1, -1, -1, -1],
        StaticArray[1, 9, 4, 1, 4, 2, 2, 4, 6, -1, -1, -1, -1, -1, -1, -1],
        StaticArray[8, 1, 3, 8, 6, 1, 8, 4, 6, 6, 10, 1, -1, -1, -1, -1],
        StaticArray[10, 1, 0, 10, 0, 6, 6, 0, 4, -1, -1, -1, -1, -1, -1, -1],
        StaticArray[4, 6, 3, 4, 3, 8, 6, 10, 3, 0, 3, 9, 10, 9, 3, -1],
        StaticArray[10, 9, 4, 6, 10, 4, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1],
        StaticArray[4, 9, 5, 7, 6, 11, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1],
        StaticArray[0, 8, 3, 4, 9, 5, 11, 7, 6, -1, -1, -1, -1, -1, -1, -1],
        StaticArray[5, 0, 1, 5, 4, 0, 7, 6, 11, -1, -1, -1, -1, -1, -1, -1],
        StaticArray[11, 7, 6, 8, 3, 4, 3, 5, 4, 3, 1, 5, -1, -1, -1, -1],
        StaticArray[9, 5, 4, 10, 1, 2, 7, 6, 11, -1, -1, -1, -1, -1, -1, -1],
        StaticArray[6, 11, 7, 1, 2, 10, 0, 8, 3, 4, 9, 5, -1, -1, -1, -1],
        StaticArray[7, 6, 11, 5, 4, 10, 4, 2, 10, 4, 0, 2, -1, -1, -1, -1],
        StaticArray[3, 4, 8, 3, 5, 4, 3, 2, 5, 10, 5, 2, 11, 7, 6, -1],
        StaticArray[7, 2, 3, 7, 6, 2, 5, 4, 9, -1, -1, -1, -1, -1, -1, -1],
        StaticArray[9, 5, 4, 0, 8, 6, 0, 6, 2, 6, 8, 7, -1, -1, -1, -1],
        StaticArray[3, 6, 2, 3, 7, 6, 1, 5, 0, 5, 4, 0, -1, -1, -1, -1],
        StaticArray[6, 2, 8, 6, 8, 7, 2, 1, 8, 4, 8, 5, 1, 5, 8, -1],
        StaticArray[9, 5, 4, 10, 1, 6, 1, 7, 6, 1, 3, 7, -1, -1, -1, -1],
        StaticArray[1, 6, 10, 1, 7, 6, 1, 0, 7, 8, 7, 0, 9, 5, 4, -1],
        StaticArray[4, 0, 10, 4, 10, 5, 0, 3, 10, 6, 10, 7, 3, 7, 10, -1],
        StaticArray[7, 6, 10, 7, 10, 8, 5, 4, 10, 4, 8, 10, -1, -1, -1, -1],
        StaticArray[6, 9, 5, 6, 11, 9, 11, 8, 9, -1, -1, -1, -1, -1, -1, -1],
        StaticArray[3, 6, 11, 0, 6, 3, 0, 5, 6, 0, 9, 5, -1, -1, -1, -1],
        StaticArray[0, 11, 8, 0, 5, 11, 0, 1, 5, 5, 6, 11, -1, -1, -1, -1],
        StaticArray[6, 11, 3, 6, 3, 5, 5, 3, 1, -1, -1, -1, -1, -1, -1, -1],
        StaticArray[1, 2, 10, 9, 5, 11, 9, 11, 8, 11, 5, 6, -1, -1, -1, -1],
        StaticArray[0, 11, 3, 0, 6, 11, 0, 9, 6, 5, 6, 9, 1, 2, 10, -1],
        StaticArray[11, 8, 5, 11, 5, 6, 8, 0, 5, 10, 5, 2, 0, 2, 5, -1],
        StaticArray[6, 11, 3, 6, 3, 5, 2, 10, 3, 10, 5, 3, -1, -1, -1, -1],
        StaticArray[5, 8, 9, 5, 2, 8, 5, 6, 2, 3, 8, 2, -1, -1, -1, -1],
        StaticArray[9, 5, 6, 9, 6, 0, 0, 6, 2, -1, -1, -1, -1, -1, -1, -1],
        StaticArray[1, 5, 8, 1, 8, 0, 5, 6, 8, 3, 8, 2, 6, 2, 8, -1],
        StaticArray[1, 5, 6, 2, 1, 6, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1],
        StaticArray[1, 3, 6, 1, 6, 10, 3, 8, 6, 5, 6, 9, 8, 9, 6, -1],
        StaticArray[10, 1, 0, 10, 0, 6, 9, 5, 0, 5, 6, 0, -1, -1, -1, -1],
        StaticArray[0, 3, 8, 5, 6, 10, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1],
        StaticArray[10, 5, 6, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1],
        StaticArray[11, 5, 10, 7, 5, 11, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1],
        StaticArray[11, 5, 10, 11, 7, 5, 8, 3, 0, -1, -1, -1, -1, -1, -1, -1],
        StaticArray[5, 11, 7, 5, 10, 11, 1, 9, 0, -1, -1, -1, -1, -1, -1, -1],
        StaticArray[10, 7, 5, 10, 11, 7, 9, 8, 1, 8, 3, 1, -1, -1, -1, -1],
        StaticArray[11, 1, 2, 11, 7, 1, 7, 5, 1, -1, -1, -1, -1, -1, -1, -1],
        StaticArray[0, 8, 3, 1, 2, 7, 1, 7, 5, 7, 2, 11, -1, -1, -1, -1],
        StaticArray[9, 7, 5, 9, 2, 7, 9, 0, 2, 2, 11, 7, -1, -1, -1, -1],
        StaticArray[7, 5, 2, 7, 2, 11, 5, 9, 2, 3, 2, 8, 9, 8, 2, -1],
        StaticArray[2, 5, 10, 2, 3, 5, 3, 7, 5, -1, -1, -1, -1, -1, -1, -1],
        StaticArray[8, 2, 0, 8, 5, 2, 8, 7, 5, 10, 2, 5, -1, -1, -1, -1],
        StaticArray[9, 0, 1, 5, 10, 3, 5, 3, 7, 3, 10, 2, -1, -1, -1, -1],
        StaticArray[9, 8, 2, 9, 2, 1, 8, 7, 2, 10, 2, 5, 7, 5, 2, -1],
        StaticArray[1, 3, 5, 3, 7, 5, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1],
        StaticArray[0, 8, 7, 0, 7, 1, 1, 7, 5, -1, -1, -1, -1, -1, -1, -1],
        StaticArray[9, 0, 3, 9, 3, 5, 5, 3, 7, -1, -1, -1, -1, -1, -1, -1],
        StaticArray[9, 8, 7, 5, 9, 7, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1],
        StaticArray[5, 8, 4, 5, 10, 8, 10, 11, 8, -1, -1, -1, -1, -1, -1, -1],
        StaticArray[5, 0, 4, 5, 11, 0, 5, 10, 11, 11, 3, 0, -1, -1, -1, -1],
        StaticArray[0, 1, 9, 8, 4, 10, 8, 10, 11, 10, 4, 5, -1, -1, -1, -1],
        StaticArray[10, 11, 4, 10, 4, 5, 11, 3, 4, 9, 4, 1, 3, 1, 4, -1],
        StaticArray[2, 5, 1, 2, 8, 5, 2, 11, 8, 4, 5, 8, -1, -1, -1, -1],
        StaticArray[0, 4, 11, 0, 11, 3, 4, 5, 11, 2, 11, 1, 5, 1, 11, -1],
        StaticArray[0, 2, 5, 0, 5, 9, 2, 11, 5, 4, 5, 8, 11, 8, 5, -1],
        StaticArray[9, 4, 5, 2, 11, 3, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1],
        StaticArray[2, 5, 10, 3, 5, 2, 3, 4, 5, 3, 8, 4, -1, -1, -1, -1],
        StaticArray[5, 10, 2, 5, 2, 4, 4, 2, 0, -1, -1, -1, -1, -1, -1, -1],
        StaticArray[3, 10, 2, 3, 5, 10, 3, 8, 5, 4, 5, 8, 0, 1, 9, -1],
        StaticArray[5, 10, 2, 5, 2, 4, 1, 9, 2, 9, 4, 2, -1, -1, -1, -1],
        StaticArray[8, 4, 5, 8, 5, 3, 3, 5, 1, -1, -1, -1, -1, -1, -1, -1],
        StaticArray[0, 4, 5, 1, 0, 5, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1],
        StaticArray[8, 4, 5, 8, 5, 3, 9, 0, 5, 0, 3, 5, -1, -1, -1, -1],
        StaticArray[9, 4, 5, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1],
        StaticArray[4, 11, 7, 4, 9, 11, 9, 10, 11, -1, -1, -1, -1, -1, -1, -1],
        StaticArray[0, 8, 3, 4, 9, 7, 9, 11, 7, 9, 10, 11, -1, -1, -1, -1],
        StaticArray[1, 10, 11, 1, 11, 4, 1, 4, 0, 7, 4, 11, -1, -1, -1, -1],
        StaticArray[3, 1, 4, 3, 4, 8, 1, 10, 4, 7, 4, 11, 10, 11, 4, -1],
        StaticArray[4, 11, 7, 9, 11, 4, 9, 2, 11, 9, 1, 2, -1, -1, -1, -1],
        StaticArray[9, 7, 4, 9, 11, 7, 9, 1, 11, 2, 11, 1, 0, 8, 3, -1],
        StaticArray[11, 7, 4, 11, 4, 2, 2, 4, 0, -1, -1, -1, -1, -1, -1, -1],
        StaticArray[11, 7, 4, 11, 4, 2, 8, 3, 4, 3, 2, 4, -1, -1, -1, -1],
        StaticArray[2, 9, 10, 2, 7, 9, 2, 3, 7, 7, 4, 9, -1, -1, -1, -1],
        StaticArray[9, 10, 7, 9, 7, 4, 10, 2, 7, 8, 7, 0, 2, 0, 7, -1],
        StaticArray[3, 7, 10, 3, 10, 2, 7, 4, 10, 1, 10, 0, 4, 0, 10, -1],
        StaticArray[1, 10, 2, 8, 7, 4, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1],
        StaticArray[4, 9, 1, 4, 1, 7, 7, 1, 3, -1, -1, -1, -1, -1, -1, -1],
        StaticArray[4, 9, 1, 4, 1, 7, 0, 8, 1, 8, 7, 1, -1, -1, -1, -1],
        StaticArray[4, 0, 3, 7, 4, 3, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1],
        StaticArray[4, 8, 7, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1],
        StaticArray[9, 10, 8, 10, 11, 8, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1],
        StaticArray[3, 0, 9, 3, 9, 11, 11, 9, 10, -1, -1, -1, -1, -1, -1, -1],
        StaticArray[0, 1, 10, 0, 10, 8, 8, 10, 11, -1, -1, -1, -1, -1, -1, -1],
        StaticArray[3, 1, 10, 11, 3, 10, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1],
        StaticArray[1, 2, 11, 1, 11, 9, 9, 11, 8, -1, -1, -1, -1, -1, -1, -1],
        StaticArray[3, 0, 9, 3, 9, 11, 1, 2, 9, 2, 11, 9, -1, -1, -1, -1],
        StaticArray[0, 2, 11, 8, 0, 11, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1],
        StaticArray[3, 2, 11, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1],
        StaticArray[2, 3, 8, 2, 8, 10, 10, 8, 9, -1, -1, -1, -1, -1, -1, -1],
        StaticArray[9, 10, 2, 0, 9, 2, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1],
        StaticArray[2, 3, 8, 2, 8, 10, 0, 1, 8, 1, 10, 8, -1, -1, -1, -1],
        StaticArray[1, 10, 2, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1],
        StaticArray[1, 3, 8, 9, 1, 8, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1],
        StaticArray[0, 9, 1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1],
        StaticArray[0, 3, 8, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1],
        StaticArray[-1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1],
      ]
    end
  end
end
