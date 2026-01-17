# Geometry utilities for 3D reconstruction
# Ported from folding project: Vec3, Distance Geometry, Kabsch alignment

module GS
  module Utils
    # Lightweight 3D vector
    struct Vec3
      getter x : Float64
      getter y : Float64
      getter z : Float64

      def initialize(@x : Float64, @y : Float64, @z : Float64)
      end

      def self.zero : Vec3
        Vec3.new(0.0, 0.0, 0.0)
      end

      def +(other : Vec3) : Vec3
        Vec3.new(@x + other.x, @y + other.y, @z + other.z)
      end

      def -(other : Vec3) : Vec3
        Vec3.new(@x - other.x, @y - other.y, @z - other.z)
      end

      def - : Vec3
        Vec3.new(-@x, -@y, -@z)
      end

      def *(scalar : Float64) : Vec3
        Vec3.new(@x * scalar, @y * scalar, @z * scalar)
      end

      def /(scalar : Float64) : Vec3
        Vec3.new(@x / scalar, @y / scalar, @z / scalar)
      end

      def dot(other : Vec3) : Float64
        @x * other.x + @y * other.y + @z * other.z
      end

      def cross(other : Vec3) : Vec3
        Vec3.new(
          @y * other.z - @z * other.y,
          @z * other.x - @x * other.z,
          @x * other.y - @y * other.x
        )
      end

      def magnitude_squared : Float64
        dot(self)
      end

      def magnitude : Float64
        Math.sqrt(magnitude_squared)
      end

      def length : Float64
        magnitude
      end

      def normalize : Vec3
        mag = magnitude
        return Vec3.zero if mag < 1e-10
        self * (1.0 / mag)
      end

      def distance_to(other : Vec3) : Float64
        (self - other).magnitude
      end

      def to_a : Array(Float64)
        [@x, @y, @z]
      end

      def dup : Vec3
        Vec3.new(@x, @y, @z)
      end
    end

    # Distance constraint for geometry optimization
    struct DistanceConstraint
      property i : Int32
      property j : Int32
      property distance : Float64
      property weight : Float64

      def initialize(@i, @j, @distance, @weight = 1.0)
      end
    end

    # Geometry utilities module
    module Geometry
      extend self

      # Compute centroid of point cloud
      def center(coords : Array(Vec3)) : Vec3
        return Vec3.zero if coords.empty?
        sum = coords.reduce(Vec3.zero) { |acc, v| acc + v }
        Vec3.new(sum.x / coords.size, sum.y / coords.size, sum.z / coords.size)
      end

      # Compute RMSD between two point sets (no alignment)
      def rmsd(coords1 : Array(Vec3), coords2 : Array(Vec3)) : Float64
        return Float64::MAX if coords1.size != coords2.size
        n = coords1.size
        return 0.0 if n == 0

        c1 = center(coords1)
        c2 = center(coords2)

        centered1 = coords1.map { |v| v - c1 }
        centered2 = coords2.map { |v| v - c2 }

        sum_sq = 0.0
        n.times { |i| sum_sq += centered1[i].distance_to(centered2[i]) ** 2 }

        Math.sqrt(sum_sq / n)
      end

      # Kabsch RMSD with optimal rotation alignment
      def kabsch_rmsd(coords1 : Array(Vec3), coords2 : Array(Vec3)) : Float64
        return Float64::MAX if coords1.size != coords2.size
        n = coords1.size
        return 0.0 if n == 0

        c1 = center(coords1)
        c2 = center(coords2)

        centered1 = coords1.map { |v| v - c1 }
        centered2 = coords2.map { |v| v - c2 }

        rotation = kabsch_rotation(centered1, centered2)

        sum_sq = 0.0
        n.times do |i|
          rotated = rotate_point(centered1[i], rotation)
          sum_sq += rotated.distance_to(centered2[i]) ** 2
        end

        kabsch_result = Math.sqrt(sum_sq / n)

        # Sanity check
        simple_sum_sq = 0.0
        n.times { |i| simple_sum_sq += centered1[i].distance_to(centered2[i]) ** 2 }
        simple_result = Math.sqrt(simple_sum_sq / n)

        kabsch_result <= simple_result ? kabsch_result : simple_result
      end

      # Align coords1 onto coords2 using Kabsch algorithm
      def kabsch_align(coords1 : Array(Vec3), coords2 : Array(Vec3)) : Array(Vec3)
        return coords1.map(&.dup) if coords1.size != coords2.size || coords1.empty?

        c1 = center(coords1)
        c2 = center(coords2)

        centered1 = coords1.map { |v| v - c1 }
        centered2 = coords2.map { |v| v - c2 }

        rotation = kabsch_rotation(centered1, centered2)

        centered1.map do |p|
          rotated = rotate_point(p, rotation)
          Vec3.new(rotated.x + c2.x, rotated.y + c2.y, rotated.z + c2.z)
        end
      end

      # Get Kabsch transform (rotation + translation)
      def kabsch_transform(coords1 : Array(Vec3), coords2 : Array(Vec3)) : NamedTuple(rotation: Array(Array(Float64)), c1: Vec3, c2: Vec3)?
        return nil if coords1.size != coords2.size || coords1.size < 3

        c1 = center(coords1)
        c2 = center(coords2)
        centered1 = coords1.map { |v| v - c1 }
        centered2 = coords2.map { |v| v - c2 }
        rotation = kabsch_rotation(centered1, centered2)
        {rotation: rotation, c1: c1, c2: c2}
      end

      # Apply Kabsch transform to a point
      def apply_transform(p : Vec3, transform : NamedTuple(rotation: Array(Array(Float64)), c1: Vec3, c2: Vec3)) : Vec3
        centered = p - transform[:c1]
        rotated = rotate_point(centered, transform[:rotation])
        Vec3.new(rotated.x + transform[:c2].x, rotated.y + transform[:c2].y, rotated.z + transform[:c2].z)
      end

      # ICP (Iterative Closest Point) registration
      # Aligns source point cloud to target
      def icp(source : Array(Vec3), target : Array(Vec3), iterations : Int32 = 50, tolerance : Float64 = 1e-6) : Array(Vec3)
        return source.map(&.dup) if source.empty? || target.empty?

        current = source.map(&.dup)
        prev_error = Float64::MAX

        iterations.times do |iter|
          # Find closest points in target for each source point
          correspondences = current.map do |p|
            closest_idx = 0
            closest_dist = Float64::MAX
            target.each_with_index do |t, i|
              d = p.distance_to(t)
              if d < closest_dist
                closest_dist = d
                closest_idx = i
              end
            end
            target[closest_idx]
          end

          # Align using Kabsch
          current = kabsch_align(current, correspondences)

          # Check convergence
          error = 0.0
          current.each_with_index { |p, i| error += p.distance_to(correspondences[i]) }
          error /= current.size

          break if (prev_error - error).abs < tolerance
          prev_error = error
        end

        current
      end

      # Estimate normal at a point using PCA of neighborhood
      def estimate_normal(point : Vec3, neighbors : Array(Vec3)) : Vec3
        return Vec3.new(0.0, 0.0, 1.0) if neighbors.size < 3

        # Compute covariance matrix
        centroid = center(neighbors)
        cov = Array.new(3) { Array.new(3, 0.0) }

        neighbors.each do |p|
          d = p - centroid
          cov[0][0] += d.x * d.x
          cov[0][1] += d.x * d.y
          cov[0][2] += d.x * d.z
          cov[1][0] += d.y * d.x
          cov[1][1] += d.y * d.y
          cov[1][2] += d.y * d.z
          cov[2][0] += d.z * d.x
          cov[2][1] += d.z * d.y
          cov[2][2] += d.z * d.z
        end

        # Find eigenvector with smallest eigenvalue (normal direction)
        normal = smallest_eigenvector(cov)
        normal.normalize
      end

      # Fit plane to points, return normal and distance from origin
      def fit_plane(points : Array(Vec3)) : {Vec3, Float64}
        return {Vec3.new(0.0, 0.0, 1.0), 0.0} if points.size < 3

        centroid = center(points)
        normal = estimate_normal(centroid, points)
        d = normal.dot(centroid)
        {normal, d}
      end

      # Distance Geometry: solve 3D coordinates from pairwise distances
      def solve_distance_geometry(n_atoms : Int32, constraints : Array(DistanceConstraint), min_coverage : Float64? = nil) : Array(Vec3)?
        actual_min_coverage = min_coverage || auto_min_coverage(n_atoms)
        return nil if n_atoms < 4 || constraints.empty?

        # Build distance matrix (squared distances)
        dist_sq = Array.new(n_atoms) { Array.new(n_atoms, -1.0) }
        n_atoms.times { |i| dist_sq[i][i] = 0.0 }

        constraints.each do |c|
          next if c.i >= n_atoms || c.j >= n_atoms
          dist_sq[c.i][c.j] = c.distance ** 2
          dist_sq[c.j][c.i] = c.distance ** 2
        end

        # Check coverage
        known_pairs = 0
        total_pairs = n_atoms * (n_atoms - 1) / 2
        n_atoms.times do |i|
          (i + 1).upto(n_atoms - 1) { |j| known_pairs += 1 if dist_sq[i][j] >= 0 }
        end

        coverage = total_pairs == 0 ? 0.0 : known_pairs.to_f / total_pairs.to_f
        return nil if coverage < actual_min_coverage

        # Fill missing distances
        fill_missing_distances!(dist_sq, n_atoms)
        return nil unless dist_sq[0].all? { |d| d >= 0.0 }

        # Convert to Gram matrix
        gram = Array.new(n_atoms) { Array.new(n_atoms, 0.0) }
        n_atoms.times do |i|
          n_atoms.times do |j|
            d0i = dist_sq[0][i]
            d0j = dist_sq[0][j]
            dij = dist_sq[i][j]
            next if d0i < 0 || d0j < 0 || dij < 0
            gram[i][j] = (d0i + d0j - dij) / 2.0
          end
        end

        return nil if gram.any? { |row| row.any? { |v| v.nan? || v.infinite? } }

        eigendecompose_3d(gram, n_atoms)
      end

      # Refine coordinates to satisfy constraints
      def refine_coordinates!(coords : Array(Vec3), constraints : Array(DistanceConstraint), iterations : Int32 = 100)
        return if constraints.empty?
        n = coords.size

        sorted_constraints = constraints.sort_by { |c| -c.weight }
        base_step = Math.min(0.2, 3.0 / Math.sqrt(n.to_f))
        min_step = 0.01

        iterations.times do |iter|
          progress = iter.to_f / iterations
          step_size = min_step + (base_step - min_step) * (1.0 + Math.cos(Math::PI * progress)) / 2.0

          # Satisfy distance constraints
          sorted_constraints.each do |c|
            next if c.i >= n || c.j >= n

            current_dist = coords[c.i].distance_to(coords[c.j])
            next if current_dist < 0.01

            error = current_dist - c.distance
            diff = coords[c.j] - coords[c.i]
            diff_len = diff.magnitude
            next if diff_len < 0.001

            direction = diff * (1.0 / diff_len)
            weight_factor = Math.min(c.weight / 5.0, 2.0)
            correction = direction * (error * step_size * weight_factor)

            coords[c.i] = coords[c.i] + correction
            coords[c.j] = coords[c.j] - correction
          end
        end
      end

      # Private helpers

      private def auto_min_coverage(n_atoms : Int32) : Float64
        return 0.0 if n_atoms <= 0
        [[1.5 / n_atoms, 0.005].max, 0.1].min
      end

      private def kabsch_rotation(p1 : Array(Vec3), p2 : Array(Vec3)) : Array(Array(Float64))
        h = Array.new(3) { Array.new(3, 0.0) }

        p1.size.times do |k|
          h[0][0] += p1[k].x * p2[k].x
          h[0][1] += p1[k].x * p2[k].y
          h[0][2] += p1[k].x * p2[k].z
          h[1][0] += p1[k].y * p2[k].x
          h[1][1] += p1[k].y * p2[k].y
          h[1][2] += p1[k].y * p2[k].z
          h[2][0] += p1[k].z * p2[k].x
          h[2][1] += p1[k].z * p2[k].y
          h[2][2] += p1[k].z * p2[k].z
        end

        u, s, v = svd_3x3(h)
        det = determinant_3x3(matrix_multiply_3x3(v, transpose_3x3(u)))

        if det < 0
          v[0][2] = -v[0][2]
          v[1][2] = -v[1][2]
          v[2][2] = -v[2][2]
        end

        matrix_multiply_3x3(v, transpose_3x3(u))
      end

      private def rotate_point(p : Vec3, r : Array(Array(Float64))) : Vec3
        Vec3.new(
          r[0][0] * p.x + r[0][1] * p.y + r[0][2] * p.z,
          r[1][0] * p.x + r[1][1] * p.y + r[1][2] * p.z,
          r[2][0] * p.x + r[2][1] * p.y + r[2][2] * p.z
        )
      end

      private def transpose_3x3(m : Array(Array(Float64))) : Array(Array(Float64))
        Array.new(3) { |i| Array.new(3) { |j| m[j][i] } }
      end

      private def matrix_multiply_3x3(a : Array(Array(Float64)), b : Array(Array(Float64))) : Array(Array(Float64))
        result = Array.new(3) { Array.new(3, 0.0) }
        3.times do |i|
          3.times do |j|
            3.times { |k| result[i][j] += a[i][k] * b[k][j] }
          end
        end
        result
      end

      private def determinant_3x3(m : Array(Array(Float64))) : Float64
        m[0][0] * (m[1][1] * m[2][2] - m[1][2] * m[2][1]) -
        m[0][1] * (m[1][0] * m[2][2] - m[1][2] * m[2][0]) +
        m[0][2] * (m[1][0] * m[2][1] - m[1][1] * m[2][0])
      end

      # SVD for 3x3 matrix using Jacobi iterations
      private def svd_3x3(m : Array(Array(Float64))) : {Array(Array(Float64)), Array(Float64), Array(Array(Float64))}
        mtm = matrix_multiply_3x3(transpose_3x3(m), m)
        v = Array.new(3) { |i| Array.new(3) { |j| i == j ? 1.0 : 0.0 } }

        30.times do
          max_val = 0.0
          p_idx = 0
          q_idx = 1

          3.times do |i|
            3.times do |j|
              next if i >= j
              if mtm[i][j].abs > max_val
                max_val = mtm[i][j].abs
                p_idx = i
                q_idx = j
              end
            end
          end

          break if max_val < 1e-10

          diff = mtm[q_idx][q_idx] - mtm[p_idx][p_idx]
          theta = if diff.abs < 1e-10
                    Math::PI / 4
                  else
                    0.5 * Math.atan2(2.0 * mtm[p_idx][q_idx], diff)
                  end

          cos_t = Math.cos(theta)
          sin_t = Math.sin(theta)

          new_mtm = mtm.map(&.dup)
          3.times do |i|
            new_mtm[i][p_idx] = cos_t * mtm[i][p_idx] - sin_t * mtm[i][q_idx]
            new_mtm[i][q_idx] = sin_t * mtm[i][p_idx] + cos_t * mtm[i][q_idx]
          end

          mtm = new_mtm.map(&.dup)
          3.times do |i|
            new_mtm[p_idx][i] = cos_t * mtm[p_idx][i] - sin_t * mtm[q_idx][i]
            new_mtm[q_idx][i] = sin_t * mtm[p_idx][i] + cos_t * mtm[q_idx][i]
          end
          mtm = new_mtm

          new_v = v.map(&.dup)
          3.times do |i|
            new_v[i][p_idx] = cos_t * v[i][p_idx] - sin_t * v[i][q_idx]
            new_v[i][q_idx] = sin_t * v[i][p_idx] + cos_t * v[i][q_idx]
          end
          v = new_v
        end

        s = [Math.sqrt(mtm[0][0].abs), Math.sqrt(mtm[1][1].abs), Math.sqrt(mtm[2][2].abs)]

        mv = matrix_multiply_3x3(m, v)
        u = Array.new(3) { Array.new(3, 0.0) }

        3.times do |j|
          if s[j] > 1e-10
            3.times { |i| u[i][j] = mv[i][j] / s[j] }
          else
            3.times { |i| u[i][j] = i == j ? 1.0 : 0.0 }
          end
        end

        u = gram_schmidt_3x3(u)
        {u, s, v}
      end

      private def gram_schmidt_3x3(m : Array(Array(Float64))) : Array(Array(Float64))
        result = m.map(&.dup)

        norm0 = Math.sqrt(result[0][0]**2 + result[1][0]**2 + result[2][0]**2)
        if norm0 > 1e-10
          3.times { |i| result[i][0] /= norm0 }
        end

        dot01 = result[0][0] * result[0][1] + result[1][0] * result[1][1] + result[2][0] * result[2][1]
        3.times { |i| result[i][1] -= dot01 * result[i][0] }
        norm1 = Math.sqrt(result[0][1]**2 + result[1][1]**2 + result[2][1]**2)
        if norm1 > 1e-10
          3.times { |i| result[i][1] /= norm1 }
        end

        result[0][2] = result[1][0] * result[2][1] - result[2][0] * result[1][1]
        result[1][2] = result[2][0] * result[0][1] - result[0][0] * result[2][1]
        result[2][2] = result[0][0] * result[1][1] - result[1][0] * result[0][1]

        result
      end

      # Find eigenvector with smallest eigenvalue (for normal estimation)
      private def smallest_eigenvector(m : Array(Array(Float64))) : Vec3
        u, s, v = svd_3x3(m)

        # Find index of smallest singular value
        min_idx = 0
        min_val = s[0]
        3.times do |i|
          if s[i] < min_val
            min_val = s[i]
            min_idx = i
          end
        end

        Vec3.new(v[0][min_idx], v[1][min_idx], v[2][min_idx])
      end

      private def eigendecompose_3d(gram : Array(Array(Float64)), n : Int32) : Array(Vec3)?
        coords = Array.new(n) { Vec3.zero }
        matrix = gram.map(&.dup)
        rng = Random::DEFAULT

        3.times do |dim|
          vec = Array.new(n) { rng.rand - 0.5 }
          normalize_array!(vec)

          50.times do
            new_vec = multiply_matrix_vector(matrix, vec)
            normalize_array!(new_vec)

            diff = 0.0
            n.times { |i| diff += (new_vec[i] - vec[i]).abs }
            vec = new_vec
            break if diff < 1e-8
          end

          mv = multiply_matrix_vector(matrix, vec)
          eigenvalue = dot_array(mv, vec)

          min_eigenvalue = dim == 2 ? 1.0 : 0.0
          effective_eigenvalue = Math.max(eigenvalue.abs, min_eigenvalue)
          scale = Math.sqrt(effective_eigenvalue)

          n.times do |i|
            case dim
            when 0 then coords[i] = Vec3.new(vec[i] * scale, coords[i].y, coords[i].z)
            when 1 then coords[i] = Vec3.new(coords[i].x, vec[i] * scale, coords[i].z)
            when 2 then coords[i] = Vec3.new(coords[i].x, coords[i].y, vec[i] * scale)
            end
          end

          n.times do |i|
            n.times { |j| matrix[i][j] -= eigenvalue * vec[i] * vec[j] }
          end
        end

        coords
      end

      private def multiply_matrix_vector(m : Array(Array(Float64)), v : Array(Float64)) : Array(Float64)
        n = v.size
        result = Array.new(n, 0.0)
        n.times { |i| n.times { |j| result[i] += m[i][j] * v[j] } }
        result
      end

      private def dot_array(a : Array(Float64), b : Array(Float64)) : Float64
        sum = 0.0
        a.size.times { |i| sum += a[i] * b[i] }
        sum
      end

      private def normalize_array!(v : Array(Float64))
        mag = Math.sqrt(dot_array(v, v))
        return if mag == 0
        v.size.times { |i| v[i] /= mag }
      end

      private def fill_missing_distances!(dist_sq : Array(Array(Float64)), n : Int32)
        5.times do
          changed = false
          n.times do |i|
            n.times do |j|
              next if i == j
              next if dist_sq[i][j] >= 0

              upper_bound = Float64::MAX
              lower_bound = 0.0

              n.times do |k|
                next if k == i || k == j
                dik = dist_sq[i][k]
                dkj = dist_sq[k][j]
                next if dik < 0 || dkj < 0

                dik_sqrt = Math.sqrt(dik)
                dkj_sqrt = Math.sqrt(dkj)

                ub = (dik_sqrt + dkj_sqrt) ** 2
                upper_bound = ub if ub < upper_bound

                lb = (dik_sqrt - dkj_sqrt).abs ** 2
                lower_bound = lb if lb > lower_bound
              end

              if upper_bound < Float64::MAX && lower_bound <= upper_bound
                dist_sq[i][j] = (upper_bound + lower_bound) / 2.0
                dist_sq[j][i] = dist_sq[i][j]
                changed = true
              end
            end
          end
          break unless changed
        end

        n.times do |i|
          n.times do |j|
            next if dist_sq[i][j] >= 0
            seq_dist = (i - j).abs * 3.8
            dist_sq[i][j] = seq_dist ** 2
            dist_sq[j][i] = dist_sq[i][j]
          end
        end
      end
    end
  end
end
