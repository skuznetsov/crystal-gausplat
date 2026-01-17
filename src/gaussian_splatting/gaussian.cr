# Gaussian3D: 3D Gaussian primitive for splatting
# Each Gaussian has position, covariance (via scale+rotation), opacity, and spherical harmonics

require "../autograd/variable"
require "../core/tensor"

module GS
  module GaussianSplatting
    # Spherical harmonics degree (0-3)
    # Degree 0: 1 coefficient
    # Degree 1: 4 coefficients
    # Degree 2: 9 coefficients
    # Degree 3: 16 coefficients
    SH_DEGREE = 3
    SH_COEFFS = (SH_DEGREE + 1) ** 2  # 16

    # 3D Gaussian parameters
    # All stored as Variables for gradient computation
    class Gaussian3D
      # Position in world space [N, 3]
      property position : Autograd::Variable

      # Log-scale for covariance [N, 3]
      # Actual scale = exp(log_scale)
      property log_scale : Autograd::Variable

      # Rotation as quaternion [N, 4] (w, x, y, z)
      # Normalized to unit quaternion
      property rotation : Autograd::Variable

      # Opacity in logit space [N, 1]
      # Actual opacity = sigmoid(logit_opacity)
      property logit_opacity : Autograd::Variable

      # Spherical harmonics coefficients [N, SH_COEFFS, 3]
      # For view-dependent color
      property sh_coeffs : Autograd::Variable

      # Number of Gaussians
      getter count : Int32

      # Protected constructor for clone/concat operations
      protected def initialize(
        @count : Int32,
        @position : Autograd::Variable,
        @log_scale : Autograd::Variable,
        @rotation : Autograd::Variable,
        @logit_opacity : Autograd::Variable,
        @sh_coeffs : Autograd::Variable
      )
      end

      # Protected setter for count (used by concat!)
      protected def count=(value : Int32)
        @count = value
      end

      def initialize(@count : Int32, device : Tensor::Device = Tensor::Device::GPU)
        # Initialize with random positions in unit cube
        @position = Autograd::Variable.randn(@count, 3, requires_grad: true, device: device)

        # Small initial scale (log-space)
        scale_init = Tensor.new(@count, 3, device: device)
        scale_init.fill!(-3.0_f32)  # exp(-3) ≈ 0.05
        @log_scale = Autograd::Variable.new(scale_init, requires_grad: true)

        # Identity rotation (w=1, x=y=z=0)
        rot_data = Tensor.zeros(@count, 4, device: device)
        rot_data.to_cpu!
        @count.times { |i| rot_data.cpu_data.not_nil![i * 4] = 1.0_f32 }  # w = 1
        rot_data.to_gpu! if device.gpu?
        @rotation = Autograd::Variable.new(rot_data, requires_grad: true)

        # Initial opacity (logit space, sigmoid(0) = 0.5)
        opacity_init = Tensor.zeros(@count, 1, device: device)
        @logit_opacity = Autograd::Variable.new(opacity_init, requires_grad: true)

        # Initialize SH with small random values
        # DC component (index 0) is the base color
        @sh_coeffs = Autograd::Variable.randn(@count, SH_COEFFS, 3, requires_grad: true, device: device)
        # Scale down initial SH
        @sh_coeffs = @sh_coeffs * 0.1_f32
      end

      # Initialize from point cloud
      def self.from_points(
        points : Tensor,       # [N, 3] positions
        colors : Tensor? = nil, # [N, 3] RGB colors (optional)
        device : Tensor::Device = Tensor::Device::GPU
      ) : Gaussian3D
        n = points.shape[0]
        gs = new(n, device)

        # Set positions
        gs.position = Autograd::Variable.new(points.clone, requires_grad: true)

        # Set DC color from input colors
        if colors
          sh_data = Tensor.zeros(n, SH_COEFFS, 3, device: device)
          sh_data.to_cpu!
          colors_cpu = colors.on_cpu? ? colors : colors.to_cpu

          # DC coefficient (index 0) from RGB
          # SH DC = color * C0 where C0 = 0.28209479177387814
          c0 = 0.28209479177387814_f32
          n.times do |i|
            3.times do |c|
              sh_data.cpu_data.not_nil![i * SH_COEFFS * 3 + c] = colors_cpu.cpu_data.not_nil![i * 3 + c] / c0
            end
          end
          sh_data.to_gpu! if device.gpu?
          gs.sh_coeffs = Autograd::Variable.new(sh_data, requires_grad: true)
        end

        gs
      end

      # Get actual scale values (exp of log_scale)
      def scale : Tensor
        log_s = @log_scale.data
        log_s_cpu = log_s.on_cpu? ? log_s : log_s.to_cpu

        result = Tensor.new(log_s.shape, log_s.dtype, Tensor::Device::CPU)
        log_s_cpu.cpu_data.not_nil!.each_with_index do |v, i|
          result.cpu_data.not_nil![i] = Math.exp(v)
        end

        log_s.on_gpu? ? result.to_gpu : result
      end

      # Get actual opacity values (sigmoid of logit_opacity)
      def opacity : Tensor
        logit = @logit_opacity.data
        logit_cpu = logit.on_cpu? ? logit : logit.to_cpu

        result = Tensor.new(logit.shape, logit.dtype, Tensor::Device::CPU)
        logit_cpu.cpu_data.not_nil!.each_with_index do |v, i|
          result.cpu_data.not_nil![i] = 1.0_f32 / (1.0_f32 + Math.exp(-v))
        end

        logit.on_gpu? ? result.to_gpu : result
      end

      # Get all trainable parameters
      def parameters : Array(Autograd::Variable)
        [@position, @log_scale, @rotation, @logit_opacity, @sh_coeffs]
      end

      # Clone Gaussians (for densification)
      def clone_gaussians(indices : Array(Int32)) : Gaussian3D
        new_count = indices.size
        device = @position.data.device

        # Helper to extract rows
        extract = ->(src : Autograd::Variable, cols : Int32) {
          src_cpu = src.data.on_cpu? ? src.data : src.data.to_cpu
          src_data = src_cpu.cpu_data.not_nil!

          result = Tensor.new(new_count, cols, device: Tensor::Device::CPU)
          result_data = result.cpu_data.not_nil!

          indices.each_with_index do |src_idx, dst_idx|
            cols.times do |c|
              result_data[dst_idx * cols + c] = src_data[src_idx * cols + c]
            end
          end

          result = result.to_gpu if device.gpu?
          Autograd::Variable.new(result, requires_grad: true)
        }

        new_position = extract.call(@position, 3)
        new_log_scale = extract.call(@log_scale, 3)
        new_rotation = extract.call(@rotation, 4)
        new_logit_opacity = extract.call(@logit_opacity, 1)

        # SH coeffs are [N, SH_COEFFS, 3], flatten for extraction
        sh_flat_cols = SH_COEFFS * 3
        sh_cpu = @sh_coeffs.data.on_cpu? ? @sh_coeffs.data : @sh_coeffs.data.to_cpu
        sh_data = sh_cpu.cpu_data.not_nil!

        sh_result = Tensor.new(new_count, SH_COEFFS, 3, device: Tensor::Device::CPU)
        sh_result_data = sh_result.cpu_data.not_nil!

        indices.each_with_index do |src_idx, dst_idx|
          sh_flat_cols.times do |c|
            sh_result_data[dst_idx * sh_flat_cols + c] = sh_data[src_idx * sh_flat_cols + c]
          end
        end

        sh_result = sh_result.to_gpu if device.gpu?
        new_sh_coeffs = Autograd::Variable.new(sh_result, requires_grad: true)

        Gaussian3D.new(new_count, new_position, new_log_scale, new_rotation, new_logit_opacity, new_sh_coeffs)
      end

      # Concatenate another Gaussian set
      def concat!(other : Gaussian3D) : Nil
        new_count = @count + other.count

        concat_var = ->(a : Autograd::Variable, b : Autograd::Variable) {
          a_cpu = a.data.on_cpu? ? a.data : a.data.to_cpu
          b_cpu = b.data.on_cpu? ? b.data : b.data.to_cpu

          # Calculate dimensions
          a_rows = a.data.shape[0]
          b_rows = b.data.shape[0]
          cols = a.data.numel // a_rows

          new_shape = [a_rows + b_rows] + a.data.shape.to_a[1..]
          result = Tensor.new(Shape.new(new_shape), a.data.dtype, Tensor::Device::CPU)
          result_data = result.cpu_data.not_nil!

          # Copy a
          (a_rows * cols).times { |i| result_data[i] = a_cpu.cpu_data.not_nil![i] }
          # Copy b
          (b_rows * cols).times { |i| result_data[a_rows * cols + i] = b_cpu.cpu_data.not_nil![i] }

          result = result.to_gpu if a.data.on_gpu?
          Autograd::Variable.new(result, requires_grad: true)
        }

        @position = concat_var.call(@position, other.position)
        @log_scale = concat_var.call(@log_scale, other.log_scale)
        @rotation = concat_var.call(@rotation, other.rotation)
        @logit_opacity = concat_var.call(@logit_opacity, other.logit_opacity)
        @sh_coeffs = concat_var.call(@sh_coeffs, other.sh_coeffs)

        @count = new_count
      end

      # Remove Gaussians by indices (for pruning)
      def remove!(indices_to_remove : Set(Int32)) : Nil
        keep_indices = (0...@count).reject { |i| indices_to_remove.includes?(i) }.to_a
        return if keep_indices.empty?

        new_count = keep_indices.size
        device = @position.data.device

        extract = ->(src : Autograd::Variable) {
          src_cpu = src.data.on_cpu? ? src.data : src.data.to_cpu
          src_data = src_cpu.cpu_data.not_nil!

          src_rows = src.data.shape[0]
          cols = src.data.numel // src_rows

          new_shape = [new_count] + src.data.shape.to_a[1..]
          result = Tensor.new(Shape.new(new_shape), src.data.dtype, Tensor::Device::CPU)
          result_data = result.cpu_data.not_nil!

          keep_indices.each_with_index do |src_idx, dst_idx|
            cols.times { |c| result_data[dst_idx * cols + c] = src_data[src_idx * cols + c] }
          end

          result = result.to_gpu if device.gpu?
          Autograd::Variable.new(result, requires_grad: true)
        }

        @position = extract.call(@position)
        @log_scale = extract.call(@log_scale)
        @rotation = extract.call(@rotation)
        @logit_opacity = extract.call(@logit_opacity)
        @sh_coeffs = extract.call(@sh_coeffs)

        @count = new_count
      end

      # Reset opacity for all Gaussians (used periodically during training)
      def reset_opacity!(value : Float32 = -2.0_f32) : Nil
        @logit_opacity.data.fill!(value)
      end

      # Normalize quaternions to unit length
      def normalize_rotations! : Nil
        rot_cpu = @rotation.data.on_cpu? ? @rotation.data : @rotation.data.to_cpu
        rot_data = rot_cpu.cpu_data.not_nil!

        @count.times do |i|
          w = rot_data[i * 4]
          x = rot_data[i * 4 + 1]
          y = rot_data[i * 4 + 2]
          z = rot_data[i * 4 + 3]

          norm = Math.sqrt(w*w + x*x + y*y + z*z)
          if norm > 1e-8_f32
            rot_data[i * 4] = w / norm
            rot_data[i * 4 + 1] = x / norm
            rot_data[i * 4 + 2] = y / norm
            rot_data[i * 4 + 3] = z / norm
          end
        end

        if @rotation.data.on_gpu?
          @rotation.data.to_gpu!
        end
      end

      # Statistics for debugging
      def stats : {min_scale: Float32, max_scale: Float32, mean_opacity: Float32, count: Int32}
        s = scale
        s_cpu = s.on_cpu? ? s : s.to_cpu
        s_data = s_cpu.cpu_data.not_nil!

        o = opacity
        o_cpu = o.on_cpu? ? o : o.to_cpu
        o_data = o_cpu.cpu_data.not_nil!

        min_s = Float32::MAX
        max_s = Float32::MIN
        s_data.each { |v| min_s = Math.min(min_s, v); max_s = Math.max(max_s, v) }

        mean_o = o_data.sum / o_data.size

        {min_scale: min_s, max_scale: max_s, mean_opacity: mean_o, count: @count}
      end
    end
  end
end
