# Gaussian Splatting Trainer
# Training loop with densification and pruning

require "../core/tensor"
require "../optim/adam"
require "../ops/loss"
require "./gaussian"
require "./camera"
require "./rasterizer"
require "./rasterizer_context"

module GS
  module GaussianSplatting
    # Training configuration
    struct TrainerConfig
      property iterations : Int32
      property lr_position : Float32
      property lr_scale : Float32
      property lr_rotation : Float32
      property lr_opacity : Float32
      property lr_sh : Float32
      property lambda_ssim : Float32

      # Densification
      property densify_from : Int32
      property densify_until : Int32
      property densify_interval : Int32
      property densify_grad_threshold : Float32
      property densify_size_threshold : Float32

      # Pruning
      property prune_interval : Int32
      property opacity_threshold : Float32
      property max_screen_size : Float32

      # Opacity reset
      property opacity_reset_interval : Int32

      def initialize(
        @iterations : Int32 = 30000,
        @lr_position : Float32 = 0.00016_f32,
        @lr_scale : Float32 = 0.005_f32,
        @lr_rotation : Float32 = 0.001_f32,
        @lr_opacity : Float32 = 0.05_f32,
        @lr_sh : Float32 = 0.0025_f32,
        @lambda_ssim : Float32 = 0.2_f32,
        @densify_from : Int32 = 500,
        @densify_until : Int32 = 15000,
        @densify_interval : Int32 = 100,
        @densify_grad_threshold : Float32 = 0.0002_f32,
        @densify_size_threshold : Float32 = 0.01_f32,
        @prune_interval : Int32 = 100,
        @opacity_threshold : Float32 = 0.005_f32,
        @max_screen_size : Float32 = 20.0_f32,
        @opacity_reset_interval : Int32 = 3000
      )
      end
    end

    # Training statistics
    struct TrainStats
      property iteration : Int32
      property loss : Float32
      property psnr : Float32
      property num_gaussians : Int32
      property lr : Float32

      def initialize(@iteration = 0, @loss = 0.0_f32, @psnr = 0.0_f32, @num_gaussians = 0, @lr = 0.0_f32)
      end

      def to_s : String
        "Iter #{@iteration}: loss=#{sprintf("%.4f", @loss)}, PSNR=#{sprintf("%.2f", @psnr)}dB, gaussians=#{@num_gaussians}"
      end
    end

    # Gradient accumulator for densification decisions
    class GradientAccumulator
      property position_grad_accum : Array(Float32)
      property position_grad_count : Array(Int32)

      def initialize(count : Int32)
        @position_grad_accum = Array(Float32).new(count, 0.0_f32)
        @position_grad_count = Array(Int32).new(count, 0)
      end

      def accumulate(grads : RasterizerGradients) : Nil
        grad_cpu = grads.dL_dmean2d.to_cpu
        grad_data = grad_cpu.cpu_data.not_nil!

        n = @position_grad_accum.size
        n.times do |i|
          # Accumulate gradient magnitude
          gx = grad_data[i * 2]
          gy = grad_data[i * 2 + 1]
          mag = Math.sqrt(gx * gx + gy * gy)
          @position_grad_accum[i] += mag
          @position_grad_count[i] += 1
        end
      end

      def get_average(index : Int32) : Float32
        count = @position_grad_count[index]
        return 0.0_f32 if count == 0
        @position_grad_accum[index] / count
      end

      def reset! : Nil
        @position_grad_accum.fill(0.0_f32)
        @position_grad_count.fill(0)
      end

      def resize(new_count : Int32) : Nil
        if new_count > @position_grad_accum.size
          (@position_grad_accum.size...new_count).each do
            @position_grad_accum << 0.0_f32
            @position_grad_count << 0
          end
        elsif new_count < @position_grad_accum.size
          @position_grad_accum = @position_grad_accum[0...new_count]
          @position_grad_count = @position_grad_count[0...new_count]
        end
      end
    end

    # Main trainer class
    class Trainer
      property gaussians : Gaussian3D
      property cameras : CameraSet
      property config : TrainerConfig

      @optimizer_position : Optim::Adam
      @optimizer_scale : Optim::Adam
      @optimizer_rotation : Optim::Adam
      @optimizer_opacity : Optim::Adam
      @optimizer_sh : Optim::Adam

      @context : RasterizerContext?
      @grad_accum : GradientAccumulator
      @iteration : Int32

      def initialize(@gaussians : Gaussian3D, @cameras : CameraSet, @config : TrainerConfig = TrainerConfig.new)
        # Create separate optimizers with different learning rates
        @optimizer_position = Optim::Adam.new([@gaussians.position], lr: @config.lr_position)
        @optimizer_scale = Optim::Adam.new([@gaussians.log_scale], lr: @config.lr_scale)
        @optimizer_rotation = Optim::Adam.new([@gaussians.rotation], lr: @config.lr_rotation)
        @optimizer_opacity = Optim::Adam.new([@gaussians.logit_opacity], lr: @config.lr_opacity)
        @optimizer_sh = Optim::Adam.new([@gaussians.sh_coeffs], lr: @config.lr_sh)

        @context = nil
        @grad_accum = GradientAccumulator.new(@gaussians.count)
        @iteration = 0
      end

      # Run full training
      def train(&block : TrainStats -> Nil) : Nil
        @config.iterations.times do |i|
          @iteration = i
          stats = train_step

          # Call callback for logging
          yield stats

          # Densification and pruning
          if i >= @config.densify_from && i < @config.densify_until
            if i % @config.densify_interval == 0
              densify_and_prune!
            end
          end

          # Opacity reset
          if i > 0 && i % @config.opacity_reset_interval == 0
            reset_opacity!
          end
        end
      end

      # Single training step
      def train_step : TrainStats
        # Sample random camera
        camera = @cameras.sample
        target_image = camera.image

        raise "Camera has no target image" unless target_image

        # Forward pass
        rendered, ctx = Rasterizer.forward(@gaussians, camera, @context, {0.0_f32, 0.0_f32, 0.0_f32})
        @context = ctx

        # Compute loss
        loss, grad_image = Loss.combined_with_grad(rendered, target_image, @config.lambda_ssim)

        # Backward pass
        grads = Rasterizer.backward(grad_image, @gaussians, camera, ctx)

        # Accumulate gradients for densification
        @grad_accum.accumulate(grads)

        # Apply gradients to Gaussian parameters
        apply_gradients!(grads)

        # Optimizer step
        zero_grads!
        optimizer_step!

        # Compute PSNR for logging
        psnr = Loss.psnr(rendered, target_image)

        TrainStats.new(
          iteration: @iteration,
          loss: loss,
          psnr: psnr,
          num_gaussians: @gaussians.count,
          lr: @config.lr_position
        )
      end

      # Apply gradients to Gaussian Variable parameters
      private def apply_gradients!(grads : RasterizerGradients) : Nil
        # Position gradients
        copy_grad!(@gaussians.position, grads.dL_dposition)

        # Scale gradients (need to chain through exp)
        # d(loss)/d(log_scale) = d(loss)/d(scale) * d(scale)/d(log_scale)
        #                      = d(loss)/d(scale) * scale
        scale = @gaussians.scale
        scale_cpu = scale.on_cpu? ? scale : scale.to_cpu
        dL_dscale_cpu = grads.dL_dcov3d.to_cpu  # Approximation - should be from cov3d backward
        # TODO: Proper scale gradient through covariance

        # Rotation gradients
        # TODO: Quaternion gradient from covariance backward

        # Opacity gradients (chain through sigmoid)
        # d(loss)/d(logit_opacity) = d(loss)/d(opacity) * d(sigmoid)/d(x)
        #                          = d(loss)/d(opacity) * opacity * (1 - opacity)
        opacity = @gaussians.opacity
        opacity_cpu = opacity.on_cpu? ? opacity : opacity.to_cpu
        dL_dop_cpu = grads.dL_dopacity.to_cpu

        dL_dlogit = Tensor.new(grads.dL_dopacity.shape, DType::F32, Tensor::Device::CPU)
        op_data = opacity_cpu.cpu_data.not_nil!
        dop_data = dL_dop_cpu.cpu_data.not_nil!
        dlogit_data = dL_dlogit.cpu_data.not_nil!

        @gaussians.count.times do |i|
          o = op_data[i]
          dlogit_data[i] = dop_data[i] * o * (1.0_f32 - o)
        end

        dL_dlogit = dL_dlogit.to_gpu if @gaussians.logit_opacity.data.on_gpu?
        copy_grad!(@gaussians.logit_opacity, dL_dlogit)

        # SH gradients
        copy_grad!(@gaussians.sh_coeffs, grads.dL_dsh)
      end

      private def copy_grad!(variable : Autograd::Variable, grad : Tensor) : Nil
        if variable.grad
          # Accumulate
          existing = variable.grad.not_nil!
          existing_cpu = existing.on_cpu? ? existing : existing.to_cpu
          grad_cpu = grad.on_cpu? ? grad : grad.to_cpu

          e_data = existing_cpu.cpu_data.not_nil!
          g_data = grad_cpu.cpu_data.not_nil!

          e_data.size.times { |i| e_data[i] += g_data[i] }

          if existing.on_gpu?
            existing.to_gpu!
          end
        else
          variable.grad = grad.clone
        end
      end

      private def zero_grads! : Nil
        @gaussians.parameters.each(&.zero_grad!)
      end

      private def optimizer_step! : Nil
        @optimizer_position.step
        @optimizer_scale.step
        @optimizer_rotation.step
        @optimizer_opacity.step
        @optimizer_sh.step
      end

      # ========================================================================
      # Densification and Pruning
      # ========================================================================

      private def densify_and_prune! : Nil
        # Find Gaussians with high gradient that need splitting/cloning
        to_clone = Array(Int32).new
        to_split = Array(Int32).new
        to_prune = Set(Int32).new

        scale = @gaussians.scale
        scale_cpu = scale.on_cpu? ? scale : scale.to_cpu
        scale_data = scale_cpu.cpu_data.not_nil!

        opacity = @gaussians.opacity
        opacity_cpu = opacity.on_cpu? ? opacity : opacity.to_cpu
        opacity_data = opacity_cpu.cpu_data.not_nil!

        @gaussians.count.times do |i|
          avg_grad = @grad_accum.get_average(i)
          max_scale = Math.max(scale_data[i * 3], Math.max(scale_data[i * 3 + 1], scale_data[i * 3 + 2]))

          if avg_grad > @config.densify_grad_threshold
            if max_scale > @config.densify_size_threshold
              # Large Gaussian with high gradient -> split
              to_split << i
            else
              # Small Gaussian with high gradient -> clone
              to_clone << i
            end
          end

          # Prune low-opacity Gaussians
          if opacity_data[i] < @config.opacity_threshold
            to_prune << i
          end
        end

        # Apply densification
        if !to_clone.empty?
          cloned = @gaussians.clone_gaussians(to_clone)
          # Add small noise to cloned positions
          add_position_noise!(cloned, 0.001_f32)
          @gaussians.concat!(cloned)
        end

        if !to_split.empty?
          split_gaussians!(to_split)
        end

        # Apply pruning
        if !to_prune.empty?
          @gaussians.remove!(to_prune)
        end

        # Update gradient accumulator size
        @grad_accum.resize(@gaussians.count)
        @grad_accum.reset!

        # Recreate optimizers with new parameter count
        recreate_optimizers!
      end

      private def split_gaussians!(indices : Array(Int32)) : Nil
        # Split each Gaussian into two smaller ones
        return if indices.empty?

        # Clone first
        split1 = @gaussians.clone_gaussians(indices)
        split2 = @gaussians.clone_gaussians(indices)

        # Reduce scale and offset positions
        scale_factor = 1.6_f32  # From original paper

        [split1, split2].each_with_index do |split, idx|
          split.log_scale.data.to_cpu!
          scale_data = split.log_scale.data.cpu_data.not_nil!

          split.position.data.to_cpu!
          pos_data = split.position.data.cpu_data.not_nil!

          split.count.times do |i|
            # Reduce scale
            3.times do |j|
              scale_data[i * 3 + j] -= Math.log(scale_factor)
            end

            # Offset position (opposite directions for the two splits)
            dir = idx == 0 ? 1.0_f32 : -1.0_f32
            offset = Math.exp(scale_data[i * 3]) * 0.5_f32 * dir
            pos_data[i * 3] += offset * (Random.rand - 0.5_f32)
            pos_data[i * 3 + 1] += offset * (Random.rand - 0.5_f32)
            pos_data[i * 3 + 2] += offset * (Random.rand - 0.5_f32)
          end

          split.log_scale.data.to_gpu! if @gaussians.log_scale.data.on_gpu?
          split.position.data.to_gpu! if @gaussians.position.data.on_gpu?
        end

        # Remove original large Gaussians and add splits
        @gaussians.remove!(indices.to_set)
        @gaussians.concat!(split1)
        @gaussians.concat!(split2)
      end

      private def add_position_noise!(gaussians : Gaussian3D, scale : Float32) : Nil
        gaussians.position.data.to_cpu!
        pos_data = gaussians.position.data.cpu_data.not_nil!

        gaussians.count.times do |i|
          3.times do |j|
            pos_data[i * 3 + j] += (Random.rand - 0.5_f32) * 2.0_f32 * scale
          end
        end

        gaussians.position.data.to_gpu! if @gaussians.position.data.on_gpu?
      end

      private def reset_opacity! : Nil
        # Reset opacity to encourage pruning
        @gaussians.reset_opacity!(-1.0_f32)  # sigmoid(-1) ≈ 0.27
      end

      private def recreate_optimizers! : Nil
        # Recreate optimizers when Gaussian count changes
        @optimizer_position = Optim::Adam.new([@gaussians.position], lr: @config.lr_position)
        @optimizer_scale = Optim::Adam.new([@gaussians.log_scale], lr: @config.lr_scale)
        @optimizer_rotation = Optim::Adam.new([@gaussians.rotation], lr: @config.lr_rotation)
        @optimizer_opacity = Optim::Adam.new([@gaussians.logit_opacity], lr: @config.lr_opacity)
        @optimizer_sh = Optim::Adam.new([@gaussians.sh_coeffs], lr: @config.lr_sh)
      end

      # ========================================================================
      # Utility
      # ========================================================================

      # Save checkpoint
      def save_checkpoint(path : String) : Nil
        # TODO: Implement serialization
      end

      # Load checkpoint
      def self.load_checkpoint(path : String) : Trainer
        # TODO: Implement deserialization
        raise "Not implemented"
      end
    end
  end
end
