# Loss functions for Gaussian Splatting training
# L1, MSE, SSIM, and combined losses

require "../core/tensor"
require "../autograd/variable"

module GS
  module Loss
    extend self

    # ========================================================================
    # L1 Loss (Mean Absolute Error)
    # ========================================================================

    def l1(pred : Tensor, target : Tensor) : Float32
      raise "Shape mismatch" unless pred.shape == target.shape

      pred_cpu = pred.on_cpu? ? pred : pred.to_cpu
      target_cpu = target.on_cpu? ? target : target.to_cpu

      pred_data = pred_cpu.cpu_data.not_nil!
      target_data = target_cpu.cpu_data.not_nil!

      sum = 0.0_f32
      pred.numel.times do |i|
        sum += (pred_data[i] - target_data[i]).abs
      end

      sum / pred.numel
    end

    # L1 loss with gradient
    def l1_with_grad(pred : Tensor, target : Tensor) : {Float32, Tensor}
      raise "Shape mismatch" unless pred.shape == target.shape

      pred_cpu = pred.on_cpu? ? pred : pred.to_cpu
      target_cpu = target.on_cpu? ? target : target.to_cpu

      pred_data = pred_cpu.cpu_data.not_nil!
      target_data = target_cpu.cpu_data.not_nil!

      grad = Tensor.new(pred.shape, pred.dtype, Tensor::Device::CPU)
      grad_data = grad.cpu_data.not_nil!

      sum = 0.0_f32
      n = pred.numel.to_f32

      pred.numel.times do |i|
        diff = pred_data[i] - target_data[i]
        sum += diff.abs
        # Gradient of |x| is sign(x)
        grad_data[i] = (diff > 0 ? 1.0_f32 : (diff < 0 ? -1.0_f32 : 0.0_f32)) / n
      end

      grad = grad.to_gpu if pred.on_gpu?
      {sum / n, grad}
    end

    # ========================================================================
    # MSE Loss (Mean Squared Error)
    # ========================================================================

    def mse(pred : Tensor, target : Tensor) : Float32
      raise "Shape mismatch" unless pred.shape == target.shape

      pred_cpu = pred.on_cpu? ? pred : pred.to_cpu
      target_cpu = target.on_cpu? ? target : target.to_cpu

      pred_data = pred_cpu.cpu_data.not_nil!
      target_data = target_cpu.cpu_data.not_nil!

      sum = 0.0_f32
      pred.numel.times do |i|
        diff = pred_data[i] - target_data[i]
        sum += diff * diff
      end

      sum / pred.numel
    end

    def mse_with_grad(pred : Tensor, target : Tensor) : {Float32, Tensor}
      raise "Shape mismatch" unless pred.shape == target.shape

      pred_cpu = pred.on_cpu? ? pred : pred.to_cpu
      target_cpu = target.on_cpu? ? target : target.to_cpu

      pred_data = pred_cpu.cpu_data.not_nil!
      target_data = target_cpu.cpu_data.not_nil!

      grad = Tensor.new(pred.shape, pred.dtype, Tensor::Device::CPU)
      grad_data = grad.cpu_data.not_nil!

      sum = 0.0_f32
      n = pred.numel.to_f32

      pred.numel.times do |i|
        diff = pred_data[i] - target_data[i]
        sum += diff * diff
        # Gradient of (x-t)² is 2(x-t)
        grad_data[i] = 2.0_f32 * diff / n
      end

      grad = grad.to_gpu if pred.on_gpu?
      {sum / n, grad}
    end

    # ========================================================================
    # SSIM Loss (Structural Similarity Index)
    # ========================================================================

    # Constants for SSIM
    SSIM_C1 = 0.01_f32 ** 2  # (k1 * L)² where k1=0.01, L=1 for normalized images
    SSIM_C2 = 0.03_f32 ** 2  # (k2 * L)² where k2=0.03

    # Compute SSIM between two images
    # Input: [H, W, C] tensors
    def ssim(pred : Tensor, target : Tensor, window_size : Int32 = 11) : Float32
      raise "Shape mismatch" unless pred.shape == target.shape
      raise "Expected 3D tensor [H, W, C]" unless pred.ndim == 3

      height = pred.shape[0]
      width = pred.shape[1]
      channels = pred.shape[2]

      pred_cpu = pred.on_cpu? ? pred : pred.to_cpu
      target_cpu = target.on_cpu? ? target : target.to_cpu

      # Create Gaussian window
      window = gaussian_window(window_size)
      pad = window_size // 2

      total_ssim = 0.0_f32
      count = 0

      # Compute SSIM per channel, then average
      channels.times do |c|
        # Extract channel
        pred_c = extract_channel(pred_cpu, c)
        target_c = extract_channel(target_cpu, c)

        # Compute local statistics using Gaussian window
        (pad...height - pad).each do |y|
          (pad...width - pad).each do |x|
            mu_x = 0.0_f32
            mu_y = 0.0_f32
            sigma_x = 0.0_f32
            sigma_y = 0.0_f32
            sigma_xy = 0.0_f32

            # Apply window
            window_size.times do |wy|
              window_size.times do |wx|
                w = window[wy * window_size + wx]
                px = pred_c[(y - pad + wy) * width + (x - pad + wx)]
                py = target_c[(y - pad + wy) * width + (x - pad + wx)]

                mu_x += w * px
                mu_y += w * py
              end
            end

            # Compute variances
            window_size.times do |wy|
              window_size.times do |wx|
                w = window[wy * window_size + wx]
                px = pred_c[(y - pad + wy) * width + (x - pad + wx)]
                py = target_c[(y - pad + wy) * width + (x - pad + wx)]

                sigma_x += w * (px - mu_x) * (px - mu_x)
                sigma_y += w * (py - mu_y) * (py - mu_y)
                sigma_xy += w * (px - mu_x) * (py - mu_y)
              end
            end

            # SSIM formula
            numerator = (2.0_f32 * mu_x * mu_y + SSIM_C1) * (2.0_f32 * sigma_xy + SSIM_C2)
            denominator = (mu_x * mu_x + mu_y * mu_y + SSIM_C1) * (sigma_x + sigma_y + SSIM_C2)

            total_ssim += numerator / denominator
            count += 1
          end
        end
      end

      count > 0 ? total_ssim / count : 0.0_f32
    end

    # SSIM loss (1 - SSIM, so lower is better)
    def ssim_loss(pred : Tensor, target : Tensor, window_size : Int32 = 11) : Float32
      1.0_f32 - ssim(pred, target, window_size)
    end

    # SSIM with gradient (approximate - using finite differences for now)
    def ssim_loss_with_grad(pred : Tensor, target : Tensor, window_size : Int32 = 11) : {Float32, Tensor}
      loss = ssim_loss(pred, target, window_size)

      # Approximate gradient using difference from target
      # This is a simplification - true SSIM gradient is more complex
      pred_cpu = pred.on_cpu? ? pred : pred.to_cpu
      target_cpu = target.on_cpu? ? target : target.to_cpu

      grad = Tensor.new(pred.shape, pred.dtype, Tensor::Device::CPU)
      pred_data = pred_cpu.cpu_data.not_nil!
      target_data = target_cpu.cpu_data.not_nil!
      grad_data = grad.cpu_data.not_nil!

      # Simplified gradient approximation
      n = pred.numel.to_f32
      pred.numel.times do |i|
        grad_data[i] = 2.0_f32 * (pred_data[i] - target_data[i]) / n
      end

      grad = grad.to_gpu if pred.on_gpu?
      {loss, grad}
    end

    # ========================================================================
    # Combined Loss (L1 + lambda * SSIM)
    # ========================================================================

    def combined(pred : Tensor, target : Tensor, lambda_ssim : Float32 = 0.2_f32) : Float32
      l1_val = l1(pred, target)
      ssim_val = ssim_loss(pred, target)
      (1.0_f32 - lambda_ssim) * l1_val + lambda_ssim * ssim_val
    end

    def combined_with_grad(pred : Tensor, target : Tensor, lambda_ssim : Float32 = 0.2_f32) : {Float32, Tensor}
      l1_val, l1_grad = l1_with_grad(pred, target)
      ssim_val, ssim_grad = ssim_loss_with_grad(pred, target)

      loss = (1.0_f32 - lambda_ssim) * l1_val + lambda_ssim * ssim_val

      # Combine gradients
      grad = Tensor.new(pred.shape, pred.dtype, l1_grad.device)
      l1_grad_cpu = l1_grad.on_cpu? ? l1_grad : l1_grad.to_cpu
      ssim_grad_cpu = ssim_grad.on_cpu? ? ssim_grad : ssim_grad.to_cpu
      grad.to_cpu! if grad.on_gpu?

      l1_d = l1_grad_cpu.cpu_data.not_nil!
      ssim_d = ssim_grad_cpu.cpu_data.not_nil!
      grad_d = grad.cpu_data.not_nil!

      pred.numel.times do |i|
        grad_d[i] = (1.0_f32 - lambda_ssim) * l1_d[i] + lambda_ssim * ssim_d[i]
      end

      grad = grad.to_gpu if pred.on_gpu?
      {loss, grad}
    end

    # ========================================================================
    # Helper Functions
    # ========================================================================

    # Generate 2D Gaussian window for SSIM
    private def gaussian_window(size : Int32, sigma : Float32 = 1.5_f32) : Array(Float32)
      window = Array(Float32).new(size * size, 0.0_f32)
      center = size // 2
      sum = 0.0_f32

      size.times do |y|
        size.times do |x|
          dx = x - center
          dy = y - center
          val = Math.exp(-(dx * dx + dy * dy) / (2.0_f32 * sigma * sigma))
          window[y * size + x] = val
          sum += val
        end
      end

      # Normalize
      window.map! { |v| v / sum }
      window
    end

    # Extract single channel from [H, W, C] tensor
    private def extract_channel(tensor : Tensor, channel : Int32) : Array(Float32)
      data = tensor.cpu_data.not_nil!
      height = tensor.shape[0]
      width = tensor.shape[1]
      channels = tensor.shape[2]

      result = Array(Float32).new(height * width, 0.0_f32)
      height.times do |y|
        width.times do |x|
          result[y * width + x] = data[(y * width + x) * channels + channel]
        end
      end
      result
    end

    # ========================================================================
    # PSNR (Peak Signal-to-Noise Ratio) for evaluation
    # ========================================================================

    def psnr(pred : Tensor, target : Tensor, max_val : Float32 = 1.0_f32) : Float32
      mse_val = mse(pred, target)
      return Float32::INFINITY if mse_val < 1e-10_f32
      10.0_f32 * Math.log10(max_val * max_val / mse_val)
    end
  end
end
