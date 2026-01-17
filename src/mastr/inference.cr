# MASt3R Inference Pipeline
# Loads pretrained weights and runs dense stereo inference

require "./encoder_v2"
require "./decoder_v2"
require "./weight_loader"
require "./weights"
require "../autograd/variable"
require "../core/tensor"
require "../nn/vit"

module GS
  module MASt3R
    # Full MASt3R model for inference
    class MASt3RInference
      getter encoder : MASt3REncoderV2
      getter decoder : MASt3RDecoder
      getter dpt_head1 : DPTHead  # For first image
      getter dpt_head2 : DPTHead  # For second image
      getter device : Tensor::Device
      @weights_loaded : Bool = false
      @patch_h : Int32
      @patch_w : Int32

      def initialize(@device : Tensor::Device = Tensor::Device::CPU)
        # Encoder: ViT-Large (1024 dim, 24 layers, 16 heads)
        encoder_config = MASt3REncoderConfigV2.new(
          img_size: 512,
          patch_size: 16,
          in_channels: 3,
          embed_dim: 1024,
          depth: 24,
          num_heads: 16,
          mlp_ratio: 4.0_f32
        )
        @encoder = MASt3REncoderV2.new(encoder_config, @device)

        # Decoder: 768 dim, 12 layers
        decoder_config = MASt3RDecoderConfig.new(
          encoder_dim: 1024,
          embed_dim: 768,
          depth: 12,
          num_heads: 12,
          mlp_ratio: 4.0_f32,
          img_size: 512,
          patch_size: 16
        )
        @decoder = MASt3RDecoder.new(decoder_config, @device)

        # DPT heads for point map prediction
        @dpt_head1 = DPTHead.new(encoder_dim: 1024, decoder_dim: 768, device: @device)
        @dpt_head2 = DPTHead.new(encoder_dim: 1024, decoder_dim: 768, device: @device)

        # Patch dimensions for 512x512 input
        @patch_h = 512 // 16  # 32
        @patch_w = 512 // 16  # 32
      end

      # Load pretrained weights
      def load_weights!(path : String)
        puts "Loading MASt3R weights from #{path}..."

        loader = WeightLoader.new(path, @device)

        # Load encoder weights
        loader.load_encoder!(@encoder)

        # Load decoder weights
        loader.load_decoder!(@decoder)

        # Load DPT head weights
        puts "Loading DPT heads..."
        loader.load_dpt_head!(@dpt_head1, "downstream_head1")
        loader.load_dpt_head!(@dpt_head2, "downstream_head2")

        # Summary
        loader.summary

        @weights_loaded = true
        puts "Weights loaded successfully!"
      end

      def weights_loaded?
        @weights_loaded
      end

      # Inference on image pair
      # img1, img2: Tensor [batch, height, width, 3] in range [0, 1]
      # Returns: {points1, conf1, points2, conf2}
      #   points: [batch, H*4, W*4, 3] - 3D coordinates (upsampled)
      #   conf: [batch, H*4, W*4, 1] - confidence
      def forward_pair(img1 : Tensor, img2 : Tensor) : {Tensor, Tensor, Tensor, Tensor}
        # Normalize images to [-1, 1] and convert to [batch, channels, height, width]
        img1_norm = preprocess_image(img1)
        img2_norm = preprocess_image(img2)

        var1 = Autograd::Variable.new(img1_norm, requires_grad: false)
        var2 = Autograd::Variable.new(img2_norm, requires_grad: false)

        puts "    Encoding images..."
        # Encode both images
        # Result: [batch, seq_len, embed_dim] where seq_len = (H/16) * (W/16)
        enc1, enc2 = @encoder.forward_pair(var1, var2)

        puts "    Decoding with cross-attention..."
        # Decode with cross-attention
        dec1, dec2 = @decoder.forward_pair(enc1, enc2)

        puts "    Running DPT heads for point maps..."
        # Use DPT heads to generate point maps
        points1_var, conf1_var = @dpt_head1.forward(dec1, @patch_h, @patch_w)
        points2_var, conf2_var = @dpt_head2.forward(dec2, @patch_h, @patch_w)

        {points1_var.data, conf1_var.data, points2_var.data, conf2_var.data}
      end

      # Inference on single image (no cross-attention)
      def forward_single(img : Tensor) : {Tensor, Tensor}
        img_norm = preprocess_image(img)
        var = Autograd::Variable.new(img_norm, requires_grad: false)

        # Encode
        enc = @encoder.forward(var)

        # Decode
        dec = @decoder.forward_single(enc)

        # Extract point map
        extract_pointmap(dec, img.shape[1], img.shape[2])
      end

      # Convert frames to point cloud
      # frames: Array of Tensor [height, width, 3]
      # Returns: points [N, 3], confidences [N]
      def frames_to_pointcloud(
        frames : Array(Video::Frame),
        conf_threshold : Float32 = 0.3_f32
      ) : {Tensor, Tensor}
        return {Tensor.new(0, 3, device: Tensor::Device::CPU), Tensor.new(0, device: Tensor::Device::CPU)} if frames.size < 2

        all_points = [] of {Float32, Float32, Float32}
        all_confs = [] of Float32

        # Process consecutive frame pairs
        (0...frames.size - 1).each do |i|
          puts "  Processing frame pair #{i + 1}/#{frames.size - 1}..."

          frame1 = frames[i].data
          frame2 = frames[i + 1].data

          # Resize to 512x512 for MASt3R
          img1 = resize_image(frame1, 512, 512)
          img2 = resize_image(frame2, 512, 512)

          # Add batch dimension: [1, H, W, 3]
          img1_batch = add_batch_dim(img1)
          img2_batch = add_batch_dim(img2)

          # Run inference
          points1, conf1, points2, conf2 = forward_pair(img1_batch, img2_batch)

          # Extract valid points from first image of pair
          collect_points(points1, conf1, conf_threshold, all_points, all_confs)

          # For last pair, also collect second image points
          if i == frames.size - 2
            collect_points(points2, conf2, conf_threshold, all_points, all_confs)
          end
        end

        # Convert to tensors
        n_points = all_points.size
        puts "  Collected #{n_points} points with conf > #{conf_threshold}"

        return {Tensor.new(0, 3, device: Tensor::Device::CPU), Tensor.new(0, device: Tensor::Device::CPU)} if n_points == 0

        points = Tensor.new(n_points, 3, device: Tensor::Device::CPU)
        confs = Tensor.new(n_points, device: Tensor::Device::CPU)

        points_d = points.cpu_data.not_nil!
        confs_d = confs.cpu_data.not_nil!

        n_points.times do |i|
          points_d[i * 3] = all_points[i][0]
          points_d[i * 3 + 1] = all_points[i][1]
          points_d[i * 3 + 2] = all_points[i][2]
          confs_d[i] = all_confs[i]
        end

        {points, confs}
      end

      private def preprocess_image(img : Tensor) : Tensor
        # img: [batch, H, W, 3] in [0, 1]
        # Output: [batch, 3, H, W] normalized to [-1, 1]

        img_cpu = img.on_cpu? ? img : img.to_cpu
        data = img_cpu.cpu_data.not_nil!

        batch = img.shape[0]
        h = img.shape[1]
        w = img.shape[2]
        c = img.shape[3]

        # ImageNet normalization
        mean = [0.485_f32, 0.456_f32, 0.406_f32]
        std = [0.229_f32, 0.224_f32, 0.225_f32]

        result = Tensor.new(batch, c, h, w, device: Tensor::Device::CPU)
        r_d = result.cpu_data.not_nil!

        batch.times do |b|
          h.times do |y|
            w.times do |x|
              c.times do |ch|
                # NHWC -> NCHW
                src_idx = b * h * w * c + y * w * c + x * c + ch
                dst_idx = b * c * h * w + ch * h * w + y * w + x

                val = data[src_idx]
                r_d[dst_idx] = (val - mean[ch]) / std[ch]
              end
            end
          end
        end

        result = result.to_gpu if @device.gpu?
        result
      end

      private def extract_pointmap(dec : Autograd::Variable, orig_h : Int32, orig_w : Int32) : {Tensor, Tensor}
        # dec: [batch, seq, embed_dim]
        # We need to project to 3D points

        dec_data = dec.data.on_cpu? ? dec.data : dec.data.to_cpu
        data = dec_data.cpu_data.not_nil!

        batch = dec_data.shape[0]
        seq_len = dec_data.shape[1]
        embed_dim = dec_data.shape[2]

        # Compute spatial dimensions
        # For 512x512 input with 16x16 patches: 32x32 = 1024 tokens
        patch_h = orig_h // 16
        patch_w = orig_w // 16

        # Output: [batch, patch_h, patch_w, 3] for points
        #         [batch, patch_h, patch_w, 1] for confidence
        points = Tensor.new(batch, patch_h, patch_w, 3, device: Tensor::Device::CPU)
        conf = Tensor.new(batch, patch_h, patch_w, 1, device: Tensor::Device::CPU)

        points_d = points.cpu_data.not_nil!
        conf_d = conf.cpu_data.not_nil!

        # Simple linear projection from decoder features to 3D + conf
        # In real MASt3R, this goes through DPT head
        # Here we use first 3 dims for XYZ and next 1 for confidence
        batch.times do |b|
          patch_h.times do |ph|
            patch_w.times do |pw|
              seq_idx = ph * patch_w + pw
              next if seq_idx >= seq_len

              base = b * seq_len * embed_dim + seq_idx * embed_dim

              # Use decoder features to generate 3D coordinates
              # Scale appropriately for world coordinates
              # XY from patch position, Z from feature magnitude
              norm_x = (pw.to_f32 / patch_w - 0.5_f32) * 2.0_f32
              norm_y = (ph.to_f32 / patch_h - 0.5_f32) * 2.0_f32

              # Depth from feature - use mean of first few features
              depth_feat = 0.0_f32
              8.times { |i| depth_feat += data[base + i].abs }
              depth = 1.0_f32 + depth_feat * 0.1_f32

              # Confidence from feature magnitude
              conf_feat = 0.0_f32
              8.times { |i| conf_feat += data[base + i] ** 2 }
              confidence = Math.min(1.0_f32, Math.sqrt(conf_feat / 8.0_f32))

              out_idx = b * patch_h * patch_w * 3 + ph * patch_w * 3 + pw * 3
              conf_idx = b * patch_h * patch_w + ph * patch_w + pw

              points_d[out_idx] = norm_x * depth
              points_d[out_idx + 1] = norm_y * depth
              points_d[out_idx + 2] = depth
              conf_d[conf_idx] = confidence
            end
          end
        end

        {points, conf}
      end

      private def collect_points(
        points : Tensor,
        conf : Tensor,
        threshold : Float32,
        all_points : Array({Float32, Float32, Float32}),
        all_confs : Array(Float32)
      )
        points_d = points.cpu_data.not_nil!
        conf_d = conf.cpu_data.not_nil!

        batch = points.shape[0]
        h = points.shape[1]
        w = points.shape[2]

        batch.times do |b|
          h.times do |y|
            w.times do |x|
              c_idx = b * h * w + y * w + x
              c = conf_d[c_idx]

              if c > threshold
                p_idx = b * h * w * 3 + y * w * 3 + x * 3
                px = points_d[p_idx]
                py = points_d[p_idx + 1]
                pz = points_d[p_idx + 2]

                all_points << {px, py, pz}
                all_confs << c
              end
            end
          end
        end
      end

      private def resize_image(img : Tensor, target_h : Int32, target_w : Int32) : Tensor
        # Simple bilinear resize
        src_h = img.shape[0]
        src_w = img.shape[1]
        c = img.shape[2]

        img_cpu = img.on_cpu? ? img : img.to_cpu
        data = img_cpu.cpu_data.not_nil!

        result = Tensor.new(target_h, target_w, c, device: Tensor::Device::CPU)
        r_d = result.cpu_data.not_nil!

        target_h.times do |ty|
          target_w.times do |tx|
            # Map to source coordinates
            sy = ty.to_f32 * (src_h - 1) / (target_h - 1)
            sx = tx.to_f32 * (src_w - 1) / (target_w - 1)

            # Bilinear interpolation
            y0 = sy.to_i32.clamp(0, src_h - 2)
            y1 = y0 + 1
            x0 = sx.to_i32.clamp(0, src_w - 2)
            x1 = x0 + 1

            fy = sy - y0
            fx = sx - x0

            c.times do |ch|
              v00 = data[y0 * src_w * c + x0 * c + ch]
              v01 = data[y0 * src_w * c + x1 * c + ch]
              v10 = data[y1 * src_w * c + x0 * c + ch]
              v11 = data[y1 * src_w * c + x1 * c + ch]

              v = v00 * (1 - fx) * (1 - fy) +
                  v01 * fx * (1 - fy) +
                  v10 * (1 - fx) * fy +
                  v11 * fx * fy

              r_d[ty * target_w * c + tx * c + ch] = v
            end
          end
        end

        result
      end

      private def add_batch_dim(img : Tensor) : Tensor
        # [H, W, C] -> [1, H, W, C]
        h = img.shape[0]
        w = img.shape[1]
        c = img.shape[2]

        img_cpu = img.on_cpu? ? img : img.to_cpu
        result = Tensor.new(1, h, w, c, device: Tensor::Device::CPU)

        src = img_cpu.cpu_data.not_nil!
        dst = result.cpu_data.not_nil!

        img.numel.times { |i| dst[i] = src[i] }

        result
      end
    end

    # Convenience function to run inference
    def self.infer_pointcloud(
      frames : Array(Video::Frame),
      weights_path : String = "models/mastr/model.safetensors",
      device : Tensor::Device = Tensor::Device::CPU,
      conf_threshold : Float32 = 0.3_f32
    ) : {Tensor, Tensor}
      model = MASt3RInference.new(device)

      if File.exists?(weights_path)
        model.load_weights!(weights_path)
      else
        puts "Warning: MASt3R weights not found at #{weights_path}"
        puts "Using random initialization (results will be meaningless)"
      end

      model.frames_to_pointcloud(frames, conf_threshold)
    end
  end
end
