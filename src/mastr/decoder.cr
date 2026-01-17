# DPT Decoder for MASt3R
# Dense Prediction Transformer decoder for pointmap/depth output

require "../nn/linear"
require "../nn/layernorm"
require "../autograd/variable"
require "../core/tensor"

module GS
  module MASt3R
    # Convolution-like operation (implemented via unfold + matmul)
    # For simplicity, uses pixel shuffle for upsampling
    class ConvBlock
      getter conv : NN::Linear
      getter norm : NN::LayerNorm?

      def initialize(
        in_channels : Int32,
        out_channels : Int32,
        with_norm : Bool = true,
        device : Tensor::Device = Tensor::Device::GPU
      )
        @conv = NN::Linear.new(in_channels, out_channels, device: device)
        @norm = with_norm ? NN::LayerNorm.new(out_channels, device: device) : nil
      end

      def forward(x : Autograd::Variable) : Autograd::Variable
        result = @conv.forward(x)
        if n = @norm
          result = n.forward(result)
        end
        gelu(result)
      end

      def parameters : Array(Autograd::Variable)
        params = @conv.parameters
        if n = @norm
          params += n.parameters
        end
        params
      end

      private def gelu(x : Autograd::Variable) : Autograd::Variable
        x_data = x.data.on_cpu? ? x.data : x.data.to_cpu
        x_d = x_data.cpu_data.not_nil!

        result = Tensor.new(x.data.shape, x.data.dtype, Tensor::Device::CPU)
        r_d = result.cpu_data.not_nil!

        sqrt_2_pi = Math.sqrt(2.0 / Math::PI).to_f32
        x.data.numel.times do |i|
          v = x_d[i]
          r_d[i] = 0.5_f32 * v * (1.0_f32 + Math.tanh(sqrt_2_pi * (v + 0.044715_f32 * v * v * v)))
        end

        result = result.to_gpu if x.data.on_gpu?
        Autograd::Variable.new(result, x.requires_grad?)
      end
    end

    # Feature fusion module (reassemble)
    class Reassemble
      getter proj : NN::Linear

      def initialize(
        in_features : Int32,
        out_features : Int32,
        device : Tensor::Device = Tensor::Device::GPU
      )
        @proj = NN::Linear.new(in_features, out_features, device: device)
      end

      # Reassemble tokens back to spatial features
      # x: [batch, num_tokens, embed_dim]
      # Returns: [batch, height, width, out_features]
      def forward(x : Autograd::Variable, height : Int32, width : Int32, skip_cls : Bool = true) : Autograd::Variable
        x_data = x.data.on_cpu? ? x.data : x.data.to_cpu
        batch = x_data.shape[0]
        seq_len = x_data.shape[1]
        embed_dim = x_data.shape[2]

        # Skip CLS token if present
        start_idx = skip_cls ? 1 : 0
        num_patches = seq_len - start_idx

        # Check spatial dimensions match
        expected_patches = height * width
        raise "Patch count mismatch: #{num_patches} vs #{expected_patches}" unless num_patches == expected_patches

        # Extract patch tokens (skip CLS)
        tokens = extract_tokens(x, start_idx, num_patches, batch, embed_dim)

        # Project
        proj_out = @proj.forward(tokens)
        out_features = proj_out.data.shape[-1]

        # Reshape to spatial: [batch, h*w, features] -> [batch, h, w, features]
        reshape_to_spatial(proj_out, batch, height, width, out_features)
      end

      def parameters : Array(Autograd::Variable)
        @proj.parameters
      end

      private def extract_tokens(x : Autograd::Variable, start : Int32, count : Int32, batch : Int32, dim : Int32) : Autograd::Variable
        x_data = x.data.on_cpu? ? x.data : x.data.to_cpu
        x_d = x_data.cpu_data.not_nil!

        result = Tensor.new(batch, count, dim, device: Tensor::Device::CPU)
        r_d = result.cpu_data.not_nil!

        seq_len = x_data.shape[1]

        batch.times do |b|
          count.times do |i|
            dim.times do |d|
              src_idx = b * seq_len * dim + (start + i) * dim + d
              dst_idx = b * count * dim + i * dim + d
              r_d[dst_idx] = x_d[src_idx]
            end
          end
        end

        result = result.to_gpu if x.data.on_gpu?
        Autograd::Variable.new(result, x.requires_grad?)
      end

      private def reshape_to_spatial(x : Autograd::Variable, batch : Int32, h : Int32, w : Int32, c : Int32) : Autograd::Variable
        x_data = x.data.on_cpu? ? x.data : x.data.to_cpu
        x_d = x_data.cpu_data.not_nil!

        result = Tensor.new(batch, h, w, c, device: Tensor::Device::CPU)
        r_d = result.cpu_data.not_nil!

        batch.times do |b|
          h.times do |i|
            w.times do |j|
              c.times do |k|
                src_idx = b * h * w * c + (i * w + j) * c + k
                dst_idx = b * h * w * c + i * w * c + j * c + k
                r_d[dst_idx] = x_d[src_idx]
              end
            end
          end
        end

        result = result.to_gpu if x.data.on_gpu?
        Autograd::Variable.new(result, x.requires_grad?)
      end
    end

    # Feature fusion block
    class FusionBlock
      getter resample : ConvBlock
      getter fuse : ConvBlock

      def initialize(features : Int32, device : Tensor::Device = Tensor::Device::GPU)
        @resample = ConvBlock.new(features, features, device: device)
        @fuse = ConvBlock.new(features * 2, features, device: device)
      end

      # Fuse two feature maps (with upsampling of lower resolution)
      def forward(x1 : Autograd::Variable, x2 : Autograd::Variable) : Autograd::Variable
        # Upsample x1 to match x2 (simplified: just interpolate)
        x1_up = upsample_2x(x1)

        # Concatenate along channel dimension
        concat = concat_features(x1_up, x2)

        # Fuse
        @fuse.forward(concat)
      end

      def parameters : Array(Autograd::Variable)
        @resample.parameters + @fuse.parameters
      end

      # Simple 2x nearest-neighbor upsample
      private def upsample_2x(x : Autograd::Variable) : Autograd::Variable
        x_data = x.data.on_cpu? ? x.data : x.data.to_cpu
        x_d = x_data.cpu_data.not_nil!

        batch = x_data.shape[0]
        h = x_data.shape[1]
        w = x_data.shape[2]
        c = x_data.shape[3]

        result = Tensor.new(batch, h * 2, w * 2, c, device: Tensor::Device::CPU)
        r_d = result.cpu_data.not_nil!

        batch.times do |b|
          h.times do |i|
            w.times do |j|
              c.times do |k|
                val = x_d[b * h * w * c + i * w * c + j * c + k]
                # Fill 2x2 block
                2.times do |di|
                  2.times do |dj|
                    dst_i = i * 2 + di
                    dst_j = j * 2 + dj
                    r_d[b * h * 2 * w * 2 * c + dst_i * w * 2 * c + dst_j * c + k] = val
                  end
                end
              end
            end
          end
        end

        result = result.to_gpu if x.data.on_gpu?
        Autograd::Variable.new(result, x.requires_grad?)
      end

      private def concat_features(x1 : Autograd::Variable, x2 : Autograd::Variable) : Autograd::Variable
        x1_data = x1.data.on_cpu? ? x1.data : x1.data.to_cpu
        x2_data = x2.data.on_cpu? ? x2.data : x2.data.to_cpu

        batch = x1_data.shape[0]
        h = x1_data.shape[1]
        w = x1_data.shape[2]
        c1 = x1_data.shape[3]
        c2 = x2_data.shape[3]

        result = Tensor.new(batch, h, w, c1 + c2, device: Tensor::Device::CPU)

        x1_d = x1_data.cpu_data.not_nil!
        x2_d = x2_data.cpu_data.not_nil!
        r_d = result.cpu_data.not_nil!

        batch.times do |b|
          h.times do |i|
            w.times do |j|
              # Copy from x1
              c1.times do |k|
                src_idx = b * h * w * c1 + i * w * c1 + j * c1 + k
                dst_idx = b * h * w * (c1 + c2) + i * w * (c1 + c2) + j * (c1 + c2) + k
                r_d[dst_idx] = x1_d[src_idx]
              end
              # Copy from x2
              c2.times do |k|
                src_idx = b * h * w * c2 + i * w * c2 + j * c2 + k
                dst_idx = b * h * w * (c1 + c2) + i * w * (c1 + c2) + j * (c1 + c2) + c1 + k
                r_d[dst_idx] = x2_d[src_idx]
              end
            end
          end
        end

        result = result.to_gpu if x1.data.on_gpu?
        Autograd::Variable.new(result, x1.requires_grad? || x2.requires_grad?)
      end
    end

    # DPT Decoder configuration
    struct DecoderConfig
      property embed_dim : Int32
      property features : Int32
      property out_channels : Int32
      property img_size : Int32
      property patch_size : Int32

      def initialize(
        @embed_dim : Int32 = 1024,
        @features : Int32 = 256,
        @out_channels : Int32 = 3,  # xyz pointmap
        @img_size : Int32 = 512,
        @patch_size : Int32 = 16
      )
      end
    end

    # DPT Decoder
    class Decoder
      getter config : DecoderConfig
      getter reassemble : Reassemble
      getter fusion1 : FusionBlock
      getter fusion2 : FusionBlock
      getter head : NN::Linear

      def initialize(@config : DecoderConfig, device : Tensor::Device = Tensor::Device::GPU)
        # Reassemble encoder features to spatial
        @reassemble = Reassemble.new(@config.embed_dim, @config.features, device: device)

        # Feature fusion blocks
        @fusion1 = FusionBlock.new(@config.features, device: device)
        @fusion2 = FusionBlock.new(@config.features, device: device)

        # Output head: features -> xyz
        @head = NN::Linear.new(@config.features, @config.out_channels, device: device)
      end

      # Decode encoder features to pointmap
      # encoder_out: [batch, num_patches + 1, embed_dim]
      # Returns: [batch, height, width, 3] - xyz coordinates
      def forward(encoder_out : Autograd::Variable) : Autograd::Variable
        patch_h = @config.img_size // @config.patch_size
        patch_w = @config.img_size // @config.patch_size

        # Reassemble to spatial features
        features = @reassemble.forward(encoder_out, patch_h, patch_w)

        # For full DPT we'd use multi-scale features from different layers
        # Simplified: just use final features

        # Output head
        @head.forward(features)
      end

      def parameters : Array(Autograd::Variable)
        @reassemble.parameters + @fusion1.parameters + @fusion2.parameters + @head.parameters
      end
    end

    # Pointmap head: converts features to 3D points with confidence
    class PointmapHead
      getter xyz_head : NN::Linear
      getter conf_head : NN::Linear

      def initialize(in_features : Int32, device : Tensor::Device = Tensor::Device::GPU)
        @xyz_head = NN::Linear.new(in_features, 3, device: device)
        @conf_head = NN::Linear.new(in_features, 1, device: device)
      end

      # Forward pass
      # features: [batch, height, width, in_features]
      # Returns: {xyz: [batch, height, width, 3], conf: [batch, height, width, 1]}
      def forward(features : Autograd::Variable) : {Autograd::Variable, Autograd::Variable}
        xyz = @xyz_head.forward(features)
        conf = sigmoid(@conf_head.forward(features))
        {xyz, conf}
      end

      def parameters : Array(Autograd::Variable)
        @xyz_head.parameters + @conf_head.parameters
      end

      private def sigmoid(x : Autograd::Variable) : Autograd::Variable
        x_data = x.data.on_cpu? ? x.data : x.data.to_cpu
        x_d = x_data.cpu_data.not_nil!

        result = Tensor.new(x.data.shape, x.data.dtype, Tensor::Device::CPU)
        r_d = result.cpu_data.not_nil!

        x.data.numel.times { |i| r_d[i] = 1.0_f32 / (1.0_f32 + Math.exp(-x_d[i])) }

        result = result.to_gpu if x.data.on_gpu?
        Autograd::Variable.new(result, x.requires_grad?)
      end
    end
  end
end
