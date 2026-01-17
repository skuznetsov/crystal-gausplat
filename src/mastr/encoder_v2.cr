# MASt3R Encoder v2
# Standard ViT encoder WITHOUT cross-attention (cross-attention is in decoder)
# Uses RoPE position embeddings

require "../nn/linear"
require "../nn/layernorm"
require "../nn/attention"
require "../nn/vit"
require "../autograd/variable"
require "../core/tensor"

module GS
  module MASt3R
    # Standard ViT encoder block (no cross-attention)
    class MASt3REncoderBlockV2
      getter self_attn : NN::MultiHeadAttention
      getter mlp : NN::MLP
      getter norm1 : NN::LayerNorm
      getter norm2 : NN::LayerNorm

      def initialize(
        embed_dim : Int32,
        num_heads : Int32,
        mlp_ratio : Float32 = 4.0_f32,
        dropout : Float32 = 0.0_f32,
        device : Tensor::Device = Tensor::Device::GPU
      )
        @self_attn = NN::MultiHeadAttention.new(embed_dim, num_heads, dropout: dropout, device: device)

        hidden_dim = (embed_dim * mlp_ratio).to_i32
        @mlp = NN::MLP.new(embed_dim, hidden_dim, embed_dim, dropout, device)

        @norm1 = NN::LayerNorm.new(embed_dim, device: device)
        @norm2 = NN::LayerNorm.new(embed_dim, device: device)
      end

      # Forward with pre-norm (like original MASt3R)
      def forward(x : Autograd::Variable) : Autograd::Variable
        # Self-attention with residual
        residual = x
        x_norm = @norm1.forward(x)
        x = add(residual, @self_attn.self_attention(x_norm))

        # MLP with residual
        residual = x
        x_norm = @norm2.forward(x)
        add(residual, @mlp.forward(x_norm))
      end

      def parameters : Array(Autograd::Variable)
        @self_attn.parameters + @mlp.parameters + @norm1.parameters + @norm2.parameters
      end

      private def add(a : Autograd::Variable, b : Autograd::Variable) : Autograd::Variable
        a_data = a.data.on_cpu? ? a.data : a.data.to_cpu
        b_data = b.data.on_cpu? ? b.data : b.data.to_cpu

        result = Tensor.new(a.data.shape, a.data.dtype, Tensor::Device::CPU)
        a_d = a_data.cpu_data.not_nil!
        b_d = b_data.cpu_data.not_nil!
        r_d = result.cpu_data.not_nil!

        a.data.numel.times { |i| r_d[i] = a_d[i] + b_d[i] }

        result = result.to_gpu if a.data.on_gpu?
        Autograd::Variable.new(result, a.requires_grad? || b.requires_grad?)
      end
    end

    # Configuration for MASt3R encoder v2
    struct MASt3REncoderConfigV2
      property img_size : Int32
      property patch_size : Int32
      property in_channels : Int32
      property embed_dim : Int32
      property depth : Int32
      property num_heads : Int32
      property mlp_ratio : Float32

      def initialize(
        @img_size : Int32 = 512,
        @patch_size : Int32 = 16,
        @in_channels : Int32 = 3,
        @embed_dim : Int32 = 1024,  # ViT-Large
        @depth : Int32 = 24,         # ViT-Large
        @num_heads : Int32 = 16,     # ViT-Large
        @mlp_ratio : Float32 = 4.0_f32
      )
      end

      def num_patches : Int32
        (img_size // patch_size) ** 2
      end
    end

    # MASt3R Encoder v2: standard ViT without cross-attention
    # Uses RoPE for position embeddings (computed, not stored)
    class MASt3REncoderV2
      getter config : MASt3REncoderConfigV2
      getter patch_embed : NN::PatchEmbedding
      getter blocks : Array(MASt3REncoderBlockV2)
      getter norm : NN::LayerNorm

      # RoPE frequencies (precomputed)
      @rope_freqs : Tensor?

      def initialize(@config : MASt3REncoderConfigV2, device : Tensor::Device = Tensor::Device::GPU)
        @patch_embed = NN::PatchEmbedding.new(
          @config.img_size,
          @config.patch_size,
          @config.in_channels,
          @config.embed_dim,
          device
        )

        # Encoder blocks (no cross-attention)
        @blocks = Array(MASt3REncoderBlockV2).new(@config.depth) do
          MASt3REncoderBlockV2.new(
            @config.embed_dim,
            @config.num_heads,
            @config.mlp_ratio,
            device: device
          )
        end

        @norm = NN::LayerNorm.new(@config.embed_dim, device: device)

        # Precompute RoPE frequencies
        @rope_freqs = compute_rope_freqs(100, @config.embed_dim // @config.num_heads, device)
      end

      # Encode single image
      # x: [batch, channels, height, width]
      # Returns: [batch, num_patches, embed_dim]
      def forward(x : Autograd::Variable) : Autograd::Variable
        # Patch embedding: [batch, num_patches, embed_dim]
        patches = @patch_embed.forward(x)

        # Apply RoPE position encoding
        patches = apply_rope(patches)

        # Encoder blocks
        @blocks.each do |block|
          patches = block.forward(patches)
        end

        @norm.forward(patches)
      end

      # Encode image pair (for stereo)
      # Simply encodes each image separately (cross-attention happens in decoder)
      def forward_pair(x1 : Autograd::Variable, x2 : Autograd::Variable) : {Autograd::Variable, Autograd::Variable}
        {forward(x1), forward(x2)}
      end

      def parameters : Array(Autograd::Variable)
        params = @patch_embed.parameters
        @blocks.each { |b| params += b.parameters }
        params += @norm.parameters
        params
      end

      # Compute RoPE frequencies for 2D positions
      # Based on MASt3R's RoPE100 implementation
      private def compute_rope_freqs(base : Int32, dim : Int32, device : Tensor::Device) : Tensor
        half_dim = dim // 2

        freqs = Tensor.new(half_dim, device: Tensor::Device::CPU)
        freqs_d = freqs.cpu_data.not_nil!

        half_dim.times do |i|
          freqs_d[i] = 1.0_f32 / (base.to_f32 ** (2.0_f32 * i / dim))
        end

        device.gpu? ? freqs.to_gpu : freqs
      end

      # Apply Rotary Position Embedding
      # For 2D images, we use separate RoPE for x and y coordinates
      private def apply_rope(x : Autograd::Variable) : Autograd::Variable
        return x unless @rope_freqs  # Skip if not initialized

        x_data = x.data.on_cpu? ? x.data : x.data.to_cpu
        batch = x_data.shape[0]
        seq_len = x_data.shape[1]
        embed_dim = x_data.shape[2]

        # Get spatial dimensions
        h = w = Math.sqrt(seq_len).to_i32
        return x unless h * w == seq_len  # Must be square

        result = Tensor.new(x.data.shape, x.data.dtype, Tensor::Device::CPU)
        x_d = x_data.cpu_data.not_nil!
        r_d = result.cpu_data.not_nil!

        # Convert RoPE freqs to CPU if needed
        rope = @rope_freqs.not_nil!
        freqs_data = rope.on_cpu? ? rope : rope.to_cpu
        freqs_d = freqs_data.cpu_data.not_nil!

        head_dim = embed_dim // @config.num_heads
        half_head_dim = head_dim // 2

        batch.times do |b|
          h.times do |y|
            w.times do |xi|
              pos = y * w + xi

              embed_dim.times do |d|
                src_idx = b * seq_len * embed_dim + pos * embed_dim + d

                # Determine which head and position within head
                head = d // head_dim
                d_in_head = d % head_dim

                if d_in_head < half_head_dim
                  # First half: apply cos/sin based on y coordinate
                  freq_idx = d_in_head
                  if freq_idx < freqs_d.size
                    theta = y.to_f32 * freqs_d[freq_idx]
                    cos_theta = Math.cos(theta).to_f32
                    sin_theta = Math.sin(theta).to_f32

                    # x * cos - x_pair * sin
                    x_val = x_d[src_idx]
                    pair_idx = b * seq_len * embed_dim + pos * embed_dim + d + half_head_dim
                    x_pair = pair_idx < x_d.size ? x_d[pair_idx] : 0.0_f32

                    r_d[src_idx] = x_val * cos_theta - x_pair * sin_theta
                  else
                    r_d[src_idx] = x_d[src_idx]
                  end
                else
                  # Second half: apply sin/cos based on x coordinate
                  freq_idx = d_in_head - half_head_dim
                  if freq_idx < freqs_d.size
                    theta = xi.to_f32 * freqs_d[freq_idx]
                    cos_theta = Math.cos(theta).to_f32
                    sin_theta = Math.sin(theta).to_f32

                    # x_pair * sin + x * cos
                    x_val = x_d[src_idx]
                    pair_idx = b * seq_len * embed_dim + pos * embed_dim + d - half_head_dim
                    x_pair = pair_idx >= 0 ? x_d[pair_idx] : 0.0_f32

                    r_d[src_idx] = x_pair * sin_theta + x_val * cos_theta
                  else
                    r_d[src_idx] = x_d[src_idx]
                  end
                end
              end
            end
          end
        end

        result = result.to_gpu if x.data.on_gpu?
        Autograd::Variable.new(result, x.requires_grad?)
      end
    end
  end
end
