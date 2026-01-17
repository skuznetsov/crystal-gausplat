# MASt3R Encoder - ViT-based backbone
# Based on CroCo/DUSt3R architecture

require "../nn/linear"
require "../nn/layernorm"
require "../nn/attention"
require "../nn/vit"
require "../autograd/variable"
require "../core/tensor"
require "./weights"

module GS
  module MASt3R
    # MASt3R encoder configuration
    struct EncoderConfig
      property img_size : Int32
      property patch_size : Int32
      property in_channels : Int32
      property embed_dim : Int32
      property depth : Int32
      property num_heads : Int32
      property mlp_ratio : Float32
      property dropout : Float32

      def initialize(
        @img_size : Int32 = 512,
        @patch_size : Int32 = 16,
        @in_channels : Int32 = 3,
        @embed_dim : Int32 = 1024,  # ViT-Large
        @depth : Int32 = 24,         # ViT-Large
        @num_heads : Int32 = 16,     # ViT-Large
        @mlp_ratio : Float32 = 4.0_f32,
        @dropout : Float32 = 0.0_f32
      )
      end

      # Preset configurations
      def self.vit_large : EncoderConfig
        new(embed_dim: 1024, depth: 24, num_heads: 16)
      end

      def self.vit_base : EncoderConfig
        new(embed_dim: 768, depth: 12, num_heads: 12)
      end

      def self.vit_small : EncoderConfig
        new(embed_dim: 384, depth: 12, num_heads: 6)
      end
    end

    # MASt3R encoder block with optional cross-attention
    class MASt3REncoderBlock
      getter self_attn : NN::MultiHeadAttention
      getter cross_attn : NN::MultiHeadAttention?
      getter mlp : NN::MLP
      getter norm1 : NN::LayerNorm
      getter norm2 : NN::LayerNorm
      getter norm3 : NN::LayerNorm?

      def initialize(
        embed_dim : Int32,
        num_heads : Int32,
        mlp_ratio : Float32 = 4.0_f32,
        with_cross_attn : Bool = false,
        dropout : Float32 = 0.0_f32,
        device : Tensor::Device = Tensor::Device::GPU
      )
        @self_attn = NN::MultiHeadAttention.new(embed_dim, num_heads, dropout: dropout, device: device)
        @mlp = NN::MLP.new(embed_dim, (embed_dim * mlp_ratio).to_i32, embed_dim, dropout, device)
        @norm1 = NN::LayerNorm.new(embed_dim, device: device)
        @norm2 = NN::LayerNorm.new(embed_dim, device: device)

        if with_cross_attn
          @cross_attn = NN::MultiHeadAttention.new(embed_dim, num_heads, dropout: dropout, device: device)
          @norm3 = NN::LayerNorm.new(embed_dim, device: device)
        else
          @cross_attn = nil
          @norm3 = nil
        end
      end

      # Forward with optional cross-attention to another view
      def forward(x : Autograd::Variable, other : Autograd::Variable? = nil) : Autograd::Variable
        # Self-attention
        residual = x
        x = @norm1.forward(x)
        x = @self_attn.self_attention(x)
        x = add(residual, x)

        # Cross-attention (if enabled and other provided)
        if (ca = @cross_attn) && (n3 = @norm3) && other
          residual = x
          x = n3.forward(x)
          x = ca.forward(x, other, other)
          x = add(residual, x)
        end

        # MLP
        residual = x
        x = @norm2.forward(x)
        x = @mlp.forward(x)
        add(residual, x)
      end

      def parameters : Array(Autograd::Variable)
        params = @self_attn.parameters + @mlp.parameters + @norm1.parameters + @norm2.parameters
        if ca = @cross_attn
          params += ca.parameters
        end
        if n3 = @norm3
          params += n3.parameters
        end
        params
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

    # MASt3R encoder: processes image pair with shared ViT + cross-attention
    class Encoder
      getter config : EncoderConfig
      getter patch_embed : NN::PatchEmbedding
      getter blocks : Array(MASt3REncoderBlock)
      getter norm : NN::LayerNorm

      # Learnable embeddings
      getter cls_token : Autograd::Variable
      getter pos_embed : Autograd::Variable

      def initialize(@config : EncoderConfig, device : Tensor::Device = Tensor::Device::GPU)
        @patch_embed = NN::PatchEmbedding.new(
          @config.img_size,
          @config.patch_size,
          @config.in_channels,
          @config.embed_dim,
          device
        )

        num_patches = @patch_embed.num_patches

        # CLS token
        cls_data = Tensor.randn(1, 1, @config.embed_dim, device: device)
        cls_data.to_cpu!
        cls_data.cpu_data.not_nil!.map! { |x| x * 0.02_f32 }
        cls_data.to_gpu! if device.gpu?
        @cls_token = Autograd::Variable.new(cls_data, requires_grad: true)

        # Position embeddings
        pos_data = Tensor.randn(1, num_patches + 1, @config.embed_dim, device: device)
        pos_data.to_cpu!
        pos_data.cpu_data.not_nil!.map! { |x| x * 0.02_f32 }
        pos_data.to_gpu! if device.gpu?
        @pos_embed = Autograd::Variable.new(pos_data, requires_grad: true)

        # Encoder blocks (cross-attention enabled for all blocks)
        @blocks = Array(MASt3REncoderBlock).new(@config.depth) do
          MASt3REncoderBlock.new(
            @config.embed_dim,
            @config.num_heads,
            @config.mlp_ratio,
            with_cross_attn: true,
            dropout: @config.dropout,
            device: device
          )
        end

        @norm = NN::LayerNorm.new(@config.embed_dim, device: device)
      end

      # Encode single image
      # x: [batch, channels, height, width]
      # Returns: [batch, num_patches + 1, embed_dim]
      def forward_single(x : Autograd::Variable) : Autograd::Variable
        batch = x.data.shape[0]

        # Patch embedding
        patches = @patch_embed.forward(x)

        # Add CLS token and position embedding
        patches = prepend_cls_token(patches, batch)
        patches = add_pos_embed(patches)

        # Encoder blocks (no cross-attention for single image)
        @blocks.each do |block|
          patches = block.forward(patches)
        end

        @norm.forward(patches)
      end

      # Encode image pair with cross-attention
      # x1, x2: [batch, channels, height, width]
      # Returns: tuple of [batch, num_patches + 1, embed_dim] for each image
      def forward_pair(x1 : Autograd::Variable, x2 : Autograd::Variable) : {Autograd::Variable, Autograd::Variable}
        batch = x1.data.shape[0]

        # Patch embedding for both
        p1 = @patch_embed.forward(x1)
        p2 = @patch_embed.forward(x2)

        # Add CLS token and position embedding
        p1 = prepend_cls_token(p1, batch)
        p2 = prepend_cls_token(p2, batch)
        p1 = add_pos_embed(p1)
        p2 = add_pos_embed(p2)

        # Encoder blocks with cross-attention
        @blocks.each do |block|
          p1_new = block.forward(p1, p2)
          p2_new = block.forward(p2, p1)
          p1 = p1_new
          p2 = p2_new
        end

        {@norm.forward(p1), @norm.forward(p2)}
      end

      # Load weights from safetensors
      def load_weights!(loader : SafetensorsLoader, prefix : String = "encoder")
        # This would map safetensors keys to model parameters
        # Implementation depends on exact key naming in weights file
        # For now, this is a placeholder
        raise "Weight loading not yet implemented - need to analyze MASt3R weight format"
      end

      def parameters : Array(Autograd::Variable)
        params = [@cls_token, @pos_embed]
        params += @patch_embed.parameters
        @blocks.each { |b| params += b.parameters }
        params += @norm.parameters
        params
      end

      private def prepend_cls_token(x : Autograd::Variable, batch : Int32) : Autograd::Variable
        x_data = x.data.on_cpu? ? x.data : x.data.to_cpu
        cls_data = @cls_token.data.on_cpu? ? @cls_token.data : @cls_token.data.to_cpu

        num_patches = x_data.shape[1]
        embed_dim = x_data.shape[2]

        result = Tensor.new(batch, num_patches + 1, embed_dim, device: Tensor::Device::CPU)
        x_d = x_data.cpu_data.not_nil!
        cls_d = cls_data.cpu_data.not_nil!
        r_d = result.cpu_data.not_nil!

        batch.times do |b|
          # CLS token (first position)
          embed_dim.times { |e| r_d[b * (num_patches + 1) * embed_dim + e] = cls_d[e] }

          # Patches
          num_patches.times do |p|
            embed_dim.times do |e|
              src_idx = b * num_patches * embed_dim + p * embed_dim + e
              dst_idx = b * (num_patches + 1) * embed_dim + (p + 1) * embed_dim + e
              r_d[dst_idx] = x_d[src_idx]
            end
          end
        end

        result = result.to_gpu if x.data.on_gpu?
        Autograd::Variable.new(result, x.requires_grad?)
      end

      private def add_pos_embed(x : Autograd::Variable) : Autograd::Variable
        x_data = x.data.on_cpu? ? x.data : x.data.to_cpu
        pos_data = @pos_embed.data.on_cpu? ? @pos_embed.data : @pos_embed.data.to_cpu

        result = Tensor.new(x.data.shape, x.data.dtype, Tensor::Device::CPU)
        x_d = x_data.cpu_data.not_nil!
        p_d = pos_data.cpu_data.not_nil!
        r_d = result.cpu_data.not_nil!

        batch = x_data.shape[0]
        seq_len = x_data.shape[1]
        embed_dim = x_data.shape[2]

        batch.times do |b|
          seq_len.times do |s|
            embed_dim.times do |e|
              idx = b * seq_len * embed_dim + s * embed_dim + e
              p_idx = s * embed_dim + e
              r_d[idx] = x_d[idx] + p_d[p_idx]
            end
          end
        end

        result = result.to_gpu if x.data.on_gpu?
        Autograd::Variable.new(result, x.requires_grad?)
      end
    end
  end
end
