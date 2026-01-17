# MASt3R Weight Loader
# Maps safetensors keys to Crystal model architecture

require "./weights"
require "./encoder_v2"
require "./decoder_v2"
require "../nn/linear"
require "../nn/layernorm"
require "../nn/attention"
require "../nn/vit"
require "../autograd/variable"

module GS
  module MASt3R
    # Weight loading utilities for MASt3R model
    class WeightLoader
      getter loader : SafetensorsLoader
      getter device : Tensor::Device
      @loaded_count : Int32 = 0
      @total_count : Int32 = 0

      def initialize(path : String, @device : Tensor::Device = Tensor::Device::CPU)
        @loader = SafetensorsLoader.new(path)
        @total_count = @loader.tensor_names.size
        puts "WeightLoader: Found #{@total_count} tensors in #{path}"
      end

      # Load encoder weights
      # Encoder structure:
      #   patch_embed.proj.weight [1024, 3, 16, 16]
      #   patch_embed.proj.bias [1024]
      #   enc_blocks.{0-23}.attn.qkv.weight [3072, 1024]
      #   enc_blocks.{0-23}.attn.qkv.bias [3072]
      #   enc_blocks.{0-23}.attn.proj.weight [1024, 1024]
      #   enc_blocks.{0-23}.attn.proj.bias [1024]
      #   enc_blocks.{0-23}.mlp.fc1.weight [4096, 1024]
      #   enc_blocks.{0-23}.mlp.fc1.bias [4096]
      #   enc_blocks.{0-23}.mlp.fc2.weight [1024, 4096]
      #   enc_blocks.{0-23}.mlp.fc2.bias [1024]
      #   enc_blocks.{0-23}.norm1.weight/bias [1024]
      #   enc_blocks.{0-23}.norm2.weight/bias [1024]
      #   enc_norm.weight/bias [1024]
      def load_encoder!(encoder : MASt3REncoderV2)
        puts "Loading encoder weights..."

        # Patch embedding - conv2d stored as [out_channels, in_channels, kH, kW]
        # Our PatchEmbedding uses Linear internally, need to reshape
        load_patch_embed!(encoder.patch_embed)

        # Encoder blocks
        encoder.blocks.each_with_index do |block, i|
          load_encoder_block!(block, i)
        end

        # Final norm
        load_layernorm!(encoder.norm, "enc_norm")

        puts "Encoder loaded: #{@loaded_count} tensors"
      end

      # Load decoder weights
      # Decoder structure:
      #   decoder_embed.weight [768, 1024]
      #   decoder_embed.bias [768]
      #   dec_blocks.{0-11}.attn.qkv.weight [2304, 768]
      #   dec_blocks.{0-11}.cross_attn.projq/projk/projv.weight [768, 768]
      #   dec_blocks.{0-11}.cross_attn.proj.weight [768, 768]
      #   dec_blocks.{0-11}.mlp.fc1/fc2.weight
      #   dec_blocks.{0-11}.norm1/norm2/norm3/norm_y.weight
      #   dec_blocks2.{0-11}... (second decoder for pair processing)
      #   dec_norm.weight/bias [768]
      def load_decoder!(decoder : MASt3RDecoder)
        puts "Loading decoder weights..."

        # Decoder embedding projection (encoder -> decoder dim)
        load_linear!(decoder.embed_proj, "decoder_embed")

        # Decoder blocks
        decoder.blocks.each_with_index do |block, i|
          load_decoder_block!(block, i, "dec_blocks")
        end

        # Second set of decoder blocks (for second image in pair)
        decoder.blocks2.each_with_index do |block, i|
          load_decoder_block!(block, i, "dec_blocks2")
        end

        # Final norm
        load_layernorm!(decoder.norm, "dec_norm")

        puts "Decoder loaded: #{@loaded_count} tensors"
      end

      # Load DPT head weights
      def load_dpt_head!(head : DPTHead, prefix : String)
        puts "Loading DPT head: #{prefix}..."

        # act_postprocess - conv layers
        4.times do |i|
          load_conv2d!(head.act_postprocess[i][0], "#{prefix}.dpt.act_postprocess.#{i}.0")
          if i != 2  # layer 2 has no second conv
            load_conv2d!(head.act_postprocess[i][1], "#{prefix}.dpt.act_postprocess.#{i}.1")
          end
        end

        # scratch layers
        4.times do |i|
          load_conv2d_no_bias!(head.scratch_layers[i], "#{prefix}.dpt.scratch.layer#{i+1}_rn")
        end

        # refinenets
        4.times do |i|
          load_refinenet!(head.refinenets[i], "#{prefix}.dpt.scratch.refinenet#{i+1}")
        end

        # output head
        load_conv2d!(head.head[0], "#{prefix}.dpt.head.0")
        load_conv2d!(head.head[1], "#{prefix}.dpt.head.2")
        load_conv2d!(head.head[2], "#{prefix}.dpt.head.4")

        # local features head
        load_linear!(head.local_features_fc1, "#{prefix}.head_local_features.fc1")
        load_linear!(head.local_features_fc2, "#{prefix}.head_local_features.fc2")

        puts "DPT head loaded"
      end

      # Tensors not needed for inference (from MAE pretraining)
      EXPECTED_UNUSED = ["mask_token"]

      # Summary
      def summary
        puts "Weight loading complete: #{@loaded_count}/#{@total_count} tensors loaded"

        # Check for unexpected unloaded tensors
        unexpected = [] of String
        @loader.tensor_names.each do |name|
          unless @loader.loaded_names.includes?(name) || EXPECTED_UNUSED.includes?(name)
            unexpected << name
          end
        end

        if unexpected.size > 0
          puts "Warning: #{unexpected.size} unexpected tensors not loaded:"
          unexpected.each { |name| puts "  Unloaded: #{name}" }
        end
      end

      # --- Private loading methods ---

      private def load_patch_embed!(patch_embed : NN::PatchEmbedding)
        # patch_embed.proj is a Conv2D but stored as Linear in our impl
        # Weights: [1024, 3, 16, 16] -> need to flatten to [1024, 768] for Linear
        weight = @loader.load_tensor("patch_embed.proj.weight", Tensor::Device::CPU)
        bias = @loader.load_tensor("patch_embed.proj.bias", Tensor::Device::CPU)

        # Reshape conv weights [out, in, kH, kW] -> [out, in*kH*kW]
        out_channels = weight.shape[0]
        in_features = weight.shape[1] * weight.shape[2] * weight.shape[3]

        weight_flat = reshape_tensor(weight, Shape.new(out_channels, in_features))

        copy_to_variable!(patch_embed.proj.weight, weight_flat)
        if b = patch_embed.proj.bias
          copy_to_variable!(b, bias)
          @loaded_count += 2
        else
          @loaded_count += 1
        end
      end

      private def load_encoder_block!(block : MASt3REncoderBlockV2, idx : Int32)
        prefix = "enc_blocks.#{idx}"

        # Self-attention with combined QKV
        load_attention_qkv!(block.self_attn, "#{prefix}.attn")

        # MLP
        load_mlp!(block.mlp, prefix)

        # LayerNorms
        load_layernorm!(block.norm1, "#{prefix}.norm1")
        load_layernorm!(block.norm2, "#{prefix}.norm2")
      end

      private def load_decoder_block!(block : MASt3RDecoderBlock, idx : Int32, prefix_base : String)
        prefix = "#{prefix_base}.#{idx}"

        # Self-attention with combined QKV
        load_attention_qkv!(block.self_attn, "#{prefix}.attn")

        # Cross-attention with separate Q, K, V projections
        load_cross_attention!(block.cross_attn, "#{prefix}.cross_attn")

        # MLP
        load_mlp!(block.mlp, prefix)

        # LayerNorms (decoder has 4: norm1, norm2, norm3, norm_y)
        load_layernorm!(block.norm1, "#{prefix}.norm1")
        load_layernorm!(block.norm2, "#{prefix}.norm2")
        load_layernorm!(block.norm3, "#{prefix}.norm3")
        load_layernorm!(block.norm_y, "#{prefix}.norm_y")
      end

      # Load attention with combined QKV weights
      private def load_attention_qkv!(attn : NN::MultiHeadAttention, prefix : String)
        embed_dim = attn.embed_dim

        # Combined QKV: [3*embed_dim, embed_dim]
        qkv_weight = @loader.load_tensor("#{prefix}.qkv.weight", Tensor::Device::CPU)
        qkv_bias = @loader.load_tensor("#{prefix}.qkv.bias", Tensor::Device::CPU)

        # Split into Q, K, V
        q_weight, k_weight, v_weight = split_qkv(qkv_weight, embed_dim)
        q_bias, k_bias, v_bias = split_qkv_bias(qkv_bias, embed_dim)

        copy_to_variable!(attn.q_proj.weight, q_weight)
        copy_to_variable!(attn.k_proj.weight, k_weight)
        copy_to_variable!(attn.v_proj.weight, v_weight)

        if b = attn.q_proj.bias
          copy_to_variable!(b, q_bias)
        end
        if b = attn.k_proj.bias
          copy_to_variable!(b, k_bias)
        end
        if b = attn.v_proj.bias
          copy_to_variable!(b, v_bias)
        end

        # Output projection
        proj_weight = @loader.load_tensor("#{prefix}.proj.weight", Tensor::Device::CPU)
        proj_bias = @loader.load_tensor("#{prefix}.proj.bias", Tensor::Device::CPU)

        copy_to_variable!(attn.out_proj.weight, proj_weight)
        if b = attn.out_proj.bias
          copy_to_variable!(b, proj_bias)
        end

        @loaded_count += 4
      end

      # Load cross-attention with separate Q, K, V projections
      private def load_cross_attention!(attn : NN::MultiHeadAttention, prefix : String)
        # Separate projections for cross-attention
        load_linear!(attn.q_proj, "#{prefix}.projq")
        load_linear!(attn.k_proj, "#{prefix}.projk")
        load_linear!(attn.v_proj, "#{prefix}.projv")

        # Output projection
        proj_weight = @loader.load_tensor("#{prefix}.proj.weight", Tensor::Device::CPU)
        proj_bias = @loader.load_tensor("#{prefix}.proj.bias", Tensor::Device::CPU)
        copy_to_variable!(attn.out_proj.weight, proj_weight)
        if b = attn.out_proj.bias
          copy_to_variable!(b, proj_bias)
        end

        @loaded_count += 2
      end

      private def load_mlp!(mlp : NN::MLP, prefix : String)
        load_linear!(mlp.fc1, "#{prefix}.mlp.fc1")
        load_linear!(mlp.fc2, "#{prefix}.mlp.fc2")
      end

      private def load_linear!(linear : NN::Linear, key : String)
        weight = @loader.load_tensor("#{key}.weight", Tensor::Device::CPU)
        bias = @loader.load_tensor("#{key}.bias", Tensor::Device::CPU)

        copy_to_variable!(linear.weight, weight)
        if b = linear.bias
          copy_to_variable!(b, bias)
          @loaded_count += 2
        else
          @loaded_count += 1
        end
      end

      private def load_layernorm!(ln : NN::LayerNorm, key : String)
        weight = @loader.load_tensor("#{key}.weight", Tensor::Device::CPU)
        bias = @loader.load_tensor("#{key}.bias", Tensor::Device::CPU)

        copy_to_variable!(ln.weight, weight)
        copy_to_variable!(ln.bias, bias)

        @loaded_count += 2
      end

      private def load_conv2d!(conv, key : String)
        weight = @loader.load_tensor("#{key}.weight", Tensor::Device::CPU)
        bias = @loader.load_tensor("#{key}.bias", Tensor::Device::CPU)

        copy_to_variable!(conv.weight, weight)
        if b = conv.bias
          copy_to_variable!(b, bias)
          @loaded_count += 2
        else
          @loaded_count += 1
        end
      end

      private def load_conv2d_no_bias!(conv, key : String)
        weight = @loader.load_tensor("#{key}.weight", Tensor::Device::CPU)
        copy_to_variable!(conv.weight, weight)
        @loaded_count += 1
      end

      private def load_refinenet!(refinenet, prefix : String)
        # out_conv
        load_conv2d!(refinenet.out_conv, "#{prefix}.out_conv")

        # resConfUnit1
        load_conv2d!(refinenet.res_unit1.conv1, "#{prefix}.resConfUnit1.conv1")
        load_conv2d!(refinenet.res_unit1.conv2, "#{prefix}.resConfUnit1.conv2")

        # resConfUnit2
        load_conv2d!(refinenet.res_unit2.conv1, "#{prefix}.resConfUnit2.conv1")
        load_conv2d!(refinenet.res_unit2.conv2, "#{prefix}.resConfUnit2.conv2")
      end

      # --- Tensor manipulation utilities ---

      # Split combined QKV weight [3*d, d] into Q, K, V each [d, d]
      private def split_qkv(qkv : Tensor, embed_dim : Int32) : {Tensor, Tensor, Tensor}
        qkv_cpu = qkv.on_cpu? ? qkv : qkv.to_cpu
        data = qkv_cpu.cpu_data.not_nil!

        q = Tensor.new(embed_dim, embed_dim, device: Tensor::Device::CPU)
        k = Tensor.new(embed_dim, embed_dim, device: Tensor::Device::CPU)
        v = Tensor.new(embed_dim, embed_dim, device: Tensor::Device::CPU)

        q_d = q.cpu_data.not_nil!
        k_d = k.cpu_data.not_nil!
        v_d = v.cpu_data.not_nil!

        size = embed_dim * embed_dim
        size.times do |i|
          q_d[i] = data[i]
          k_d[i] = data[size + i]
          v_d[i] = data[2 * size + i]
        end

        {q, k, v}
      end

      private def split_qkv_bias(qkv : Tensor, embed_dim : Int32) : {Tensor, Tensor, Tensor}
        qkv_cpu = qkv.on_cpu? ? qkv : qkv.to_cpu
        data = qkv_cpu.cpu_data.not_nil!

        q = Tensor.new(embed_dim, device: Tensor::Device::CPU)
        k = Tensor.new(embed_dim, device: Tensor::Device::CPU)
        v = Tensor.new(embed_dim, device: Tensor::Device::CPU)

        q_d = q.cpu_data.not_nil!
        k_d = k.cpu_data.not_nil!
        v_d = v.cpu_data.not_nil!

        embed_dim.times do |i|
          q_d[i] = data[i]
          k_d[i] = data[embed_dim + i]
          v_d[i] = data[2 * embed_dim + i]
        end

        {q, k, v}
      end

      private def reshape_tensor(t : Tensor, shape : Shape) : Tensor
        raise "Cannot reshape: numel mismatch" unless t.numel == shape.numel

        t_cpu = t.on_cpu? ? t : t.to_cpu
        result = Tensor.new(shape, t.dtype, Tensor::Device::CPU)

        src = t_cpu.cpu_data.not_nil!
        dst = result.cpu_data.not_nil!
        t.numel.times { |i| dst[i] = src[i] }

        result
      end

      private def copy_to_variable!(var : Autograd::Variable, src : Tensor)
        dst = var.data
        dst_cpu = dst.on_cpu? ? dst : dst.to_cpu
        src_cpu = src.on_cpu? ? src : src.to_cpu

        raise "Shape mismatch: expected #{dst.shape}, got #{src.shape}" unless dst.numel == src.numel

        dst_d = dst_cpu.cpu_data.not_nil!
        src_d = src_cpu.cpu_data.not_nil!

        dst.numel.times { |i| dst_d[i] = src_d[i] }

        # Copy back to GPU if needed
        if dst.on_gpu?
          dst.to_gpu!
        end
      end
    end
  end
end
