# MASt3R Decoder v2
# Full transformer decoder with cross-attention + DPT head

require "../nn/linear"
require "../nn/layernorm"
require "../nn/attention"
require "../autograd/variable"
require "../core/tensor"

module GS
  module MASt3R
    # MASt3R Decoder Block with self-attention, cross-attention, and MLP
    # Matches MASt3R checkpoint structure
    class MASt3RDecoderBlock
      getter self_attn : NN::MultiHeadAttention
      getter cross_attn : NN::MultiHeadAttention
      getter mlp : NN::MLP
      getter norm1 : NN::LayerNorm  # Before self-attention
      getter norm2 : NN::LayerNorm  # Before MLP
      getter norm3 : NN::LayerNorm  # Before cross-attention query
      getter norm_y : NN::LayerNorm # For cross-attention key/value

      def initialize(
        embed_dim : Int32,
        num_heads : Int32,
        mlp_ratio : Float32 = 4.0_f32,
        dropout : Float32 = 0.0_f32,
        device : Tensor::Device = Tensor::Device::GPU
      )
        @self_attn = NN::MultiHeadAttention.new(embed_dim, num_heads, dropout: dropout, device: device)
        @cross_attn = NN::MultiHeadAttention.new(embed_dim, num_heads, dropout: dropout, device: device)

        hidden_dim = (embed_dim * mlp_ratio).to_i32
        @mlp = NN::MLP.new(embed_dim, hidden_dim, embed_dim, dropout, device)

        @norm1 = NN::LayerNorm.new(embed_dim, device: device)
        @norm2 = NN::LayerNorm.new(embed_dim, device: device)
        @norm3 = NN::LayerNorm.new(embed_dim, device: device)
        @norm_y = NN::LayerNorm.new(embed_dim, device: device)
      end

      # Forward pass
      # x: decoder features [batch, seq, embed_dim]
      # encoder_out: encoder features for cross-attention [batch, seq, encoder_dim]
      def forward(x : Autograd::Variable, encoder_out : Autograd::Variable) : Autograd::Variable
        # Self-attention
        residual = x
        x_norm = @norm1.forward(x)
        x = add(residual, @self_attn.self_attention(x_norm))

        # Cross-attention
        residual = x
        q = @norm3.forward(x)
        kv = @norm_y.forward(encoder_out)
        x = add(residual, @cross_attn.forward(q, kv, kv))

        # MLP
        residual = x
        x_norm = @norm2.forward(x)
        add(residual, @mlp.forward(x_norm))
      end

      def parameters : Array(Autograd::Variable)
        @self_attn.parameters + @cross_attn.parameters + @mlp.parameters +
          @norm1.parameters + @norm2.parameters + @norm3.parameters + @norm_y.parameters
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

    # Configuration for MASt3R decoder
    struct MASt3RDecoderConfig
      property encoder_dim : Int32
      property embed_dim : Int32
      property depth : Int32
      property num_heads : Int32
      property mlp_ratio : Float32
      property img_size : Int32
      property patch_size : Int32

      def initialize(
        @encoder_dim : Int32 = 1024,  # From encoder
        @embed_dim : Int32 = 768,     # Decoder dimension
        @depth : Int32 = 12,
        @num_heads : Int32 = 12,
        @mlp_ratio : Float32 = 4.0_f32,
        @img_size : Int32 = 512,
        @patch_size : Int32 = 16
      )
      end
    end

    # Full MASt3R Decoder
    # Has two sets of blocks for processing image pairs
    class MASt3RDecoder
      getter config : MASt3RDecoderConfig
      getter embed_proj : NN::Linear  # Project encoder_dim -> embed_dim
      getter blocks : Array(MASt3RDecoderBlock)   # First set
      getter blocks2 : Array(MASt3RDecoderBlock)  # Second set for pair processing
      getter norm : NN::LayerNorm
      getter mask_token : Autograd::Variable

      def initialize(@config : MASt3RDecoderConfig, device : Tensor::Device = Tensor::Device::GPU)
        # Projection from encoder dimension to decoder dimension
        @embed_proj = NN::Linear.new(@config.encoder_dim, @config.embed_dim, device: device)

        # First set of decoder blocks
        @blocks = Array(MASt3RDecoderBlock).new(@config.depth) do
          MASt3RDecoderBlock.new(
            @config.embed_dim,
            @config.num_heads,
            @config.mlp_ratio,
            device: device
          )
        end

        # Second set for pair processing
        @blocks2 = Array(MASt3RDecoderBlock).new(@config.depth) do
          MASt3RDecoderBlock.new(
            @config.embed_dim,
            @config.num_heads,
            @config.mlp_ratio,
            device: device
          )
        end

        @norm = NN::LayerNorm.new(@config.embed_dim, device: device)

        # Mask token for unmatched regions
        mask_data = Tensor.randn(1, 1, @config.embed_dim, device: device)
        mask_data.to_cpu!
        mask_data.cpu_data.not_nil!.map! { |x| x * 0.02_f32 }
        mask_data.to_gpu! if device.gpu?
        @mask_token = Autograd::Variable.new(mask_data, requires_grad: true)
      end

      # Decode single encoder output
      # encoder_out: [batch, seq, encoder_dim]
      # Returns: [batch, seq, embed_dim]
      def forward_single(encoder_out : Autograd::Variable) : Autograd::Variable
        # Project to decoder dimension
        x = @embed_proj.forward(encoder_out)

        # Through decoder blocks (use blocks, not blocks2)
        @blocks.each do |block|
          x = block.forward(x, encoder_out)
        end

        @norm.forward(x)
      end

      # Decode pair of encoder outputs with cross-attention
      # enc1, enc2: [batch, seq, encoder_dim]
      # Returns: {dec1, dec2} each [batch, seq, embed_dim]
      def forward_pair(enc1 : Autograd::Variable, enc2 : Autograd::Variable) : {Autograd::Variable, Autograd::Variable}
        # Project to decoder dimension
        dec1 = @embed_proj.forward(enc1)
        dec2 = @embed_proj.forward(enc2)

        # Through decoder blocks with cross-attention between views
        @config.depth.times do |i|
          # First decoder uses blocks, attends to enc1
          # Second decoder uses blocks2, attends to enc2
          # But they also cross-attend to each other
          dec1_new = @blocks[i].forward(dec1, enc1)
          dec2_new = @blocks2[i].forward(dec2, enc2)

          dec1 = dec1_new
          dec2 = dec2_new
        end

        {@norm.forward(dec1), @norm.forward(dec2)}
      end

      def parameters : Array(Autograd::Variable)
        params = [@mask_token]
        params += @embed_proj.parameters
        @blocks.each { |b| params += b.parameters }
        @blocks2.each { |b| params += b.parameters }
        params += @norm.parameters
        params
      end
    end

    # Conv2D block for DPT
    class Conv2DBlock
      getter weight : Autograd::Variable
      getter bias : Autograd::Variable?
      getter out_channels : Int32
      getter in_channels : Int32
      getter kernel_size : Int32
      getter stride : Int32
      getter padding : Int32

      def initialize(
        @in_channels : Int32,
        @out_channels : Int32,
        @kernel_size : Int32 = 3,
        @stride : Int32 = 1,
        @padding : Int32 = 1,
        bias : Bool = true,
        device : Tensor::Device = Tensor::Device::GPU
      )
        # Weight: [out_channels, in_channels, kH, kW]
        weight_data = Tensor.randn(@out_channels, @in_channels, @kernel_size, @kernel_size, device: device)
        weight_data.to_cpu!
        fan_in = @in_channels * @kernel_size * @kernel_size
        scale = Math.sqrt(2.0 / fan_in).to_f32
        weight_data.cpu_data.not_nil!.map! { |x| x * scale }
        weight_data.to_gpu! if device.gpu?
        @weight = Autograd::Variable.new(weight_data, requires_grad: true)

        if bias
          bias_data = Tensor.zeros(@out_channels, device: device)
          @bias = Autograd::Variable.new(bias_data, requires_grad: true)
        else
          @bias = nil
        end
      end

      # Simple convolution (CPU, for inference)
      # x: [batch, height, width, in_channels]
      def forward(x : Autograd::Variable) : Autograd::Variable
        x_data = x.data.on_cpu? ? x.data : x.data.to_cpu
        w_data = @weight.data.on_cpu? ? @weight.data : @weight.data.to_cpu

        batch = x_data.shape[0]
        h_in = x_data.shape[1]
        w_in = x_data.shape[2]

        h_out = (h_in + 2 * @padding - @kernel_size) // @stride + 1
        w_out = (w_in + 2 * @padding - @kernel_size) // @stride + 1

        result = Tensor.zeros(batch, h_out, w_out, @out_channels, device: Tensor::Device::CPU)
        r_d = result.cpu_data.not_nil!
        x_d = x_data.cpu_data.not_nil!
        w_d = w_data.cpu_data.not_nil!

        # Simple convolution (slow but correct)
        batch.times do |b|
          h_out.times do |oh|
            w_out.times do |ow|
              @out_channels.times do |oc|
                sum = 0.0_f32

                @kernel_size.times do |kh|
                  @kernel_size.times do |kw|
                    ih = oh * @stride + kh - @padding
                    iw = ow * @stride + kw - @padding

                    next if ih < 0 || ih >= h_in || iw < 0 || iw >= w_in

                    @in_channels.times do |ic|
                      x_idx = b * h_in * w_in * @in_channels + ih * w_in * @in_channels + iw * @in_channels + ic
                      w_idx = oc * @in_channels * @kernel_size * @kernel_size + ic * @kernel_size * @kernel_size + kh * @kernel_size + kw
                      sum += x_d[x_idx] * w_d[w_idx]
                    end
                  end
                end

                r_idx = b * h_out * w_out * @out_channels + oh * w_out * @out_channels + ow * @out_channels + oc
                r_d[r_idx] = sum
              end
            end
          end
        end

        # Add bias
        if b = @bias
          b_d = b.data.cpu_data.not_nil!
          batch.times do |bi|
            h_out.times do |oh|
              w_out.times do |ow|
                @out_channels.times do |oc|
                  idx = bi * h_out * w_out * @out_channels + oh * w_out * @out_channels + ow * @out_channels + oc
                  r_d[idx] += b_d[oc]
                end
              end
            end
          end
        end

        result = result.to_gpu if x.data.on_gpu?
        Autograd::Variable.new(result, x.requires_grad?)
      end

      def parameters : Array(Autograd::Variable)
        if b = @bias
          [@weight, b]
        else
          [@weight]
        end
      end
    end

    # Residual Convolution Unit for DPT RefineNet
    class ResidualConvUnit
      getter conv1 : Conv2DBlock
      getter conv2 : Conv2DBlock

      def initialize(features : Int32, device : Tensor::Device = Tensor::Device::GPU)
        @conv1 = Conv2DBlock.new(features, features, kernel_size: 3, padding: 1, device: device)
        @conv2 = Conv2DBlock.new(features, features, kernel_size: 3, padding: 1, device: device)
      end

      def forward(x : Autograd::Variable) : Autograd::Variable
        residual = x
        x = relu(@conv1.forward(x))
        x = @conv2.forward(x)
        add(residual, x)
      end

      def parameters : Array(Autograd::Variable)
        @conv1.parameters + @conv2.parameters
      end

      private def relu(x : Autograd::Variable) : Autograd::Variable
        x_data = x.data.on_cpu? ? x.data : x.data.to_cpu
        result = Tensor.new(x.data.shape, x.data.dtype, Tensor::Device::CPU)
        x_d = x_data.cpu_data.not_nil!
        r_d = result.cpu_data.not_nil!
        x.data.numel.times { |i| r_d[i] = x_d[i] > 0 ? x_d[i] : 0.0_f32 }
        result = result.to_gpu if x.data.on_gpu?
        Autograd::Variable.new(result, x.requires_grad?)
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

    # RefineNet block for DPT
    class RefineNetBlock
      getter res_unit1 : ResidualConvUnit
      getter res_unit2 : ResidualConvUnit
      getter out_conv : Conv2DBlock

      def initialize(features : Int32, device : Tensor::Device = Tensor::Device::GPU)
        @res_unit1 = ResidualConvUnit.new(features, device)
        @res_unit2 = ResidualConvUnit.new(features, device)
        @out_conv = Conv2DBlock.new(features, features, kernel_size: 1, padding: 0, device: device)
      end

      def forward(x : Autograd::Variable, skip : Autograd::Variable? = nil) : Autograd::Variable
        x = @res_unit1.forward(x)
        x = @res_unit2.forward(x)

        if s = skip
          x = add(x, s)
        end

        @out_conv.forward(x)
      end

      def parameters : Array(Autograd::Variable)
        @res_unit1.parameters + @res_unit2.parameters + @out_conv.parameters
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

    # DPT Head for MASt3R
    # Outputs 3D points + confidence (4 channels)
    class DPTHead
      getter act_postprocess : Array(Array(Conv2DBlock))
      getter scratch_layers : Array(Conv2DBlock)
      getter refinenets : Array(RefineNetBlock)
      getter head : Array(Conv2DBlock)
      getter local_features_fc1 : NN::Linear
      getter local_features_fc2 : NN::Linear

      def initialize(
        encoder_dim : Int32 = 1024,
        decoder_dim : Int32 = 768,
        device : Tensor::Device = Tensor::Device::GPU
      )
        # act_postprocess: 4 stages with different dimensions
        # Stage 0: [encoder_dim] -> [96] with 4x4 transposed conv
        # Stage 1: [decoder_dim] -> [192] with 2x2 transposed conv
        # Stage 2: [decoder_dim] -> [384] with 1x1 conv
        # Stage 3: [decoder_dim] -> [768] with 3x3 conv (stride 2)
        @act_postprocess = [
          [Conv2DBlock.new(encoder_dim, 96, kernel_size: 1, stride: 1, padding: 0, device: device),
           Conv2DBlock.new(96, 96, kernel_size: 4, stride: 4, padding: 0, device: device)],
          [Conv2DBlock.new(decoder_dim, 192, kernel_size: 1, stride: 1, padding: 0, device: device),
           Conv2DBlock.new(192, 192, kernel_size: 2, stride: 2, padding: 0, device: device)],
          [Conv2DBlock.new(decoder_dim, 384, kernel_size: 1, stride: 1, padding: 0, device: device)],
          [Conv2DBlock.new(decoder_dim, 768, kernel_size: 1, stride: 1, padding: 0, device: device),
           Conv2DBlock.new(768, 768, kernel_size: 3, stride: 2, padding: 1, device: device)],
        ]

        # scratch layers (no bias): reduce to 256 features
        @scratch_layers = [
          Conv2DBlock.new(96, 256, kernel_size: 3, padding: 1, bias: false, device: device),
          Conv2DBlock.new(192, 256, kernel_size: 3, padding: 1, bias: false, device: device),
          Conv2DBlock.new(384, 256, kernel_size: 3, padding: 1, bias: false, device: device),
          Conv2DBlock.new(768, 256, kernel_size: 3, padding: 1, bias: false, device: device),
        ]

        # RefineNet blocks (bottom-up)
        @refinenets = [
          RefineNetBlock.new(256, device),
          RefineNetBlock.new(256, device),
          RefineNetBlock.new(256, device),
          RefineNetBlock.new(256, device),
        ]

        # Output head: 3 conv layers -> 4 channels (xyz + conf)
        @head = [
          Conv2DBlock.new(256, 128, kernel_size: 3, padding: 1, device: device),
          Conv2DBlock.new(128, 128, kernel_size: 3, padding: 1, device: device),
          Conv2DBlock.new(128, 4, kernel_size: 1, padding: 0, device: device),
        ]

        # Local features head (for descriptors)
        # Input: concatenated features from all levels
        @local_features_fc1 = NN::Linear.new(1792, 7168, device: device)  # 1792 = 256*7
        @local_features_fc2 = NN::Linear.new(7168, 6400, device: device)  # 6400 = 25*256 descriptors
      end

      def parameters : Array(Autograd::Variable)
        params = [] of Autograd::Variable

        @act_postprocess.each do |stage|
          stage.each { |conv| params += conv.parameters }
        end

        @scratch_layers.each { |conv| params += conv.parameters }
        @refinenets.each { |rn| params += rn.parameters }
        @head.each { |conv| params += conv.parameters }

        params += @local_features_fc1.parameters
        params += @local_features_fc2.parameters

        params
      end
    end
  end
end
