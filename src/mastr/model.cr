# Full MASt3R Model
# Dense stereo reconstruction from image pairs

require "./encoder"
require "./decoder"
require "./weights"
require "../autograd/variable"
require "../core/tensor"

module GS
  module MASt3R
    # MASt3R model configuration
    struct ModelConfig
      property img_size : Int32
      property patch_size : Int32
      property in_channels : Int32
      property embed_dim : Int32
      property encoder_depth : Int32
      property decoder_features : Int32
      property num_heads : Int32
      property mlp_ratio : Float32

      def initialize(
        @img_size : Int32 = 512,
        @patch_size : Int32 = 16,
        @in_channels : Int32 = 3,
        @embed_dim : Int32 = 1024,      # ViT-Large
        @encoder_depth : Int32 = 24,     # ViT-Large
        @decoder_features : Int32 = 256,
        @num_heads : Int32 = 16,
        @mlp_ratio : Float32 = 4.0_f32
      )
      end

      def self.vit_large : ModelConfig
        new(embed_dim: 1024, encoder_depth: 24, num_heads: 16)
      end

      def self.vit_base : ModelConfig
        new(embed_dim: 768, encoder_depth: 12, num_heads: 12)
      end

      def self.vit_small : ModelConfig
        new(embed_dim: 384, encoder_depth: 12, num_heads: 6, decoder_features: 128)
      end
    end

    # Full MASt3R model for dense stereo
    class Model
      getter config : ModelConfig
      getter encoder : Encoder
      getter decoder : Decoder
      getter pointmap_head : PointmapHead

      def initialize(@config : ModelConfig, device : Tensor::Device = Tensor::Device::GPU)
        encoder_config = EncoderConfig.new(
          img_size: @config.img_size,
          patch_size: @config.patch_size,
          in_channels: @config.in_channels,
          embed_dim: @config.embed_dim,
          depth: @config.encoder_depth,
          num_heads: @config.num_heads,
          mlp_ratio: @config.mlp_ratio
        )

        decoder_config = DecoderConfig.new(
          embed_dim: @config.embed_dim,
          features: @config.decoder_features,
          out_channels: @config.decoder_features,  # Output features, not final xyz
          img_size: @config.img_size,
          patch_size: @config.patch_size
        )

        @encoder = Encoder.new(encoder_config, device)
        @decoder = Decoder.new(decoder_config, device)
        @pointmap_head = PointmapHead.new(@config.decoder_features, device)
      end

      # Process a single image (no cross-attention)
      # img: [batch, channels, height, width]
      # Returns: {xyz: [batch, h, w, 3], conf: [batch, h, w, 1]}
      def forward_single(img : Autograd::Variable) : {Autograd::Variable, Autograd::Variable}
        # Encode
        features = @encoder.forward_single(img)

        # Decode
        decoded = @decoder.forward(features)

        # Pointmap head
        @pointmap_head.forward(decoded)
      end

      # Process image pair with cross-attention (full MASt3R)
      # img1, img2: [batch, channels, height, width]
      # Returns: {xyz1, conf1, xyz2, conf2}
      def forward_pair(
        img1 : Autograd::Variable,
        img2 : Autograd::Variable
      ) : {Autograd::Variable, Autograd::Variable, Autograd::Variable, Autograd::Variable}
        # Encode with cross-attention
        enc1, enc2 = @encoder.forward_pair(img1, img2)

        # Decode each
        dec1 = @decoder.forward(enc1)
        dec2 = @decoder.forward(enc2)

        # Pointmap heads
        xyz1, conf1 = @pointmap_head.forward(dec1)
        xyz2, conf2 = @pointmap_head.forward(dec2)

        {xyz1, conf1, xyz2, conf2}
      end

      # Convenience method: get merged point cloud from image pair
      # Returns single point cloud with confidence filtering
      def predict_pointcloud(
        img1 : Autograd::Variable,
        img2 : Autograd::Variable,
        conf_threshold : Float32 = 0.5_f32
      ) : {Tensor, Tensor}
        xyz1, conf1, xyz2, conf2 = forward_pair(img1, img2)

        # Merge point clouds, filter by confidence
        merge_pointclouds(xyz1.data, conf1.data, xyz2.data, conf2.data, conf_threshold)
      end

      # Load weights from safetensors file
      def load_weights!(path : String)
        loader = SafetensorsLoader.new(path)

        # Log available tensors for debugging
        puts "Loading weights from #{path}"
        puts "Found #{loader.tensor_names.size} tensors"

        # Weight mapping would go here
        # This requires knowing the exact key naming convention in the checkpoint
        # For now, this is a placeholder that shows the structure

        # Example mapping (would need to match actual checkpoint):
        # encoder.patch_embed.proj.weight -> @encoder.patch_embed...
        # encoder.blocks.0.attn.qkv.weight -> @encoder.blocks[0].self_attn...
        # etc.

        puts "Warning: Weight loading not yet implemented - need to analyze MASt3R checkpoint format"
      end

      def parameters : Array(Autograd::Variable)
        @encoder.parameters + @decoder.parameters + @pointmap_head.parameters
      end

      # Merge two point clouds with confidence filtering
      private def merge_pointclouds(
        xyz1 : Tensor,
        conf1 : Tensor,
        xyz2 : Tensor,
        conf2 : Tensor,
        threshold : Float32
      ) : {Tensor, Tensor}
        xyz1_cpu = xyz1.on_cpu? ? xyz1 : xyz1.to_cpu
        conf1_cpu = conf1.on_cpu? ? conf1 : conf1.to_cpu
        xyz2_cpu = xyz2.on_cpu? ? xyz2 : xyz2.to_cpu
        conf2_cpu = conf2.on_cpu? ? conf2 : conf2.to_cpu

        batch = xyz1_cpu.shape[0]
        h = xyz1_cpu.shape[1]
        w = xyz1_cpu.shape[2]

        # Count valid points
        valid_count = 0
        conf1_d = conf1_cpu.cpu_data.not_nil!
        conf2_d = conf2_cpu.cpu_data.not_nil!

        total_points = batch * h * w
        total_points.times do |i|
          valid_count += 1 if conf1_d[i] > threshold
          valid_count += 1 if conf2_d[i] > threshold
        end

        # Allocate output
        points = Tensor.new(valid_count, 3, device: Tensor::Device::CPU)
        confidences = Tensor.new(valid_count, device: Tensor::Device::CPU)

        points_d = points.cpu_data.not_nil!
        conf_out_d = confidences.cpu_data.not_nil!
        xyz1_d = xyz1_cpu.cpu_data.not_nil!
        xyz2_d = xyz2_cpu.cpu_data.not_nil!

        idx = 0
        batch.times do |b|
          h.times do |i|
            w.times do |j|
              flat_idx = b * h * w + i * w + j
              xyz_idx = b * h * w * 3 + i * w * 3 + j * 3

              # From first image
              if conf1_d[flat_idx] > threshold
                points_d[idx * 3] = xyz1_d[xyz_idx]
                points_d[idx * 3 + 1] = xyz1_d[xyz_idx + 1]
                points_d[idx * 3 + 2] = xyz1_d[xyz_idx + 2]
                conf_out_d[idx] = conf1_d[flat_idx]
                idx += 1
              end

              # From second image
              if conf2_d[flat_idx] > threshold
                points_d[idx * 3] = xyz2_d[xyz_idx]
                points_d[idx * 3 + 1] = xyz2_d[xyz_idx + 1]
                points_d[idx * 3 + 2] = xyz2_d[xyz_idx + 2]
                conf_out_d[idx] = conf2_d[flat_idx]
                idx += 1
              end
            end
          end
        end

        {points, confidences}
      end
    end

    # Convenience function to create model with default config
    def self.create_model(
      variant : Symbol = :vit_large,
      device : Tensor::Device = Tensor::Device::GPU
    ) : Model
      config = case variant
               when :vit_large then ModelConfig.vit_large
               when :vit_base  then ModelConfig.vit_base
               when :vit_small then ModelConfig.vit_small
               else                 ModelConfig.vit_large
               end

      Model.new(config, device)
    end

    # Load model from checkpoint
    def self.load_model(
      weights_path : String,
      variant : Symbol = :vit_large,
      device : Tensor::Device = Tensor::Device::GPU
    ) : Model
      model = create_model(variant, device)
      model.load_weights!(weights_path)
      model
    end
  end
end
