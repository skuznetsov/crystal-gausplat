require "../spec_helper"
require "../../src/mastr/weights"
require "../../src/mastr/encoder_v2"
require "../../src/mastr/decoder_v2"
require "../../src/mastr/weight_loader"

describe GS::MASt3R::WeightLoader do
  model_path = "models/mastr/model.safetensors"

  describe "SafetensorsLoader" do
    it "parses header and lists tensors" do
      pending "Model file not present" unless File.exists?(model_path)

      loader = GS::MASt3R::SafetensorsLoader.new(model_path)

      loader.tensor_names.size.should be > 1000  # MASt3R has ~1009 tensors

      # Check some expected keys exist
      loader.tensor_names.should contain("patch_embed.proj.weight")
      loader.tensor_names.should contain("enc_blocks.0.attn.qkv.weight")
      loader.tensor_names.should contain("dec_blocks.0.attn.qkv.weight")
      loader.tensor_names.should contain("enc_norm.weight")
    end

    it "loads tensor with correct shape" do
      pending "Model file not present" unless File.exists?(model_path)

      loader = GS::MASt3R::SafetensorsLoader.new(model_path)

      # Load patch embedding weight
      info = loader.tensor_info("patch_embed.proj.weight")
      info.should_not be_nil

      if i = info
        # patch_embed.proj.weight should be [1024, 3, 16, 16]
        i.shape.should eq([1024_i64, 3_i64, 16_i64, 16_i64])

        tensor = loader.load_tensor("patch_embed.proj.weight")
        tensor.shape.numel.should eq(1024 * 3 * 16 * 16)
      end
    end

    it "loads encoder block weights" do
      pending "Model file not present" unless File.exists?(model_path)

      loader = GS::MASt3R::SafetensorsLoader.new(model_path)

      # enc_blocks.0.attn.qkv.weight should be [3072, 1024]
      qkv_info = loader.tensor_info("enc_blocks.0.attn.qkv.weight")
      qkv_info.should_not be_nil

      if i = qkv_info
        i.shape.should eq([3072_i64, 1024_i64])
      end

      # Load and verify
      qkv = loader.load_tensor("enc_blocks.0.attn.qkv.weight")
      qkv.shape.numel.should eq(3072 * 1024)
    end

    it "loads decoder block weights" do
      pending "Model file not present" unless File.exists?(model_path)

      loader = GS::MASt3R::SafetensorsLoader.new(model_path)

      # dec_blocks.0.attn.qkv.weight should be [2304, 768]
      qkv_info = loader.tensor_info("dec_blocks.0.attn.qkv.weight")
      qkv_info.should_not be_nil

      if i = qkv_info
        i.shape.should eq([2304_i64, 768_i64])
      end

      # Cross-attention weights
      projq_info = loader.tensor_info("dec_blocks.0.cross_attn.projq.weight")
      projq_info.should_not be_nil

      if i = projq_info
        i.shape.should eq([768_i64, 768_i64])
      end
    end
  end

  describe "WeightLoader" do
    it "initializes and reports tensor count" do
      pending "Model file not present" unless File.exists?(model_path)

      weight_loader = GS::MASt3R::WeightLoader.new(model_path, GS::Tensor::Device::CPU)

      # Should have loaded the safetensors header
      weight_loader.loader.tensor_names.size.should eq(1009)
    end

    # Full encoder loading test (takes more memory/time)
    pending "loads encoder weights" do
      pending "Model file not present" unless File.exists?(model_path)

      config = GS::MASt3R::MASt3REncoderConfigV2.new
      encoder = GS::MASt3R::MASt3REncoderV2.new(config, GS::Tensor::Device::CPU)

      weight_loader = GS::MASt3R::WeightLoader.new(model_path, GS::Tensor::Device::CPU)
      weight_loader.load_encoder!(encoder)

      # Verify some weights were loaded
      # Check first encoder block attention weights
      first_block = encoder.blocks[0]
      first_block.self_attn.q_proj.weight.data.numel.should eq(1024 * 1024)
    end
  end
end
